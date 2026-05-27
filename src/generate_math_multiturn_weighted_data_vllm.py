#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
import os
import re
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sympy as sp
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from sympy.parsing.latex import parse_latex

    PARSE_LATEX_AVAILABLE = True
except Exception:
    parse_latex = None
    PARSE_LATEX_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert in math reasoning, You should think step by step and give the final answer in the format <answer>ANSWER</answer>.
"""

DEFAULT_FIX_MESSAGE = (
    "Your answer is incorrect. Try again"
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\(boxed|fbox)\s*\{", re.DOTALL)
STRICT_BOXED_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)
ANSWER_TAG_PATTERNS = ("<answer>", "</answer>")

SYM_LOCALS = {"pi": sp.pi, "E": sp.E, "sqrt": sp.sqrt}
CORRECT_REWARD = 2.0
STRICT_FORMAT_REWARD = 0.8
LOOSE_FORMAT_REWARD = 0.2
XML_REWARD_MAX = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-turn weighted SFT data for math datasets with vLLM."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["math", "gsm8k"],
        default="math",
        help="Dataset to use for data synthesis.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset config (GSM8K: main/socratic; MATH: subset).",
    )
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--n", type=int, default=8, help="Trajectories per prompt.")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--sympy_timeout", type=float, default=1.0)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class SympyTimeoutError(RuntimeError):
    pass


@contextmanager
def time_limit(seconds: Optional[float]):
    if seconds is None or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise SympyTimeoutError("sympy timeout")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = (0.0, 0.0)
    try:
        signal.signal(signal.SIGALRM, _handler)
        try:
            old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
        except Exception:
            old_timer = (0.0, 0.0)
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0 or old_timer[1] > 0:
            try:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
            except Exception:
                pass


def load_math_dataset(config: Optional[str], split: str) -> Iterable[Dict[str, Any]]:
    dataset_name = "nlile/hendrycks-MATH-benchmark"
    configs = get_dataset_config_names(dataset_name)
    if config:
        if config not in configs:
            raise ValueError(
                f"Unknown MATH config '{config}'. Available: {', '.join(configs)}"
            )
        LOGGER.info("Loading MATH dataset config: %s", config)
        return load_dataset(dataset_name, config, split=split)
    if "all" in configs:
        LOGGER.info("Loading MATH dataset with config 'all'.")
        return load_dataset(dataset_name, "all", split=split)

    datasets = []
    for cfg in configs:
        LOGGER.info("Loading MATH dataset config: %s", cfg)
        ds = load_dataset(dataset_name, cfg, split=split)
        datasets.append(ds)
    return concatenate_datasets(datasets)


def load_gsm8k_dataset(config: Optional[str], split: str) -> Iterable[Dict[str, Any]]:
    dataset_name = "gsm8k"
    configs = get_dataset_config_names(dataset_name)
    config = config or "main"
    if config not in configs:
        raise ValueError(
            f"Unknown GSM8K config '{config}'. Available: {', '.join(configs)}"
        )
    LOGGER.info("Loading GSM8K dataset config: %s", config)
    return load_dataset(dataset_name, config, split=split)


def load_synthesis_dataset(
    dataset_name: str, dataset_config: Optional[str]
) -> Iterable[Dict[str, Any]]:
    if dataset_name == "gsm8k":
        return load_gsm8k_dataset(dataset_config, split="train")
    return load_math_dataset(dataset_config, split="train")


def extract_answer_block(text: str) -> Optional[str]:
    if text is None:
        return None
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1)
    if "<answer>" in text:
        after = text.split("<answer>", 1)[1]
        if "</answer>" in after:
            return after.split("</answer>", 1)[0]
        return after
    return None


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def extract_strict_boxed_answer(text: str) -> Optional[str]:
    answer_block = extract_answer_block(text)
    if answer_block is None:
        return None
    stripped = answer_block.strip()
    if not stripped:
        return None
    match = STRICT_BOXED_RE.search(stripped)
    if not match:
        return None
    brace_start = stripped.find("{", match.end() - 1)
    if brace_start == -1:
        return None
    content, end_idx = extract_braced_content(stripped, brace_start)
    if content is None:
        return None
    prefix = stripped[:match.start()].strip()
    suffix = stripped[end_idx:].strip()
    if prefix or suffix:
        return None
    content = content.strip()
    return content if content else None


def compute_xml_score(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for pattern in ANSWER_TAG_PATTERNS if pattern in text)
    return XML_REWARD_MAX * hits / len(ANSWER_TAG_PATTERNS)


def analyze_model_output(text: str) -> Dict[str, Any]:
    text = text or ""
    strict_boxed_answer = extract_strict_boxed_answer(text)
    answer_block = extract_answer_block(text)
    answer_block_stripped = answer_block.strip() if answer_block is not None else ""
    boxed_in_answer = (
        extract_boxed_answer(answer_block_stripped) if answer_block_stripped else None
    )
    primary_answer = ""
    if boxed_in_answer:
        primary_answer = boxed_in_answer.strip()
    elif answer_block_stripped:
        primary_answer = answer_block_stripped

    boxed_full = extract_boxed_answer(text) if text else None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not primary_answer:
        if boxed_full:
            primary_answer = boxed_full.strip()
        else:
            primary_answer = lines[-1] if lines else ""

    candidates = _dedupe_keep_order(
        [
            answer_block_stripped,
            boxed_in_answer.strip() if boxed_in_answer else "",
            boxed_full.strip() if boxed_full else "",
            lines[-1] if lines else "",
        ]
    )

    strict_format_ok = strict_boxed_answer is not None
    xml_score = compute_xml_score(text)
    return {
        "primary_answer": primary_answer,
        "candidates": candidates,
        "has_answer_tag": answer_block is not None,
        "strict_format_ok": strict_format_ok,
        "strict_boxed_answer": strict_boxed_answer,
        "xml_score": xml_score,
    }


def compute_turn_reward(
    output: str,
    gold: str,
    sympy_timeout: Optional[float],
) -> Dict[str, Any]:
    analysis = analyze_model_output(output)
    candidates = analysis["candidates"]
    correct = any(
        normalize_and_compare(candidate, gold, sympy_timeout) for candidate in candidates
    )

    strict_format_ok = analysis["strict_format_ok"]
    strict_format_reward = STRICT_FORMAT_REWARD if strict_format_ok else 0.0
    soft_format_ok = (not strict_format_ok) and bool(analysis["primary_answer"])
    loose_format_reward = LOOSE_FORMAT_REWARD if soft_format_ok else 0.0
    xml_reward = analysis["xml_score"]
    correct_reward = CORRECT_REWARD if correct else 0.0

    answer_norm = normalize_text_for_string(analysis["primary_answer"])

    total_reward = correct_reward + strict_format_reward + loose_format_reward + xml_reward
    return {
        "analysis": analysis,
        "correct": correct,
        "answer_norm": answer_norm,
        "reward": total_reward,
        "components": {
            "correctness": correct_reward,
            "strict_format": strict_format_reward,
            "loose_format": loose_format_reward,
            "xml": xml_reward,
            "repeat_penalty": 0.0,
        },
    }


def extract_braced_content(text: str, start_idx: int) -> Tuple[Optional[str], int]:
    if start_idx >= len(text) or text[start_idx] != "{":
        return None, start_idx
    depth = 0
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1 : idx], idx + 1
    return None, start_idx


def extract_boxed_answer(solution: str) -> Optional[str]:
    if not solution:
        return None
    matches = list(BOXED_RE.finditer(solution))
    if not matches:
        return None
    last = matches[-1]
    brace_start = solution.find("{", last.end() - 1)
    if brace_start == -1:
        return None
    content, _ = extract_braced_content(solution, brace_start)
    return content


def extract_answer_field(answer_text: str) -> str:
    text = str(answer_text).strip() if answer_text is not None else ""
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    return text


def extract_ground_truth_answer(example: Dict[str, Any]) -> str:
    if "answer" in example and example["answer"]:
        parsed = extract_answer_field(example["answer"])
        if parsed:
            return parsed
    solution = example.get("solution", "")
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed.strip()
    if solution:
        hash_match = re.findall(r"####\s*(.+)", solution)
        if hash_match:
            return hash_match[-1].strip()
        lines = [line.strip() for line in solution.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return ""


def extract_problem_text(example: Dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return str(example.get("question", "")).strip()
    return str(example.get("problem", "")).strip()


def strip_math_delimiters(text: str) -> str:
    text = text.strip()
    if text.startswith("$") and text.endswith("$"):
        text = text.strip("$")
    if text.startswith("\\(") and text.endswith("\\)"):
        text = text[2:-2]
    if text.startswith("\\[") and text.endswith("\\]"):
        text = text[2:-2]
    return text.strip()


def replace_frac(text: str) -> str:
    for cmd in ("\\frac", "\\dfrac", "\\tfrac"):
        while True:
            idx = text.find(cmd)
            if idx == -1:
                break
            brace_start = text.find("{", idx + len(cmd))
            if brace_start == -1:
                break
            num, end_num = extract_braced_content(text, brace_start)
            if num is None:
                break
            brace_start = text.find("{", end_num)
            if brace_start == -1:
                break
            den, end_den = extract_braced_content(text, brace_start)
            if den is None:
                break
            text = text[:idx] + f"(({num})/({den}))" + text[end_den:]
    return text


def replace_sqrt(text: str) -> str:
    while True:
        idx = text.find("\\sqrt")
        if idx == -1:
            break
        brace_start = text.find("{", idx + len("\\sqrt"))
        if brace_start == -1:
            break
        rad, end_rad = extract_braced_content(text, brace_start)
        if rad is None:
            break
        text = text[:idx] + f"sqrt({rad})" + text[end_rad:]
    return text


def replace_powers(text: str) -> str:
    text = re.sub(r"\^\{([^}]+)\}", r"**(\1)", text)
    text = re.sub(r"\^([A-Za-z0-9]+)", r"**(\1)", text)
    return text


def normalize_text_for_sympy(text: str) -> str:
    text = strip_math_delimiters(text)
    text = text.replace("\u2212", "-")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = text.replace("\\!", "").replace("\\,", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\pi", "pi")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = replace_frac(text)
    text = replace_sqrt(text)
    text = replace_powers(text)
    if "=" in text and "==" not in text:
        text = text.split("=")[-1]
    return text.strip()


def normalize_text_for_string(text: str) -> str:
    text = strip_math_delimiters(text)
    text = text.replace("\u2212", "-")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "", text)
    return text


def to_sympy_expr(text: str, timeout: Optional[float]) -> Optional[sp.Expr]:
    text = text.strip()
    if not text:
        return None
    if PARSE_LATEX_AVAILABLE and "\\" in text:
        try:
            with time_limit(timeout):
                return parse_latex(text)
        except (SympyTimeoutError, Exception):
            pass
    normalized = normalize_text_for_sympy(text)
    if not normalized:
        return None
    try:
        with time_limit(timeout):
            return sp.sympify(normalized, locals=SYM_LOCALS)
    except (SympyTimeoutError, Exception):
        return None


def safe_simplify(expr: sp.Expr, timeout: Optional[float]) -> Optional[sp.Expr]:
    try:
        with time_limit(timeout):
            return sp.simplify(expr)
    except (SympyTimeoutError, Exception):
        return None


def normalize_and_compare(pred: str, gold: str, sympy_timeout: Optional[float]) -> bool:
    pred = pred.strip() if pred else ""
    gold = gold.strip() if gold else ""
    if not pred or not gold:
        return False

    pred_expr = to_sympy_expr(pred, sympy_timeout)
    gold_expr = to_sympy_expr(gold, sympy_timeout)
    if pred_expr is not None and gold_expr is not None:
        try:
            if pred_expr.is_number and gold_expr.is_number:
                diff = safe_simplify(pred_expr - gold_expr, sympy_timeout)
                if diff == 0:
                    return True
            diff = safe_simplify(pred_expr - gold_expr, sympy_timeout)
            if diff == 0:
                return True
            ratio = safe_simplify(pred_expr / gold_expr, sympy_timeout)
            if ratio == 1:
                return True
        except Exception:
            pass

    pred_norm = normalize_text_for_string(pred)
    gold_norm = normalize_text_for_string(gold)
    return pred_norm == gold_norm


def render_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Optional[Any],
    use_chat_template: bool,
) -> str:
    if not use_chat_template:
        raise ValueError("Prompt rendering requires --use_chat_template (fallback disabled).")
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; cannot render prompts.")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_batches(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_size: int,
    progress_bar: Optional[Any] = None,
) -> List[List[str]]:
    outputs: List[List[str]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_outputs = llm.generate(batch, sampling_params, use_tqdm=False)
        batch_count = 0
        for out in batch_outputs:
            if out.outputs:
                outputs.append([o.text for o in out.outputs])
                batch_count += len(out.outputs)
            else:
                outputs.append([""])
                batch_count += 1
        if progress_bar is not None:
            progress_bar.update(batch_count)
    return outputs


def iter_prompt_groups(
    dataset: Iterable[Dict[str, Any]], max_prompts: Optional[int], group_size: int
) -> Iterable[List[Tuple[int, Dict[str, Any]]]]:
    group: List[Tuple[int, Dict[str, Any]]] = []
    count = 0
    for idx, example in enumerate(dataset):
        if max_prompts is not None and count >= max_prompts:
            break
        group.append((idx, example))
        count += 1
        if len(group) >= group_size:
            yield group
            group = []
    if group:
        yield group


def create_llm(args: argparse.Namespace) -> LLM:
    kwargs = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    try:
        sig = inspect.signature(LLM.__init__)
        if "disable_log_stats" in sig.parameters:
            kwargs["disable_log_stats"] = True
        if "disable_log_requests" in sig.parameters:
            kwargs["disable_log_requests"] = True
    except Exception:
        pass
    return LLM(**kwargs)


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.max_turns < 1:
        raise ValueError("--max_turns must be >= 1")

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_DISABLE_LOG_STATS", "1")
    os.environ.setdefault("VLLM_DISABLE_LOG_REQUESTS", "1")
    for name in ("vllm", "vllm.engine", "vllm.worker"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    dataset = load_synthesis_dataset(args.dataset, args.dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    llm = create_llm(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=1,
    )
    sampling_params_turn1 = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=args.n,
    )

    total_prompts_target = None
    if args.max_prompts is not None:
        total_prompts_target = args.max_prompts
    elif hasattr(dataset, "__len__"):
        try:
            total_prompts_target = len(dataset)
        except Exception:
            total_prompts_target = None

    overall_bar = tqdm(
        total=total_prompts_target,
        desc="overall_prompts",
        unit="prompt",
        ascii=True,
        position=1,
        leave=True,
        file=sys.stdout,
    )

    total_prompts = 0
    total_trajectories = 0
    turn_attempts: Dict[int, int] = {}
    turn_correct: Dict[int, int] = {}
    success_within = 0

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for group in iter_prompt_groups(dataset, args.max_prompts, args.batch_size):
            prompts: List[Dict[str, Any]] = []
            for global_idx, example in group:
                problem = extract_problem_text(example, args.dataset)
                if not problem:
                    LOGGER.warning("Skipping empty problem at index %d.", global_idx)
                    continue
                gold = extract_ground_truth_answer(example)
                base_id = example.get("problem_id") or example.get("id") or example.get("idx")
                if base_id is None:
                    base_id = "sample"
                prompt_id = f"{base_id}_{global_idx}"
                prompts.append({"prompt_id": str(prompt_id), "problem": problem, "gold": gold})

            if not prompts:
                continue

            trajectories: List[Dict[str, Any]] = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt["problem"]},
                ]
                prompt["messages"] = messages

            total_prompts += len(prompts)
            overall_bar.update(len(prompts))

            turn1_prompts = [
                render_prompt(prompt["messages"], tokenizer, args.use_chat_template)
                for prompt in prompts
            ]
            turn1_total = len(prompts) * args.n
            turn1_bar = tqdm(
                total=turn1_total,
                desc="turn1",
                unit="traj",
                ascii=True,
                position=0,
                leave=False,
                file=sys.stdout,
            )
            turn1_outputs = generate_batches(
                llm,
                turn1_prompts,
                sampling_params_turn1,
                args.batch_size,
                progress_bar=turn1_bar,
            )
            turn1_bar.close()
            if len(turn1_outputs) != len(prompts):
                raise RuntimeError("Turn1 outputs length mismatch.")

            for prompt, outputs in zip(prompts, turn1_outputs):
                if len(outputs) != args.n:
                    LOGGER.warning(
                        "Prompt %s expected %d outputs, got %d.",
                        prompt["prompt_id"],
                        args.n,
                        len(outputs),
                    )
                    if len(outputs) < args.n:
                        outputs = outputs + [""] * (args.n - len(outputs))
                    else:
                        outputs = outputs[: args.n]
                for traj_idx, output in enumerate(outputs):
                    traj = {
                        "prompt_id": prompt["prompt_id"],
                        "gold": prompt["gold"],
                        "messages": list(prompt["messages"]),
                        "verifier": {},
                        "turn_rewards": {},
                        "reward_details": {},
                        "answer_norms": [],
                        "t_star": None,
                        "done": False,
                        "traj_idx": traj_idx,
                    }
                    traj["messages"].append({"role": "assistant", "content": output})
                    result = compute_turn_reward(
                        output,
                        traj["gold"],
                        args.sympy_timeout,
                    )
                    correct = result["correct"]
                    traj["verifier"]["t1"] = 1 if correct else 0
                    traj["turn_rewards"]["t1"] = result["reward"]
                    traj["reward_details"]["t1"] = result["components"]
                    traj["answer_norms"].append(result["answer_norm"])
                    if correct:
                        traj["t_star"] = 1
                        traj["done"] = True
                        turn_correct[1] = turn_correct.get(1, 0) + 1
                    else:
                        if args.max_turns == 1:
                            traj["done"] = True
                        else:
                            traj["messages"].append(
                                {"role": "user", "content": DEFAULT_FIX_MESSAGE}
                            )
                    trajectories.append(traj)

            total_trajectories += len(trajectories)
            turn_attempts[1] = turn_attempts.get(1, 0) + len(trajectories)

            for turn in range(2, args.max_turns + 1):
                active = [traj for traj in trajectories if not traj["done"]]
                if not active:
                    break
                turn_attempts[turn] = turn_attempts.get(turn, 0) + len(active)
                turn_prompts = [
                    render_prompt(traj["messages"], tokenizer, args.use_chat_template)
                    for traj in active
                ]
                turn_bar = tqdm(
                    total=len(active),
                    desc=f"turn{turn}",
                    unit="traj",
                    ascii=True,
                    position=0,
                    leave=False,
                    file=sys.stdout,
                )
                outputs = generate_batches(
                    llm,
                    turn_prompts,
                    sampling_params,
                    args.batch_size,
                    progress_bar=turn_bar,
                )
                turn_bar.close()
                if len(outputs) != len(active):
                    raise RuntimeError(f"Turn{turn} outputs length mismatch.")

                for traj, output_list in zip(active, outputs):
                    output = output_list[0] if output_list else ""
                    traj["messages"].append({"role": "assistant", "content": output})
                    result = compute_turn_reward(
                        output,
                        traj["gold"],
                        args.sympy_timeout,
                    )
                    correct = result["correct"]
                    traj["verifier"][f"t{turn}"] = 1 if correct else 0
                    traj["turn_rewards"][f"t{turn}"] = result["reward"]
                    traj["reward_details"][f"t{turn}"] = result["components"]
                    traj["answer_norms"].append(result["answer_norm"])
                    if correct:
                        traj["t_star"] = turn
                        traj["done"] = True
                        turn_correct[turn] = turn_correct.get(turn, 0) + 1
                    else:
                        if turn == args.max_turns:
                            traj["done"] = True
                        else:
                            traj["messages"].append(
                                {"role": "user", "content": DEFAULT_FIX_MESSAGE}
                            )
            for traj in trajectories:
                for turn in range(1, args.max_turns + 1):
                    traj["verifier"].setdefault(f"t{turn}", 0)
                    traj["turn_rewards"].setdefault(f"t{turn}", 0.0)
            for traj in trajectories:
                if traj["t_star"] is not None:
                    success_within += 1
                verifier = {k: int(v) for k, v in traj["verifier"].items()}
                record = {
                    "prompt_id": traj["prompt_id"],
                    "messages": traj["messages"],
                    "verifier": verifier,
                    "turn_rewards": traj["turn_rewards"],
                    "reward_details": traj["reward_details"],
                    "t_star": traj["t_star"],
                    "answer_norms": traj["answer_norms"],
                }
                f.write(json.dumps(record) + "\n")

    overall_bar.close()

    success_rate = success_within / total_trajectories if total_trajectories else 0.0
    tqdm.write("Summary:")
    tqdm.write(f"  total prompts processed: {total_prompts}")
    tqdm.write(f"  total trajectories: {total_trajectories}")
    for turn in range(1, args.max_turns + 1):
        attempts = turn_attempts.get(turn, 0)
        correct = turn_correct.get(turn, 0)
        acc = correct / attempts if attempts else 0.0
        if turn == 1:
            label = "turn1 accuracy"
        else:
            label = f"turn{turn} accuracy (conditional on previous wrong)"
        tqdm.write(f"  {label}: {acc:.4f}")
    tqdm.write(f"  overall success rate (<= {args.max_turns} turns): {success_rate:.4f}")


if __name__ == "__main__":
    main()
