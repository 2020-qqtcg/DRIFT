#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from prompting import (
    ACTION_SEP_DEFAULT,
    DEFAULT_ENV_INSTRUCTION,
    DEFAULT_INCORRECT_OBS,
    MultiTurnAnswerEpisode,
    build_initial_messages,
    build_reward_message,
    clone_messages,
    normalize_answer,
    parse_response,
    render_prompt,
    run_actions,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LOGGER = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"The answer is: (.*?)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-turn weighted data for MetaMathQA with vLLM."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="meta-math/MetaMathQA")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--type_prefix", type=str, default="MATH_")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--n", type=int, default=8, help="Trajectories per prompt.")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_actions_per_traj", type=int, default=5)
    parser.add_argument("--max_actions_per_turn", type=int, default=5)
    parser.add_argument("--format_penalty", type=float, default=-0.1)
    parser.add_argument("--instruction_max_tokens", type=int, default=1000)
    parser.add_argument("--action_sep", type=str, default=ACTION_SEP_DEFAULT)
    parser.add_argument("--disable_think", action="store_true")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="full",
        choices=["full", "simple"],
        help=(
            "Prompting mode. "
            "'full' uses the original state/reward/turn prompt. "
            "'simple' uses a minimal prompt; after an incorrect attempt, the next user message is only "
            f"'{DEFAULT_INCORRECT_OBS}'."
        ),
    )
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_metamathqa_dataset(
    dataset_path: str,
    split: str,
    cache_dir: Optional[str],
    type_prefix: str,
    seed: Optional[int],
) -> Iterable[Dict[str, Any]]:
    LOGGER.info("Loading dataset %s (%s)", dataset_path, split)
    dataset = load_dataset(dataset_path, split=split, cache_dir=cache_dir)
    if type_prefix:
        dataset = dataset.filter(
            lambda example: str(example.get("type", "")).startswith(type_prefix)
        )
    if seed is not None:
        try:
            dataset = dataset.shuffle(seed=seed)
        except Exception as exc:
            LOGGER.warning("Shuffle failed: %s", exc)
    return dataset


def extract_question(example: Dict[str, Any]) -> str:
    return str(example.get("query", "")).strip()


def extract_answer(example: Dict[str, Any]) -> str:
    response = str(example.get("response", ""))
    match = ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return ""


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


def process_turn(
    traj: Dict[str, Any],
    raw_output: str,
    turn_idx: int,
    args: argparse.Namespace,
    enable_think: bool,
) -> bool:
    response = ("<think>" + (raw_output or "")) if enable_think else ("<answer>" + (raw_output or ""))
    llm_response, actions = parse_response(
        response, enable_think, args.action_sep, args.max_actions_per_turn
    )
    actions_left_before = traj["actions_left"]

    env_reward = 0.0
    done = False
    info: Dict[str, Any] = {}
    executed_actions: List[str] = []
    format_penalty = 0.0

    if actions and actions_left_before > 0:
        available = min(actions_left_before, len(actions))
        env_reward, done, info, executed_actions = run_actions(
            traj["episode"], actions[:available]
        )
    else:
        format_penalty = args.format_penalty

    traj["actions_left"] = max(args.max_actions_per_traj - traj["episode"].step_num, 0)
    if traj["actions_left"] <= 0 and not done:
        done = True

    for action in executed_actions:
        traj["answer_norms"].append(normalize_answer(action))
    traj["format_penalty"] += format_penalty
    turn_reward = env_reward
    traj["messages"].append({"role": "assistant", "content": llm_response})
    traj["turn_rewards"][f"t{turn_idx}"] = turn_reward
    traj["reward_details"][f"t{turn_idx}"] = {
        "env_reward": env_reward,
        "format_penalty": format_penalty,
        "actions": executed_actions,
        "parsed_actions": actions,
        "info": info,
    }

    correct = bool(info.get("success"))
    traj["verifier"][f"t{turn_idx}"] = 1 if correct else 0
    if correct and traj["t_star"] is None:
        traj["t_star"] = turn_idx
        done = True

    if not done and turn_idx < args.max_turns:
        traj["messages"].append(
            build_reward_message(
                env_reward,
                turn_idx + 1,
                traj["episode"].state,
                traj["actions_left"],
                args.instruction_max_tokens,
                enable_think,
                args.prompt_mode,
            )
        )

    traj["done"] = done or (turn_idx >= args.max_turns)
    return correct


def main() -> None:
    args = parse_args()
    setup_logging()
    enable_think = not args.disable_think

    if args.max_turns < 1:
        raise ValueError("--max_turns must be >= 1")
    if args.max_actions_per_traj < 1:
        raise ValueError("--max_actions_per_traj must be >= 1")

    dataset = load_metamathqa_dataset(
        args.dataset_path,
        args.dataset_split,
        args.cache_dir,
        args.type_prefix,
        args.seed,
    )
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
                question = extract_question(example)
                if not question:
                    LOGGER.warning("Skipping empty question at index %d.", global_idx)
                    continue
                gold = extract_answer(example)
                base_id = example.get("id") or example.get("idx") or example.get("problem_id")
                if base_id is None:
                    base_id = "sample"
                prompt_id = f"{base_id}_{global_idx}"
                messages = build_initial_messages(
                    question,
                    args.max_actions_per_traj,
                    args.instruction_max_tokens,
                    enable_think,
                    args.prompt_mode,
                    DEFAULT_ENV_INSTRUCTION,
                )
                prompts.append(
                    {
                        "prompt_id": str(prompt_id),
                        "question": question,
                        "gold": gold,
                        "messages": messages,
                    }
                )

            if not prompts:
                continue

            total_prompts += len(prompts)
            overall_bar.update(len(prompts))

            turn1_prompts = [
                render_prompt(p["messages"], tokenizer, args.use_chat_template, enable_think)
                for p in prompts
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

            trajectories: List[Dict[str, Any]] = []
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
                    episode = MultiTurnAnswerEpisode(
                        prompt["question"], prompt["gold"], args.max_actions_per_traj
                    )
                    traj = {
                        "prompt_id": prompt["prompt_id"],
                        "gold": prompt["gold"],
                        "messages": clone_messages(prompt["messages"]),
                        "verifier": {},
                        "turn_rewards": {},
                        "reward_details": {},
                        "t_star": None,
                        "done": False,
                        "traj_idx": traj_idx,
                        "episode": episode,
                        "actions_left": args.max_actions_per_traj,
                        "answer_norms": [],
                        "format_penalty": 0.0,
                    }
                    correct = process_turn(traj, output, 1, args, enable_think)
                    if correct:
                        turn_correct[1] = turn_correct.get(1, 0) + 1
                    trajectories.append(traj)

            total_trajectories += len(trajectories)
            turn_attempts[1] = turn_attempts.get(1, 0) + len(trajectories)

            for turn in range(2, args.max_turns + 1):
                active = [traj for traj in trajectories if not traj["done"]]
                if not active:
                    break
                turn_attempts[turn] = turn_attempts.get(turn, 0) + len(active)
                turn_prompts = [
                    render_prompt(
                        traj["messages"], tokenizer, args.use_chat_template, enable_think
                    )
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
                    correct = process_turn(traj, output, turn, args, enable_think)
                    if correct:
                        turn_correct[turn] = turn_correct.get(turn, 0) + 1

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
                    "gold": traj["gold"],
                    "messages": traj["messages"],
                    "verifier": verifier,
                    "turn_rewards": traj["turn_rewards"],
                    "reward_details": traj["reward_details"],
                    "t_star": traj["t_star"],
                    "answer_norms": traj["answer_norms"],
                    "format_penalty": traj["format_penalty"],
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
        label = "turn1 accuracy" if turn == 1 else f"turn{turn} accuracy (conditional)"
        tqdm.write(f"  {label}: {acc:.4f}")
    tqdm.write(f"  overall success rate (<= {args.max_turns} turns): {success_rate:.4f}")


if __name__ == "__main__":
    main()
