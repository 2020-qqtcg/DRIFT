#!/usr/bin/env python3
"""
Dataset loading and preprocessing utilities for eval_math_multiturn_vllm.py.
"""
import hashlib
import json
import logging
import random
import re
import string
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset

LOGGER = logging.getLogger(__name__)

ENV_INSTRUCTION = (
    "You are solving Math problems. Only give the final answer between <answer> and </answer>."
)
MMLU_INSTRUCTION = (
    "You are solving multiple-choice questions. Only give the final answer letter (A-D) "
    "between <answer> and </answer>."
)
GPQA_INSTRUCTION = (
    "You are solving multiple-choice questions. Only give the final answer letter (A-D) "
    "between <answer> and </answer>."
)
MMLU_PRO_INSTRUCTION = (
    "You are solving multiple-choice questions. Only give the final answer letter (A-J) "
    "between <answer> and </answer>."
)
MMLU_DATASET_PATH = "cais/mmlu"
MMLU_REDUX_DATASET_PATH = "edinburgh-dawg/mmlu-redux-2.0"
GPQA_DATASET_PATH = "Idavidrein/gpqa"
GPQA_DEFAULT_CONFIG = "gpqa_diamond"
MMLU_PRO_DATASET_PATH = "TIGER-Lab/MMLU-Pro"
THEOREMQA_DATASET_PATH = "TIGER-Lab/TheoremQA"
MATH500_DATASET_PATH = "HuggingFaceH4/MATH-500"
HENDRYCKS_MATH_DATASET_PATH = "EleutherAI/hendrycks_math"
MTU_BENCH_DATASET_PATH = "wpei/MTU-Bench"

# MTU-Bench specific instruction for tool use evaluation
MTU_BENCH_INSTRUCTION = (
    "You are a helpful assistant with access to tools. "
    "When you need to use a tool, output in the following format:\n"
    "Thought: [Your reasoning about what to do]\n"
    "Action: [tool_name]\n"
    "Action Input: [parameters as JSON]\n"
    "If you don't need to use a tool, just respond directly."
)

# ReAct format patterns for parsing tool calls
REACT_ACTION_RE = re.compile(r"Action:\s*([^\n]+)", re.IGNORECASE)
REACT_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+?)(?=(?:Thought:|Action:|Observation:|$))", re.DOTALL | re.IGNORECASE)
REACT_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=(?:Action:|Observation:|$))", re.DOTALL | re.IGNORECASE)
REACT_OBSERVATION_RE = re.compile(r"Observation:\s*(.+?)(?=(?:Thought:|Action:|$))", re.DOTALL | re.IGNORECASE)

HENDRYCKS_MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

ANSWER_RE = re.compile(r"The answer is: (.*?)$", re.DOTALL)
CHOICE_ANSWER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)
BOXED_RE = re.compile(r"\\(boxed|fbox)\s*\{", re.DOTALL)
NUMERIC_RE = re.compile(r"(\d+(?:\.\d+)?)")


def load_metamathqa_dataset(
    dataset_path: str,
    split: str,
    cache_dir: Optional[str],
    type_prefix: str,
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
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
    samples: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and len(samples) >= max_samples:
            break
        question = extract_metamathqa_question(example)
        if not question:
            continue
        gold = extract_metamathqa_answer(example)
        base_id = example.get("id") or example.get("idx") or example.get("problem_id") or idx
        samples.append({
            "id": str(base_id),
            "problem": question,
            "gold": gold,
            "source": "MetaMathQA",
        })
    return samples


def load_math500_dataset(max_samples: Optional[int], seed: Optional[int]) -> List[Dict[str, Any]]:
    LOGGER.info("Loading %s test split", MATH500_DATASET_PATH)
    dataset = load_dataset(MATH500_DATASET_PATH, split="test")
    if seed is not None:
        try:
            dataset = dataset.shuffle(seed=seed)
        except Exception as exc:
            LOGGER.warning("Shuffle failed: %s", exc)
    samples: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and len(samples) >= max_samples:
            break
        problem = str(example.get("problem", "")).strip()
        if not problem:
            continue
        gold = extract_ground_truth_answer(example)
        base_id = example.get("id") or idx
        samples.append({
            "id": str(base_id),
            "problem": problem,
            "gold": gold,
            "source": "MATH-500",
        })
    return samples


def load_gsm8k_dataset(
    config: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    dataset_name = "gsm8k"
    configs = get_dataset_config_names(dataset_name)
    if config is None or config.lower() == "default":
        config = "main"
    if config not in configs:
        LOGGER.error("Unknown GSM8K config '%s'. Available: %s", config, ", ".join(configs))
        return []
    LOGGER.info("Loading GSM8K config: %s (test split)", config)
    dataset = load_dataset(dataset_name, config, split="test")
    if seed is not None:
        try:
            dataset = dataset.shuffle(seed=seed)
        except Exception as exc:
            LOGGER.warning("Shuffle failed: %s", exc)
    samples: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and len(samples) >= max_samples:
            break
        problem = str(example.get("question", "")).strip()
        if not problem:
            continue
        gold = extract_ground_truth_answer(example)
        base_id = example.get("id") or idx
        samples.append({
            "id": str(base_id),
            "problem": problem,
            "gold": gold,
            "source": f"GSM8K-{config}",
        })
    return samples


def load_theoremqa_dataset(
    dataset_path: str,
    split: str,
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    LOGGER.info("Loading TheoremQA (%s split)", split)
    try:
        dataset = load_dataset(dataset_path, split=split, cache_dir=cache_dir)
    except Exception as exc:
        LOGGER.warning("Failed to load TheoremQA split '%s': %s", split, exc)
        dataset = None
        for fallback_split in ("test", "validation", "dev", "train"):
            if fallback_split == split:
                continue
            try:
                dataset = load_dataset(
                    dataset_path, split=fallback_split, cache_dir=cache_dir
                )
                LOGGER.warning("Falling back to TheoremQA split '%s'", fallback_split)
                break
            except Exception:
                dataset = None
        if dataset is None:
            return []

    if seed is not None:
        try:
            dataset = dataset.shuffle(seed=seed)
        except Exception as exc:
            LOGGER.warning("Shuffle failed: %s", exc)
    samples: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and len(samples) >= max_samples:
            break
        question = extract_theoremqa_question(example)
        if not question:
            continue
        gold = extract_theoremqa_answer(example)
        base_id = example.get("id") or example.get("uid") or example.get("question_id") or idx
        samples.append({
            "id": str(base_id),
            "problem": question,
            "gold": gold,
            "source": "TheoremQA",
        })
    return samples


def load_gpqa_dataset(
    config: Optional[str],
    split: str,
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    dataset_name = GPQA_DATASET_PATH
    
    dataset = load_dataset(dataset_name, "gpqa_diamond", split="train")
    

    if seed is not None:
        try:
            dataset = dataset.shuffle(seed=seed)
        except Exception as exc:
            LOGGER.warning("Shuffle failed: %s", exc)
    samples: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and len(samples) >= max_samples:
            break
        question, gold = build_gpqa_qa(example)
        if not question or not gold:
            continue
        base_id = example.get("id") or example.get("uid") or idx
        samples.append({
            "id": str(base_id),
            "problem": question,
            "gold": gold,
            "source": f"GPQA-{config}",
        })
    return samples


def load_mmlu_dataset(
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
    dataset_config: Optional[str],
) -> List[Dict[str, Any]]:
    if dataset_config and dataset_config.lower() != "all":
        LOGGER.warning("MMLU ignores --dataset_config; loading all subsets.")
    try:
        subsets = get_dataset_config_names(MMLU_DATASET_PATH)
    except Exception as exc:
        LOGGER.warning("Failed to fetch MMLU config names: %s", exc)
        subsets = []
    if not subsets:
        LOGGER.error("No MMLU subsets available.")
        return []

    samples: List[Dict[str, Any]] = []
    for subset in subsets:
        LOGGER.info("Loading MMLU subset: %s (test split)", subset)
        try:
            dataset = load_dataset(
                MMLU_DATASET_PATH,
                subset,
                split="test",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            LOGGER.warning("Failed to load subset '%s': %s", subset, exc)
            continue
        for idx, example in enumerate(dataset):
            question = extract_mmlu_question(example)
            if not question:
                continue
            gold = extract_mmlu_answer(example)
            base_id = example.get("id") or example.get("uid") or idx
            samples.append({
                "id": f"{subset}-{base_id}",
                "problem": question,
                "gold": gold,
                "source": f"MMLU-{subset}",
            })

    if seed is not None and samples:
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_mmlu_redux_dataset(
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
    dataset_config: Optional[str],
) -> List[Dict[str, Any]]:
    if dataset_config and dataset_config.lower() != "all":
        LOGGER.warning("MMLU-Redux ignores --dataset_config; loading all subsets.")
    try:
        subsets = get_dataset_config_names(MMLU_REDUX_DATASET_PATH)
    except Exception as exc:
        LOGGER.warning("Failed to fetch MMLU-Redux config names: %s", exc)
        subsets = []
    if not subsets:
        LOGGER.error("No MMLU-Redux subsets available.")
        return []

    samples: List[Dict[str, Any]] = []
    for subset in subsets:
        LOGGER.info("Loading MMLU-Redux subset: %s (test split)", subset)
        try:
            dataset = load_dataset(
                MMLU_REDUX_DATASET_PATH,
                subset,
                split="test",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            LOGGER.warning("Failed to load subset '%s': %s", subset, exc)
            continue
        for idx, example in enumerate(dataset):
            question = extract_mmlu_question(example)
            if not question:
                continue
            gold = extract_mmlu_answer(example)
            base_id = example.get("id") or example.get("uid") or idx
            samples.append({
                "id": f"{subset}-{base_id}",
                "problem": question,
                "gold": gold,
                "source": f"MMLU-Redux-{subset}",
            })

    if seed is not None and samples:
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_mmlu_pro_dataset(
    config: Optional[str],
    split: str,
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    dataset_name = MMLU_PRO_DATASET_PATH
    config = config or "default"
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception as exc:
        configs = []
        LOGGER.warning("Failed to fetch MMLU-Pro config names: %s", exc)
    if configs and config not in configs:
        LOGGER.error("Unknown MMLU-Pro config '%s'. Available: %s", config, ", ".join(configs))
        return []
    LOGGER.info("Loading MMLU-Pro config: %s (%s split)", config, split)
    try:
        dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
    except Exception as exc:
        LOGGER.warning("Failed to load MMLU-Pro split '%s': %s", split, exc)
        dataset = None
        for fallback_split in ("test", "validation", "dev"):
            if fallback_split == split:
                continue
            try:
                dataset = load_dataset(
                    dataset_name, config, split=fallback_split, cache_dir=cache_dir
                )
                LOGGER.warning("Falling back to MMLU-Pro split '%s'", fallback_split)
                break
            except Exception:
                dataset = None
        if dataset is None:
            return []

    if seed is not None:
        try:
            dataset = dataset.shuffle(seed=seed)
        except Exception as exc:
            LOGGER.warning("Shuffle failed: %s", exc)
    samples: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and len(samples) >= max_samples:
            break
        problem = extract_mmlu_pro_question(example)
        if not problem:
            continue
        gold = extract_mmlu_pro_answer(example)
        base_id = example.get("id") or example.get("uid") or example.get("problem_id") or idx
        samples.append({
            "id": str(base_id),
            "problem": problem,
            "gold": gold,
            "source": f"MMLU-Pro-{config}",
        })
    return samples


def load_hendrycks_math_dataset(
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
    dataset_config: Optional[str],
) -> List[Dict[str, Any]]:
    if dataset_config and dataset_config.lower() != "all":
        LOGGER.warning("Hendrycks MATH ignores --dataset_config; loading all subsets.")
    try:
        subsets = get_dataset_config_names(HENDRYCKS_MATH_DATASET_PATH)
    except Exception as exc:
        LOGGER.warning("Failed to fetch Hendrycks MATH config names: %s", exc)
        subsets = []
    if not subsets:
        subsets = list(HENDRYCKS_MATH_SUBSETS)

    samples: List[Dict[str, Any]] = []
    for subset in subsets:
        LOGGER.info("Loading Hendrycks MATH subset: %s (test split)", subset)
        try:
            dataset = load_dataset(
                HENDRYCKS_MATH_DATASET_PATH,
                subset,
                split="test",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            LOGGER.warning("Failed to load subset '%s': %s", subset, exc)
            continue
        for idx, example in enumerate(dataset):
            question = str(example.get("problem", "")).strip()
            if not question:
                continue
            gold = extract_ground_truth_answer(example)
            base_id = example.get("id") or example.get("problem_id") or idx
            samples.append({
                "id": f"{subset}-{base_id}",
                "problem": question,
                "gold": gold,
                "source": f"HendrycksMATH-{subset}",
            })

    if seed is not None and samples:
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_mtu_bench_dataset(
    dataset_path: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: Optional[str],
    max_samples: Optional[int],
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    dataset_name = dataset_path
    config = dataset_config
    if config is not None and config.lower() == "default":
        config = None

    def load_with_fallback(config_name: Optional[str]):
        try:
            if config_name is None:
                return load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            return load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)
        except Exception as exc:
            suffix = f" config '{config_name}'" if config_name else ""
            LOGGER.warning("Failed to load MTU-Bench%s split '%s': %s", suffix, split, exc)
            for fallback_split in ("test", "validation", "dev", "train"):
                if fallback_split == split:
                    continue
                try:
                    if config_name is None:
                        dataset = load_dataset(
                            dataset_name, split=fallback_split, cache_dir=cache_dir
                        )
                    else:
                        dataset = load_dataset(
                            dataset_name, config_name, split=fallback_split, cache_dir=cache_dir
                        )
                    LOGGER.warning("Falling back to MTU-Bench split '%s'", fallback_split)
                    return dataset
                except Exception:
                    continue
        return None

    def append_samples(dataset, source_label: str, id_prefix: Optional[str]):
        if seed is not None:
            try:
                dataset = dataset.shuffle(seed=seed)
            except Exception as exc:
                LOGGER.warning("Shuffle failed: %s", exc)
        samples: List[Dict[str, Any]] = []
        for idx, example in enumerate(dataset):
            if max_samples is not None and len(samples) >= max_samples:
                break
            question = extract_mtu_bench_question(example)
            if not question:
                continue

            # For MTU-Bench, 'output' field contains ReAct-style tool calls
            gold_output = example.get("output", "")
            if isinstance(gold_output, str):
                gold_output = gold_output.strip()
            else:
                gold_output = str(gold_output) if gold_output else ""

            # MTU-Bench stores ground truth tool calls in various formats:
            # 1. Direct 'gold' field with JSON string like '{"ToolName": {"param": "value"}}'
            # 2. 'Action' field in the conversation history
            # Try multiple sources to find the ground truth tool calls
            gold_field = ""

            # First try direct fields that might contain JSON tool calls
            for field_name in ("gold", "Action", "action", "target", "label", "answer"):
                field_val = example.get(field_name)
                if field_val:
                    if isinstance(field_val, dict):
                        gold_field = json.dumps(field_val, ensure_ascii=False)
                        break
                    elif isinstance(field_val, str):
                        field_val = field_val.strip()
                        # Check if it looks like a JSON tool call dict
                        if field_val.startswith("{") and field_val.endswith("}"):
                            gold_field = field_val
                            break

            # If no direct field found, use extract_mtu_bench_answer as fallback
            if not gold_field:
                gold_field = extract_mtu_bench_answer(example)

            base_id = (
                example.get("id")
                or example.get("uid")
                or example.get("problem_id")
                or example.get("question_id")
                or idx
            )
            sample_id = f"{id_prefix}-{base_id}" if id_prefix else str(base_id)
            samples.append({
                "id": sample_id,
                "problem": question,
                "gold": gold_field,  # JSON format tool calls or extracted answer
                "gold_output": gold_output,  # ReAct-style output (if any)
                "source": source_label,
                # Store additional metadata for scenario classification
                "scenario": example.get("scenario", "unknown"),
                "difficulty": example.get("difficulty", example.get("split", "normal")),
            })
        return samples

    if config is not None and config.lower() == "all":
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception as exc:
            LOGGER.warning("Failed to fetch MTU-Bench config names: %s", exc)
            configs = []
        if not configs:
            LOGGER.error("No MTU-Bench configs available for --dataset_config=all.")
            return []
        samples: List[Dict[str, Any]] = []
        for cfg in configs:
            LOGGER.info("Loading MTU-Bench config: %s (%s split)", cfg, split)
            dataset = load_with_fallback(cfg)
            if dataset is None:
                continue
            samples.extend(append_samples(dataset, f"MTU-Bench-{cfg}", cfg))
        if seed is not None and samples:
            random.Random(seed).shuffle(samples)
        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    if config is None:
        LOGGER.info("Loading MTU-Bench (%s split)", split)
        dataset = load_with_fallback(None)
        if dataset is None:
            try:
                configs = get_dataset_config_names(dataset_name)
            except Exception as exc:
                LOGGER.warning("Failed to fetch MTU-Bench config names: %s", exc)
                configs = []
            if configs:
                config = configs[0]
                LOGGER.warning("MTU-Bench requires a config; falling back to '%s'", config)
                dataset = load_with_fallback(config)
        if dataset is None:
            return []
        source = f"MTU-Bench-{config}" if config else "MTU-Bench"
        return append_samples(dataset, source, config)

    LOGGER.info("Loading MTU-Bench config: %s (%s split)", config, split)
    dataset = load_with_fallback(config)
    if dataset is None:
        return []
    return append_samples(dataset, f"MTU-Bench-{config}", config)


def load_samples(args: Any) -> List[Dict[str, Any]]:
    if args.dataset == "metamathqa":
        return load_metamathqa_dataset(
            args.dataset_path,
            args.dataset_split,
            args.cache_dir,
            args.type_prefix,
            args.max_samples,
            args.seed,
        )
    if args.dataset == "math500":
        return load_math500_dataset(args.max_samples, args.seed)
    if args.dataset == "gsm8k":
        return load_gsm8k_dataset(args.dataset_config, args.max_samples, args.seed)
    if args.dataset == "theoremqa":
        return load_theoremqa_dataset(
            resolve_dataset_path(args.dataset, args.dataset_path),
            args.dataset_split,
            args.cache_dir,
            args.max_samples,
            args.seed,
        )
    if args.dataset == "gpqa":
        return load_gpqa_dataset(
            args.dataset_config,
            args.dataset_split,
            args.cache_dir,
            args.max_samples,
            args.seed,
        )
    if args.dataset == "mmlu":
        return load_mmlu_dataset(
            args.cache_dir,
            args.max_samples,
            args.seed,
            args.dataset_config,
        )
    if args.dataset in ("mmlu-redux", "mmlu_redux"):
        return load_mmlu_redux_dataset(
            args.cache_dir,
            args.max_samples,
            args.seed,
            args.dataset_config,
        )
    if args.dataset == "mmlu_pro":
        return load_mmlu_pro_dataset(
            args.dataset_config,
            args.dataset_split,
            args.cache_dir,
            args.max_samples,
            args.seed,
        )
    if args.dataset == "hendrycks_math":
        return load_hendrycks_math_dataset(
            args.cache_dir,
            args.max_samples,
            args.seed,
            args.dataset_config,
        )
    if args.dataset in ("mtu_bench", "mtu-bench"):
        return load_mtu_bench_dataset(
            resolve_dataset_path(args.dataset, args.dataset_path),
            args.dataset_config,
            args.dataset_split,
            args.cache_dir,
            args.max_samples,
            args.seed,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def extract_metamathqa_question(example: Dict[str, Any]) -> str:
    return str(example.get("query", "")).strip()


def extract_metamathqa_answer(example: Dict[str, Any]) -> str:
    response = str(example.get("response", ""))
    match = ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return ""


def extract_theoremqa_question(example: Dict[str, Any]) -> str:
    question = str(example.get("Question", example.get("question", ""))).strip()
    picture = example.get("Picture", example.get("picture"))
    if picture not in (None, "", "None"):
        return f"<image>\n{question}" if question else "<image>"
    return question


def extract_theoremqa_answer(example: Dict[str, Any]) -> str:
    answer = example.get("Answer", example.get("answer", ""))
    if isinstance(answer, (list, tuple)):
        for item in answer:
            text = str(item).strip()
            if text:
                return text
        return ""
    if isinstance(answer, dict):
        for key in ("answer", "value", "text"):
            if key in answer:
                text = str(answer[key]).strip()
                if text:
                    return text
        return str(answer).strip()
    text = str(answer).strip()
    parsed = extract_answer_field(text)
    return parsed if parsed else text


def extract_mtu_bench_question(example: Dict[str, Any]) -> str:
    instruction = example.get("instruction")
    input_text = example.get("input")
    instruction = instruction.strip() if isinstance(instruction, str) else ""
    input_text = input_text.strip() if isinstance(input_text, str) else ""
    if instruction or input_text:
        parts = [part for part in (instruction, input_text) if part]
        return "\n".join(parts).strip()

    for key in ("question", "problem", "prompt", "query", "context", "text", "task"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    convo = (
        example.get("conversations")
        or example.get("conversation")
        or example.get("messages")
        or example.get("dialogue")
        or example.get("turns")
        or example.get("history")
    )
    if isinstance(convo, list):
        user_messages: List[str] = []
        fallback_messages: List[str] = []
        for turn in convo:
            if isinstance(turn, dict):
                role = str(turn.get("role") or turn.get("from") or turn.get("speaker") or "").lower()
                content = (
                    turn.get("content")
                    or turn.get("value")
                    or turn.get("text")
                    or turn.get("message")
                )
                text = str(content).strip() if content is not None else ""
                if not text:
                    continue
                fallback_messages.append(text)
                if role in ("user", "human", "prompt", "question", "input", "instruction", "system"):
                    user_messages.append(text)
            else:
                text = str(turn).strip()
                if text:
                    fallback_messages.append(text)
                    user_messages.append(text)
        if user_messages:
            if len(user_messages) == 1:
                return user_messages[0]
            return "\n".join(f"User: {msg}" for msg in user_messages)
        if fallback_messages:
            return "\n".join(fallback_messages)

    return ""


def extract_mtu_bench_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            text = extract_mtu_bench_answer_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("answer", "final_answer", "final", "value", "text", "output", "response", "label", "target"):
            if key in value:
                text = extract_mtu_bench_answer_text(value.get(key))
                if text:
                    return text
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = ANSWER_RE.search(text)
    if match:
        text = match.group(1).strip()
    parsed = extract_answer_field(text)
    boxed = extract_boxed_answer(parsed)
    if boxed:
        return boxed.strip()
    return parsed


def extract_mtu_bench_answer(example: Dict[str, Any]) -> str:
    for key in (
        "answer",
        "final_answer",
        "final",
        "target",
        "label",
        "output",
        "response",
        "completion",
        "ground_truth",
        "gold",
        "solution",
        "reference",
    ):
        if key in example:
            text = extract_mtu_bench_answer_text(example.get(key))
            if text:
                return text

    convo = (
        example.get("conversations")
        or example.get("conversation")
        or example.get("messages")
        or example.get("dialogue")
        or example.get("turns")
        or example.get("history")
    )
    if isinstance(convo, list):
        for turn in reversed(convo):
            if isinstance(turn, dict):
                role = str(turn.get("role") or turn.get("from") or turn.get("speaker") or "").lower()
                if role in ("assistant", "gpt", "bot", "model", "answer"):
                    content = (
                        turn.get("content")
                        or turn.get("value")
                        or turn.get("text")
                        or turn.get("message")
                    )
                    text = extract_mtu_bench_answer_text(content)
                    if text:
                        return text
    return ""


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
            boxed = extract_boxed_answer(parsed)
            if boxed:
                return boxed.strip()
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


def extract_mmlu_question(example: Dict[str, Any]) -> str:
    question = str(example.get("question", "")).strip()
    options = example.get("choices")
    if options is None:
        options = []
    elif isinstance(options, tuple):
        options = list(options)
    elif not isinstance(options, list):
        options = [str(options)]
    lines = [question] if question else []
    for idx, option in enumerate(options):
        lines.append(f"{chr(65 + idx)}. {option}")
    return "\n".join(lines).strip()


def extract_mmlu_answer(example: Dict[str, Any]) -> str:
    answer_field = example.get("answer", example.get("answer_index"))
    if answer_field is None:
        return ""
    if isinstance(answer_field, str):
        match = CHOICE_ANSWER_RE.search(answer_field.strip().upper())
        if match:
            return match.group(1)
        try:
            answer_idx = int(answer_field)
        except ValueError:
            return ""
    else:
        try:
            answer_idx = int(answer_field)
        except (TypeError, ValueError):
            return ""
    if 0 <= answer_idx <= 3:
        return chr(65 + answer_idx)
    return ""


def build_gpqa_qa(example: Dict[str, Any]) -> Tuple[str, str]:
    question = str(
        example.get("Question", example.get("question", ""))
    ).strip()
    choices = example.get("choices", example.get("options"))
    answer_field = example.get("answer", example.get("answer_index"))
    seed_key = str(
        example.get("id")
        or example.get("uid")
        or example.get("question")
        or question
    )

    correct_text = example.get(
        "Correct Answer",
        example.get("correct_answer"),
    )
    incorrect = example.get(
        "Incorrect Answers",
        example.get("incorrect_answers"),
    )

    options: List[str] = []
    correct_idx: Optional[int] = None
    if isinstance(choices, dict):
        dict_options = choices.get("text") or choices.get("choices") or choices.get("options")
        if dict_options is not None:
            choices = dict_options

    if choices is None:
        if incorrect is None:
            incorrect = []
        elif isinstance(incorrect, tuple):
            incorrect = list(incorrect)
        elif not isinstance(incorrect, list):
            incorrect = [str(incorrect)]

        if not incorrect:
            for key in (
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
                "incorrect_answer_1",
                "incorrect_answer_2",
                "incorrect_answer_3",
            ):
                value = example.get(key)
                if value:
                    incorrect.append(value)

        if correct_text is None:
            return question, ""
        options = [str(correct_text)] + [str(opt) for opt in incorrect]
        correct_idx = 0
    else:
        if isinstance(choices, tuple):
            options = [str(opt) for opt in choices]
        elif isinstance(choices, list):
            options = [str(opt) for opt in choices]
        else:
            options = [str(choices)]

        correct_idx = None
        if answer_field is not None:
            if isinstance(answer_field, str):
                match = CHOICE_ANSWER_RE.search(answer_field.strip().upper())
                if match:
                    correct_idx = ord(match.group(1)) - 65
                else:
                    try:
                        correct_idx = int(answer_field)
                    except ValueError:
                        correct_idx = None
            else:
                try:
                    correct_idx = int(answer_field)
                except (TypeError, ValueError):
                    correct_idx = None
        if correct_idx is None and correct_text is not None:
            for idx, option in enumerate(options):
                if str(option).strip() == str(correct_text).strip():
                    correct_idx = idx
                    break

    filtered_options: List[str] = []
    filtered_correct_idx: Optional[int] = None
    for idx, option in enumerate(options):
        text = str(option).strip()
        if not text:
            continue
        if correct_idx is not None and idx == correct_idx:
            filtered_correct_idx = len(filtered_options)
        filtered_options.append(text)
    options = filtered_options
    if not options:
        return question, ""
    if correct_idx is not None:
        correct_idx = filtered_correct_idx

    seed_int = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    indices = list(range(len(options)))
    rng.shuffle(indices)
    options = [options[i] for i in indices]
    if correct_idx is not None:
        correct_idx = indices.index(correct_idx)

    lines = [question] if question else []
    for idx, option in enumerate(options):
        lines.append(f"{chr(65 + idx)}. {option}")
    formatted_question = "\n".join(lines).strip()

    if correct_idx is None or correct_idx < 0 or correct_idx >= len(options):
        return formatted_question, ""
    return formatted_question, chr(65 + correct_idx)


def extract_mmlu_pro_question(example: Dict[str, Any]) -> str:
    question = str(example.get("question", "")).strip()
    options = example.get("options", example.get("choices"))
    if options is None:
        options = []
    elif isinstance(options, tuple):
        options = list(options)
    elif not isinstance(options, list):
        options = [str(options)]
    lines = [question] if question else []
    for idx, option in enumerate(options):
        lines.append(f"{chr(65 + idx)}. {option}")
    return "\n".join(lines).strip()


def extract_mmlu_pro_answer(example: Dict[str, Any]) -> str:
    answer_field = example.get("answer", example.get("answer_index"))
    if answer_field is None:
        return ""
    if isinstance(answer_field, str):
        match = CHOICE_ANSWER_RE.search(answer_field.strip().upper())
        if match:
            return match.group(1)
        try:
            answer_idx = int(answer_field)
        except ValueError:
            return ""
    else:
        try:
            answer_idx = int(answer_field)
        except (TypeError, ValueError):
            return ""
    if 0 <= answer_idx <= 9:
        return chr(65 + answer_idx)
    return ""


def normalize_theoremqa_answer(text: str) -> str:
    if not text:
        return ""
    numeric = extract_numeric_value(text)
    if numeric is not None:
        return format_numeric_value(numeric)
    return normalize_text(str(text))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return text.translate(str.maketrans("", "", string.punctuation))


def extract_numeric_value(text: str) -> Optional[float]:
    if not text:
        return None
    match = NUMERIC_RE.search(str(text))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def format_numeric_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = str(value)
    if "." in text:
        return text.rstrip("0").rstrip(".")
    return text


def check_numeric_answer(prediction: str, label: str) -> Tuple[bool, bool]:
    pred_value = extract_numeric_value(prediction)
    label_value = extract_numeric_value(label)
    is_valid = pred_value is not None
    is_correct = False
    if pred_value is not None and label_value is not None:
        is_correct = pred_value == label_value
    if not is_correct:
        is_correct = normalize_text(prediction or "") == normalize_text(label or "")
    return is_correct, is_valid


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text.strip().lower())


def normalize_choice_answer(text: str) -> str:
    if not text:
        return ""
    matches = CHOICE_ANSWER_RE.findall(text.strip())
    if matches:
        return matches[-1].lower()
    return normalize_answer(text)


def get_env_instruction(dataset: str) -> str:
    if dataset in ("mmlu-redux", "mmlu_redux"):
        return MMLU_INSTRUCTION
    if dataset == "gpqa":
        return GPQA_INSTRUCTION
    if dataset == "mmlu":
        return MMLU_INSTRUCTION
    if dataset == "mmlu_pro":
        return MMLU_PRO_INSTRUCTION
    if dataset in ("mtu_bench", "mtu-bench"):
        return MTU_BENCH_INSTRUCTION
    return ENV_INSTRUCTION


def get_answer_normalizer(dataset: str) -> Callable[[str], str]:
    if dataset in ("mmlu-redux", "mmlu_redux"):
        return normalize_choice_answer
    if dataset == "gpqa":
        return normalize_choice_answer
    if dataset == "mmlu":
        return normalize_choice_answer
    if dataset == "mmlu_pro":
        return normalize_choice_answer
    if dataset == "theoremqa":
        return normalize_theoremqa_answer
    return normalize_answer


def get_answer_checker(dataset: str) -> Optional[Callable[[str, str], Tuple[bool, bool]]]:
    if dataset == "theoremqa":
        return check_numeric_answer
    return None


def resolve_dataset_path(dataset: str, dataset_path: str) -> str:
    if dataset in ("mmlu-redux", "mmlu_redux"):
        return MMLU_REDUX_DATASET_PATH
    if dataset == "gpqa":
        return GPQA_DATASET_PATH
    if dataset == "mmlu":
        return MMLU_DATASET_PATH
    if dataset == "mmlu_pro":
        return MMLU_PRO_DATASET_PATH
    if dataset == "hendrycks_math":
        return HENDRYCKS_MATH_DATASET_PATH
    if dataset == "theoremqa":
        return THEOREMQA_DATASET_PATH
    if dataset in ("mtu_bench", "mtu-bench"):
        return MTU_BENCH_DATASET_PATH
    return dataset_path


def get_dataset_label(
    dataset: str,
    dataset_path: str,
    dataset_config: Optional[str],
) -> str:
    if dataset == "metamathqa":
        return dataset_path
    if dataset == "gsm8k":
        config = dataset_config
        if config is None or config.lower() == "default":
            config = "main"
        return f"gsm8k/{config}"
    if dataset in ("mmlu-redux", "mmlu_redux"):
        return f"{MMLU_REDUX_DATASET_PATH}/all"
    if dataset == "gpqa":
        config = dataset_config
        if config is None or config.lower() == "default":
            config = GPQA_DEFAULT_CONFIG
        return f"{GPQA_DATASET_PATH}/{config}"
    if dataset == "mmlu":
        return f"{MMLU_DATASET_PATH}/all"
    if dataset == "mmlu_pro":
        return f"{MMLU_PRO_DATASET_PATH}/{dataset_config or 'default'}"
    if dataset == "hendrycks_math":
        return f"{HENDRYCKS_MATH_DATASET_PATH}/all"
    if dataset == "theoremqa":
        return THEOREMQA_DATASET_PATH
    if dataset in ("mtu_bench", "mtu-bench"):
        if dataset_config and dataset_config.lower() == "all":
            return f"{MTU_BENCH_DATASET_PATH}/all"
        if dataset_config and dataset_config.lower() != "default":
            return f"{MTU_BENCH_DATASET_PATH}/{dataset_config}"
        return MTU_BENCH_DATASET_PATH
    return MATH500_DATASET_PATH


# =============================================================================
# MTU-Bench Tool Use Evaluation Functions
# =============================================================================

@dataclass
class ToolCall:
    """Represents a single tool call extracted from model output."""
    tool_name: str
    parameters: Dict[str, Any]
    thought: Optional[str] = None
    raw_text: Optional[str] = None


@dataclass
class MTUBenchTurn:
    """Represents a single turn in MTU-Bench evaluation."""
    tool_calls: List[ToolCall]
    raw_output: str


@dataclass
class MTUBenchMetrics:
    """Metrics for MTU-Bench evaluation."""
    tool_selection_accuracy: float  # TS: correct tool name selection
    parameter_selection_accuracy: float  # PS: correct parameters
    tool_number_accuracy: float  # TN: correct number of tools
    tool_order_accuracy: float  # TO: correct order of tools
    turn_success_rate: float  # ATS: averaged turn success rate
    soft_turn_success_rate: float  # SATS: soft averaged turn success rate
    success_rate: float  # SR: overall task success rate
    task_process_rate: float  # TPR: task process rate

    def to_dict(self) -> Dict[str, float]:
        return {
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "parameter_selection_accuracy": self.parameter_selection_accuracy,
            "tool_number_accuracy": self.tool_number_accuracy,
            "tool_order_accuracy": self.tool_order_accuracy,
            "turn_success_rate": self.turn_success_rate,
            "soft_turn_success_rate": self.soft_turn_success_rate,
            "success_rate": self.success_rate,
            "task_process_rate": self.task_process_rate,
        }


def parse_json_params(text: str) -> Dict[str, Any]:
    """Parse JSON parameters from action input text."""
    text = text.strip()
    if not text:
        return {}
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in text
    brace_start = text.find("{")
    if brace_start != -1:
        brace_end = text.rfind("}")
        if brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
    # Try key=value format
    params = {}
    for match in re.finditer(r'(\w+)\s*[=:]\s*(["\']?)([^,\n]+)\2', text):
        key, _, value = match.groups()
        value = value.strip().strip("\"'")
        params[key] = value
    return params


def parse_tool_calls_from_text(text: str) -> List[ToolCall]:
    """Parse tool calls from ReAct-style text output."""
    tool_calls = []
    if not text:
        return tool_calls

    # Find all Action: patterns
    action_matches = list(REACT_ACTION_RE.finditer(text))
    if not action_matches:
        return tool_calls

    for i, action_match in enumerate(action_matches):
        tool_name = action_match.group(1).strip()
        # Clean tool name
        tool_name = re.sub(r'[\[\]"\']', '', tool_name).strip()

        # Find corresponding Action Input
        start_pos = action_match.end()
        end_pos = action_matches[i + 1].start() if i + 1 < len(action_matches) else len(text)
        segment = text[start_pos:end_pos]

        # Extract Action Input
        input_match = REACT_ACTION_INPUT_RE.search(segment)
        params = {}
        if input_match:
            params = parse_json_params(input_match.group(1))

        # Extract Thought (from before this action)
        thought = None
        if i == 0:
            pre_text = text[:action_match.start()]
        else:
            pre_text = text[action_matches[i - 1].end():action_match.start()]
        thought_match = REACT_THOUGHT_RE.search(pre_text)
        if thought_match:
            thought = thought_match.group(1).strip()

        tool_calls.append(ToolCall(
            tool_name=tool_name,
            parameters=params,
            thought=thought,
            raw_text=text[action_match.start():end_pos].strip(),
        ))

    return tool_calls


def parse_mtu_bench_ground_truth(output_text: str) -> List[ToolCall]:
    """Parse ground truth tool calls from MTU-Bench output field (ReAct format)."""
    return parse_tool_calls_from_text(output_text)


# Pattern to parse <think>...<answer>... format
THINK_ANSWER_PATTERN = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE
)
# Pattern to parse tool calls like: tool_name(param1=value1, param2="value2")
TOOL_CALL_PATTERN = re.compile(
    r"(\w+)\s*\(\s*(.*?)\s*\)",
    re.DOTALL
)
# Pattern to parse individual parameters
PARAM_PATTERN = re.compile(
    r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|([^,\)]+))',
    re.DOTALL
)


def parse_tool_calls_from_answer_tag(text: str) -> Tuple[str, List[ToolCall]]:
    """
    Parse tool calls from <think>...<answer>... format.

    Returns:
        Tuple of (thought_content, list_of_tool_calls)
    """
    tool_calls = []
    thought = ""

    if not text:
        return thought, tool_calls

    # Add <think> prefix if not present (model output starts after <think>)
    if not text.strip().startswith("<think>"):
        text = "<think>" + text

    # Find <think>...<answer>... pattern
    match = THINK_ANSWER_PATTERN.search(text)
    if not match:
        # Try to find just <answer>...</answer>
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # No valid format found, treat entire text as answer
            answer_content = text.strip()
    else:
        thought = match.group(1).strip()
        answer_content = match.group(2).strip()

    # Clean special tokens from thought and answer
    for token in ("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"):
        thought = thought.replace(token, "").strip()
        answer_content = answer_content.replace(token, "").strip()

    if not answer_content:
        return thought, tool_calls

    # Try to parse tool calls from answer content
    # Format: tool_name(param1=value1, param2="value2")
    tool_matches = list(TOOL_CALL_PATTERN.finditer(answer_content))

    for tool_match in tool_matches:
        tool_name = tool_match.group(1).strip()
        params_str = tool_match.group(2).strip()

        # Parse parameters
        params = {}
        if params_str:
            # Try to parse as key=value pairs
            for param_match in PARAM_PATTERN.finditer(params_str):
                param_name = param_match.group(1).strip()
                # Get value from whichever group matched
                param_value = (
                    param_match.group(2) or  # Double quoted
                    param_match.group(3) or  # Single quoted
                    param_match.group(4) or  # Unquoted
                    ""
                )
                if param_value:
                    param_value = param_value.strip()
                params[param_name] = param_value

        tool_calls.append(ToolCall(
            tool_name=tool_name,
            parameters=params,
            thought=thought,
            raw_text=tool_match.group(0),
        ))

    return thought, tool_calls


def parse_mtu_bench_gold_field(gold_text: str) -> List[ToolCall]:
    """
    Parse ground truth tool calls from MTU-Bench 'gold' field.

    The gold field is a JSON string like:
    - '{"": {}}' means no tool call needed
    - '{"ToolName": {"param1": "value1", ...}}' for single tool
    - '{"Tool1": {...}, "Tool2": {...}}' for multiple tools
    - '{"Tool1": {...}, " Tool2": {...}}' (note: sometimes has leading space in key)
    """
    if not gold_text:
        return []

    gold_text = gold_text.strip()
    if not gold_text:
        return []

    try:
        gold_dict = json.loads(gold_text)
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            # Sometimes the JSON has issues, try to parse anyway
            gold_text = gold_text.replace("'", '"')
            gold_dict = json.loads(gold_text)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse gold field as JSON: %s", gold_text[:100])
            return []

    if not isinstance(gold_dict, dict):
        return []

    tool_calls = []
    for tool_name, params in gold_dict.items():
        # Skip empty tool names (means no tool call needed)
        tool_name = tool_name.strip()
        if not tool_name:
            continue

        if not isinstance(params, dict):
            params = {}

        tool_calls.append(ToolCall(
            tool_name=tool_name,
            parameters=params,
            thought=None,
            raw_text=None,
        ))

    return tool_calls


def normalize_tool_name(name: str) -> str:
    """Normalize tool name for comparison."""
    if not name:
        return ""
    # Convert to lowercase, remove special chars, normalize whitespace
    name = name.lower().strip()
    name = re.sub(r'[_\-\s]+', '_', name)
    name = re.sub(r'[^\w]', '', name)
    return name


def normalize_param_value(value: Any) -> Any:
    """Normalize parameter value for comparison."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (list, tuple)):
        return [normalize_param_value(v) for v in value]
    if isinstance(value, dict):
        return {k.lower(): normalize_param_value(v) for k, v in value.items()}
    return value


def compare_tool_calls(
    pred: ToolCall,
    gt: ToolCall,
    strict_params: bool = False,
) -> Tuple[bool, bool]:
    """
    Compare predicted tool call with ground truth.
    Returns (tool_name_correct, params_correct).
    """
    pred_name = normalize_tool_name(pred.tool_name)
    gt_name = normalize_tool_name(gt.tool_name)
    tool_correct = pred_name == gt_name

    if not tool_correct:
        return False, False

    # Compare parameters
    if strict_params:
        pred_params = {k.lower(): normalize_param_value(v) for k, v in pred.parameters.items()}
        gt_params = {k.lower(): normalize_param_value(v) for k, v in gt.parameters.items()}
        params_correct = pred_params == gt_params
    else:
        # Lenient comparison: check if all GT params are present in pred
        params_correct = True
        for key, gt_val in gt.parameters.items():
            pred_val = pred.parameters.get(key) or pred.parameters.get(key.lower())
            if pred_val is None:
                params_correct = False
                break
            if normalize_param_value(pred_val) != normalize_param_value(gt_val):
                params_correct = False
                break

    return tool_correct, params_correct


def compute_mtu_bench_turn_metrics(
    pred_calls: List[ToolCall],
    gt_calls: List[ToolCall],
) -> Dict[str, float]:
    """Compute metrics for a single turn."""
    if not gt_calls:
        return {
            "tool_selection": 1.0 if not pred_calls else 0.0,
            "param_selection": 1.0 if not pred_calls else 0.0,
            "tool_number": 1.0 if len(pred_calls) == 0 else 0.0,
            "tool_order": 1.0,
            "turn_success": 1.0 if not pred_calls else 0.0,
        }

    n_gt = len(gt_calls)
    n_pred = len(pred_calls)

    # Tool Number Accuracy
    tool_number_correct = 1.0 if n_pred == n_gt else 0.0

    # Tool Selection and Parameter Selection
    tool_correct_count = 0
    param_correct_count = 0
    min_len = min(n_pred, n_gt)

    for i in range(min_len):
        tool_ok, param_ok = compare_tool_calls(pred_calls[i], gt_calls[i])
        if tool_ok:
            tool_correct_count += 1
        if param_ok:
            param_correct_count += 1

    tool_selection = tool_correct_count / n_gt if n_gt > 0 else 0.0
    param_selection = param_correct_count / n_gt if n_gt > 0 else 0.0

    # Tool Order Accuracy
    pred_tools = [normalize_tool_name(c.tool_name) for c in pred_calls]
    gt_tools = [normalize_tool_name(c.tool_name) for c in gt_calls]
    tool_order_correct = 1.0 if pred_tools == gt_tools else 0.0

    # Turn Success: all tools correct with correct params
    turn_success = 1.0 if (
        tool_correct_count == n_gt and
        param_correct_count == n_gt and
        n_pred == n_gt
    ) else 0.0

    return {
        "tool_selection": tool_selection,
        "param_selection": param_selection,
        "tool_number": tool_number_correct,
        "tool_order": tool_order_correct,
        "turn_success": turn_success,
    }


def compute_mtu_bench_metrics(
    pred_turns: List[List[ToolCall]],
    gt_turns: List[List[ToolCall]],
    decay_factor: float = 0.9,
) -> MTUBenchMetrics:
    """
    Compute all MTU-Bench metrics for a dialogue.

    Args:
        pred_turns: List of predicted tool calls per turn
        gt_turns: List of ground truth tool calls per turn
        decay_factor: Decay factor for SATS computation

    Returns:
        MTUBenchMetrics with all computed metrics
    """
    n_turns = max(len(gt_turns), 1)
    n_pred_turns = len(pred_turns)

    # Pad predictions if fewer turns
    while len(pred_turns) < len(gt_turns):
        pred_turns.append([])

    # Compute per-turn metrics
    turn_metrics = []
    for pred_calls, gt_calls in zip(pred_turns[:n_turns], gt_turns):
        metrics = compute_mtu_bench_turn_metrics(pred_calls, gt_calls)
        turn_metrics.append(metrics)

    if not turn_metrics:
        return MTUBenchMetrics(
            tool_selection_accuracy=0.0,
            parameter_selection_accuracy=0.0,
            tool_number_accuracy=0.0,
            tool_order_accuracy=0.0,
            turn_success_rate=0.0,
            soft_turn_success_rate=0.0,
            success_rate=0.0,
            task_process_rate=0.0,
        )

    # Aggregate metrics
    ts_sum = sum(m["tool_selection"] for m in turn_metrics)
    ps_sum = sum(m["param_selection"] for m in turn_metrics)
    tn_sum = sum(m["tool_number"] for m in turn_metrics)
    to_sum = sum(m["tool_order"] for m in turn_metrics)
    turn_success_sum = sum(m["turn_success"] for m in turn_metrics)

    # ATS: Averaged Turn Success Rate
    ats = turn_success_sum / n_turns

    # SATS: Soft Averaged Turn Success Rate (with decay)
    # Earlier turns have higher weight; errors early on affect more
    weights = [decay_factor ** i for i in range(n_turns)]
    total_weight = sum(weights)
    sats = sum(
        w * m["turn_success"] for w, m in zip(weights, turn_metrics)
    ) / total_weight if total_weight > 0 else 0.0

    # SR: Success Rate (all turns must be successful)
    sr = 1.0 if all(m["turn_success"] == 1.0 for m in turn_metrics) else 0.0

    # TPR: Task Process Rate (fraction of turns completed successfully)
    tpr = ats  # Same as ATS for now

    return MTUBenchMetrics(
        tool_selection_accuracy=ts_sum / n_turns,
        parameter_selection_accuracy=ps_sum / n_turns,
        tool_number_accuracy=tn_sum / n_turns,
        tool_order_accuracy=to_sum / n_turns,
        turn_success_rate=ats,
        soft_turn_success_rate=sats,
        success_rate=sr,
        task_process_rate=tpr,
    )


def aggregate_mtu_bench_metrics(
    metrics_list: List[MTUBenchMetrics],
) -> Dict[str, float]:
    """Aggregate metrics across multiple dialogues."""
    if not metrics_list:
        return MTUBenchMetrics(
            tool_selection_accuracy=0.0,
            parameter_selection_accuracy=0.0,
            tool_number_accuracy=0.0,
            tool_order_accuracy=0.0,
            turn_success_rate=0.0,
            soft_turn_success_rate=0.0,
            success_rate=0.0,
            task_process_rate=0.0,
        ).to_dict()

    n = len(metrics_list)
    return {
        "tool_selection_accuracy": sum(m.tool_selection_accuracy for m in metrics_list) / n,
        "parameter_selection_accuracy": sum(m.parameter_selection_accuracy for m in metrics_list) / n,
        "tool_number_accuracy": sum(m.tool_number_accuracy for m in metrics_list) / n,
        "tool_order_accuracy": sum(m.tool_order_accuracy for m in metrics_list) / n,
        "turn_success_rate": sum(m.turn_success_rate for m in metrics_list) / n,
        "soft_turn_success_rate": sum(m.soft_turn_success_rate for m in metrics_list) / n,
        "success_rate": sum(m.success_rate for m in metrics_list) / n,
        "task_process_rate": sum(m.task_process_rate for m in metrics_list) / n,
    }


def is_mtu_bench_dataset(dataset: str) -> bool:
    """Check if the dataset is MTU-Bench."""
    return dataset in ("mtu_bench", "mtu-bench")
