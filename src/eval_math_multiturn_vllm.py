#!/usr/bin/env python3
"""
Multi-turn evaluation for MetaMathQA/MATH-500/GSM8K/TheoremQA/GPQA/MMLU/MMLU-Redux/MMLU-Pro/
Hendrycks-MATH/MTU-Bench with vLLM. Aligned to unary-feedback rollout prompt and reward parsing.
"""
import argparse
import inspect
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from benchmark import (
    MTUBenchMetrics,
    aggregate_mtu_bench_metrics,
    compute_mtu_bench_metrics,
    get_answer_checker,
    get_answer_normalizer,
    get_dataset_label,
    get_env_instruction,
    is_mtu_bench_dataset,
    load_samples,
    parse_mtu_bench_gold_field,
    parse_mtu_bench_ground_truth,
    parse_tool_calls_from_answer_tag,
    resolve_dataset_path,
)
from prompting import (
    ACTION_SEP_DEFAULT,
    DEFAULT_INCORRECT_OBS,
    MultiTurnAnswerEpisode,
    build_initial_messages,
    build_reward_message,
    parse_response,
    render_prompt,
    run_actions,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.lora.request import LoRARequest
    LORA_AVAILABLE = True
except ImportError:
    LoRARequest = None
    LORA_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

INCORRECT_OBS = os.environ.get("INCORRECT_OBS", DEFAULT_INCORRECT_OBS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-turn evaluation for MetaMathQA/MATH-500/GSM8K/TheoremQA/GPQA/MMLU/MMLU-Redux/"
            "MMLU-Pro/Hendrycks-MATH/MTU-Bench with vLLM."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Base model path or HuggingFace model name")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file for results")
    parser.add_argument("--dataset", type=str, default="metamathqa",
                        choices=[
                            "metamathqa",
                            "math500",
                            "gsm8k",
                            "theoremqa",
                            "gpqa",
                            "mmlu",
                            "mmlu-redux",
                            "mmlu_redux",
                            "mmlu_pro",
                            "hendrycks_math",
                            "mtu_bench",
                            "mtu-bench",
                        ],
                        help="Dataset to evaluate")
    parser.add_argument("--dataset_path", type=str, default="meta-math/MetaMathQA")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help=(
                            "Optional dataset config (GSM8K: main/socratic; "
                            "GPQA: gpqa_diamond; MMLU/MMLU-Redux: all subsets; "
                            "MMLU-Pro: default; Hendrycks MATH: all subsets; "
                            "MTU-Bench: config name or 'all' if available)."
                        ))
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--type_prefix", type=str, default="MATH_")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Maximum number of turns for correction")
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
            f"'{INCORRECT_OBS}'."
        ),
    )
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use tokenizer's chat template")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--fix_mistral_regex",
        dest="fix_mistral_regex",
        action="store_true",
        default=True,
        help="Fix incorrect Mistral tokenizer regex (enabled by default).",
    )
    parser.add_argument(
        "--no-fix-mistral-regex",
        dest="fix_mistral_regex",
        action="store_false",
        help="Disable Mistral tokenizer regex fix.",
    )
    parser.add_argument("--max_lora_rank", type=int, default=64,
                        help="Maximum LoRA rank for vLLM (must match your LoRA adapter's rank)")
    parser.add_argument("--max_num_seqs", type=int, default=256,
                        help="Max number of sequences for vLLM (reduce if OOM with LoRA)")
    parser.add_argument("--lora_extra_vocab_size", type=int, default=0,
                        help="Extra vocab size for LoRA (set if vocab was extended during training)")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_llm(args: argparse.Namespace) -> LLM:
    kwargs = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "max_num_seqs": args.max_num_seqs,
    }

    if args.lora_path is not None:
        if not LORA_AVAILABLE:
            LOGGER.warning("LoRA support not available in this vLLM version, ignoring --lora_path")
        else:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = args.max_lora_rank
            kwargs["max_loras"] = 1
            if args.lora_extra_vocab_size > 0:
                kwargs["lora_extra_vocab_size"] = args.lora_extra_vocab_size

    try:
        # kwargs["dtype"] = "bfloat16"
        sig = inspect.signature(LLM.__init__)
        if "disable_log_stats" in sig.parameters:
            kwargs["disable_log_stats"] = True
        if "disable_log_requests" in sig.parameters:
            kwargs["disable_log_requests"] = True
    except Exception:
        pass

    return LLM(**kwargs)


def generate_batches(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_size: int,
    lora_request: Optional[Any] = None,
    progress_bar: Optional[Any] = None,
) -> List[str]:
    outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        if lora_request is not None:
            batch_outputs = llm.generate(batch, sampling_params, lora_request=lora_request, use_tqdm=False)
        else:
            batch_outputs = llm.generate(batch, sampling_params, use_tqdm=False)

        for out in batch_outputs:
            if out.outputs:
                outputs.append(out.outputs[0].text)
            else:
                outputs.append("")

        if progress_bar is not None:
            progress_bar.update(len(batch))

    return outputs


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

    for action in executed_actions:
        traj["answer_norms"].append(traj["episode"].answer_normalizer(action))
    traj["format_penalty"] += format_penalty

    traj["actions_left"] = max(args.max_actions_per_traj - traj["episode"].step_num, 0)
    if traj["actions_left"] <= 0 and not done:
        done = True

    traj["messages"].append({"role": "assistant", "content": llm_response})
    traj["responses"].append(raw_output)
    traj["predictions"].append((" " + args.action_sep + " ").join(actions) if actions else "")
    traj["parsed_actions"].append(actions)
    traj["executed_actions"].append(executed_actions)
    traj["turn_rewards"].append(env_reward)

    correct = bool(info.get("success"))
    traj["correct_at_turn"].append(correct)
    if correct and traj["first_correct_turn"] is None:
        traj["first_correct_turn"] = turn_idx
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
                INCORRECT_OBS,
            )
        )

    traj["done"] = done or (turn_idx >= args.max_turns)
    return correct


def evaluate_multiturn(
    llm: LLM,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    lora_request: Optional[Any] = None,
) -> Dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=1,
    )

    env_instruction = get_env_instruction(args.dataset)
    answer_normalizer = get_answer_normalizer(args.dataset)
    answer_checker = get_answer_checker(args.dataset)
    trajectories = []
    for sample in samples:
        problem = sample.get("problem", "").strip()
        gold = sample.get("gold", "")
        if not problem:
            LOGGER.warning("Skipping empty problem: %s", sample.get("id", "unknown"))
            continue

        episode = MultiTurnAnswerEpisode(
            problem,
            gold,
            args.max_actions_per_traj,
            answer_normalizer=answer_normalizer,
            answer_checker=answer_checker,
            incorrect_obs=INCORRECT_OBS,
        )
        messages = build_initial_messages(
            problem,
            args.max_actions_per_traj,
            args.instruction_max_tokens,
            not args.disable_think,
            args.prompt_mode,
            env_instruction,
        )
        trajectories.append({
            "id": sample["id"],
            "problem": problem,
            "gold": gold,
            "source": sample.get("source", "unknown"),
            "messages": messages,
            "responses": [],
            "predictions": [],
            "parsed_actions": [],
            "executed_actions": [],
            "turn_rewards": [],
            "correct_at_turn": [],
            "first_correct_turn": None,
            "done": False,
            "episode": episode,
            "actions_left": args.max_actions_per_traj,
            "answer_norms": [],
            "format_penalty": 0.0,
        })

    turn_stats = {t: {"attempts": 0, "correct": 0} for t in range(1, args.max_turns + 1)}

    for turn in range(1, args.max_turns + 1):
        active_indices = [i for i, traj in enumerate(trajectories) if not traj["done"]]
        if not active_indices:
            break
        turn_stats[turn]["attempts"] = len(active_indices)

        prompts = []
        for idx in active_indices:
            traj = trajectories[idx]
            prompt = render_prompt(
                traj["messages"], tokenizer, args.use_chat_template, not args.disable_think
            )
            prompts.append(prompt)

        turn_bar = tqdm(
            total=len(prompts),
            desc=f"Turn {turn}",
            unit="sample",
            ascii=True,
            position=0,
            leave=False,
            file=sys.stdout,
        )

        outputs = generate_batches(
            llm,
            prompts,
            sampling_params,
            args.batch_size,
            lora_request=lora_request,
            progress_bar=turn_bar,
        )
        turn_bar.close()

        for idx, output in zip(active_indices, outputs):
            traj = trajectories[idx]
            correct = process_turn(traj, output, turn, args, not args.disable_think)
            if correct:
                turn_stats[turn]["correct"] += 1

    return {
        "trajectories": trajectories,
        "turn_stats": turn_stats,
    }


def compute_metrics(
    trajectories: List[Dict[str, Any]],
    turn_stats: Dict[int, Dict[str, int]],
    max_turns: int,
) -> Dict[str, Any]:
    total = len(trajectories)
    if total == 0:
        return {"error": "No samples evaluated"}

    metrics: Dict[str, Any] = {
        "total_samples": total,
        "max_turns": max_turns,
        "per_turn": {},
        "cumulative": {},
        "correction_rate": {},
    }

    for turn in range(1, max_turns + 1):
        attempts = turn_stats[turn]["attempts"]
        correct = turn_stats[turn]["correct"]
        if attempts > 0:
            metrics["per_turn"][f"turn_{turn}"] = {
                "attempts": attempts,
                "correct": correct,
                "accuracy": correct / attempts,
            }

    cumulative_correct = 0
    for turn in range(1, max_turns + 1):
        first_correct_at_turn = sum(
            1 for traj in trajectories
            if traj["first_correct_turn"] == turn
        )
        cumulative_correct += first_correct_at_turn
        metrics["cumulative"][f"by_turn_{turn}"] = {
            "correct": cumulative_correct,
            "accuracy": cumulative_correct / total,
        }

    for turn in range(2, max_turns + 1):
        wrong_at_prev = turn_stats[turn]["attempts"]
        corrected_at_turn = turn_stats[turn]["correct"]
        if wrong_at_prev > 0:
            metrics["correction_rate"][f"turn_{turn}"] = {
                "wrong_before": wrong_at_prev,
                "corrected": corrected_at_turn,
                "rate": corrected_at_turn / wrong_at_prev,
            }

    final_correct = sum(1 for traj in trajectories if traj["first_correct_turn"] is not None)
    metrics["overall"] = {
        "correct": final_correct,
        "accuracy": final_correct / total,
    }

    return metrics


# =============================================================================
# MTU-Bench Specific Evaluation Functions
# =============================================================================

MTU_BENCH_SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to tools. "
    "Always output in the following format:\n"
    "<think> [Your reasoning about what to do] </think> "
    "<answer> [tool_name(param1=value1, param2=value2)] </answer>\n"
    "If you don't need to use a tool, just put your response in <answer> tags.\n"
    "Example with tool: <think>I need to search for weather</think><answer>get_weather(city=\"Beijing\")</answer>\n"
    "Example without tool: <think>This is a simple greeting</think><answer>Hello! How can I help you?</answer>"
)

MTU_BENCH_OBSERVATION_CORRECT = "Correct!"
MTU_BENCH_OBSERVATION_INCORRECT = "Incorrect. Please think again."


def build_mtu_bench_initial_messages(
    problem: str,
    tools_description: str,
) -> List[Dict[str, str]]:
    """Build initial messages for MTU-Bench evaluation."""
    system_content = MTU_BENCH_SYSTEM_MESSAGE
    if tools_description:
        system_content += f"\n\nAvailable tools:\n{tools_description}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": problem},
    ]


def build_mtu_bench_feedback_message(
    success: bool,
    observation: str = "",
) -> Dict[str, str]:
    """Build feedback message for MTU-Bench turn."""
    if success:
        content = f"Observation: {observation or MTU_BENCH_OBSERVATION_CORRECT}"
    else:
        content = f"Observation: {observation or MTU_BENCH_OBSERVATION_INCORRECT}"
    return {"role": "user", "content": content}


def render_mtu_bench_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Optional[Any],
    use_chat_template: bool,
    enable_think: bool = True,
) -> str:
    """Render prompt for MTU-Bench with <think>/<answer> format."""
    return render_prompt(messages, tokenizer, use_chat_template, enable_think)


def evaluate_mtu_bench(
    llm: LLM,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    lora_request: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    MTU-Bench specific evaluation with tool use metrics.

    This implements the official MTU-Eval evaluation method:
    - Tool Selection Accuracy (TS)
    - Parameter Selection Accuracy (PS)
    - Tool Number Accuracy (TN)
    - Tool Order Accuracy (TO)
    - Averaged Turn Success Rate (ATS)
    - Soft Averaged Turn Success Rate (SATS)
    - Success Rate (SR)
    - Task Process Rate (TPR)
    """
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=1,
    )

    trajectories = []
    for sample in samples:
        problem = sample.get("problem", "").strip()
        gold_output = sample.get("gold_output", "")  # Full output with tool calls (ReAct format)
        gold_field = sample.get("gold", "")  # JSON format tool calls
        if not problem:
            LOGGER.warning("Skipping empty problem: %s", sample.get("id", "unknown"))
            continue

        # Parse ground truth tool calls
        # MTU-Bench stores tool calls in 'gold' field as JSON, e.g.:
        # '{"ToolName": {"param": "value"}}' or '{"": {}}' for no tool call
        gt_tool_calls = parse_mtu_bench_gold_field(gold_field)
        # Fallback to parsing gold_output if gold field didn't have tool calls
        if not gt_tool_calls and gold_output:
            gt_tool_calls = parse_mtu_bench_ground_truth(gold_output)

        # Build initial messages (no special format tokens)
        messages = build_mtu_bench_initial_messages(problem, "")

        trajectories.append({
            "id": sample["id"],
            "problem": problem,
            "gold": gold_field,
            "gold_output": gold_output,
            "gt_tool_calls": gt_tool_calls,
            "source": sample.get("source", "unknown"),
            "scenario": sample.get("scenario", "unknown"),
            "difficulty": sample.get("difficulty", "normal"),
            "messages": messages,
            "responses": [],
            "thoughts": [],  # Parsed thoughts from <think> tags
            "pred_tool_calls_per_turn": [],  # List of tool calls per turn
            "turn_metrics": [],  # Metrics per turn
            "correct_at_turn": [],  # Whether correct at each turn
            "first_correct_turn": None,  # First turn where correct
            "done": False,
        })

    # Track metrics by scenario and difficulty
    scenario_metrics: Dict[str, List[MTUBenchMetrics]] = defaultdict(list)
    difficulty_metrics: Dict[str, List[MTUBenchMetrics]] = defaultdict(list)
    all_metrics: List[MTUBenchMetrics] = []

    # Track per-turn statistics (like other benchmarks)
    turn_stats: Dict[int, Dict[str, Any]] = {
        t: {"attempts": 0, "tool_calls_made": 0, "correct_tools": 0, "correct_params": 0, "correct": 0}
        for t in range(1, args.max_turns + 1)
    }

    for turn in range(1, args.max_turns + 1):
        active_indices = [i for i, traj in enumerate(trajectories) if not traj["done"]]
        if not active_indices:
            break

        prompts = []
        enable_think = not args.disable_think
        for idx in active_indices:
            traj = trajectories[idx]
            prompt = render_mtu_bench_prompt(
                traj["messages"], tokenizer, args.use_chat_template, enable_think
            )
            prompts.append(prompt)

        turn_bar = tqdm(
            total=len(prompts),
            desc=f"MTU-Bench Turn {turn}",
            unit="sample",
            ascii=True,
            position=0,
            leave=False,
            file=sys.stdout,
        )

        outputs = generate_batches(
            llm,
            prompts,
            sampling_params,
            args.batch_size,
            lora_request=lora_request,
            progress_bar=turn_bar,
        )
        turn_bar.close()

        for idx, output in zip(active_indices, outputs):
            traj = trajectories[idx]

            # Parse tool calls from model output using <think>/<answer> format
            thought, pred_tool_calls = parse_tool_calls_from_answer_tag(output)
            traj["responses"].append(output)
            traj["thoughts"].append(thought)
            traj["pred_tool_calls_per_turn"].append(pred_tool_calls)

            # Update turn statistics
            turn_stats[turn]["attempts"] += 1
            turn_stats[turn]["tool_calls_made"] += len(pred_tool_calls)

            # Check tool correctness for this turn
            gt_calls = traj["gt_tool_calls"]
            all_tools_correct = True
            all_params_correct = True

            # Case 1: GT has no tool calls
            if not gt_calls:
                # Success if model also didn't call any tools
                all_tools_correct = (len(pred_tool_calls) == 0)
                all_params_correct = all_tools_correct
            # Case 2: GT has tool calls
            else:
                # Check if number of tool calls matches
                if len(pred_tool_calls) != len(gt_calls):
                    all_tools_correct = False
                    all_params_correct = False
                else:
                    # Check each tool call
                    for i, pred_call in enumerate(pred_tool_calls):
                        if i < len(gt_calls):
                            from benchmark import compare_tool_calls
                            tool_ok, param_ok = compare_tool_calls(pred_call, gt_calls[i])
                            if tool_ok:
                                turn_stats[turn]["correct_tools"] += 1
                            else:
                                all_tools_correct = False
                            if param_ok:
                                turn_stats[turn]["correct_params"] += 1
                            else:
                                all_params_correct = False
                        else:
                            all_tools_correct = False
                            all_params_correct = False

            # Add assistant response to messages
            traj["messages"].append({"role": "assistant", "content": output})

            # Determine success: all tool calls correct (name + params)
            is_success = all_tools_correct and all_params_correct
            traj["correct_at_turn"].append(is_success)

            if is_success:
                # Success! Mark as done and record the turn
                traj["done"] = True
                if traj["first_correct_turn"] is None:
                    traj["first_correct_turn"] = turn
                turn_stats[turn]["correct"] = turn_stats[turn].get("correct", 0) + 1
            elif turn < args.max_turns:
                # Not correct, add feedback for retry
                if not gt_calls:
                    feedback_msg = "You should not call any tool for this request. Please respond directly without tool calls."
                elif not pred_tool_calls:
                    feedback_msg = "You need to call a tool to complete this task. Please try again with the appropriate tool call."
                else:
                    feedback_msg = "The tool call was incorrect. Please check the tool name and parameters, then try again."
                feedback = build_mtu_bench_feedback_message(
                    success=False,
                    observation=feedback_msg
                )
                traj["messages"].append(feedback)
            else:
                # Last turn, mark as done even if not successful
                traj["done"] = True

    # Compute final metrics for each trajectory
    total_correct = 0
    for traj in trajectories:
        gt_calls = traj["gt_tool_calls"]
        pred_calls_per_turn = traj["pred_tool_calls_per_turn"]

        # Determine if this trajectory was successful (correct at any turn)
        is_correct = traj["first_correct_turn"] is not None
        if is_correct:
            total_correct += 1

        # Use the turn where we got it right, or the last turn if never correct
        if traj["first_correct_turn"] is not None:
            correct_turn_idx = traj["first_correct_turn"] - 1  # Convert to 0-indexed
            final_pred_calls = pred_calls_per_turn[correct_turn_idx] if correct_turn_idx < len(pred_calls_per_turn) else []
        else:
            # Use the last turn's predictions
            final_pred_calls = pred_calls_per_turn[-1] if pred_calls_per_turn else []

        # Compute metrics using the final prediction
        pred_turns = [final_pred_calls]
        gt_turns = [gt_calls] if gt_calls else [[]]

        metrics = compute_mtu_bench_metrics(pred_turns, gt_turns)
        
        # Override success rate with our new definition
        metrics.success_rate = 1.0 if is_correct else 0.0
        metrics.task_process_rate = 1.0 if is_correct else 0.0
        
        traj["metrics"] = metrics.to_dict()
        traj["is_correct"] = is_correct
        all_metrics.append(metrics)

        # Group by scenario and difficulty
        scenario = traj.get("scenario", "unknown")
        difficulty = traj.get("difficulty", "normal")
        scenario_metrics[scenario].append(metrics)
        difficulty_metrics[difficulty].append(metrics)

    # Aggregate metrics
    aggregated = aggregate_mtu_bench_metrics(all_metrics)

    # Aggregate by scenario
    scenario_aggregated = {}
    for scenario, metrics_list in scenario_metrics.items():
        scenario_aggregated[scenario] = aggregate_mtu_bench_metrics(metrics_list)
        scenario_aggregated[scenario]["count"] = len(metrics_list)

    # Aggregate by difficulty
    difficulty_aggregated = {}
    for difficulty, metrics_list in difficulty_metrics.items():
        difficulty_aggregated[difficulty] = aggregate_mtu_bench_metrics(metrics_list)
        difficulty_aggregated[difficulty]["count"] = len(metrics_list)

    return {
        "trajectories": trajectories,
        "aggregated_metrics": aggregated,
        "by_scenario": scenario_aggregated,
        "by_difficulty": difficulty_aggregated,
        "turn_stats": turn_stats,
    }


def print_mtu_bench_summary(metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print MTU-Bench evaluation summary."""
    print("\n" + "=" * 70)
    print("MTU-Bench Evaluation Summary")
    print("=" * 70)
    print(f"Model: {args.model_name_or_path}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    dataset_label = get_dataset_label(args.dataset, args.dataset_path, args.dataset_config)
    print(f"Dataset: {dataset_label}")
    print(f"Total samples: {len(metrics.get('trajectories', []))}")
    print(f"Max turns: {args.max_turns}")
    print()

    # Print per-turn statistics with correct/cumulative accuracy
    turn_stats = metrics.get("turn_stats", {})
    total_samples = len(metrics.get('trajectories', []))
    cumulative_correct = 0
    
    if turn_stats:
        print("Per-turn Statistics (like other benchmarks: success -> done, failure -> retry):")
        for turn in range(1, args.max_turns + 1):
            stats = turn_stats.get(turn, {})
            attempts = stats.get("attempts", 0)
            if attempts > 0:
                correct_this_turn = stats.get("correct", 0)
                cumulative_correct += correct_this_turn
                tool_calls = stats.get("tool_calls_made", 0)
                correct_tools = stats.get("correct_tools", 0)
                correct_params = stats.get("correct_params", 0)
                tool_acc = correct_tools / tool_calls if tool_calls > 0 else 0
                param_acc = correct_params / tool_calls if tool_calls > 0 else 0
                acc_this_turn = correct_this_turn / attempts if attempts > 0 else 0
                cumulative_acc = cumulative_correct / total_samples if total_samples > 0 else 0
                print(f"  Turn {turn}: {attempts} attempts, {correct_this_turn} correct (acc={acc_this_turn:.4f}), "
                      f"cumulative {cumulative_correct}/{total_samples} (acc={cumulative_acc:.4f})")
        print()

    agg = metrics.get("aggregated_metrics", {})
    print("Overall Metrics:")
    print(f"  Tool Selection Accuracy (TS):      {agg.get('tool_selection_accuracy', 0):.4f}")
    print(f"  Parameter Selection Accuracy (PS): {agg.get('parameter_selection_accuracy', 0):.4f}")
    print(f"  Tool Number Accuracy (TN):         {agg.get('tool_number_accuracy', 0):.4f}")
    print(f"  Tool Order Accuracy (TO):          {agg.get('tool_order_accuracy', 0):.4f}")
    print(f"  Turn Success Rate (ATS):           {agg.get('turn_success_rate', 0):.4f}")
    print(f"  Soft Turn Success Rate (SATS):     {agg.get('soft_turn_success_rate', 0):.4f}")
    print(f"  Success Rate (SR):                 {agg.get('success_rate', 0):.4f}")
    print(f"  Task Process Rate (TPR):           {agg.get('task_process_rate', 0):.4f}")
    print()

    # Print by scenario if available
    by_scenario = metrics.get("by_scenario", {})
    if by_scenario and len(by_scenario) > 1:
        print("Metrics by Scenario:")
        for scenario, scenario_metrics in sorted(by_scenario.items()):
            count = scenario_metrics.get("count", 0)
            sr = scenario_metrics.get("success_rate", 0)
            ts = scenario_metrics.get("tool_selection_accuracy", 0)
            print(f"  {scenario} (n={count}): SR={sr:.4f}, TS={ts:.4f}")
        print()

    # Print by difficulty if available
    by_difficulty = metrics.get("by_difficulty", {})
    if by_difficulty and len(by_difficulty) > 1:
        print("Metrics by Difficulty:")
        for difficulty, diff_metrics in sorted(by_difficulty.items()):
            count = diff_metrics.get("count", 0)
            sr = diff_metrics.get("success_rate", 0)
            ts = diff_metrics.get("tool_selection_accuracy", 0)
            print(f"  {difficulty} (n={count}): SR={sr:.4f}, TS={ts:.4f}")
        print()

    print("=" * 70)


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.max_turns < 1:
        raise ValueError("--max_turns must be >= 1")
    if args.prompt_mode == "simple" and args.disable_think:
        LOGGER.info("prompt_mode=simple with disable_think enabled.")

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_DISABLE_LOG_STATS", "1")
    os.environ.setdefault("VLLM_DISABLE_LOG_REQUESTS", "1")
    for name in ("vllm", "vllm.engine", "vllm.worker"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    LOGGER.info("Loading dataset: %s", args.dataset)
    samples = load_samples(args)
    if not samples:
        LOGGER.error("No samples loaded!")
        sys.exit(1)
    LOGGER.info("Total samples to evaluate: %d", len(samples))

    LOGGER.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=args.fix_mistral_regex,
    )

    LOGGER.info("Creating vLLM engine...")
    llm = create_llm(args)

    lora_request = None
    if args.lora_path is not None and LORA_AVAILABLE:
        LOGGER.info("Loading LoRA adapter from: %s", args.lora_path)
        lora_request = LoRARequest(
            lora_name="eval_lora",
            lora_int_id=1,
            lora_path=args.lora_path,
        )

    # Check if this is MTU-Bench dataset - use specialized evaluation
    if is_mtu_bench_dataset(args.dataset):
        LOGGER.info("Starting MTU-Bench tool use evaluation...")
        overall_bar = tqdm(
            total=len(samples),
            desc="MTU-Bench Eval",
            unit="sample",
            ascii=True,
            position=1,
            leave=True,
            file=sys.stdout,
        )

        results = evaluate_mtu_bench(llm, tokenizer, samples, args, lora_request)
        overall_bar.update(len(samples))
        overall_bar.close()

        # Build output for MTU-Bench
        output = {
            "config": {
                "model_name_or_path": args.model_name_or_path,
                "lora_path": args.lora_path,
                "dataset": args.dataset,
                "dataset_path": resolve_dataset_path(args.dataset, args.dataset_path),
                "dataset_split": args.dataset_split,
                "dataset_config": args.dataset_config,
                "max_samples": args.max_samples,
                "max_turns": args.max_turns,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "metrics": results["aggregated_metrics"],
            "turn_stats": {
                str(k): v for k, v in results["turn_stats"].items()
            },  # Per-turn statistics
            "metrics_by_scenario": results["by_scenario"],
            "metrics_by_difficulty": results["by_difficulty"],
            "trajectories": [
                {
                    "id": traj["id"],
                    "problem": traj["problem"],
                    "gold": traj["gold"],
                    "gold_output": traj.get("gold_output", ""),
                    "source": traj["source"],
                    "scenario": traj.get("scenario", "unknown"),
                    "difficulty": traj.get("difficulty", "normal"),
                    "responses": traj["responses"],
                    "messages": traj["messages"],
                    "is_correct": traj.get("is_correct", False),
                    "first_correct_turn": traj.get("first_correct_turn"),
                    "correct_at_turn": traj.get("correct_at_turn", []),
                    "metrics": traj.get("metrics", {}),
                }
                for traj in results["trajectories"]
            ],
        }

        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        LOGGER.info("Results saved to: %s", args.output_json)
        print_mtu_bench_summary(results, args)

    else:
        # Standard math evaluation for other datasets
        LOGGER.info("Starting multi-turn evaluation...")
        overall_bar = tqdm(
            total=len(samples),
            desc="Overall",
            unit="sample",
            ascii=True,
            position=1,
            leave=True,
            file=sys.stdout,
        )

        results = evaluate_multiturn(llm, tokenizer, samples, args, lora_request)
        overall_bar.update(len(samples))
        overall_bar.close()

        metrics = compute_metrics(
            results["trajectories"],
            results["turn_stats"],
            args.max_turns,
        )

        output = {
            "config": {
                "model_name_or_path": args.model_name_or_path,
                "lora_path": args.lora_path,
                "dataset": args.dataset,
                "dataset_path": resolve_dataset_path(args.dataset, args.dataset_path),
                "dataset_split": args.dataset_split,
                "dataset_config": args.dataset_config,
                "type_prefix": args.type_prefix,
                "max_samples": args.max_samples,
                "max_turns": args.max_turns,
                "prompt_mode": args.prompt_mode,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "timestamp": datetime.now().isoformat(),
            },
            "metrics": metrics,
            "trajectories": [
                {
                    "id": traj["id"],
                    "problem": traj["problem"],
                    "gold": traj["gold"],
                    "source": traj["source"],
                    "predictions": traj["predictions"],
                    "correct_at_turn": traj["correct_at_turn"],
                    "first_correct_turn": traj["first_correct_turn"],
                    "messages": traj["messages"],
                    "format_penalty": traj["format_penalty"],
                }
                for traj in results["trajectories"]
            ],
        }

        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        LOGGER.info("Results saved to: %s", args.output_json)

        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Model: {args.model_name_or_path}")
        if args.lora_path:
            print(f"LoRA: {args.lora_path}")
        dataset_label = get_dataset_label(
            args.dataset,
            args.dataset_path,
            args.dataset_config,
        )
        print(f"Dataset: {dataset_label}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Max turns: {args.max_turns}")
        print()

        print("Per-turn Accuracy (samples reaching that turn):")
        for turn in range(1, args.max_turns + 1):
            key = f"turn_{turn}"
            if key in metrics["per_turn"]:
                data = metrics["per_turn"][key]
                print(f"  Turn {turn}: {data['correct']}/{data['attempts']} = {data['accuracy']:.4f}")
        print()

        print("Cumulative Success Rate:")
        for turn in range(1, args.max_turns + 1):
            key = f"by_turn_{turn}"
            if key in metrics["cumulative"]:
                data = metrics["cumulative"][key]
                print(f"  By Turn {turn}: {data['correct']}/{metrics['total_samples']} = {data['accuracy']:.4f}")
        print()

        print("Correction Rate (wrong → correct):")
        for turn in range(2, args.max_turns + 1):
            key = f"turn_{turn}"
            if key in metrics["correction_rate"]:
                data = metrics["correction_rate"][key]
                print(f"  Turn {turn}: {data['corrected']}/{data['wrong_before']} = {data['rate']:.4f}")
        print()

        print(f"Overall Accuracy: {metrics['overall']['correct']}/{metrics['total_samples']} = {metrics['overall']['accuracy']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
