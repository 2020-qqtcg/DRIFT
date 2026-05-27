#!/usr/bin/env python3
import argparse
import inspect
import logging
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import IterableDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    PEFT_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

ROLE_TOKENS_FALLBACK = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
}

TRAJECTORY_PENALTY_LAMBDA = 0.5


def compute_weights(returns: List[float], beta: float) -> List[float]:
    if beta <= 0:
        raise ValueError("--beta must be > 0")
    scaled = [r / beta for r in returns]
    m = max(scaled) if scaled else 0.0
    exp_shift = [math.exp(s - m) for s in scaled]
    denom = (sum(exp_shift) / max(len(exp_shift), 1)) + 1e-12
    return [v / denom for v in exp_shift]


def compute_trajectory_penalty(answer_norms: Any) -> float:
    if not isinstance(answer_norms, list):
        return 0.0
    total_turns = len(answer_norms)
    if total_turns <= 0:
        return 0.0
    seen: Set[str] = set()
    new_count = 0
    for answer_norm in answer_norms:
        if not answer_norm:
            continue
        answer_text = str(answer_norm)
        if answer_text in seen:
            continue
        seen.add(answer_text)
        new_count += 1
    penalty = TRAJECTORY_PENALTY_LAMBDA * (1.0 - (new_count / total_turns))
    return max(0.0, penalty)


def compute_discounted_return(turn_rewards: Any, gamma: float) -> float:
    if not isinstance(turn_rewards, dict):
        return 0.0
    turns: List[Tuple[int, float]] = []
    for key, value in turn_rewards.items():
        if not isinstance(key, str) or not key.startswith("t"):
            continue
        turn_id = key[1:]
        if not turn_id.isdigit():
            continue
        try:
            reward = float(value)
        except (TypeError, ValueError):
            reward = 0.0
        turns.append((int(turn_id), reward))
    if not turns:
        return 0.0
    total = 0.0
    for turn_idx, reward in sorted(turns, key=lambda item: item[0]):
        total += (gamma ** (turn_idx - 1)) * reward
    return total


def add_weights_to_dataset(dataset, args: argparse.Namespace):
    if "prompt_id" not in dataset.column_names:
        raise ValueError("prompt_id is required to compute weights.")
    if "turn_rewards" not in dataset.column_names:
        raise ValueError("turn_rewards is required to compute weights.")

    if "weight" in dataset.column_names:
        dataset = dataset.remove_columns(["weight"])

    LOGGER.info("Computing weights with gamma=%.4f, beta=%.4f", args.gamma, args.beta)
    prompt_to_indices: Dict[str, List[int]] = {}
    returns: List[float] = []

    for example in dataset:
        prompt_id = example.get("prompt_id")
        if prompt_id is None or prompt_id == "":
            raise ValueError("prompt_id must be non-empty for weight computation.")
        idx = len(returns)
        prompt_key = str(prompt_id)
        prompt_to_indices.setdefault(prompt_key, []).append(idx)

        discounted_return = compute_discounted_return(example.get("turn_rewards"), args.gamma)
        repeat_penalty = 0.0
        if "answer_norms" in example:
            repeat_penalty = compute_trajectory_penalty(example.get("answer_norms"))

        format_penalty = 0.0
        if "format_penalty" in example:
            try:
                format_penalty = float(example.get("format_penalty") or 0.0)
            except (TypeError, ValueError):
                format_penalty = 0.0
        elif "trajectory_penalty" in example:
            try:
                format_penalty = float(example.get("trajectory_penalty") or 0.0)
            except (TypeError, ValueError):
                format_penalty = 0.0

        returns.append(discounted_return + format_penalty - repeat_penalty)

    weights = [1.0] * len(returns)
    for indices in prompt_to_indices.values():
        group_returns = [returns[i] for i in indices]
        group_weights = compute_weights(group_returns, args.beta)
        for i, weight in zip(indices, group_weights):
            weights[i] = float(weight)

    dataset = dataset.add_column("weight", weights)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weighted SFT for last-assistant responses with prompt/response masking."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_split_ratio", type=float, default=0.0)
    parser.add_argument("--eval_split_seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--disable_shuffle", "--no_shuffle", action="store_true")
    parser.add_argument("--deepspeed_config_path", type=str, default=None)
    parser.add_argument("--zero_stage", type=int, choices=[2, 3], default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--truncation_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--lora_bias", type=str, choices=["none", "all", "lora_only"], default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_determinism(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            LOGGER.warning("torch.use_deterministic_algorithms(True) not supported on this setup.")


def get_ds_config_path(args: argparse.Namespace) -> Optional[str]:
    if args.deepspeed_config_path:
        return args.deepspeed_config_path
    filename = f"deepspeed_zero{args.zero_stage}.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if os.path.exists(path):
        return path
    LOGGER.warning("DeepSpeed config not found at %s; proceeding without DeepSpeed config.", path)
    return None


def filter_training_args_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        LOGGER.warning("Dropping unsupported TrainingArguments keys: %s", ", ".join(dropped))
    return filtered


def maybe_wrap_with_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for --use_lora. Install with: pip install peft")
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name.")
    try:
        task_type = TaskType[args.lora_task_type.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported --lora_task_type {args.lora_task_type}") from exc

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.lora_bias,
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized = []
    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be a dict")
        role = str(msg.get("role", "user")).lower()
        if role not in valid_roles:
            LOGGER.warning("Unknown role '%s'; treating as 'user'.", role)
            role = "user"
        content = msg.get("content", "")
        if content is None:
            content = ""
        normalized.append({"role": role, "content": str(content)})
    return normalized


def split_prompt_answer_messages(
    messages: Any,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    messages = normalize_messages(messages)
    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        return None, None
    if last_assistant_idx < len(messages) - 1:
        LOGGER.warning(
            "Ignoring %d messages after last assistant turn.",
            len(messages) - 1 - last_assistant_idx,
        )
    prompt_messages = messages[:last_assistant_idx]
    answer = messages[last_assistant_idx]["content"]
    return prompt_messages, answer


def render_fallback_prompt(messages: List[Dict[str, str]]) -> str:
    rendered = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prefix = ROLE_TOKENS_FALLBACK.get(role, ROLE_TOKENS_FALLBACK["user"])
        rendered.append(f"{prefix}{content}\n")
    rendered.append(ROLE_TOKENS_FALLBACK["assistant"])
    return "".join(rendered)


def render_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Optional[Any],
) -> str:
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return render_fallback_prompt(messages)


def prepare_dataset(dataset):
    original_len = len(dataset)

    def _split(example: Dict[str, Any]) -> Dict[str, Any]:
        if "weight" not in example:
            raise ValueError("Each example must include 'weight'")
        messages = example.get("messages")
        prompt_messages, response = split_prompt_answer_messages(messages)
        valid = prompt_messages is not None and response is not None
        return {
            "prompt_messages": prompt_messages or [],
            "response": response or "",
            "weight": float(example["weight"]),
            "valid": bool(valid and response),
        }

    dataset = dataset.map(
        _split,
        remove_columns=dataset.column_names,
        desc="Preparing last-assistant samples",
        load_from_cache_file=False,
    )
    dataset = dataset.filter(lambda ex: ex["valid"], desc="Filtering invalid samples")
    dataset = dataset.remove_columns(["valid"])
    filtered_len = len(dataset)
    filtered_out = original_len - filtered_len
    if filtered_out > 0:
        LOGGER.warning(
            "Filtered out %d/%d samples without last assistant responses.",
            filtered_out,
            original_len,
        )
    return dataset


class WeightedPromptAnswerCollator:
    def __init__(self, tokenizer, max_prompt_length: int, max_completion_length: int):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts: List[str] = []
        texts: List[str] = []
        weights: List[float] = []
        eos_token = self.tokenizer.eos_token or ""

        for feature in features:
            prompt_messages = feature.get("prompt_messages")
            response = feature.get("response", "")
            if not isinstance(prompt_messages, list):
                raise ValueError("prompt_messages must be a list")
            prompt = render_prompt(prompt_messages, self.tokenizer)
            prompts.append(prompt)
            texts.append(prompt + response + eos_token)
            weights.append(float(feature.get("weight", 1.0)))

        weights_t = torch.tensor(weights, dtype=torch.float32)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length + self.max_completion_length,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()

        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        prompt_lens = prompt_tokens["attention_mask"].sum(dim=1)

        for i in range(labels.size(0)):
            labels[i, : prompt_lens[i]] = -100

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "weights": weights_t,
        }


class WeightedSFTTrainer(Trainer):
    def __init__(self, *args, disable_shuffle: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_shuffle = disable_shuffle

    def get_train_sampler(self):
        if not self.disable_shuffle:
            return super().get_train_sampler()

        if self.train_dataset is None:
            return None
        if isinstance(self.train_dataset, IterableDataset):
            return None

        world_size = getattr(self.args, "world_size", 1)
        if world_size <= 1:
            return SequentialSampler(self.train_dataset)

        rank = getattr(self.args, "process_index", None)
        if rank is None:
            rank = getattr(self.args, "local_rank", 0)
        if rank is None or rank < 0:
            rank = 0

        return DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        mask = (shift_labels != -100).float()
        sum_loss = (token_loss * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        per_sample_loss = sum_loss / denom

        if os.environ.get("BATCH_WEIGHT_NORMAL", "False") == "False":
            weights = weights.to(per_sample_loss.device).to(per_sample_loss.dtype)
            loss = (per_sample_loss * weights).mean()
        else:
            weights = weights.to(per_sample_loss.device).to(per_sample_loss.dtype)
            weighted_sum = (per_sample_loss * weights).sum()
            weight_sum = weights.sum().clamp(min=1e-8)

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(weighted_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(weight_sum, op=torch.distributed.ReduceOp.SUM)

            loss = weighted_sum / weight_sum
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    setup_logging()
    set_determinism(args.seed, args.deterministic)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = args.truncation_side
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            LOGGER.warning("Tokenizer has no pad_token or eos_token; using 0 as pad_token_id.")

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    attn_impl = args.attn_implementation or None
    if attn_impl and attn_impl.lower() in ("none", "null"):
        attn_impl = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
    )
    model = maybe_wrap_with_lora(model, args)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    ds_config_path = get_ds_config_path(args)

    raw_train_dataset = load_dataset("json", data_files={"train": args.train_jsonl})["train"]
    raw_train_dataset = add_weights_to_dataset(raw_train_dataset, args)

    eval_dataset = None
    if args.eval_split_ratio and args.eval_split_ratio > 0:
        split_seed = args.eval_split_seed if args.eval_split_seed is not None else args.seed
        split = raw_train_dataset.train_test_split(
            test_size=args.eval_split_ratio, seed=split_seed, shuffle=False
        )
        train_dataset = prepare_dataset(split["train"])
        eval_dataset = prepare_dataset(split["test"])
        if len(eval_dataset) == 0:
            LOGGER.warning("Eval split is empty after filtering; disabling eval.")
            eval_dataset = None
    else:
        train_dataset = prepare_dataset(raw_train_dataset)

    if len(train_dataset) == 0:
        raise ValueError("No train samples after filtering (missing last assistant).")

    eval_batch_size = args.per_device_eval_batch_size or args.per_device_train_batch_size
    eval_steps = args.eval_steps or args.logging_steps

    report_to = [] if args.report_to == "none" else args.report_to.split(",")
    training_args_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": report_to,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "logging_first_step": True,
        "deepspeed": ds_config_path,
        "seed": args.seed,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if eval_dataset is not None:
        if "evaluation_strategy" in params:
            training_args_kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in params:
            training_args_kwargs["eval_strategy"] = "steps"
        elif "do_eval" in params:
            training_args_kwargs["do_eval"] = True
        if "eval_steps" in params:
            training_args_kwargs["eval_steps"] = eval_steps
    else:
        if "evaluation_strategy" in params:
            training_args_kwargs["evaluation_strategy"] = "no"
        elif "eval_strategy" in params:
            training_args_kwargs["eval_strategy"] = "no"
        elif "do_eval" in params:
            training_args_kwargs["do_eval"] = False

    if args.run_name and "run_name" in params:
        training_args_kwargs["run_name"] = args.run_name

    training_args = TrainingArguments(**filter_training_args_kwargs(training_args_kwargs))

    data_collator = WeightedPromptAnswerCollator(
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        disable_shuffle=args.disable_shuffle,
    )
    try:
        trainer = WeightedSFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = WeightedSFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
