#!/usr/bin/env python3
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

SYSTEM_MESSAGE = "You're a helpful assistant. "
DEFAULT_ENV_INSTRUCTION = (
    "You are solving Math problems. Only give the final answer between <answer> and </answer>."
)
ACTION_SEP_DEFAULT = "||"
SPECIAL_TOKENS = ("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>")
DEFAULT_INCORRECT_OBS = "Incorrect. Please think again."
DONE_OBS = "Correct!"
PENALTY_LAMBDA = 0.5


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text.strip().lower())


def build_turn_block(
    turn: int,
    state: str,
    actions_left: int,
    instruction_max_tokens: int,
    enable_think: bool,
) -> str:
    format_prompt = (
        "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
        if enable_think
        else "<answer> [your answer] </answer>"
    )
    length_prompt = f"Max response length: {instruction_max_tokens} words (tokens)."
    return (
        f"Turn {turn}:\n"
        f"State:\n{state}\n"
        f"You have {actions_left} actions left. Always output: {format_prompt} "
        f"with no extra text. Strictly follow this format. {length_prompt}\n"
    )


def build_initial_messages(
    state: str,
    actions_left: int,
    instruction_max_tokens: int,
    enable_think: bool,
    prompt_mode: str,
    env_instruction: str = DEFAULT_ENV_INSTRUCTION,
    system_message: str = SYSTEM_MESSAGE,
) -> List[Dict[str, str]]:
    format_prompt = (
        "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
        if enable_think
        else "<answer> [your answer] </answer>"
    )

    if prompt_mode == "simple":
        user_content = (
            env_instruction
            + "\n\n"
            + "Problem:\n"
            + state
            + "\n\n"
            + f"Always output: {format_prompt} with no extra text. Strictly follow this format."
        )
    else:
        user_content = env_instruction + "\n" + build_turn_block(
            1, state, actions_left, instruction_max_tokens, enable_think
        )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_reward_message(
    reward: float,
    turn: int,
    state: str,
    actions_left: int,
    instruction_max_tokens: int,
    enable_think: bool,
    prompt_mode: str,
    incorrect_obs: str = DEFAULT_INCORRECT_OBS,
) -> Dict[str, str]:
    if prompt_mode == "simple":
        return {"role": "user", "content": incorrect_obs}
    content = f"Reward:\n{reward}\n" + build_turn_block(
        turn, state, actions_left, instruction_max_tokens, enable_think
    )
    return {"role": "user", "content": content}


def parse_response(
    response: str,
    enable_think: bool,
    action_sep: str,
    max_actions_per_turn: int,
) -> Tuple[str, List[str]]:
    pattern = (
        r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
        if enable_think
        else r"<answer>(.*?)</answer>"
    )
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return response, []
    if enable_think:
        think_content = match.group(1)
        action_content = match.group(2)
    else:
        think_content = ""
        action_content = match.group(1)

    for token in SPECIAL_TOKENS:
        think_content = think_content.replace(token, "").strip()
        action_content = action_content.replace(token, "").strip()

    actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
    if max_actions_per_turn > 0 and len(actions) > max_actions_per_turn:
        actions = actions[:max_actions_per_turn]
        action_content = (" " + action_sep + " ").join(actions)

    if enable_think:
        llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
    else:
        llm_response = f"<answer>{action_content}</answer>"
    return llm_response, actions


class MultiTurnAnswerEpisode:
    def __init__(
        self,
        question: str,
        answer: str,
        max_steps: int,
        answer_normalizer: Optional[Callable[[str], str]] = None,
        answer_checker: Optional[Callable[[str, str], Tuple[bool, bool]]] = None,
        incorrect_obs: str = DEFAULT_INCORRECT_OBS,
        done_obs: str = DONE_OBS,
        penalty_lambda: float = PENALTY_LAMBDA,
    ):
        self.question = question
        self.correct_answer = answer or ""
        self.max_steps = max_steps
        self.answer_normalizer = answer_normalizer or normalize_answer
        self.answer_checker = answer_checker
        self.incorrect_obs = incorrect_obs
        self.done_obs = done_obs
        self.penalty_lambda = penalty_lambda
        self.step_num = 0
        self.unique_answers_count = defaultdict(int)
        self.total_valid_answers = 0
        self.step_rewards: List[float] = []
        self.state = question

    def _check_answer(self, user_answer: str) -> Tuple[bool, bool]:
        user_answer = user_answer.strip()
        if self.answer_checker is not None:
            return self.answer_checker(user_answer, self.correct_answer)
        normalized = self.answer_normalizer(user_answer)
        is_valid = normalized != ""
        if not self.correct_answer:
            return False, is_valid
        is_correct = normalized == self.answer_normalizer(self.correct_answer)
        return is_correct, is_valid

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        is_correct, is_valid = self._check_answer(action)
        reward = 1.0 if is_correct else 0.0
        info: Dict[str, Any] = {"action_is_valid": is_valid, "success": is_correct}

        if is_valid:
            normalized = self.answer_normalizer(action)
            self.unique_answers_count[normalized] += 1
            self.total_valid_answers += 1
            self.step_rewards.append(reward)
            unique_ratio = (
                len(self.unique_answers_count) / self.total_valid_answers
                if self.total_valid_answers > 0
                else 0.0
            )
            info["per_question_unique_answers_ratio"] = unique_ratio
            self.step_num += 1

        if is_correct or self.step_num >= self.max_steps:
            penalty = (
                self.penalty_lambda
                * (1 - (len(self.unique_answers_count) / self.total_valid_answers))
                if self.total_valid_answers > 0
                else 0.0
            )
            total_reward = sum(self.step_rewards) - penalty
            info["global_repetition_penalty"] = penalty
            info["final_total_reward"] = total_reward
            self.state = self.done_obs
            return self.state, reward, True, info

        self.state = self.incorrect_obs
        return self.state, reward, False, info


def render_prompt(
    messages: List[Dict[str, str]],
    tokenizer: Optional[Any],
    use_chat_template: bool,
    enable_think: bool,
) -> str:
    if not use_chat_template:
        raise ValueError("Prompt rendering requires --use_chat_template.")
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; cannot render prompts.")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text + ("<think>" if enable_think else "<answer>")


def clone_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


def run_actions(
    episode: MultiTurnAnswerEpisode,
    actions: List[str],
) -> Tuple[float, bool, Dict[str, Any], List[str]]:
    acc_reward = 0.0
    done = False
    info: Dict[str, Any] = {}
    executed_actions: List[str] = []
    for action in actions:
        _, reward, done, info = episode.step(action)
        acc_reward += reward
        executed_actions.append(action)
        if done:
            break
    return acc_reward, done, info, executed_actions
