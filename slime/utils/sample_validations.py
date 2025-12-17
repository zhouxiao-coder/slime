"""Utilities that validate :class:`slime.utils.types.Sample` objects.

These helpers focus on catching subtle mismatches in the fields that downstream
training and logging utilities assume (e.g., loss mask shape, reward layout,
and rollout metadata).  They are intentionally dependency-light so they can be
run as a smoke test after refactoring functions that consume or return
``Sample``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import torch

from slime.utils.types import Sample


@dataclass
class ValidationArgs:
    """Minimal argument container for validation.

    The actual training/rollout code relies on a large ``args`` object.  For
    validation we only care about the subset of fields that affect Sample
    handling.  ``ValidationArgs`` mirrors the behaviour of
    ``RolloutManager._post_process_rewards`` so downstream consumers can use it
    as a lightweight drop-in during contract checks.
    """

    reward_key: str | None = None
    advantage_estimator: str = "ppo"
    rewards_normalization: bool = False
    n_samples_per_prompt: int = 1
    rollout_batch_size: int = 1
    grpo_std_normalization: bool = False


def _normalize_rewards(args: ValidationArgs | SimpleNamespace, raw_rewards: list[float]):
    """Normalize rewards exactly like the rollout conversion step.

    Returns both the raw rewards and the possibly normalized rewards list so
    callers can surface mismatches in either quantity.
    """

    adv = getattr(args, "advantage_estimator", None)
    if adv not in ["grpo", "gspo", "reinforce_plus_plus_baseline"]:
        return raw_rewards, raw_rewards

    if not getattr(args, "rewards_normalization", False):
        return raw_rewards, raw_rewards

    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
    rollout_batch_size = getattr(args, "rollout_batch_size", 1)
    if rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size:
        rewards = rewards.reshape(-1, n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])

    rewards = rewards - rewards.mean(dim=-1, keepdim=True)
    if adv in ["grpo", "gspo"] and getattr(args, "grpo_std_normalization", False):
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)

    return raw_rewards, rewards.flatten().tolist()


def validate_samples_for_training(
    samples: Iterable[Sample], args: ValidationArgs | SimpleNamespace | None = None
):
    """Validate and export a batch of samples.

    The validation mirrors ``RolloutManager._convert_samples_to_train_data`` so
    regressions in Sample manipulation are caught early without the Ray or
    SGLang dependencies.  ``args`` may be a ``ValidationArgs``, a thin
    ``SimpleNamespace`` that provides the same attributes, or ``None`` (which
    disables reward normalization and defaults to ``reward_key=None``).

    A ``ValueError`` is raised when invariants are violated.
    """

    samples = list(samples)
    if not samples:
        raise ValueError("Samples cannot be empty")

    args = args or ValidationArgs()
    raw_rewards, rewards = _normalize_rewards(args, [s.get_reward_value(args) for s in samples])

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    loss_masks = []
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError(
                f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            )
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if samples[0].multimodal_inputs is not None:
        train_data["multimodal_inputs"] = [sample.multimodal_inputs for sample in samples]

    if "teacher_log_probs" in samples[0].__dict__:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    return train_data
