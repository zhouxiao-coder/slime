import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.data_source import RolloutDataSource
from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from slime.utils.sample_validations import ValidationArgs, validate_samples_for_training
from slime.utils.types import Sample


def _stub_sglang_dependencies():
    """Install lightweight stubs for optional SGLang imports used in rollout code."""

    def ensure_module(name: str):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]

    sglang = ensure_module("sglang")
    srt = ensure_module("sglang.srt")
    sglang.srt = srt

    server_args = ensure_module("sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    utils = ensure_module("sglang.srt.utils")
    utils.kill_process_tree = lambda *_args, **_kwargs: None

    entrypoints = ensure_module("sglang.srt.entrypoints")
    http_server = ensure_module("sglang.srt.entrypoints.http_server")
    http_server.launch_server = lambda *_args, **_kwargs: None
    entrypoints.http_server = http_server

    srt.server_args = server_args
    srt.utils = utils
    srt.entrypoints = entrypoints

    ensure_module("sglang_router")


@pytest.fixture(scope="module")
def rollout_module():
    _stub_sglang_dependencies()
    sys.modules.pop("slime.ray.rollout", None)
    return importlib.import_module("slime.ray.rollout")


def test_sample_round_trip_preserves_spec_and_status():
    sample = Sample(
        group_index=2,
        index=7,
        response="hello",
        response_length=5,
        reward=1.5,
        status=Sample.Status.TRUNCATED,
    )
    sample.spec_info.add({"spec_accept_token_num": 2, "spec_draft_token_num": 4, "spec_verify_ct": 1}, 5)

    clone = Sample.from_dict(sample.to_dict())

    assert clone.status == Sample.Status.TRUNCATED
    assert clone.spec_info.spec_accept_rate == pytest.approx(0.5)
    assert clone.spec_info.spec_accept_length == pytest.approx(5.0)


def test_rollout_data_source_assigns_group_and_sample_indices():
    args = SimpleNamespace(
        rollout_global_dataset=False,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
    )

    data_source = RolloutDataSource(args)
    first = data_source.get_samples(2)
    second = data_source.get_samples(1)

    assert [[s.group_index for s in group] for group in first] == [[0, 0], [1, 1]]
    assert [s.index for group in first for s in group] == [0, 1, 2, 3]

    assert [[s.group_index for s in second[0]]] == [[2, 2]]
    assert [s.index for s in second[0]] == [4, 5]


def test_validation_exports_training_ready_payload_and_masks():
    samples = [
        Sample(
            index=0,
            tokens=[1, 2, 3, 4, 5],
            response_length=3,
            loss_mask=[1, 0, 1],
            reward={"score": 1.0},
            metadata={"raw_reward": 42, "round_number": 3},
            rollout_log_probs=[0.1, 0.2, 0.3],
            rollout_routed_experts=[[0], [1], [2]],
            train_metadata={"loss": "ce"},
            multimodal_inputs={"image": torch.zeros(1)},
        ),
        Sample(
            index=1,
            tokens=[6, 7, 8],
            response_length=2,
            loss_mask=None,
            reward={"score": 2.0},
            metadata={"raw_reward": 42, "round_number": 3},
            rollout_log_probs=[0.4, 0.5],
            rollout_routed_experts=[[3], [4]],
            train_metadata={"loss": "kl"},
            multimodal_inputs={"audio": torch.ones(1)},
        ),
    ]

    args = ValidationArgs(reward_key="score", advantage_estimator="gspo", rewards_normalization=True, n_samples_per_prompt=2)
    train_payload = validate_samples_for_training(samples, args)

    assert train_payload["loss_masks"][0] == [1, 0, 1]
    assert train_payload["loss_masks"][1] == [1, 1]
    assert train_payload["raw_reward"] == [42, 42]
    assert train_payload["response_lengths"] == [3, 2]
    assert "rollout_log_probs" in train_payload and "rollout_routed_experts" in train_payload
    assert train_payload["truncated"] == [0, 0]
    assert train_payload["sample_indices"] == [0, 1]
    assert "metadata" in train_payload and "multimodal_inputs" in train_payload


def test_check_reward_zero_std_flags(monkeypatch):
    args = SimpleNamespace(reward_key=None)
    samples = [Sample(reward=1.0), Sample(reward=1.0)]

    output = check_reward_nonzero_std(args, samples)
    assert not output.keep
    assert output.reason.startswith("zero_std")


def test_compute_metrics_respects_effective_length_and_categories(rollout_module):
    args = SimpleNamespace(
        reward_key="score",
        advantage_estimator="grpo",
        sglang_speculative_algorithm="spec",
        log_reward_category="cat",
    )

    samples = [
        Sample(
            group_index=0,
            response="abc",
            response_length=5,
            loss_mask=[1, 1, 0, 0, 0],
            reward={"score": 1.0, "cat": "ok"},
            status=Sample.Status.TRUNCATED,
        ),
        Sample(
            group_index=0,
            response="abcd",
            response_length=4,
            reward={"score": 1.0, "cat": "ok"},
            status=Sample.Status.COMPLETED,
        ),
        Sample(
            group_index=1,
            response="xyz",
            response_length=3,
            loss_mask=[1, 1, 1],
            reward={"score": 2.0, "cat": "bad"},
            status=Sample.Status.COMPLETED,
        ),
    ]

    samples[0].spec_info.add({"spec_accept_token_num": 1, "spec_draft_token_num": 2, "spec_verify_ct": 1}, 5)
    samples[1].spec_info.add({"spec_accept_token_num": 3, "spec_draft_token_num": 6, "spec_verify_ct": 2}, 4)

    metrics = rollout_module.compute_metrics_from_samples(args, samples)

    assert metrics["response_len/mean"] == pytest.approx(3.0)
    assert metrics["response_len/median"] == pytest.approx(3.0)
    assert metrics["truncated_ratio"] == pytest.approx(1 / 3)
    assert metrics["rollout/spec_accept_rate"] == pytest.approx((0.5 + 0.5 + 0) / 3)
    assert metrics["rollout/spec_accept_length"] == pytest.approx((5 + 2 + 0) / 3)
    assert metrics["error_cat/ok"] == pytest.approx(2 / 3)
    assert metrics["error_cat/bad"] == pytest.approx(1 / 3)
    assert metrics["repetition_frac"] == 0
    assert metrics["zero_std/count_1.0"] == 1
    assert metrics["zero_std/count_2.0"] == 1
