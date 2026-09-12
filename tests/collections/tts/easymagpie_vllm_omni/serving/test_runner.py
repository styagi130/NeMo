# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for EasyMagpie streaming metadata on vLLM-Omni 0.26."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.runner import (
    EasyMagpieGPUARModelRunner,
    GPUARModelRunner,
    merge_streaming_additional_information,
)

WORKER_CLS = "easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker"


@pytest.mark.parametrize("padded,active", [(8, 3), (16, 9), (40, 34), (128, 65)])
def test_async_output_preserves_distinct_rows_after_contraction_and_slot_reuse(padded, active):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(omni_pooler_payload_include_hidden=False)
    hidden = torch.zeros(padded, 3)
    for epoch, count in enumerate((active, 1, active)):
        codes = torch.arange(padded * 2).view(padded, 2) + epoch * 1000
        snapshot = runner._build_omni_async_snapshot_payload(
            hidden_states=hidden,
            staged_hidden_states_cpu=None,
            multimodal_outputs={"codes": {"audio": codes}},
        )
        carrier = snapshot.get("hidden_states", hidden[:0])
        assert carrier.shape == (padded, 0)
        for row in range(count):
            output = runner._build_omni_mm_payload(
                combined_multimodal_outputs=None,
                mm_cpu={"codes.audio": codes},
                rid=f"request-{epoch}-{row}",
                idx=row,
                start=row,
                end=row + 1,
                audio_sparse_output=False,
                sparse_mm_index={},
                hidden_seq_len=carrier.shape[0],
                scheduled_seq_len=count,
            )
            torch.testing.assert_close(output["codes.audio"], codes[row : row + 1])


def test_async_payload_retains_requested_hidden_states():
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(omni_pooler_payload_include_hidden=True)
    hidden = torch.ones(8, 3)
    snapshot = runner._build_omni_async_snapshot_payload(
        hidden_states=hidden, staged_hidden_states_cpu=hidden, multimodal_outputs={}
    )
    assert snapshot["hidden_states"] is hidden
    assert snapshot["staged_hidden_states_cpu"] is hidden


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resident_updates_own_storage_across_steps_and_requests(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.requests = {key: SimpleNamespace() for key in ("a", "b")}
    runner.model_intermediate_buffer = {}
    runner.model = SimpleNamespace(gpu_resident_buffer_keys={"last_audio_codes", ("hidden_states", "last")})
    codes = torch.tensor([[1, 2]], device=device)
    hidden = torch.tensor([[3.0]], device=device)
    for req_id in ("a", "b"):
        runner._update_intermediate_buffer(req_id, {"last_audio_codes": codes, "hidden_states": {"last": hidden}})
    codes.fill_(9)
    hidden.fill_(10)
    runner._update_intermediate_buffer("a", {"last_audio_codes": codes, "decode_offset": 2})

    a, b = (runner.model_intermediate_buffer[key] for key in ("a", "b"))
    torch.testing.assert_close(a["last_audio_codes"], codes)
    torch.testing.assert_close(b["last_audio_codes"], torch.tensor([[1, 2]], device=device))
    for key, cached in (("a", a), ("b", b)):
        torch.testing.assert_close(cached["hidden_states"]["last"], torch.tensor([[3.0]], device=device))
        assert runner.requests[key].additional_information_cpu is cached
        assert cached["last_audio_codes"].device == codes.device
        assert cached["last_audio_codes"].data_ptr() != codes.data_ptr()
    assert a["last_audio_codes"].data_ptr() != b["last_audio_codes"].data_ptr()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eager_postprocess_owns_feedback_before_producer_and_slot_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.has_phoneme = True
    model._dec_phoneme_tokens = torch.arange(10, 18, device=device).view(8, 1)
    codes = torch.arange(16, device=device).view(8, 2)
    hidden = torch.zeros(8, 3, device=device)
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = model
    runner.requests = {key: SimpleNamespace() for key in ("a", "b")}
    runner.model_intermediate_buffer = {}
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="latent"))
    runner._request_needs_downstream_stage_payload = lambda req_id: True

    def eager_postprocess(req_ids, lengths, offsets):
        return runner._maybe_run_eager_omni_postprocess_before_async_output(
            hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": codes}},
            num_scheduled_tokens_np=np.array(lengths),
            scheduler_output=None,
            req_ids_output_copy=req_ids,
            query_start_loc_cpu=np.array(offsets),
        )

    # Mixed two-token prefill plus one-token decode, padded to eight rows.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        assert eager_postprocess(["a", "b"], [2, 1], [0, 2, 3])
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::cat") == 1
    model._dec_phoneme_tokens.fill_(99)
    codes.fill_(99)
    for key, phoneme, audio in (("a", 11, [2, 3]), ("b", 12, [4, 5])):
        cached = runner.model_intermediate_buffer[key]
        torch.testing.assert_close(cached["last_phoneme_token"], torch.tensor([[phoneme]], device=device))
        torch.testing.assert_close(cached["last_audio_codes"], torch.tensor([audio], device=device))

    # The remaining request moves to row zero without overwriting the other cache.
    assert eager_postprocess(["b"], [1], [0, 1])
    model._dec_phoneme_tokens.zero_()
    codes.zero_()
    cached = runner.model_intermediate_buffer
    torch.testing.assert_close(cached["a"]["last_audio_codes"], torch.tensor([[2, 3]], device=device))
    torch.testing.assert_close(cached["b"]["last_audio_codes"], torch.tensor([[99, 99]], device=device))
    torch.testing.assert_close(cached["b"]["last_phoneme_token"], torch.tensor([[99]], device=device))
    assert "last_hidden" not in model.gpu_resident_buffer_keys

    runner._request_needs_downstream_stage_payload = lambda req_id: req_id != "a"
    codes.fill_(7)
    assert eager_postprocess(["a", "b"], [1, 1], [0, 1, 2])
    torch.testing.assert_close(cached["a"]["last_audio_codes"], torch.tensor([[2, 3]], device=device))
    runner.requests["c"] = SimpleNamespace()
    codes.fill_(8)
    assert eager_postprocess(["c"], [1], [0, 1])
    torch.testing.assert_close(cached["b"]["last_audio_codes"], torch.tensor([[7, 7]], device=device))
    torch.testing.assert_close(cached["c"]["last_audio_codes"], torch.tensor([[8, 8]], device=device))


@pytest.mark.parametrize("count", [1, 32, 64, 128])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eager_feedback_packs_by_dtype_and_keeps_writable_rows_independent(monkeypatch, count, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = _feedback_runner(count)
    integers = torch.arange(count * 3, device=device).view(count, 3)
    flags = torch.ones(count, dtype=torch.bool, device=device)

    def postprocess(self, **kwargs):
        for index in range(count):
            self._update_intermediate_buffer(
                str(index), {"last_audio_codes": integers[index : index + 1], "phoneme_ended": flags[index]}
            )
        return True

    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", postprocess)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        assert runner._maybe_run_eager_omni_postprocess_before_async_output()
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::cat") == 2
    integers.fill_(-1)
    flags.zero_()
    for index in range(count):
        cached = runner.model_intermediate_buffer[str(index)]
        expected = torch.arange(index * 3, index * 3 + 3, device=device).view(1, 3)
        torch.testing.assert_close(cached["last_audio_codes"], expected)
        assert cached["phoneme_ended"].dtype == torch.bool
        assert cached["phoneme_ended"].shape == ()
        assert cached["phoneme_ended"].item()
        assert runner.requests[str(index)].additional_information_cpu is cached
        cached["last_audio_codes"].fill_(100)


def test_eager_feedback_cuda_snapshot_survives_next_step_and_copy_stream(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = _feedback_runner(2)
    work_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    copied = [torch.empty((1, 3), dtype=torch.long, pin_memory=True) for _ in range(2)]

    def postprocess(self, **kwargs):
        for index in range(2):
            self._update_intermediate_buffer(str(index), {"last_audio_codes": codes[index : index + 1]})

    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", postprocess)
    with torch.cuda.stream(work_stream):
        codes = torch.arange(6, device="cuda").view(2, 3)
        runner._maybe_run_eager_omni_postprocess_before_async_output()
        copy_stream.wait_stream(work_stream)
        with torch.cuda.stream(copy_stream):
            for index, destination in enumerate(copied):
                snapshot = runner.model_intermediate_buffer[str(index)]["last_audio_codes"]
                destination.copy_(snapshot, non_blocking=True)
                snapshot.record_stream(copy_stream)
        del snapshot
        codes.fill_(99)
        runner._maybe_run_eager_omni_postprocess_before_async_output()
    copy_stream.synchronize()
    work_stream.synchronize()
    for index, destination in enumerate(copied):
        torch.testing.assert_close(destination, torch.arange(index * 3, index * 3 + 3).view(1, 3))
        torch.testing.assert_close(
            runner.model_intermediate_buffer[str(index)]["last_audio_codes"],
            torch.full((1, 3), 99, dtype=torch.long, device="cuda"),
        )


def test_eager_feedback_does_not_defer_other_threads_or_ordinary_stores(monkeypatch):
    runner = _feedback_runner(2)
    codes = torch.tensor([[1, 2]])

    def other_thread():
        runner._update_intermediate_buffer("1", {"last_audio_codes": codes})
        codes.fill_(9)
        torch.testing.assert_close(runner.model_intermediate_buffer["1"]["last_audio_codes"], torch.tensor([[1, 2]]))

    def postprocess(self, **kwargs):
        self._update_intermediate_buffer("0", {"last_audio_codes": torch.tensor([[3, 4]])})
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(other_thread).result()
        return True

    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", postprocess)
    assert runner._maybe_run_eager_omni_postprocess_before_async_output()
    runner._update_intermediate_buffer("0", {"last_audio_codes": codes})
    codes.zero_()
    torch.testing.assert_close(runner.model_intermediate_buffer["0"]["last_audio_codes"], torch.tensor([[9, 9]]))


def test_eager_feedback_preserves_partial_updates_and_cleans_up_after_error(monkeypatch):
    runner = _feedback_runner(1)
    codes = torch.tensor([[1, 2]])

    def postprocess(self, **kwargs):
        self._update_intermediate_buffer("0", {"last_audio_codes": codes})
        raise ValueError("postprocess failed")

    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", postprocess)
    with pytest.raises(ValueError, match="postprocess failed"):
        runner._maybe_run_eager_omni_postprocess_before_async_output()
    codes.fill_(9)
    torch.testing.assert_close(runner.model_intermediate_buffer["0"]["last_audio_codes"], torch.tensor([[1, 2]]))
    runner._update_intermediate_buffer("0", {"last_audio_codes": codes})
    codes.zero_()
    torch.testing.assert_close(runner.model_intermediate_buffer["0"]["last_audio_codes"], torch.tensor([[9, 9]]))


def test_eager_feedback_packing_error_never_publishes_unowned_views(monkeypatch):
    runner = _feedback_runner(1)
    codes = torch.tensor([[1, 2]])
    concatenate = torch.cat
    calls = []

    def postprocess(self, **kwargs):
        self._update_intermediate_buffer("0", {"last_audio_codes": codes, "phoneme_ended": torch.tensor(True)})

    def fail_pack(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("packing failed")
        return concatenate(*args, **kwargs)

    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", postprocess)
    monkeypatch.setattr(torch, "cat", fail_pack)
    with pytest.raises(RuntimeError, match="packing failed"):
        runner._maybe_run_eager_omni_postprocess_before_async_output()
    assert runner.model_intermediate_buffer["0"] == {}
    runner._update_intermediate_buffer("0", {"last_audio_codes": codes})
    codes.zero_()
    torch.testing.assert_close(runner.model_intermediate_buffer["0"]["last_audio_codes"], torch.tensor([[1, 2]]))


def test_eager_feedback_keeps_last_write_and_unsupported_layout_fallback(monkeypatch):
    runner = _feedback_runner(1)
    noncontiguous = torch.arange(6).view(2, 3).t()

    def postprocess(self, **kwargs):
        self._update_intermediate_buffer("0", {"last_audio_codes": torch.tensor([[1, 2]])})
        self._update_intermediate_buffer("0", {"last_audio_codes": None})
        self._update_intermediate_buffer("0", {"last_phoneme_token": noncontiguous})
        noncontiguous.fill_(9)

    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", postprocess)
    runner._maybe_run_eager_omni_postprocess_before_async_output()
    cached = runner.model_intermediate_buffer["0"]
    assert cached["last_audio_codes"] is None
    torch.testing.assert_close(cached["last_phoneme_token"], torch.arange(6).view(2, 3).t())


@pytest.mark.parametrize("nested", ["preprocess", "postprocess"])
def test_preprocess_copy_scope_is_nested_and_thread_local(monkeypatch, nested):
    runner = _feedback_runner(3)
    runner._update_intermediate_buffer("0", {"phoneme_ended": torch.tensor(False)})
    flag, codes = torch.tensor(True), torch.tensor([[3, 4]])
    depth = 0

    def other_thread():
        other = torch.tensor(True)
        runner._update_intermediate_buffer("2", {"phoneme_ended": other})
        other.fill_(False)
        assert runner.model_intermediate_buffer["2"]["phoneme_ended"]

    def inner(self, **kwargs):
        self._update_intermediate_buffer("1", {"phoneme_ended": flag})
        raise ValueError("nested failure")

    def preprocess(self, *args, **kwargs):
        nonlocal depth
        self._maybe_run_batch_preprocess([], torch.device("cpu"))
        if depth:
            return inner(self)
        depth += 1
        self._update_intermediate_buffer("0", {"phoneme_ended": flag, "last_audio_codes": codes, "decode_offset": 9})
        assert self.model_intermediate_buffer["0"]["decode_offset"] == 9
        codes.fill_(99)
        torch.testing.assert_close(self.model_intermediate_buffer["0"]["last_audio_codes"], torch.tensor([[3, 4]]))
        with pytest.raises(ValueError, match="nested failure"):
            if nested == "preprocess":
                self._preprocess(None, 0)
            else:
                self._maybe_run_eager_omni_postprocess_before_async_output()
        assert self._feedback_copy_context.phase == "preprocess"
        assert self.model_intermediate_buffer["1"]["phoneme_ended"]
        assert self.model_intermediate_buffer["0"]["phoneme_ended"]
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(other_thread).result()
        return "prepared"

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", inner)
    assert runner._preprocess(None, 0) == "prepared"
    flag.fill_(False)
    assert all(info["phoneme_ended"] for info in runner.model_intermediate_buffer.values())
    assert runner._feedback_copy_context.pending is None and runner._feedback_copy_context.phase is None


@pytest.mark.parametrize("nested", ["preprocess", "postprocess"])
@pytest.mark.parametrize("replacement", [None, False])
def test_nested_copy_scope_keeps_same_target_last_write(monkeypatch, nested, replacement):
    runner = _feedback_runner(1)
    depth = 0

    def inner(self, **kwargs):
        value = None if replacement is None else torch.tensor(replacement)
        self._update_intermediate_buffer("0", {"phoneme_ended": value})

    def preprocess(self, *args, **kwargs):
        nonlocal depth
        self._maybe_run_batch_preprocess([], torch.device("cpu"))
        if depth:
            return inner(self)
        depth += 1
        self._update_intermediate_buffer("0", {"phoneme_ended": torch.tensor(True)})
        if nested == "preprocess":
            self._preprocess(None, 0)
        else:
            self._maybe_run_eager_omni_postprocess_before_async_output()

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    monkeypatch.setattr(GPUARModelRunner, "_maybe_run_eager_omni_postprocess_before_async_output", inner)
    runner._preprocess(None, 0)
    value = runner.model_intermediate_buffer["0"]["phoneme_ended"]
    assert value is None if replacement is None else not value


@pytest.mark.parametrize("replacement", [None, torch.tensor(7), torch.tensor([True, False])])
def test_preprocess_copy_scope_preserves_last_write_and_unsupported_flag_fallback(monkeypatch, replacement):
    runner = _feedback_runner(1)
    flag = torch.tensor(True)
    expected = replacement.clone() if isinstance(replacement, torch.Tensor) else replacement

    def preprocess(self, *args, **kwargs):
        self._maybe_run_batch_preprocess([], torch.device("cpu"))
        self._update_intermediate_buffer("0", {"phoneme_ended": flag})
        self._update_intermediate_buffer("0", {"phoneme_ended": replacement})
        if isinstance(replacement, torch.Tensor):
            replacement.zero_()
        return None

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    runner._preprocess(None, 0)
    actual = runner.model_intermediate_buffer["0"]["phoneme_ended"]
    if expected is None:
        assert actual is None
    else:
        torch.testing.assert_close(actual, expected)


def test_preprocess_ended_cuda_snapshot_survives_next_step_and_copy_stream(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = _feedback_runner(2)
    work, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    copied = [torch.empty((), dtype=torch.bool, pin_memory=True) for _ in range(2)]

    def preprocess(self, *args, **kwargs):
        self._maybe_run_batch_preprocess([], flags.device)
        for index in range(2):
            self._update_intermediate_buffer(str(index), {"phoneme_ended": flags[index]})

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    with torch.cuda.stream(work):
        flags = torch.tensor([True, False], device="cuda")
        runner._preprocess(None, 0)
        copy_stream.wait_stream(work)
        with torch.cuda.stream(copy_stream):
            for index, destination in enumerate(copied):
                snapshot = runner.model_intermediate_buffer[str(index)]["phoneme_ended"]
                destination.copy_(snapshot, non_blocking=True)
                snapshot.record_stream(copy_stream)
        flags.logical_not_()
        runner._preprocess(None, 0)
    copy_stream.synchronize()
    work.synchronize()
    assert [bool(value) for value in copied] == [True, False]
    assert [bool(runner.model_intermediate_buffer[str(i)]["phoneme_ended"]) for i in range(2)] == [False, True]


@pytest.mark.parametrize("async_chunk", [False, True])
def test_async_snapshot_reader_does_not_consume_pending_ended_flag(monkeypatch, async_chunk):
    from vllm_omni.worker import gpu_ar_model_runner

    runner = _feedback_runner(1)

    class NoFeedbackRead(dict):
        def get(self, key, default=None):
            assert key != "phoneme_ended"
            return super().get(key, default)

        def __getitem__(self, key):
            assert key != "phoneme_ended"
            return super().__getitem__(key)

        def items(self):
            raise AssertionError("Async output must not enumerate live feedback")

    state = NoFeedbackRead(phoneme_ended=torch.tensor(False), omni_final_stage_id=1)
    runner.model_intermediate_buffer["0"] = state
    runner.requests["0"].additional_information_cpu = state
    runner.model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(runner.model)
    runner.model_config = SimpleNamespace(
        engine_output_type="latent", stage_id=0, stage_connector_config={"extra": {"role": "sender"}}
    )
    runner.vllm_config = SimpleNamespace(model_config=runner.model_config)
    runner.omni_prefix_cache = None
    runner._async_chunk = async_chunk
    runner.supports_mm_inputs = False
    runner._downstream_payload_cache = {}
    runner._stage_deferred_prefix_cache_mm_outputs = lambda **kwargs: None
    runner._should_accumulate_full_payload_output = lambda: False
    runner._should_return_omni_routed_experts = lambda: False
    runner.get_omni_connector_output = lambda: None
    partition_calls = []
    partition = gpu_ar_model_runner.partition_payload_list

    def partition_output(payload):
        partition_calls.append(True)
        return partition(payload)

    monkeypatch.setattr(gpu_ar_model_runner, "partition_payload_list", partition_output)

    def output():
        assert runner.model.eager_omni_postprocess_before_async_output
        assert not runner.model.postprocess_uses_req_infos
        return runner._build_omni_model_runner_output_from_snapshot(
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=1),
            hidden_states=torch.empty(1, 0),
            staged_hidden_states_cpu=None,
            multimodal_outputs={"codes.audio": torch.tensor([[4, 5]])},
            req_ids_output_copy=["0"],
            req_id_to_index_output_copy={"0": 0},
            valid_sampled_token_ids=[[0]],
            logprobs_lists=None,
            prompt_logprobs_dict={},
            num_nans_in_logits=None,
            kv_connector_output=None,
            ec_connector_output=None,
            cudagraph_stats=None,
            kv_extracted_req_ids=None,
            num_scheduled_tokens_np=np.array([1]),
            query_start_loc_cpu=np.array([0, 1]),
            postprocess_already_applied=True,
        )

    def preprocess(self, *args, **kwargs):
        # No model hook is needed by this reader-only fixture.
        monkeypatch.setattr(self.model, "preprocess_batch", lambda **kwargs: None)
        self._maybe_run_batch_preprocess([], torch.device("cpu"))
        self._update_intermediate_buffer("0", {"phoneme_ended": torch.tensor(True)})
        assert not dict.__getitem__(state, "phoneme_ended")
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(output).result()
        assert result.req_ids == ["0"]
        torch.testing.assert_close(result.inter_stage_outputs[0]["codes.audio"], torch.tensor([[4, 5]]))

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    runner._preprocess(None, 0)
    assert dict.__getitem__(state, "phoneme_ended")
    assert bool(partition_calls) == async_chunk


def test_streaming_update_preserves_model_state_and_replaces_latest_chunk():
    cached = {
        "decode_offset": 7,
        "text_tokens": [10, 20],
        "text_token": [20],
        "meta": {"num_processed_tokens": 3},
    }

    merged = merge_streaming_additional_information(cached, {"text_token": [30]})

    assert merged["decode_offset"] == 7
    assert merged["text_tokens"] == [10, 20]
    assert merged["text_token"] == [30]
    assert merged["meta"]["num_processed_tokens"] == 0
    assert merged["meta"]["resumable"] is True


def test_streaming_update_accumulates_declared_tensor_keys():
    cached = {"hidden_states": {"output": torch.tensor([[1.0]])}}
    incoming = {"hidden_states": {"output": torch.tensor([[2.0]])}}

    merged = merge_streaming_additional_information(
        cached,
        incoming,
        accumulated_keys={("hidden_states", "output")},
    )

    torch.testing.assert_close(merged["hidden_states"]["output"], torch.tensor([[1.0], [2.0]]))


def test_deploy_configs_select_compatibility_worker_for_lm():
    for filename in ("easymagpie_lm.yaml", "easymagpie.yaml"):
        deploy = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / filename).read_text())
        lm_stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 0)
        assert lm_stage["engine_extras"]["worker_cls"] == WORKER_CLS


def _feedback_runner(count):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.requests = {str(index): SimpleNamespace() for index in range(count)}
    runner.model_intermediate_buffer = {}
    runner.model = SimpleNamespace(
        gpu_resident_buffer_keys={"last_audio_codes", "last_phoneme_token", "phoneme_ended"}
    )
    return runner
