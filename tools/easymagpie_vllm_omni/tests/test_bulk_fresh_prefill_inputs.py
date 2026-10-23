# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Paired actual native preprocessing on tiny CPU heads, without a forward.

The distributed first-rank query and preexisting connector collection leaves
are isolated. The installed local hook, native Omni runner loop, native model
preprocessing, metadata store, and original complete-Q1 path are exercised.
No learned checkpoint, CUDA, serving, PCM or inherited test suite is executed.
"""
import copy
import gc
import weakref
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import vllm_omni.worker.gpu_model_runner as runtime
from easymagpie_vllm_omni import bulk_fresh_prefill_inputs as H
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration as Model
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner as Runner
from torch import nn

ENV = "EASYMAGPIE_BULK_FRESH_PREFILL_INPUTS"
CPU = torch.device("cpu")


class NoForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sentinel", torch.arange(13, dtype=torch.float32))
        self.temperature = 0.9
        self.top_k = 17

    def forward(self, *args, **kwargs):
        raise AssertionError("preprocessing called a learned forward")


class ContextTokenizer:
    def encode_context(self, text):
        return [2, 3] if text == "[EN]" else [4, 5]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [6, 7] if text else []


def bits(tensor):
    tensor = tensor.detach().contiguous().reshape(-1)
    return str(tensor.dtype), tuple(tensor.shape), bytes(tensor.view(torch.uint8).tolist())


def frozen(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), bits(value))
    if isinstance(value, dict):
        return {key: frozen(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(frozen(item) for item in value))
    return value


def make_model(tmp_path, dtype=torch.float16, *, deployed=False):
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.embedding_dim = 8
    stacks = 1 if deployed else 2
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=stacks, condition_on_user_speech=False)
    model.num_task_embeddings = 0 if deployed else 2
    model.task_embedding = None if deployed else nn.Embedding(2, 8, dtype=dtype)
    model.text_embedding = nn.Embedding(32, 8, dtype=dtype)
    model.phoneme_embeddings = nn.ModuleList([nn.Embedding(8, 8, dtype=dtype) for _ in range(stacks)])
    model.has_phoneme = True
    model.phonemes_delay = 2
    model.phoneme_bos_id = 1
    model.text_eos_id = 31
    model.register_buffer("_combined_embeddings", torch.zeros(1, 8, dtype=dtype))
    model._prefill_cache = {}
    model._text_tokenizer = ContextTokenizer()
    model.model_path = str(tmp_path)
    model.backbone = NoForward()
    model.code_predictor = NoForward()
    model.has_preprocess = True
    model._batched_decode_inputs_enabled = False
    model._device_decode_state = None
    model._prefill_graphs_enabled = False
    model._prepared_decode_updates = {}
    model._decode_zero_embedding = torch.zeros((1, 8), dtype=dtype)
    # Only the existing GPU DDS preparation leaf is suppressed. Its real
    # complete-Q1 consumer remains installed and is exercised separately.
    model.prepare_decode_batch = None
    directory = tmp_path / "speaker_embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    speaker_rows = 65 if deployed else 64
    speaker = (torch.arange(speaker_rows * 8, dtype=torch.float32).reshape(speaker_rows, 8) - 256) / 100
    torch.save(speaker, directory / "eng.pt")
    with torch.no_grad():
        layers = [
            layer
            for layer in (model.task_embedding, model.text_embedding, *model.phoneme_embeddings)
            if layer is not None
        ]
        for index, layer in enumerate(layers):
            values = (torch.arange(layer.weight.numel()).reshape_as(layer.weight) % 19 - 9) / 16
            layer.weight.copy_(values + index / 32)
            layer.weight[0].zero_()
            layer.weight[0, 1] = -0.0
    model.eval()
    with torch.inference_mode():
        value = Model._build_prefill_embeds(model, CPU, info())
        assert value.shape == (71, 8)
    return model


def info(prefix=(6, 7, 8, 9), **changes):
    result = dict(
        speaker_id="eng",
        context_text="[EN]",
        task_mode_id=0,
        text_prefill_num=4,
        prefill_text_tokens=list(prefix),
        prefill_offset=0,
        text_tokens=[6, 7, 8, 9, 31],
        temperature=0.7,
        top_k=11,
        untouched={"value": [1, 2, 3]},
    )
    result.update(changes)
    return result


def empty_runner(model):
    runner = Runner.__new__(Runner)
    runner.model = model
    runner.device = CPU
    runner.requests = {}
    runner.input_batch = SimpleNamespace(req_ids=[], num_reqs=0, num_computed_tokens_cpu=np.empty(0, dtype=np.int32))
    runner.execute_model_state = None
    runner.model_intermediate_buffer = {}
    runner.has_talker_mtp = False
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(is_encoder_decoder=False, async_chunk=True)
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    runner.vllm_config = SimpleNamespace(model_config=runner.model_config, parallel_config=runner.parallel_config)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner._leaf_calls = []
    runner._kwargs = {"fixture_untouched": object()}
    runner._init_model_kwargs = MethodType(lambda self: self._kwargs, runner)
    runner._collect_additional_information_for_prefill = MethodType(
        lambda self, counts: self._leaf_calls.append(("collect", tuple(map(int, counts)))), runner
    )
    runner._update_additional_information = MethodType(
        lambda self, output: self._leaf_calls.append(("additional", output.total_num_scheduled_tokens)), runner
    )
    return runner


def populate(runner, requests, counts=None, computed=None, padding=5):
    count = len(requests)
    counts = [71] * count if counts is None else list(counts)
    computed = [0] * count if computed is None else list(computed)
    req_ids = ["owner-" + str(index) for index in range(count)]
    starts = [0]
    for length in counts:
        starts.append(starts[-1] + length)
    length = starts[-1] + padding
    runner.requests = {
        key: SimpleNamespace(prompt_token_ids=[0] * 71, additional_information_cpu=None) for key in req_ids
    }
    runner.input_batch = SimpleNamespace(
        req_ids=req_ids, num_reqs=count, num_computed_tokens_cpu=np.array(computed, dtype=np.int32)
    )
    runner.model_intermediate_buffer = {key: copy.deepcopy(value) for key, value in zip(req_ids, requests)}
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor(starts, dtype=torch.int32))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(length, dtype=torch.int32) + 100)
    runner.inputs_embeds = SimpleNamespace(
        gpu=torch.full((length, 8), -37.25, dtype=runner.model._combined_embeddings.dtype)
    )
    runner.positions = torch.arange(length, dtype=torch.int64) + 31
    output = SimpleNamespace(total_num_scheduled_tokens=starts[-1], num_scheduled_tokens=dict(zip(req_ids, counts)))
    return output, length


def install(runner):
    state = H.BulkFreshInputs(runner, runner.get_model())
    runner._bulk_fresh_prefill_inputs = state
    return state


def worker(runner):
    return SimpleNamespace(model_runner=runner, is_codec=False)


def native_state(runner):
    model = runner.model
    return dict(
        parameters={name: frozen(value) for name, value in model.named_parameters()},
        context=frozen(model._prefill_cache),
        backbone=frozen(model.backbone.state_dict()),
        predictor=frozen(model.code_predictor.state_dict()),
        RNG=frozen(torch.random.get_rng_state()),
    )


def observations(runner):
    return dict(
        ids=frozen(runner.input_ids.gpu),
        embeds=frozen(runner.inputs_embeds.gpu),
        positions=frozen(runner.positions),
        infos=frozen(runner.model_intermediate_buffer),
        requests={key: frozen(value.additional_information_cpu) for key, value in runner.requests.items()},
        prepared=frozen(runner.model._prepared_decode_updates),
        sampling=(runner.model.code_predictor.temperature, runner.model.code_predictor.top_k),
        leaves=runner._leaf_calls,
    )


def invoke(runner, output, length):
    before = native_state(runner)
    result = runtime.OmniGPUModelRunner._preprocess(runner, output, length)
    assert result[0].data_ptr() == runner.input_ids.gpu.data_ptr()
    assert result[1].data_ptr() == runner.inputs_embeds.gpu.data_ptr()
    assert result[2].data_ptr() == runner.positions.data_ptr()
    assert result[3] is None and result[4] is runner._kwargs and result[5] is None
    assert native_state(runner) == before
    for key, request in runner.requests.items():
        assert request.additional_information_cpu is runner.model_intermediate_buffer[key]
    return result


def pair(
    monkeypatch,
    tmp_path,
    requests,
    *,
    dtype=torch.float16,
    counts=None,
    computed=None,
    padding=5,
    setup=None,
    deployed=False,
):
    native = empty_runner(make_model(tmp_path / "native", dtype, deployed=deployed))
    candidate = empty_runner(make_model(tmp_path / "candidate", dtype, deployed=deployed))
    monkeypatch.setenv(ENV, "1")
    assert install(candidate) is not None
    a = populate(native, requests, counts, computed, padding)
    b = populate(candidate, requests, counts, computed, padding)
    if setup is not None:
        setup(native)
        setup(candidate)
    before = frozen(candidate.input_ids.gpu), frozen(candidate.inputs_embeds.gpu)
    with torch.inference_mode():
        invoke(native, *a)
        invoke(candidate, *b)
    assert observations(native) == observations(candidate)
    return native, candidate, a, before


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(runtime, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    yield
    assert not torch.cuda.is_initialized()


def test_arbitrary_prefix_lengths_and_order_are_byte_exact_fp16(monkeypatch, tmp_path):
    requests = [info(prefix) for prefix in ((), (0,), (9, 0), (0, 9, 0), (9, 8, 7, 6), (0, 0, 0, 0))]
    _, candidate, _, _ = pair(monkeypatch, tmp_path, requests, deployed=True)
    assert candidate.model.task_embedding is None and len(candidate.model.phoneme_embeddings) == 1
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"]["bulk_rows"] == len(requests)


def test_arbitrary_prefix_and_phoneme_add_order_are_byte_exact_fp32(monkeypatch, tmp_path):
    _, candidate, _, _ = pair(monkeypatch, tmp_path, [info((0, 2, 0, 3)), info((3, 0))], dtype=torch.float32)
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"]["bulk_rows"] == 2


def test_maximum_115_rows_nonzero_offsets_and_padded_tail(monkeypatch, tmp_path):
    requests = [info(tuple((index + item) % 32 for item in range(index % 5))) for index in range(115)]
    _, candidate, (output, _), _ = pair(monkeypatch, tmp_path, requests, padding=11, deployed=True)
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"]["bulk_rows"] == 115
    end = output.total_num_scheduled_tokens
    assert torch.equal(candidate.input_ids.gpu[end:], torch.arange(end, end + 11, dtype=torch.int32) + 100)
    assert torch.equal(candidate.inputs_embeds.gpu[end:], torch.full((11, 8), -37.25, dtype=torch.float16))


def test_metadata_mirrors_and_sampling_setters_preserve_native_row_order(monkeypatch, tmp_path):
    requests = [
        info(temperature=0.25, top_k=3),
        info(temperature=None, top_k=None, topk=21),
        info(temperature=1.5, top_k=7),
        info(temperature=None, top_k=None),
    ]
    # None top_k has native precedence over topk. Missing keys preserve state.
    requests[1].pop("top_k")
    requests[3].pop("top_k")
    _, candidate, _, _ = pair(monkeypatch, tmp_path, requests)
    assert candidate.model.code_predictor.temperature == 1.5
    assert candidate.model.code_predictor.top_k == 7
    for index, key in enumerate(candidate.input_batch.req_ids):
        value = candidate.model_intermediate_buffer[key]
        assert value["request_id"] == key and value["prefill_offset"] == 0
        assert value["decode_offset"] == 4 and value["_omni_num_computed_tokens"] == 0
        assert value["_omni_prompt_len"] == value["duplex_prompt_len"] == 71
        assert value["_omni_is_prefill"] is True


def test_original_complete_q1_hook_runs_first_and_consumes_updates_once(monkeypatch, tmp_path):
    def setup(runner):
        model = runner.model
        model._batched_decode_inputs_enabled = True
        model._device_decode_state = object()
        model._prepared_decode_updates = {index: {"decode_offset": index + 8} for index in range(3)}

    _, candidate, _, _ = pair(monkeypatch, tmp_path, [info()] * 3, counts=[1] * 3, computed=[71] * 3, setup=setup)
    assert candidate._batched_decode_input_hits == 1
    assert candidate._batched_decode_input_requests == 3
    assert candidate.model._prepared_decode_updates == {}
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"].get("bulk_rows", 0) == 0


@pytest.mark.parametrize("multimodal", [False, True])
def test_named_speaker_bulk_prefill_requires_no_request_multimodal_features(tmp_path, multimodal):
    runner = empty_runner(make_model(tmp_path))
    state = install(runner)
    populate(runner, [info(), info()])
    runner.supports_mm_inputs = True
    if multimodal:
        runner.requests["owner-0"].mm_features = [object()]
    before = observations(runner)
    with torch.inference_mode():
        prepared = state.try_prepare(runner.input_ids.gpu, runner.inputs_embeds.gpu, np.array([71, 71], np.int32))
    assert prepared is not multimodal
    if multimodal:
        assert observations(runner) == before


@pytest.mark.parametrize("replacement", [{"text": "replacement"}, {"text_token": [9, 10], "text_token_start": 0}])
def test_fresh_prefill_replaces_previous_target_text(monkeypatch, tmp_path, replacement):
    _, candidate, _, _ = pair(monkeypatch, tmp_path, [info(**replacement), info()])
    assert candidate._bulk_fresh_prefill_inputs.counts["bulk_rows"] == 0
    expected = [6, 7, 31] if "text" in replacement else [9, 10]
    assert candidate.model_intermediate_buffer["owner-0"]["text_tokens"] == expected


def test_unsupported_task_cannot_reuse_a_warm_context(tmp_path):
    runner = empty_runner(make_model(tmp_path, deployed=True))
    state = install(runner)
    populate(runner, [info(task_mode_id=1), info(task_mode_id=1)])
    before = observations(runner)
    with torch.inference_mode():
        assert not state.try_prepare(runner.input_ids.gpu, runner.inputs_embeds.gpu, np.array([71, 71], np.int32))
        with pytest.raises(ValueError, match="Only task_mode_id=0"):
            runner.model._build_prefill_embeds(CPU, info(task_mode_id=1))
    assert observations(runner) == before


def test_mixed_q1_and_cold_rows_fall_back_to_entire_native_loop(monkeypatch, tmp_path):
    def setup(runner):
        runner.model._prepared_decode_updates = {71: {"decode_offset": 9, "text_tokens": [6, 7, 31]}}

    _, candidate, _, _ = pair(
        monkeypatch, tmp_path, [info(), info(), info((7, 8))], counts=[71, 1, 71], computed=[0, 71, 0], setup=setup
    )
    assert candidate.model._prepared_decode_updates == {}
    counts = candidate._bulk_fresh_prefill_inputs.status()["counts"]
    assert counts["bulk_rows"] == 0 and counts["mixed_batches"] == 1
    assert counts["mixed_fresh_rows"] == counts["candidate_fresh_rows"] == 2
    assert counts["attempted_prefill_batches"] == counts["native_fallback_batches"] == 1
    assert counts["native_fallback_rows"] == 3


def test_singleton_and_partial_resumed_native_fallback(monkeypatch, tmp_path):
    for index, (requests, counts, computed) in enumerate(
        [
            ([info()], [71], [0]),
            ([info(prefill_offset=1), info(prefill_offset=1)], [70, 70], [1, 1]),
        ]
    ):
        _, candidate, _, _ = pair(monkeypatch, tmp_path / str(index), requests, counts=counts, computed=computed)
        assert candidate._bulk_fresh_prefill_inputs.status()["counts"].get("bulk_rows", 0) == 0

    def one_token_prefill(runner):
        runner.model._prepared_decode_updates = {0: {"decode_offset": 1}}

    _, candidate, _, _ = pair(
        monkeypatch,
        tmp_path / "one-token-prefill",
        [info(), info()],
        counts=[1, 71],
        computed=[0, 0],
        setup=one_token_prefill,
    )
    assert candidate.model_intermediate_buffer["owner-0"]["_omni_is_prefill"] is True
    # Prefill does not consume a decode update; the runner clears it after the batch.
    assert candidate.model._prepared_decode_updates == {0: {"decode_offset": 1}}
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"]["bulk_rows"] == 0


def test_nested_information_and_raw_reference_keep_native_fallback(monkeypatch, tmp_path):
    requests = [
        info(additional_information={"temperature": 0.1, "prefill_text_tokens": [1]}),
        info(speaker_id=None, reference_audio_num_rows=64),
    ]
    _, candidate, _, _ = pair(monkeypatch, tmp_path, requests)
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"].get("bulk_rows", 0) == 0
    assert candidate.model.code_predictor.temperature == 0.7
    _, candidate, _, _ = pair(
        monkeypatch,
        tmp_path / "text-updates",
        [
            info(text_tokens=[], text="whole-text fixture"),
            info(text_tokens=[], text_token=[9, 10], text_token_start=0),
        ],
    )
    assert candidate._bulk_fresh_prefill_inputs.status()["counts"]["bulk_rows"] == 0
    assert candidate.model_intermediate_buffer["owner-0"]["text_tokens"] == [6, 7, 31]
    assert candidate.model_intermediate_buffer["owner-1"]["text_tokens"] == [9, 10]


def test_invalid_middle_row_preserves_error_and_prefix_of_native_side_effects(monkeypatch, tmp_path):
    a = empty_runner(make_model(tmp_path / "native"))
    b = empty_runner(make_model(tmp_path / "candidate"))
    monkeypatch.setenv(ENV, "1")
    assert install(b) is not None
    requests = [
        info(temperature=0.2, top_k=3),
        info((1, 2, 3, 4, 5), temperature=0.4, top_k=5),
        info(temperature=0.6, top_k=7),
    ]
    args_a, args_b = populate(a, requests), populate(b, requests)
    initial_a, initial_b = native_state(a), native_state(b)
    with torch.inference_mode():
        with pytest.raises(AssertionError) as expected:
            runtime.OmniGPUModelRunner._preprocess(a, *args_a)
        with pytest.raises(type(expected.value)) as actual:
            runtime.OmniGPUModelRunner._preprocess(b, *args_b)
    assert str(actual.value) == str(expected.value)
    assert observations(a) == observations(b)
    assert native_state(a) == initial_a and native_state(b) == initial_b
    assert b.model.code_predictor.temperature == 0.4 and b.model.code_predictor.top_k == 5
    assert b.requests["owner-0"].additional_information_cpu is b.model_intermediate_buffer["owner-0"]
    assert b.requests["owner-1"].additional_information_cpu is None
    assert "request_id" not in b.model_intermediate_buffer["owner-2"]
    assert b._bulk_fresh_prefill_inputs.status()["counts"].get("bulk_rows", 0) == 0

    # A repeated ID or shared metadata dictionary is not a valid bulk cohort.
    # Preserve native absolute-offset processing by declining the whole batch.
    for case in ("repeated_ID", "shared_info"):
        left = empty_runner(make_model(tmp_path / case / "native"))
        right = empty_runner(make_model(tmp_path / case / "candidate"))
        assert install(right) is not None
        left_args = populate(left, [info(), info(), info()])
        right_args = populate(right, [info(), info(), info()])
        for runner in (left, right):
            if case == "repeated_ID":
                runner.input_batch.req_ids[1] = runner.input_batch.req_ids[0]
            else:
                runner.model_intermediate_buffer["owner-1"] = runner.model_intermediate_buffer["owner-0"]
        with torch.inference_mode():
            runtime.OmniGPUModelRunner._preprocess(left, *left_args)
            runtime.OmniGPUModelRunner._preprocess(right, *right_args)
        assert observations(left) == observations(right)
        assert right._bulk_fresh_prefill_inputs.status()["counts"].get("bulk_rows", 0) == 0


def test_reused_workspace_never_aliases_sources_and_close_has_no_request_retention(monkeypatch, tmp_path):
    a, b, _, _ = pair(monkeypatch, tmp_path, [info((0, 1)), info((2, 3, 4, 5))])
    source_a, source_b = native_state(a), native_state(b)
    args_a = populate(a, [info((9, 8, 7, 6)), info(())], padding=9)
    args_b = populate(b, [info((9, 8, 7, 6)), info(())], padding=9)
    with torch.inference_mode():
        invoke(a, *args_a)
        invoke(b, *args_b)
    assert observations(a) == observations(b)
    assert native_state(a) == source_a and native_state(b) == source_b
    assert b._bulk_fresh_prefill_inputs.status()["counts"]["bulk_rows"] == 4
    view = b._bulk_fresh_prefill_inputs.close()
    assert view["closed"] is view["binding_matched"] is True
    assert view["close_calls"] == 1 and view["retained_requests"] == view["retained_tensors"] == 0
    assert b._try_complete_decode_preprocess.__func__ is Runner._try_complete_decode_preprocess
    assert b._bulk_fresh_prefill_inputs.close() == view
    state = b._bulk_fresh_prefill_inputs
    owner = weakref.ref(b)
    model = weakref.ref(b.model)
    del a, b
    gc.collect()
    assert owner() is None and model() is None
    del state


def test_nonempty_install_rejected_before_instance_hook_mutation(monkeypatch, tmp_path):
    runner = empty_runner(make_model(tmp_path))
    populate(runner, [info(), info()])
    original = runner._try_complete_decode_preprocess.__func__
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError):
        install(runner)
    assert runner._try_complete_decode_preprocess.__func__ is original
    assert not hasattr(runner, "_bulk_fresh_prefill_inputs")
    for index, (target, name) in enumerate(
        (("model", "_preprocess_prefill"), ("runner", "_update_intermediate_buffer"))
    ):
        clean = empty_runner(make_model(tmp_path / ("override-" + str(index))))
        owner = clean.model if target == "model" else clean

        def altered(self, *args, **kwargs):
            raise AssertionError("unsupported preinstall delegate was executed")

        setattr(owner, name, MethodType(altered, owner))
        old_hook = clean._try_complete_decode_preprocess.__func__
        with pytest.raises(RuntimeError):
            install(clean)
        assert getattr(owner, name).__func__ is altered
        assert clean._try_complete_decode_preprocess.__func__ is old_hook
        assert not hasattr(clean, "_bulk_fresh_prefill_inputs")
