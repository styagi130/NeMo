"""Native sender, exact-owner publication, fallback and cancellation lifecycles."""

import threading
import time
import uuid
from collections import defaultdict, deque
from copy import copy
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

vllm.platforms._current_platform = CpuPlatform()
from easymagpie_vllm_omni import scheduler as sched
from easymagpie_vllm_omni import stage_processors
from easymagpie_vllm_omni.bulk_sender import install_bulk_sender
from easymagpie_vllm_omni.pooled_publication import PooledConnector, StaleDescriptor
from easymagpie_vllm_omni.pooled_publication_v11 import _TOKEN, install_owned_fallback_publication
from vllm.v1.request import RequestStatus
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


def make_adapter(stage):
    a = object.__new__(OmniChunkTransferAdapter)
    a.config = NS(hf_config=NS(streaming_speech_delay=0))
    a.connector = SharedMemoryConnector(
        {'stage_id': stage, 'extra': {'codec_chunk_frames': 8, 'codec_startup_chunk_frames': [2, 6]}}
    )
    a.custom_process_next_stage_input_func = stage_processors.talker2code2wav_async_chunk
    for name in ('put_req_chunk', 'ramp_chunk_count', 'requests_num_chunks_sent'):
        setattr(a, name, defaultdict(int))
    a.code_prompt_token_ids = defaultdict(list)
    a.request_payload = {}
    a._pending_streaming_prefills = {}
    a._pending_save_reqs = deque()
    a._save_cond = threading.Condition()
    a._confirmed_num_computed_tokens = lambda req: 0
    a.cleanup = lambda internal, external=None: a.cleanup_sender(external or internal)
    a._active_window = False
    a.process_pending_chunks = lambda *args, **kwargs: None
    a._poll_single_request = lambda *args, **kwargs: None
    if stage == 0:
        a._send_single_request = MethodType(sched._send_talker_chunk, a)
    return a


@pytest.fixture
def publication(monkeypatch):
    options = dict(
        EASYMAGPIE_BULK_SENDER='1',
        EASYMAGPIE_BULK_SENDER_BACKEND='native',
        EASYMAGPIE_BULK_FRAME_VIEWLESS='1',
        EASYMAGPIE_READY_CHUNK_DEADLINES='1',
        EASYMAGPIE_METADATA_WIRE_CACHE='1',
        EASYMAGPIE_POOLED_PUBLICATION='1',
        EASYMAGPIE_PUBLICATION_NAMESPACE='cpu-integrated-' + uuid.uuid4().hex,
        EASYMAGPIE_PUBLICATION_SLOTS='16',
        EASYMAGPIE_PUBLICATION_TIMEOUT_S='.01',
        EASYMAGPIE_POOLED_PUBLICATION_EXACT_OWNERS='1',
    )
    for name, value in options.items():
        monkeypatch.setenv(name, value)
    a, b = make_adapter(0), make_adapter(1)
    s = NS(
        chunk_transfer_adapter=a,
        vllm_config=NS(model_config=NS(stage_id=0)),
        requests={},
        update_from_output=lambda *args, **kwargs: None,
    )

    def finish(ids, state):
        for rid in ids:
            s.requests[rid].status = state

    s.finish_requests = finish
    controller = install_bulk_sender(s)
    install_owned_fallback_publication(controller)

    def stub_init(self, *args, **kwargs):
        self.chunk_transfer_adapter = b
        self.vllm_config = NS(model_config=NS(hf_config=NS(num_stacked_codebooks=16)))

    monkeypatch.setattr(sched.OmniGenerationScheduler, '__init__', stub_init)
    receiver = sched.EasyMagpieAsyncCodecScheduler()
    assert isinstance(a.connector, PooledConnector) and isinstance(b.connector, PooledConnector)
    assert a.connector.pool.epoch == b.connector.pool.epoch
    assert a.connector.original.serialize_obj.__self__ is controller._metadata_wire_cache
    assert b._poll_single_request.__func__ is sched._poll_fenced_codec_chunk
    try:
        yield s, controller, a, b, receiver
    finally:
        pool = a.connector.pool
        a.connector.close()
        b.connector.close()
        pool.unlink()


def request(s, rid):
    req = NS(
        request_id=rid,
        external_req_id=rid,
        additional_information=None,
        status=RequestStatus.RUNNING,
        resumable=True,
        output_token_ids=[],
    )
    req.is_finished = lambda: req.status in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_ABORTED)
    s.requests[rid] = req
    return req


def send(c, req, start, count, terminal=False, segment=False, dtype=torch.long):
    for i in range(count):
        c.send_envelope(
            {
                'request': req,
                'multimodal_output': {'codes.audio': torch.arange(16, dtype=dtype).reshape(1, 16) + start + i},
                'is_finished': terminal and i == count - 1,
                'is_segment_finished': segment and i == count - 1,
                'enqueue_ns': time.monotonic_ns(),
            }
        )


def get(b, rid, seq):
    return b.connector.get('0', '1', f'{rid}_0_{seq}')


def test_ordinary_terminal_and_reuse(publication):
    s, c, a, b, receiver = publication
    req = request(s, 'normal')
    send(c, req, 1, 2)
    first, _ = get(b, 'normal', 0)
    assert torch.equal(first['codes']['audio'], torch.stack([torch.arange(16) + 1, torch.arange(16) + 2]))
    send(c, req, 3, 6)
    second, _ = get(b, 'normal', 1)
    assert second['codes']['audio'].shape == (6, 16)
    send(c, req, 9, 8, terminal=True)
    assert not c.states and a.connector.pool.status()['occupied'] == 1
    final, _ = get(b, 'normal', 2)
    assert final['meta']['finished'].item() and final['codes']['audio'].shape == (8, 16)
    replacement = request(s, 'normal')
    send(c, replacement, 21, 2, terminal=True)
    reused, _ = get(b, 'normal', 0)
    assert reused['meta']['finished'].item() and reused['codes']['audio'][0, 0].item() == 21
    assert a.connector.pool.status()['occupied'] == 0
    assert a.connector.pool.counts['puts'] == 4
    assert not a.connector.pool.counts['fallback_context_puts']
    assert c._metadata_wire_cache.cache_hits == 8
    assert c.status()['ready_deadlines']['counters']['offered_records'] == 4


def test_scheduler_cancellation_preserves_segment(publication):
    s, c, a, b, receiver = publication
    req = request(s, 'cancel')
    send(c, req, 1, 2)
    send(c, req, 3, 6, segment=True)
    assert a.connector.pool.status()['occupied'] == 2
    s.finish_requests(['cancel'], RequestStatus.FINISHED_ABORTED)
    while a._pending_save_reqs:
        a._send_single_request(a._pending_save_reqs.popleft())
    assert not c.states
    assert get(b, 'cancel', 0) is None
    control, _ = get(b, 'cancel', 1)
    assert control['meta']['is_segment_finished'].item() is True
    assert a.connector.pool.status()['occupied'] == 0
    assert a.connector.pool.counts['cancelled_ordinary_records'] == 1


def test_mixed_native_to_original_fallback(publication):
    s, c, a, b, receiver = publication
    req = request(s, 'fallback')
    send(c, req, 1, 2)
    old_generation = c.states[id(req)].generation
    send(c, req, 3, 6, dtype=torch.int32)
    assert a.connector.pool.status()['occupied'] == 2
    assert a.connector.pool._owners['fallback'][0] == old_generation
    assert not a.connector.pool.counts['fallback_context_puts']
    first, _ = get(b, 'fallback', 0)
    second, _ = get(b, 'fallback', 1)
    assert first['codes']['audio'][0, 0].item() == 1
    assert second['codes']['audio'].shape == (6, 16)
    assert torch.equal(second['codes']['audio'], torch.stack([torch.arange(16) + i for i in range(3, 9)]))
    send(c, req, 9, 8, terminal=True, dtype=torch.int32)
    final, _ = get(b, 'fallback', 2)
    assert final['meta']['finished'].item() and final['codes']['audio'].shape == (8, 16)
    assert not c.states and a.connector.pool.status()['occupied'] == 0
    assert c.exact_original_owner.counts['original_path_puts'] == 2


def test_failed_terminal_after_release(publication):
    s, c, a, b, _ = publication
    req = request(s, 'failed-terminal')
    send(c, req, 1, 2)
    generation = c.states[id(req)].generation
    req.status = RequestStatus.FINISHED_ABORTED
    c.send_envelope(
        {
            'request': req,
            'multimodal_output': None,
            'is_finished': True,
            'is_segment_finished': False,
            'enqueue_ns': time.monotonic_ns(),
        }
    )
    assert not c.states
    assert get(b, 'failed-terminal', 0) is None
    terminal, _ = get(b, 'failed-terminal', 1)
    assert terminal['meta']['finished'].item()
    assert a.connector.pool._owners['failed-terminal'][0] == generation
    assert a.connector.pool.status()['occupied'] == 0
    assert c.exact_original_owner.counts['retained_request_token_uses'] >= 1


def test_original_only_terminal_and_stale_copy(publication):
    s, c, a, b, _ = publication
    req = request(s, 'control-only')
    req.status = RequestStatus.FINISHED_ABORTED
    task = {
        'request': req,
        'multimodal_output': None,
        'is_finished': True,
        'is_segment_finished': False,
        'enqueue_ns': time.monotonic_ns(),
    }
    c.send_envelope(task)
    final, _ = get(b, 'control-only', 0)
    assert final['meta']['finished'].item()
    assert c.exact_original_owner.counts['original_only_control_owners'] == 1
    assert not c.states
    copied = copy(req)
    task2 = {**task, 'request': copied}
    try:
        c.send_envelope(task2)
    except StaleDescriptor:
        pass
    else:
        raise AssertionError('stale copied owner token inherited without live generation')
    assert a.connector.pool.status()['occupied'] == 0
    assert a.connector.pool.counts['fallback_context_puts'] == 0


def test_copied_live_terminal_sentinel(publication):
    s, c, a, b, _ = publication
    req = request(s, 'copy-terminal')
    send(c, req, 1, 2)
    generation = c.states[id(req)].generation
    # Enter real original fallback with a partial tail, then flush it with
    # the copied Request sentinel accepted by BulkSender._state.
    send(c, req, 3, 2, dtype=torch.int32)
    sentinel = copy(req)
    c.send_envelope(
        {
            'request': sentinel,
            'multimodal_output': None,
            'is_finished': True,
            'is_segment_finished': False,
            'enqueue_ns': time.monotonic_ns(),
        }
    )
    assert a.connector.pool._owners['copy-terminal'][0] == generation
    assert getattr(sentinel, _TOKEN)[1] == id(sentinel)
    first, _ = get(b, 'copy-terminal', 0)
    final, _ = get(b, 'copy-terminal', 1)
    assert final['meta']['finished'].item()
    assert final['codes']['audio'].shape == (8, 16)
    assert final['codes']['audio'][0, 0].item() == 3
    assert not c.states and a.connector.pool.status()['occupied'] == 0
