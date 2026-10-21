"""Bounded no-fill publication and injected partial-failure protocol tests."""

import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1] / 'easymagpie_vllm_omni'
PACKAGE = '_pool_v2_test_package'
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = pkg


def load(name):
    fullname = PACKAGE + '.' + name
    spec = importlib.util.spec_from_file_location(fullname, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


v1 = load('pooled_publication')
v2 = load('pooled_publication_v2')
deadlines = load('ready_deadlines')


class Original:
    stage_id = 0
    serialize_obj = staticmethod(lambda value: json.dumps(value).encode())
    deserialize_obj = staticmethod(lambda value: json.loads(value))


class Controller:
    def __init__(self, pool):
        self.consumer = Counter()
        self.deadline_policy = deadlines.ReadyDeadlineQueue()
        self.released = []
        self.fail_cleanup = False
        self.adapter = types.SimpleNamespace(
            connector=v1.PooledConnector(Original(), pool, 'sender'),
            put_req_chunk=defaultdict(int),
            ramp_chunk_count=defaultdict(int),
            code_prompt_token_ids={},
            requests_num_chunks_sent={},
            _cached_ic={},
        )
        self.adapter.cleanup = self.cleanup

    def cleanup(self, internal, rid):
        if self.fail_cleanup:
            self.fail_cleanup = False
            raise RuntimeError('injected cleanup failure')
        self.adapter.put_req_chunk.pop(rid, None)
        self.adapter.ramp_chunk_count.pop(rid, None)

    def _release(self, state):
        if v2.BatchedReadyPublisher._failed(state.request):
            self.deadline_policy.cancel((state.request.external_req_id, state.generation))
        self.released.append(state)

    def status(self):
        return {}

    def owner(self, rid, generation=1):
        request = types.SimpleNamespace(
            external_req_id=rid, request_id=rid + '-internal', status=types.SimpleNamespace(name='RUNNING')
        )
        state = types.SimpleNamespace(request=request, generation=generation)
        self.deadline_policy.register((rid, generation), request, arrival_ns=time.monotonic_ns())
        return state

    def record(self, state, seq, *, final=False, segment=False):
        request = state.request
        return {
            'owner': (request.external_req_id, state.generation),
            'request': request,
            'state': state,
            'task': {'request': request, 'is_finished': final, 'is_segment_finished': segment},
            'payload': {'rid': request.external_req_id, 'seq': seq, 'frames': (2, 6, 8)[seq % 3]},
            'produced_frames': (2, 6, 8)[seq % 3],
            'is_finished': final,
            'is_segment_finished': segment,
            'enqueue_ns': time.monotonic_ns(),
            'ready_ns': time.monotonic_ns(),
        }


def batch_consumer(directory, rounds, queue):
    pool = v1.SharedPublicationPool('v2', slots=8, payload_bytes=4096, directory=directory, timeout=2)
    received = []
    deadline = time.monotonic() + 20
    for sequence in range(rounds):
        for owner in range(4):
            key = f'batch-{owner}_0_{sequence}'
            while True:
                result = pool.get(key, lambda body: body)
                if result is not None:
                    received.append(result[0])
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('batch consumer timeout')
                time.sleep(0.00005)
    queue.put(received)
    pool.close()


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pool = v1.SharedPublicationPool('v2', slots=8, payload_bytes=4096, directory=self.tmp.name, timeout=0.01)
        self.peer = v1.SharedPublicationPool('v2', slots=8, payload_bytes=4096, directory=self.tmp.name, timeout=0.01)
        self.controller = Controller(self.pool)
        self.publisher = v2.BatchedReadyPublisher(self.controller, max_batch=4)

    def tearDown(self):
        self.pool.close()
        self.peer.close()
        self.tmp.cleanup()

    def read(self, rid, seq):
        result = self.peer.get(f'{rid}_0_{seq}', json.loads)
        return None if result is None else result[0]

    def test_ready_batch_uses_one_lock_and_no_fill_wait(self):
        before = self.pool.counts['persistent_lock_acquisitions']
        records = [self.controller.record(self.controller.owner(f'r{i}'), 0, final=True) for i in range(4)]
        self.publisher.publish_ready(records)
        self.assertEqual(self.pool.counts['persistent_lock_acquisitions'] - before, 1)
        self.assertEqual(self.pool.counts['maximum_put_batch'], 4)
        self.assertEqual(self.controller.deadline_policy.status()['inflight_records'], 0)
        for i in range(4):
            self.assertEqual(self.read(f'r{i}', 0)['rid'], f'r{i}')
        lone = self.controller.record(self.controller.owner('lone'), 0, final=True)
        self.publisher.publish_ready([lone])
        self.assertEqual(self.read('lone', 0)['rid'], 'lone')
        self.assertEqual(self.publisher.status()['retry_records'], 0)

    def test_multiple_chunks_per_owner_keep_order_and_terminal_drains(self):
        a = self.controller.owner('a')
        b = self.controller.owner('b')
        records = [self.controller.record(owner, seq, final=seq == 2) for seq in range(3) for owner in (a, b)]
        self.publisher.publish_ready(records)
        for seq in range(3):
            for rid in ('a', 'b'):
                self.assertEqual(self.read(rid, seq)['seq'], seq)
        state = self.controller.deadline_policy.status()
        self.assertEqual(state['live_generations'], 0)
        self.assertEqual(state['queued_records'], 0)
        self.assertEqual(state['counters'].get('fifo_violations', 0), 0)
        self.assertEqual(self.publisher.counts['committed_records'], 6)

    def test_validation_exception_retains_exact_record_and_no_false_ack(self):
        state = self.controller.owner('a')
        record = self.controller.record(state, 0, final=True)
        record['payload'] = object()
        self.publisher.publish_ready([record])
        self.assertEqual(self.publisher.status()['retry_records'], 1)
        self.assertEqual(self.controller.deadline_policy.status()['inflight_records'], 1)
        self.assertEqual(self.controller.deadline_policy.status()['counters'].get('successful_publications', 0), 0)
        self.assertEqual(self.pool.status()['occupied'], 0)
        record['payload'] = {'repaired': True}
        self.publisher.publish_ready([])
        self.assertEqual(self.read('a', 0), {'repaired': True})
        self.assertEqual(self.publisher.status()['retry_records'], 0)
        self.assertEqual(self.controller.deadline_policy.status()['live_generations'], 0)

    def test_partial_store_exception_only_acks_actual_commits_and_retries_rest(self):
        records = [self.controller.record(self.controller.owner(f'r{i}'), 0, final=True) for i in range(4)]
        original = self.pool._store_one
        calls = [0]

        def inject(item):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError('injected store failure before write')
            return original(item)

        self.pool._store_one = inject
        self.publisher.publish_ready(records)
        self.assertEqual(self.publisher.status()['retry_records'], 3)
        state = self.controller.deadline_policy.status()
        self.assertEqual(state['inflight_records'], 3)
        self.assertEqual(state['counters']['successful_publications'], 1)
        self.assertEqual(self.pool.status()['occupied'], 1)
        self.pool._store_one = original
        self.publisher.publish_ready([])
        self.assertEqual(self.pool.status()['occupied'], 4)
        for i in range(4):
            self.assertEqual(self.read(f'r{i}', 0)['rid'], f'r{i}')
        self.assertEqual(self.pool.counts['puts'], 4)
        self.assertEqual(self.publisher.status()['retry_records'], 0)
        self.assertEqual(self.controller.deadline_policy.status()['live_generations'], 0)

    def test_exception_after_fully_committed_store_recovers_receipt_not_duplicate(self):
        records = [self.controller.record(self.controller.owner(f'r{i}'), 0, final=True) for i in range(4)]
        original = self.pool._store_one
        calls = [0]

        def inject(item):
            result = original(item)
            calls[0] += 1
            if calls[0] == 2:
                raise OSError('injected exception after full commit')
            return result

        self.pool._store_one = inject
        self.publisher.publish_ready(records)
        self.assertEqual(self.pool.counts['puts'], 4)
        self.assertEqual(self.pool.counts['recovered_postcommit_store_exceptions'], 1)
        self.assertEqual(self.publisher.status()['retry_records'], 0)
        self.assertEqual(self.controller.deadline_policy.status()['live_generations'], 0)
        for i in range(4):
            self.assertEqual(self.read(f'r{i}', 0)['rid'], f'r{i}')

    def test_cross_process_many_batches_keep_payloads_owned_and_ordered(self):
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        rounds = 100
        child = ctx.Process(target=batch_consumer, args=(self.tmp.name, rounds, queue))
        child.start()
        expected = []
        for seq in range(rounds):
            items = []
            for owner in range(4):
                body = bytes([(seq + owner) % 256]) * ((2, 6, 8)[seq % 3] * 16 * 8)
                expected.append(body)
                items.append(
                    {
                        'key': f'batch-{owner}_0_{seq}',
                        'payload': body,
                        'request_generation': 1,
                        'flags': 1 if seq == rounds - 1 else 0,
                    }
                )
            results = self.pool.put_many(items, timeout=10)
            self.assertTrue(all('descriptor' in result for result in results))
        self.assertEqual(queue.get(timeout=10), expected)
        child.join(10)
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(self.pool.status()['occupied'], 0)
        self.assertEqual(self.pool.counts['puts'], 400)
        queue.close()

    def test_postcommit_cleanup_exception_retries_bookkeeping_without_resend(self):
        record = self.controller.record(self.controller.owner('a'), 0, final=True)
        self.controller.fail_cleanup = True
        self.publisher.publish_ready([record])
        self.assertEqual(self.pool.counts['puts'], 1)
        self.assertEqual(self.publisher.status()['retry_records'], 1)
        self.assertEqual(self.controller.deadline_policy.status()['inflight_records'], 1)
        self.assertEqual(self.read('a', 0)['rid'], 'a')
        self.publisher.publish_ready([])
        self.assertEqual(self.pool.counts['puts'], 1)
        self.assertIsNone(self.read('a', 0))
        self.assertEqual(self.publisher.status()['retry_records'], 0)
        self.assertEqual(self.controller.consumer['published_chunks'], 1)
        self.assertEqual(self.controller.deadline_policy.status()['live_generations'], 0)

    def test_batch_full_pool_backpressure_releases_lock_for_receiver(self):
        for seq in range(8):
            self.pool.put(f'old_0_{seq}', b'{}', request_generation=1)
        items = [{'key': f'new-{i}_0_0', 'payload': b'{}', 'request_generation': 2} for i in range(4)]
        outputs = []
        thread = threading.Thread(target=lambda: outputs.append(self.pool.put_many(items, timeout=2)))
        thread.start()
        time.sleep(0.02)
        for seq in range(4):
            self.peer.get(f'old_0_{seq}', json.loads)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outputs[0]), 4)
        self.assertEqual(self.pool.counts['backpressured_batches'], 1)

    def test_failed_batch_keeps_terminal_owner_until_actual_retry(self):
        for seq in range(8):
            self.pool.put(f'old_0_{seq}', b'{}', request_generation=1)
        record = self.controller.record(self.controller.owner('new'), 0, final=True)
        self.publisher.publish_ready([record])
        self.assertEqual(self.publisher.status()['retry_records'], 1)
        self.assertEqual(self.controller.deadline_policy.status()['retained_closed_generations'], 1)
        self.assertEqual(self.controller.deadline_policy.status()['counters'].get('terminal_retirements', 0), 0)
        self.peer.get('old_0_0', json.loads)
        self.publisher.publish_ready([])
        self.assertEqual(self.read('new', 0)['rid'], 'new')
        self.assertEqual(self.controller.deadline_policy.status()['counters']['terminal_retirements'], 1)

    def test_cancelled_ordinary_record_is_not_published(self):
        state = self.controller.owner('a')
        record = self.controller.record(state, 0)
        state.request.status.name = 'FINISHED_ABORTED'
        self.publisher.publish_ready([record])
        self.assertIsNone(self.read('a', 0))
        self.assertEqual(self.controller.deadline_policy.status()['live_generations'], 0)
        self.assertEqual(self.publisher.status()['retry_records'], 0)

    def test_default_flag_off_and_explicit_install(self):
        old = os.environ.pop('EASYMAGPIE_POOLED_PUBLICATION_BATCH', None)
        try:
            self.assertIsNone(v2.install_batched_publication(self.controller))
            os.environ['EASYMAGPIE_POOLED_PUBLICATION_BATCH'] = '1'
            installed = v2.install_batched_publication(self.controller)
            self.assertIs(installed, v2.install_batched_publication(self.controller))
            self.assertTrue(self.controller.status()['batched_publication']['enabled'])
        finally:
            os.environ.pop('EASYMAGPIE_POOLED_PUBLICATION_BATCH', None)
            if old is not None:
                os.environ['EASYMAGPIE_POOLED_PUBLICATION_BATCH'] = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
