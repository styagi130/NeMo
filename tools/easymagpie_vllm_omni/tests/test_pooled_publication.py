"""CPU-only protocol tests; runnable standalone without importing torch/vLLM."""

import importlib.util
import json
import multiprocessing as mp
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

PATH = Path(__file__).parents[1] / 'easymagpie_vllm_omni/pooled_publication.py'
spec = importlib.util.spec_from_file_location('publication_pool_test_module', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Pool, Busy, Stale = module.SharedPublicationPool, module.PoolBusy, module.StaleDescriptor


def process_consumer(directory, count, queue):
    pool = Pool('process', slots=8, payload_bytes=4096, directory=directory)
    rows = []
    deadline = time.monotonic() + 20
    for seq in range(count):
        while True:
            result = pool.get(f'process-rid_0_{seq}', lambda b: b)
            if result is not None:
                rows.append(result[0])
                break
            if time.monotonic() > deadline:
                raise RuntimeError('consumer timeout')
            time.sleep(0.00005)
    queue.put((rows, pool.status()))
    pool.close()


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pool = Pool('test', slots=4, payload_bytes=4096, directory=self.tmp.name, timeout=0.02)
        self.peer = Pool('test', slots=4, payload_bytes=4096, directory=self.tmp.name, timeout=0.02)

    def tearDown(self):
        self.pool.close()
        self.peer.close()
        self.tmp.cleanup()

    def test_exact_payloads_and_owned_bytes_after_reuse(self):
        retained = []
        for seq in range(80):
            frames = (2, 6, 8)[seq % 3]
            payload = bytes((i + seq) % 256 for i in range(frames * 16 * 8))
            desc = self.pool.put(f'voice_0_{seq}', payload, request_generation=7)
            result = self.peer.get(f'voice_0_{seq}', lambda b: b, descriptor=desc)
            self.assertEqual(result[0], payload)
            retained.append((result[0], payload))
        self.assertTrue(all(got == expected for got, expected in retained))
        self.assertEqual(self.pool.status()['occupied'], 0)

    def test_bounded_backpressure_no_overwrite(self):
        for seq in range(4):
            self.pool.put(f'a_0_{seq}', bytes([seq]), request_generation=1)
        with self.assertRaises(Busy):
            self.pool.put('a_0_4', b'four', request_generation=1)
        for seq in range(4):
            self.assertEqual(self.peer.get(f'a_0_{seq}', lambda b: b)[0], bytes([seq]))
        self.assertEqual(self.pool.status()['occupied'], 0)

    def test_no_reuse_before_successful_decoder_ack(self):
        entered, proceed = threading.Event(), threading.Event()
        self.pool.put('a_0_0', b'original', request_generation=1)

        def decoder(body):
            entered.set()
            self.assertTrue(proceed.wait(2))
            return body

        output = []
        thread = threading.Thread(target=lambda: output.append(self.peer.get('a_0_0', decoder)))
        thread.start()
        self.assertTrue(entered.wait(2))
        self.assertIsNone(self.pool.get('a_0_0', lambda b: b))
        with self.assertRaises(Busy):
            self.pool.put('a_0_0', b'replacement', request_generation=2)
        proceed.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(output[0][0], b'original')
        self.pool.put('a_0_0', b'replacement', request_generation=2)
        self.assertEqual(self.peer.get('a_0_0', lambda b: b)[0], b'replacement')

    def test_decode_exception_does_not_ack_or_lose_payload(self):
        desc = self.pool.put('a_0_0', b'encoded', request_generation=1)

        def fail(body):
            raise ValueError('injected decoder failure')

        with self.assertRaises(ValueError):
            self.peer.get('a_0_0', fail)
        self.assertEqual(self.pool.status()['occupied'], 1)
        self.assertEqual(self.peer.get('a_0_0', lambda b: b, descriptor=desc)[0], b'encoded')

    def test_descriptor_replay_rejected_after_slot_and_id_reuse(self):
        old = self.pool.put('a_0_0', b'old', request_generation=1)
        self.peer.get('a_0_0', lambda b: b)
        new = self.pool.put('a_0_0', b'new', request_generation=2)
        self.assertNotEqual(old['slot_generation'], new['slot_generation'])
        with self.assertRaises(Stale):
            self.peer.get('a_0_0', lambda b: b, descriptor=old)
        self.assertEqual(self.peer.get('a_0_0', lambda b: b, descriptor=new)[0], b'new')

    def test_reused_id_waits_for_all_old_chunks_not_only_same_key(self):
        self.pool.put('a_0_9', b'old tail', request_generation=1, flags=1)
        with self.assertRaises(Busy):
            self.pool.put('a_0_0', b'new head', request_generation=2)
        self.assertEqual(self.peer.get('a_0_9', lambda b: b)[0], b'old tail')
        self.pool.put('a_0_0', b'new head', request_generation=2)
        self.assertEqual(self.peer.get('a_0_0', lambda b: b)[0], b'new head')

    def test_cancel_retains_segment_terminal_and_late_audio_is_rejected(self):
        for seq, flags in enumerate((0, 2, 1)):
            self.pool.put(f'a_0_{seq}', bytes([seq]), request_generation=9, flags=flags)
        self.assertEqual(self.pool.cancel('a', 9), 1)
        self.assertIsNone(self.peer.get('a_0_0', lambda b: b))
        self.assertIsNone(self.pool.put('a_0_3', b'late audio', request_generation=9))
        self.assertEqual(self.peer.get('a_0_1', lambda b: b)[0], b'\x01')
        self.assertEqual(self.peer.get('a_0_2', lambda b: b)[0], b'\x02')
        self.assertEqual(self.pool.status()['occupied'], 0)
        self.pool.put('a_0_0', b'new', request_generation=10)
        self.assertEqual(self.peer.get('a_0_0', lambda b: b)[0], b'new')

    def test_cancel_during_decode_does_not_reuse_claim(self):
        entered, proceed = threading.Event(), threading.Event()
        self.pool.put('a_0_0', b'claimed', request_generation=9)
        result = []

        def decode(body):
            entered.set()
            self.assertTrue(proceed.wait(2))
            return body

        t = threading.Thread(target=lambda: result.append(self.peer.get('a_0_0', decode)))
        t.start()
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.pool.cancel('a', 9), 0)
        self.assertEqual(self.pool.status()['occupied'], 1)
        proceed.set()
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(result[0][0], b'claimed')
        self.assertEqual(self.pool.status()['occupied'], 0)

    def test_bad_dimensions_and_overlarge_payload_fail_without_partial_publication(self):
        with self.assertRaises(ValueError):
            Pool('test', slots=8, payload_bytes=4096, directory=self.tmp.name)
        with self.assertRaises(ValueError):
            self.pool.put('a_0_0', b'x' * 4097, request_generation=1)
        self.assertIsNone(self.peer.get('a_0_0', lambda b: b))
        self.assertEqual(self.pool.status()['occupied'], 0)

    def test_different_namespaces_do_not_cross_read(self):
        other = Pool('other', slots=4, payload_bytes=4096, directory=self.tmp.name)
        try:
            self.pool.put('a_0_0', b'a', request_generation=1)
            self.assertIsNone(other.get('a_0_0', lambda b: b))
        finally:
            other.close()

    def test_cross_process_steady_reuse_exact_2_6_8_frames(self):
        count = 500
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        sender = Pool('process', slots=8, payload_bytes=4096, directory=self.tmp.name, timeout=10)
        child = ctx.Process(target=process_consumer, args=(self.tmp.name, count, queue))
        child.start()
        expected = []
        for seq in range(count):
            frames = (2, 6, 8)[seq % 3]
            data = bytes([seq % 256]) * (frames * 16 * 8)
            expected.append(data)
            sender.put(f'process-rid_0_{seq}', data, request_generation=12, flags=1 if seq == count - 1 else 0)
        rows, state = queue.get(timeout=10)
        child.join(10)
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(rows, expected)
        self.assertEqual(state['counts']['gets'], count)
        self.assertEqual(sender.status()['occupied'], 0)
        self.assertLessEqual(sender.status()['counts']['maximum_occupied'], 8)
        sender.close()
        queue.close()

    def test_owner_tracking_is_bounded_under_sustained_unique_ids(self):
        for index in range(1000):
            rid = f'unique-{index}'
            self.pool.put(f'{rid}_0_0', b'one', request_generation=index + 1)
            self.peer.get(f'{rid}_0_0', lambda b: b)
        self.assertLessEqual(self.pool.status()['retained_owner_entries'], 8)
        self.assertEqual(self.pool.status()['occupied'], 0)

    def test_older_generation_cannot_return_after_new_owner(self):
        self.pool.put('a_0_0', b'new', request_generation=2)
        self.peer.get('a_0_0', lambda b: b)
        with self.assertRaises(Stale):
            self.pool.put('a_0_1', b'stale old', request_generation=1)
        self.assertEqual(self.pool.status()['occupied'], 0)

    def test_instance_proxy_scopes_only_edge_and_cleanup_keeps_final(self):
        class Original:
            stage_id = 0
            serialize_obj = staticmethod(lambda x: json.dumps(x).encode())
            deserialize_obj = staticmethod(lambda x: json.loads(x))

            def put(self, *args):
                return 'original-put'

            def get(self, *args):
                return 'original-get'

            def cleanup(self, rid):
                return 'original-cleanup'

        sender = module.PooledConnector(Original(), self.pool, 'sender')
        receiver = module.PooledConnector(Original(), self.peer, 'receiver')
        with sender.publication_context(('a', 4), 1):
            success, size, meta = sender.put('0', '1', 'a_0_0', {'final': True})
        self.assertTrue(success)
        sender.cleanup('a')
        self.assertEqual(receiver.get('0', '1', 'a_0_0', meta)[0], {'final': True})
        self.assertEqual(sender.put('1', '2', 'x', {}), 'original-put')
        self.assertEqual(receiver.get('1', '2', 'x'), 'original-get')


if __name__ == '__main__':
    unittest.main(verbosity=2)
