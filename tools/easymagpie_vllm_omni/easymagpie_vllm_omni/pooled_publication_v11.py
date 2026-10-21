# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact request ownership for native, original-fallback and cancellation puts.

This opt-in repair composes with the frozen v1 transport. Ownership is attached
only to the existing Request object and dies with it: there is no global map
retaining requests. A copied Request must prove it is the existing generation's
terminal sentinel before inheriting that generation.
"""
import os
from collections import Counter

from .pooled_publication import PooledConnector, StaleDescriptor

_TOKEN = '_easymagpie_pooled_publication_owner_v11'


class OwnedPooledConnector(PooledConnector):
    def put(self, from_stage, to_stage, put_key, data):
        targeted = self.role == 'sender' and (str(from_stage), str(to_stage)) == ('0', '1')
        if targeted and getattr(self._local, 'context', None) is None:
            self.pool.counts['unowned_put_rejections'] += 1
            raise RuntimeError('talker publication requires an exact request-generation context')
        result = super().put(from_stage, to_stage, put_key, data)
        receipt = getattr(self._local, 'original_path_receipt', None)
        if targeted and receipt is not None and result[0]:
            receipt['successful_puts'] += 1
        return result


class ExactOriginalOwner:
    def __init__(self, controller):
        self.controller = controller
        self.proxy = controller.adapter.connector
        if not isinstance(self.proxy, PooledConnector) or self.proxy.role != 'sender':
            raise ValueError('install sender publication pool before exact ownership repair')
        if type(self.proxy) is not PooledConnector:
            raise ValueError('unexpected publication proxy subclass')
        self.proxy.__class__ = OwnedPooledConnector
        self.counts = Counter()
        self.epoch = self.proxy.pool.epoch
        self.original_state = controller._state
        self.original_release = controller._release
        self.original_sender = controller._send_original
        controller._state = self.state
        controller._release = self.release
        controller._send_original = self.send_original

    def _tag(self, request, generation):
        token = (self.epoch, id(request), request.request_id, request.external_req_id, generation)
        setattr(request, _TOKEN, token)
        return request.external_req_id, generation

    def _read_tag(self, request):
        token = getattr(request, _TOKEN, None)
        if token is None:
            return None
        if (
            type(token) is not tuple
            or len(token) != 5
            or token[:4] != (self.epoch, id(request), request.request_id, request.external_req_id)
            or type(token[4]) is not int
            or token[4] <= 0
        ):
            return None
        return request.external_req_id, token[4]

    def _copied_terminal(self, task, current):
        request = task['request']
        if not (
            current is not None
            and task.get('is_finished')
            and task.get('multimodal_output') is None
            and request.request_id == current.request.request_id
            and request.external_req_id == current.request.external_req_id
        ):
            return False
        token = getattr(request, _TOKEN, None)
        return token is None or token == (
            self.epoch,
            id(current.request),
            current.request.request_id,
            current.request.external_req_id,
            current.generation,
        )

    def state(self, task):
        request = task['request']
        if getattr(request, _TOKEN, None) is not None and self._read_tag(request) is None:
            current = self.controller.current_owner.get(request.external_req_id)
            if not self._copied_terminal(task, current):
                self.counts['stale_copied_token_rejections'] += 1
                raise StaleDescriptor('stale copied request token cannot create a replacement owner')
        state = self.original_state(task)
        self._tag(state.request, state.generation)
        request = task['request']
        if request is not state.request:
            if not self._copied_terminal(task, state):
                raise StaleDescriptor('different request object cannot inherit publication generation')
            self._tag(request, state.generation)
            self.counts['validated_copied_terminal_owners'] += 1
        return state

    def release(self, state):
        # Preserve the exact generation on the same request object before v1
        # cancellation removes BulkSender.states/current_owner. The object is
        # already owned by queued tasks; no additional lifetime is introduced.
        self._tag(state.request, state.generation)
        return self.original_release(state)

    def owner(self, task):
        request = task['request']
        live = self.controller.states.get(id(request))
        if live is not None and live.request is request:
            return self._tag(request, live.generation)
        tagged = self._read_tag(request)
        if tagged is not None:
            self.counts['retained_request_token_uses'] += 1
            return tagged
        current = self.controller.current_owner.get(request.external_req_id)
        if self._copied_terminal(task, current):
            self.counts['validated_copied_terminal_owners'] += 1
            return self._tag(request, current.generation)
        if getattr(request, _TOKEN, None) is not None:
            self.counts['stale_copied_token_rejections'] += 1
            raise StaleDescriptor('copied or stale request token has no matching live terminal owner')
        if not (task.get('is_finished') or task.get('is_segment_finished')):
            self.counts['unowned_audio_rejections'] += 1
            raise RuntimeError('original audio fallback must already own a BulkSender generation')
        # A never-native terminal or segment can legitimately bypass _state.
        # It gets one identity-bound generation from the same monotonic counter,
        # so subsequent controls and later native ownership cannot go backwards.
        with self.controller._lock:
            self.controller.next_generation += 1
            generation = self.controller.next_generation
        self.counts['original_only_control_owners'] += 1
        return self._tag(request, generation)

    def send_original(self, task):
        owner = self.owner(task)
        flags = int(bool(task['is_finished'])) | (int(bool(task['is_segment_finished'])) << 1)
        receipt = {'successful_puts': 0}
        previous = getattr(self.proxy._local, 'original_path_receipt', None)
        self.proxy._local.original_path_receipt = receipt
        try:
            with self.proxy.publication_context(owner, flags):
                result = self.original_sender(task)
        finally:
            self.proxy._local.original_path_receipt = previous
        self.counts['original_path_puts'] += receipt['successful_puts']
        return result

    def status(self):
        return {
            'enabled': True,
            'version': '1.1',
            'counts': dict(self.counts),
        }


def install_owned_fallback_publication(controller):
    if os.environ.get('EASYMAGPIE_POOLED_PUBLICATION_EXACT_OWNERS', '0') != '1':
        return None
    if hasattr(controller, 'exact_original_owner'):
        return controller.exact_original_owner
    repair = ExactOriginalOwner(controller)
    controller.exact_original_owner = repair
    original_status = controller.status

    def status():
        result = original_status()
        result['pooled_publication_exact_owners'] = repair.status()
        return result

    controller.status = status
    return repair
