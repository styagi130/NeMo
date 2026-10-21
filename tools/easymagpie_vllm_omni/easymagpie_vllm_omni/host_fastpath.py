"""Cache only the two immutable BulkSender metadata scalars on the wire."""

import os


def _enabled(name):
    value = os.environ.get(name, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return value == '1'


class MetadataWireCache:
    def __init__(self, owner, serializer):
        import msgspec
        import torch

        self._true, self._false = owner._true, owner._false
        for value, expected in ((self._true, True), (self._false, False)):
            if not (
                isinstance(value, torch.Tensor)
                and value.device.type == 'cpu'
                and value.dtype == torch.bool
                and value.shape == ()
                and value.item() is expected
            ):
                raise ValueError('metadata cache requires the two owned CPU bool scalars')
        self._versions = (self._true._version, self._false._version)
        self._original_encoder = serializer.encoder
        self._true_wire = serializer.encoder._encode_tensor(self._true)
        self._false_wire = serializer.encoder._encode_tensor(self._false)
        self._encoder = msgspec.msgpack.Encoder(enc_hook=self._hook)
        self.calls = 0
        self.cache_hits = 0
        self.generic_hook_calls = 0
        self.mutation_rejections = 0

    def _hook(self, obj):
        if obj is self._true:
            self.cache_hits += 1
            return self._true_wire
        if obj is self._false:
            self.cache_hits += 1
            return self._false_wire
        self.generic_hook_calls += 1
        return self._original_encoder._enc_hook(obj)

    def encode(self, obj):
        if (self._true._version, self._false._version) != self._versions:
            self.mutation_rejections += 1
            raise RuntimeError('owned metadata scalars were mutated after wire-cache installation')
        self.calls += 1
        return self._encoder.encode(obj)

    def status(self):
        return {
            'enabled': True,
            'scope': 'two exact BulkSender-owned CPU bool tensors',
            'strong_references': True,
            'mutation_guard': 'tensor version',
            'calls': self.calls,
            'cache_hits': self.cache_hits,
            'generic_hook_calls': self.generic_hook_calls,
            'mutation_rejections': self.mutation_rejections,
        }


def install_metadata_wire_cache(controller):
    """Call after BulkSender creates its immutable scalars, before publication."""
    if not _enabled('EASYMAGPIE_METADATA_WIRE_CACHE'):
        return None
    existing = getattr(controller, '_metadata_wire_cache', None)
    if existing is not None:
        return existing
    from vllm_omni.distributed.omni_connectors.connectors import shm_connector
    from vllm_omni.distributed.omni_connectors.utils import serialization

    connector = controller.adapter.connector
    if type(connector) is not shm_connector.SharedMemoryConnector:
        raise TypeError('metadata wire cache requires the supported SharedMemoryConnector')
    if 'serialize_obj' in vars(connector):
        raise RuntimeError('connector serialization was already overridden')
    cache = MetadataWireCache(controller, serialization.OmniSerializer)
    connector.serialize_obj = cache.encode
    controller._metadata_wire_cache = cache
    return cache


def host_fastpath_status(runner=None, controller=None):
    cache = getattr(controller, '_metadata_wire_cache', None)
    return {'metadata_wire_cache': cache.status() if cache is not None else {'enabled': False}}
