# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CUDA-IPC transport for EasyMagpie's Stage-0 codec payloads.

vLLM-Omni's stock shared-memory connector serializes every tensor through
``tensor.cpu()``.  That is appropriate for general inter-stage transport, but
is especially expensive for EasyMagpie: Stage 0 and the TensorRT codec are
local CUDA processes on the same GPU and exchange a tiny code tensor for every
audio frame.  This connector retains the normal POSIX-SHM control payload and
replaces only ``codes.audio`` with a CUDA IPC storage descriptor.

The producer keeps the source allocation alive until the consumer drops its
IPC tensor.  A best-effort POSIX-SHM acknowledgement releases that lease, with
a configurable timeout as a safety valve for cancelled requests.
"""
from __future__ import annotations

import copy
import os
import select
import socket
import struct
import threading
import time
import weakref
from multiprocessing import shared_memory
from typing import Any

import torch
from easymagpie_vllm_omni.profiling import maybe_start_cuda_profile, nvtx_profiled, nvtx_range
from vllm.logger import init_logger

_IPC_KEY = "_easymagpie_cuda_ipc"
_ACK_PREFIX = "easymagpie_cuda_ipc_ack_"
_NOTIFY_PREFIX = "easymagpie_cuda_ipc_notify_stage_"
_DATAGRAM_MAGIC = b"EMD1"
_DATAGRAM_HEADER = struct.Struct("!HI")
logger = init_logger(__name__)


def _ack_name(put_key: str) -> str:
    # POSIX SHM names are deliberately short enough for all supported systems.
    return f"{_ACK_PREFIX}{put_key}"[:240]


def _write_ack(put_key: str) -> None:
    """Notify the producer that the receiver has released one IPC tensor."""
    name = _ack_name(put_key)
    try:
        segment = shared_memory.SharedMemory(name=name, create=True, size=1)
    except FileExistsError:
        return
    except Exception:
        return
    try:
        segment.buf[0] = 1
    finally:
        segment.close()


def _notification_path(stage_id: int) -> str:
    """Return the local Unix-datagram endpoint for one receiving stage.

    CUDA IPC is local to the server container, so a Unix datagram gives the
    consumer an inexpensive wake-up signal without changing the existing SHM
    payload format or relying on a busy polling loop.  The SHM key remains the
    source of truth: notifications may be coalesced or lost safely.
    """
    directory = os.environ.get("EASYMAGPIE_CUDA_IPC_NOTIFY_DIR", "/tmp")
    return os.path.join(directory, f"{_NOTIFY_PREFIX}{int(stage_id)}.sock")


def _export_cuda_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Return the stable subset of PyTorch's CUDA IPC tensor reduction."""
    storage = tensor._typed_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    return {
        "device": int(device),
        "handle": bytes(handle),
        "storage_size_bytes": int(storage_size_bytes),
        "storage_offset_bytes": int(storage_offset_bytes),
        "ref_counter_handle": bytes(ref_counter_handle),
        "ref_counter_offset": int(ref_counter_offset),
        "event_handle": bytes(event_handle),
        "event_sync_required": bool(event_sync_required),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
    }


def _import_cuda_tensor(descriptor: dict[str, Any]) -> torch.Tensor:
    """Rebuild a CUDA tensor from :func:`_export_cuda_tensor` metadata."""
    # Delegate to PyTorch's own IPC rebuild helper. Besides constructing the
    # storage, it maintains the per-process CUDA-handle cache required by the
    # caching allocator (opening the same cudaIpcMemHandle twice is unsafe).
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    dtype = getattr(torch, str(descriptor["dtype"]))
    return rebuild_cuda_tensor(
        torch.Tensor,
        tuple(int(v) for v in descriptor["shape"]),
        tuple(int(v) for v in descriptor["stride"]),
        int(descriptor["storage_offset"]),
        torch.storage.TypedStorage,
        dtype,
        int(descriptor["device"]),
        bytes(descriptor["handle"]),
        int(descriptor["storage_size_bytes"]),
        int(descriptor["storage_offset_bytes"]),
        False,
        bytes(descriptor["ref_counter_handle"]),
        int(descriptor["ref_counter_offset"]),
        bytes(descriptor["event_handle"]),
        bool(descriptor["event_sync_required"]),
    )


class EasyMagpieCudaIpcConnector:
    """SharedMemoryConnector wrapper that carries ``codes.audio`` via CUDA IPC."""

    def __init__(self, config: dict[str, Any]) -> None:
        from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

        # Composition keeps this extension isolated from vLLM-Omni internals.
        connector_config = dict(config)
        poll_override = os.environ.get("EASYMAGPIE_CONNECTOR_GET_SLEEP_S")
        if poll_override is not None:
            connector_config["connector_get_sleep_s"] = float(poll_override)
        self._shm = SharedMemoryConnector(connector_config)
        self.config = connector_config
        self.stage_id = self._shm.stage_id
        self.device = self._shm.device
        self._exports: dict[str, tuple[torch.Tensor, float]] = {}
        self._exports_lock = threading.Lock()
        self._cuda_exports_total = 0
        self._cuda_imports_total = 0
        self._shm_fallback_total = 0
        self._notifications_sent = 0
        self._notifications_received = 0
        self._notification_fallback_waits = 0
        self._direct_control_sent = 0
        self._direct_control_received = 0
        self._direct_control_fallbacks = 0
        self._direct_control_pending: dict[str, bytes] = {}
        self._direct_control_fallback_scan_requested = False
        self._put_diagnostics_total = 0
        self._closed = False
        self._lease_s = max(1.0, float(connector_config.get("cuda_ipc_lease_seconds", 30.0)))
        direct_override = os.environ.get("EASYMAGPIE_CUDA_IPC_DIRECT_CONTROL")
        self._direct_control = (
            bool(connector_config.get("cuda_ipc_direct_control", False))
            if direct_override is None
            else direct_override.strip().lower() in {"1", "true", "yes", "on"}
        )
        self._direct_control_max_bytes = max(
            4096,
            int(connector_config.get("cuda_ipc_direct_control_max_bytes", 60 * 1024)),
        )
        # The EasyMagpie stage processor already creates an owning contiguous
        # window before calling ``put``.  Keep the conservative extra clone by
        # default for other potential users of this connector, but allow the
        # EasyMagpie deployment to prove that this second tiny D2D copy is
        # unnecessary.
        clone_override = os.environ.get("EASYMAGPIE_CUDA_IPC_CLONE_PAYLOAD")
        self._clone_payload = (
            bool(connector_config.get("cuda_ipc_clone_payload", True))
            if clone_override is None
            else clone_override.strip().lower() in {"1", "true", "yes", "on"}
        )
        self._reaper = threading.Thread(target=self._reap_loop, name="easymagpie_cuda_ipc_reaper", daemon=True)
        self._reaper.start()
        self._notify_socket: socket.socket | None = None
        self._notify_path: str | None = None
        # Publishing a codec window used to create and close a Unix socket for
        # every ``put``. Stage-0 has one connector for its lifetime, so retain
        # a nonblocking sender and make publishing a single sendto syscall.
        # This is independent from the receiver socket below.
        self._send_socket: socket.socket | None = None
        try:
            self._send_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._send_socket.setblocking(False)
            self._send_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except OSError:
            logger.warning("EasyMagpie CUDA IPC could not create the notify sender; receiver polling remains available.")
        # Only downstream stages receive connector chunks.  Bind the socket
        # before the adapter starts polling so the producer can wake it as soon
        # as Stage-0 publishes the first CUDA descriptor.
        if int(self.stage_id) > 0:
            self._init_notification_socket()

    def _init_notification_socket(self) -> None:
        path = _notification_path(int(self.stage_id))
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("EasyMagpie CUDA IPC could not remove stale notify socket %s", path, exc_info=True)
            return
        try:
            notify_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            notify_socket.setblocking(False)
            notify_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            notify_socket.bind(path)
        except OSError:
            logger.warning("EasyMagpie CUDA IPC event wake-up is unavailable; using polling.", exc_info=True)
            return
        self._notify_socket = notify_socket
        self._notify_path = path

    def _notify_stage(self, stage_id: int) -> None:
        """Best-effort nonblocking wake-up for a local downstream stage."""
        path = _notification_path(stage_id)
        sender = self._send_socket
        if sender is None:
            return
        try:
            sender.sendto(b"1", path)
        except (FileNotFoundError, ConnectionRefusedError, BlockingIOError, OSError):
            # The receiver may still be starting, or several producer events
            # may have coalesced.  The SHM key is authoritative and the small
            # fallback timeout in ``wait_for_data`` keeps this loss harmless.
            pass

    def wake_receiver(self) -> None:
        """Interrupt a receiver wait when a new request starts polling."""
        if self._notify_socket is not None:
            self._notify_stage(int(self.stage_id))

    @staticmethod
    def _encode_direct_control(put_key: str, payload: bytes) -> bytes:
        key = put_key.encode("utf-8")
        if len(key) > 0xFFFF:
            raise ValueError("CUDA IPC control key is too long for a local datagram")
        return _DATAGRAM_MAGIC + _DATAGRAM_HEADER.pack(len(key), len(payload)) + key + payload

    @staticmethod
    def _decode_direct_control(message: bytes) -> tuple[str, bytes] | None:
        prefix_size = len(_DATAGRAM_MAGIC) + _DATAGRAM_HEADER.size
        if len(message) < prefix_size or not message.startswith(_DATAGRAM_MAGIC):
            return None
        key_size, payload_size = _DATAGRAM_HEADER.unpack_from(message, len(_DATAGRAM_MAGIC))
        if len(message) != prefix_size + key_size + payload_size:
            return None
        key_start = prefix_size
        key_end = key_start + key_size
        try:
            key = message[key_start:key_end].decode("utf-8")
        except UnicodeDecodeError:
            return None
        return key, message[key_end:]

    def _drain_notification_socket(self) -> int:
        """Drain wake-ups and cache any direct control envelopes by key."""
        notify_socket = self._notify_socket
        if notify_socket is None:
            return 0
        received = 0
        while True:
            try:
                message = notify_socket.recv(self._direct_control_max_bytes)
            except BlockingIOError:
                break
            except OSError:
                break
            received += 1
            decoded = self._decode_direct_control(message)
            if decoded is not None:
                key, payload = decoded
                self._direct_control_pending[key] = payload
                self._direct_control_received += 1
            else:
                # Content-free notifications identify a producer that used
                # the lossless SHM fallback (or a newly admitted request).
                # Request one full key scan rather than incorrectly assuming
                # every ready payload arrived through the direct datagram.
                self._direct_control_fallback_scan_requested = True
        self._notifications_received += received
        return received

    def direct_control_key_ready(self, get_key: str) -> bool:
        """Return whether the datagram receiver already has ``get_key``."""
        return get_key in self._direct_control_pending

    def consume_direct_control_fallback_scan(self) -> bool:
        """Consume the one-shot request to check the SHM fallback path."""
        requested = self._direct_control_fallback_scan_requested
        self._direct_control_fallback_scan_requested = False
        return requested

    def wait_for_data(self, timeout_s: float) -> bool:
        """Wait for a producer notification, returning ``True`` on wake-up.

        The adapter always performs a key lookup after this method returns.
        Therefore a timeout has no correctness impact; it merely recovers from
        startup races and external connectors that bypass ``put``.
        """
        notify_socket = self._notify_socket
        if notify_socket is None:
            time.sleep(max(0.0, timeout_s))
            self._notification_fallback_waits += 1
            return False
        try:
            ready, _, _ = select.select([notify_socket], [], [], max(0.0, timeout_s))
        except (OSError, ValueError):
            self._notification_fallback_waits += 1
            return False
        if not ready:
            self._notification_fallback_waits += 1
            return False
        # Most datagrams are content-free wake-ups. The optional direct
        # control path carries the tiny serialized envelope in the same local
        # socket and caches it for the next key lookup.
        received = self._drain_notification_socket()
        return received > 0

    def _reap_loop(self) -> None:
        while not self._closed:
            self._reap_exports()
            time.sleep(0.25)

    def _reap_exports(self) -> None:
        now = time.monotonic()
        released: list[str] = []
        with self._exports_lock:
            for put_key, (_tensor, created) in tuple(self._exports.items()):
                name = _ack_name(put_key)
                acknowledged = False
                try:
                    ack = shared_memory.SharedMemory(name=name)
                    ack.close()
                    ack.unlink()
                    acknowledged = True
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                if acknowledged or now - created >= self._lease_s:
                    released.append(put_key)
            for put_key in released:
                self._exports.pop(put_key, None)

    @staticmethod
    def _as_payload_dict(data: Any) -> dict[str, Any] | None:
        try:
            from vllm_omni.data_entry_keys import to_dict

            if hasattr(data, "__struct_fields__"):
                return to_dict(data)
        except Exception:
            pass
        return dict(data) if isinstance(data, dict) else None

    @nvtx_profiled("EM.stage0.ipc.put")
    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any):
        payload = self._as_payload_dict(data)
        # vLLM-Omni flattens an ``OmniPayload`` at the transfer boundary on
        # newer releases, while older builds retain the nested mapping.  The
        # connector must support both representations; otherwise the native
        # codec path silently falls back with ``codes.audio is None``.
        codes = payload.get("codes") if isinstance(payload, dict) else None
        audio = codes.get("audio") if isinstance(codes, dict) else None
        structured_codes = False
        if audio is None and codes is not None and hasattr(codes, "audio"):
            # ``OmniPayloadStruct`` retains its generated ``CodesStruct`` at
            # the connector boundary on vLLM-Omni 0.24.
            audio = codes.audio
            structured_codes = True
        flattened_codes = False
        if audio is None and isinstance(payload, dict):
            audio = payload.get("codes.audio")
            flattened_codes = audio is not None
        flat_audio_codes = False
        if audio is None and isinstance(payload, dict):
            # The audio-sparse path in vLLM-Omni 0.24 uses this flat key.
            audio = payload.get("audio_codes")
            flat_audio_codes = audio is not None
        self._put_diagnostics_total += 1
        if self._put_diagnostics_total <= 3:
            audio_summary = (
                f"shape={tuple(audio.shape)}, device={audio.device}, dtype={audio.dtype}"
                if isinstance(audio, torch.Tensor)
                else type(audio).__name__
            )
            logger.warning(
                "EasyMagpie Stage-0 connector put[%d]: payload_type=%s keys=%s codes_type=%s codes.audio=%s",
                self._put_diagnostics_total,
                type(data).__name__,
                sorted(payload.keys()) if isinstance(payload, dict) else [],
                type(codes).__name__,
                audio_summary,
            )
        if not isinstance(audio, torch.Tensor) or not audio.is_cuda or audio.numel() == 0:
            self._shm_fallback_total += 1
            if self._shm_fallback_total <= 3:
                logger.warning(
                    "EasyMagpie CUDA IPC inactive for this payload: codes.audio is %s; keys=%s; using shared-memory tensor transport.",
                    (f"cuda={audio.is_cuda}, shape={tuple(audio.shape)}" if isinstance(audio, torch.Tensor) else type(audio).__name__),
                    sorted(payload.keys()) if isinstance(payload, dict) else [],
                )
            result = self._shm.put(from_stage, to_stage, put_key, data)
            if result[0]:
                self._notify_stage(int(to_stage))
                self._notifications_sent += 1
            return result

        # A contiguous clone owns a stable allocation after a generic vLLM
        # output buffer is recycled. EasyMagpie's stage processor has already
        # materialized an owning contiguous window, so its experiment can
        # safely retain that allocation and avoid a second D2D copy.
        if maybe_start_cuda_profile():
            logger.warning("EasyMagpie request-triggered CUDA profile capture started.")
        with nvtx_range("EM.stage0.ipc.export"):
            exported = audio.detach().contiguous()
            if self._clone_payload:
                exported = exported.clone()
            descriptor = _export_cuda_tensor(exported)
        envelope = copy.copy(payload)
        if flattened_codes:
            envelope.pop("codes.audio", None)
        elif flat_audio_codes:
            envelope.pop("audio_codes", None)
        elif structured_codes:
            # Replace the generated struct by a serializable mapping without
            # the CUDA allocation. Stage 1 restores the leaf after IPC import.
            envelope["codes"] = {}
        else:
            envelope_codes = dict(codes)
            envelope_codes.pop("audio", None)
            envelope["codes"] = envelope_codes
        kv_metadata = dict(envelope.get("kv_metadata") or {})
        kv_metadata[_IPC_KEY] = descriptor
        envelope["kv_metadata"] = kv_metadata
        with self._exports_lock:
            self._exports[put_key] = (exported, time.monotonic())
            self._cuda_exports_total += 1
            if self._cuda_exports_total == 1:
                logger.warning(
                    "EasyMagpie CUDA IPC active: exporting Stage-0 codes.audio directly from CUDA (shape=%s, dtype=%s).",
                    tuple(exported.shape),
                    exported.dtype,
                )
        result = None
        if self._direct_control and self._send_socket is not None:
            try:
                payload_bytes = self._shm.serialize_obj(envelope)
                message = self._encode_direct_control(put_key, payload_bytes)
                if len(message) <= self._direct_control_max_bytes:
                    self._send_socket.sendto(message, _notification_path(int(to_stage)))
                    self._direct_control_sent += 1
                    self._notifications_sent += 1
                    result = (True, len(payload_bytes), {"direct_datagram": True, "size": len(payload_bytes)})
            except (FileNotFoundError, ConnectionRefusedError, BlockingIOError, OSError, ValueError):
                # SHM remains the authoritative, lossless fallback when the
                # receiver is starting or the bounded datagram queue is full.
                self._direct_control_fallbacks += 1
        if result is None:
            with nvtx_range("EM.stage0.ipc.shm_put"):
                result = self._shm.put(from_stage, to_stage, put_key, envelope)
            if result[0]:
                self._notify_stage(int(to_stage))
                self._notifications_sent += 1
        return result

    @nvtx_profiled("EM.stage1.ipc.get")
    def get(self, from_stage: str, to_stage: str, get_key: str, metadata=None):
        result = None
        if self._direct_control:
            self._drain_notification_socket()
            payload_bytes = self._direct_control_pending.pop(get_key, None)
            if payload_bytes is not None:
                try:
                    result = (self._shm.deserialize_obj(payload_bytes), len(payload_bytes))
                except Exception:
                    logger.warning("EasyMagpie direct CUDA-IPC control decode failed for %s; checking SHM.", get_key)
        if result is None:
            with nvtx_range("EM.stage1.ipc.shm_get"):
                result = self._shm.get(from_stage, to_stage, get_key, metadata)
        if result is None:
            return None
        payload, size = result
        if not isinstance(payload, dict):
            return result
        kv_metadata = payload.get("kv_metadata")
        descriptor = kv_metadata.get(_IPC_KEY) if isinstance(kv_metadata, dict) else None
        if not isinstance(descriptor, dict):
            return result
        try:
            with nvtx_range("EM.stage1.ipc.import"):
                tensor = _import_cuda_tensor(descriptor)
        except Exception:
            # Do not silently substitute corrupted audio. Returning the normal
            # payload lets the caller surface a useful missing-code warning.
            return result
        # Preserve the representation seen by the receiver.  The native
        # transfer callback in vLLM 0.24 consumes flattened keys; legacy
        # callbacks consume the nested mapping.
        if "codes" in payload:
            codes = dict(payload.get("codes") or {})
            codes["audio"] = tensor
            payload["codes"] = codes
        elif "audio_codes" in payload:
            payload["audio_codes"] = tensor
        else:
            payload["codes.audio"] = tensor
        payload["kv_metadata"] = {key: value for key, value in kv_metadata.items() if key != _IPC_KEY}
        self._cuda_imports_total += 1
        if self._cuda_imports_total == 1:
            logger.warning(
                "EasyMagpie CUDA IPC active: imported Stage-1 codes.audio directly on CUDA (shape=%s, dtype=%s).",
                tuple(tensor.shape),
                tensor.dtype,
            )
        weakref.finalize(tensor, _write_ack, get_key)
        return payload, size

    def cleanup(self, request_id: str) -> None:
        self._shm.cleanup(request_id)
        stale = [
            key
            for key in self._direct_control_pending
            if key == request_id or key.startswith(request_id + "_") or key.endswith("_" + request_id)
        ]
        for key in stale:
            self._direct_control_pending.pop(key, None)
        self._reap_exports()

    def close(self) -> None:
        self._closed = True
        self._reap_exports()
        if self._notify_socket is not None:
            self._notify_socket.close()
            self._notify_socket = None
        if self._send_socket is not None:
            self._send_socket.close()
            self._send_socket = None
        if self._notify_path is not None:
            try:
                os.unlink(self._notify_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("EasyMagpie CUDA IPC could not remove notify socket %s", self._notify_path)
            self._notify_path = None
        self._shm.close()

    def health(self) -> dict[str, Any]:
        result = self._shm.health()
        with self._exports_lock:
            result.update(
                {
                    "cuda_ipc_exports": len(self._exports),
                    "cuda_ipc_exports_total": self._cuda_exports_total,
                    "cuda_ipc_imports_total": self._cuda_imports_total,
                    "cuda_ipc_shm_fallback_total": self._shm_fallback_total,
                    "cuda_ipc_lease_s": self._lease_s,
                    "cuda_ipc_notifications_sent": self._notifications_sent,
                    "cuda_ipc_notifications_received": self._notifications_received,
                    "cuda_ipc_notification_fallback_waits": self._notification_fallback_waits,
                    "cuda_ipc_direct_control": self._direct_control,
                    "cuda_ipc_direct_control_sent": self._direct_control_sent,
                    "cuda_ipc_direct_control_received": self._direct_control_received,
                    "cuda_ipc_direct_control_fallbacks": self._direct_control_fallbacks,
                    "cuda_ipc_direct_control_pending": len(self._direct_control_pending),
                    "cuda_ipc_direct_control_fallback_scan_requested": self._direct_control_fallback_scan_requested,
                }
            )
        return result


def install_cuda_ipc_connector() -> bool:
    """Register the connector and keep CUDA tensors in Stage-1 request state."""
    try:
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
        from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
    except Exception:
        return False

    if "EasyMagpieCudaIpcConnector" not in OmniConnectorFactory._registry:
        OmniConnectorFactory.register_connector("EasyMagpieCudaIpcConnector", EasyMagpieCudaIpcConnector)

    if getattr(OmniChunkTransferAdapter, "_easymagpie_cuda_ipc_patched", False):
        return True
    original_poll = OmniChunkTransferAdapter._poll_single_request
    original_load_async = OmniChunkTransferAdapter.load_async
    original_recv_loop = OmniTransferAdapterBase.recv_loop

    def load_async(self, request):
        result = original_load_async(self, request)
        connector = getattr(self, "connector", None)
        if isinstance(connector, EasyMagpieCudaIpcConnector):
            connector.wake_receiver()
        return result

    def recv_loop(self):
        """Use producer notifications instead of failed-SHM polling for Stage 1."""
        connector = getattr(self, "connector", None)
        if not isinstance(connector, EasyMagpieCudaIpcConnector) or int(getattr(connector, "stage_id", -1)) <= 0:
            return original_recv_loop(self)

        # A notification should arrive for every successful ``put``. Keep a
        # bounded fallback for server-start races and for a producer that dies
        # between writing SHM and sending its datagram.
        timeout_s = max(0.001, float(connector.config.get("cuda_ipc_notify_fallback_ms", 20)) / 1000.0)
        force_fallback_scan = True
        while not self.stop_event.is_set():
            # Direct-control datagrams already carry the exact connector key.
            # Drain once, then skip the O(pending requests) set of failed SHM
            # lookups for keys that have not arrived. A content-free wake-up
            # or periodic timeout still forces a complete fallback scan.
            connector._drain_notification_socket()
            fallback_scan = force_fallback_scan or connector.consume_direct_control_fallback_scan()
            force_fallback_scan = False
            n = len(self._pending_load_reqs)
            any_success = False
            for _ in range(n):
                if not self._pending_load_reqs:
                    break
                request = self._pending_load_reqs.popleft()
                request_id = request.request_id
                if request_id in self._cancelled_load_reqs:
                    self._cancelled_load_reqs.discard(request_id)
                    continue
                self.request_ids_mapping[request_id] = request.external_req_id
                target_stage_id = connector.stage_id - 1
                external_req_id = self.request_ids_mapping.get(request_id, request_id)
                connector_get_key = f"{external_req_id}_{target_stage_id}_{self.get_req_chunk[request_id]}"
                if (
                    connector._direct_control
                    and not fallback_scan
                    and not connector.direct_control_key_ready(connector_get_key)
                ):
                    self._pending_load_reqs.append(request)
                    continue
                try:
                    if self._poll_single_request(request):
                        any_success = True
                    else:
                        self._pending_load_reqs.append(request)
                except Exception:
                    self._pending_load_reqs.append(request)
                    logger.warning("Error receiving EasyMagpie CUDA-IPC data for %s", request_id, exc_info=True)

            if not self._pending_load_reqs:
                with self._recv_cond:
                    if not self.stop_event.is_set() and not self._pending_load_reqs:
                        self._recv_cond.wait(timeout=0.1)
            elif not any_success:
                with nvtx_range("EM.stage1.ipc.event_wait"):
                    if not connector.wait_for_data(timeout_s):
                        force_fallback_scan = True

    def _poll_single_request(self, request):
        # Only intercept the custom connector. Other pipelines retain the
        # stock adapter unchanged.
        if not isinstance(getattr(self, "connector", None), EasyMagpieCudaIpcConnector):
            return original_poll(self, request)

        req_id = request.request_id
        target_stage_id = self.connector.stage_id - 1
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{self.get_req_chunk[req_id]}"
        result = self.connector.get(str(target_stage_id), str(self.connector.stage_id), connector_get_key)
        if result is None:
            return False
        payload_data, _size = result
        if not payload_data:
            return False
        self.get_req_chunk[req_id] += 1
        meta = payload_data.get("meta", {})
        if self.model_mode == "ar":
            request.additional_information = payload_data
            if meta.get("finished"):
                self.finished_requests.add(req_id)
        else:
            if meta.get("finished"):
                self.finished_requests.add(req_id)
            audio = payload_data.get("codes", {}).get("audio")
            if isinstance(audio, torch.Tensor) and audio.is_cuda:
                # The codec model reads this tensor from runtime additional
                # information. Keep a one-token scheduler placeholder instead
                # of materializing its contents on CPU with ``tolist()``.
                request.prompt_token_ids = [0]
            else:
                request.prompt_token_ids = audio.tolist() if isinstance(audio, torch.Tensor) else (audio or [])
            previous = getattr(request, "additional_information", None)
            info = dict(previous) if isinstance(previous, dict) else {}
            codes = payload_data.get("codes")
            if isinstance(codes, dict):
                info["codes"] = dict(codes)
            for key, value in payload_data.items():
                if key == "codes":
                    continue
                if isinstance(value, dict):
                    sub = dict(info.get(key) or {})
                    for subkey, subvalue in value.items():
                        if key == "meta" and subkey == "finished":
                            continue
                        sub[subkey] = subvalue
                    info[key] = sub
                else:
                    info[key] = value
            request.additional_information = info
            request.num_computed_tokens = 0
            if not request.prompt_token_ids and not meta.get("finished"):
                return True
        self._finished_load_reqs.add(req_id)
        return True

    OmniChunkTransferAdapter.load_async = load_async
    OmniChunkTransferAdapter._poll_single_request = _poll_single_request
    OmniTransferAdapterBase.recv_loop = recv_loop
    OmniChunkTransferAdapter._easymagpie_cuda_ipc_patched = True
    return True
