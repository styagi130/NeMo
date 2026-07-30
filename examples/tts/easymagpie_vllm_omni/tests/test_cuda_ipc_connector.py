from easymagpie_vllm_omni.cuda_ipc_connector import EasyMagpieCudaIpcConnector


def test_direct_control_datagram_round_trip():
    key = "speech-request_0_17"
    payload = b"\x00codec-control\xff"

    message = EasyMagpieCudaIpcConnector._encode_direct_control(key, payload)

    assert EasyMagpieCudaIpcConnector._decode_direct_control(message) == (key, payload)


def test_direct_control_datagram_rejects_truncated_payload():
    message = EasyMagpieCudaIpcConnector._encode_direct_control("request_0_1", b"payload")

    assert EasyMagpieCudaIpcConnector._decode_direct_control(message[:-1]) is None


def test_direct_control_ready_key_and_fallback_scan_are_one_shot():
    connector = EasyMagpieCudaIpcConnector.__new__(EasyMagpieCudaIpcConnector)
    connector._direct_control_pending = {"request_0_2": b"payload"}
    connector._direct_control_fallback_scan_requested = True

    assert connector.direct_control_key_ready("request_0_2")
    assert not connector.direct_control_key_ready("request_0_3")
    assert connector.consume_direct_control_fallback_scan()
    assert not connector.consume_direct_control_fallback_scan()
