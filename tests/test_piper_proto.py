import io

import pytest

from hosaka.server.engines.piper_proto import (
    PiperProtocolError,
    PiperSidecarError,
    encode_request,
    pack_audio,
    pack_end,
    pack_error,
    parse_request,
    read_response,
)


def _stream(*frames):
    return io.BytesIO(b"".join(frames))


def test_single_audio_chunk_then_end():
    pcm = b"\x00\x01\x02\x03"
    out = list(read_response(_stream(pack_audio(pcm), pack_end())))
    assert out == [pcm]


def test_multiple_audio_chunks_preserve_order():
    a, b = b"\xaa\xbb", b"\xcc\xdd\xee\xff"
    out = list(read_response(_stream(pack_audio(a), pack_audio(b), pack_end())))
    assert out == [a, b]


def test_end_only_yields_nothing():
    assert list(read_response(_stream(pack_end()))) == []


def test_error_frame_raises_with_message():
    with pytest.raises(PiperSidecarError, match="kaboom"):
        list(read_response(_stream(pack_audio(b"\x00\x00"), pack_error("kaboom"))))


def test_truncated_stream_raises_not_silent():
    # Sidecar died mid-utterance (stream ends with no end marker). Must surface,
    # not return a silently-truncated success -- same contract as a mid-stream
    # engine error on the GPU path.
    with pytest.raises(PiperProtocolError):
        list(read_response(_stream(pack_audio(b"\x00\x01\x02\x03"))))


def test_partial_header_raises():
    # A tag byte followed by a chopped length must not be mistaken for valid.
    with pytest.raises(PiperProtocolError):
        list(read_response(_stream(b"A\x00\x00")))  # 'A' + 2 of 4 length bytes


def test_request_roundtrips():
    raw = encode_request("Hello.", voice="glados", length_scale=1.2, noise_scale=0.6, noise_w=0.8)
    assert raw.endswith(b"\n")
    d = parse_request(raw)
    assert d["text"] == "Hello."
    assert d["voice"] == "glados"
    assert d["length_scale"] == 1.2


def test_request_carries_selected_voice():
    # Multi-voice: the sidecar picks the model by this field.
    d = parse_request(
        encode_request("hi", voice="glados_high", length_scale=1.0, noise_scale=0.6, noise_w=0.8)
    )
    assert d["voice"] == "glados_high"


def test_request_flattens_embedded_newlines():
    # Text with newlines must not break the one-line-per-request framing.
    raw = encode_request("a\nb", voice="glados", length_scale=1.0, noise_scale=0.6, noise_w=0.8)
    assert raw.count(b"\n") == 1  # only the terminator
    assert parse_request(raw)["text"] == "a b"
