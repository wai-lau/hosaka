import io

import pytest

from hosaka.server.engines.rvc_proto import (
    RvcProtocolError,
    RvcSidecarError,
    encode_request,
    pack_audio,
    pack_end,
    pack_error,
    read_request,
    read_response,
)


def _stream(*frames):
    return io.BytesIO(b"".join(frames))


# --- response framing (sidecar -> server), same contract as piper_proto ---
def test_single_audio_chunk_then_end():
    pcm = b"\x00\x01\x02\x03"
    assert list(read_response(_stream(pack_audio(pcm), pack_end()))) == [pcm]


def test_multiple_chunks_preserve_order():
    a, b = b"\xaa\xbb", b"\xcc\xdd\xee\xff"
    assert list(read_response(_stream(pack_audio(a), pack_audio(b), pack_end()))) == [a, b]


def test_end_only_yields_nothing():
    assert list(read_response(_stream(pack_end()))) == []


def test_error_frame_raises_with_message():
    with pytest.raises(RvcSidecarError, match="kaboom"):
        list(read_response(_stream(pack_error("kaboom"))))


def test_truncated_stream_raises_not_silent():
    with pytest.raises(RvcProtocolError):
        list(read_response(_stream(pack_audio(b"\x00\x01\x02\x03"))))


def test_partial_header_raises():
    with pytest.raises(RvcProtocolError):
        list(read_response(_stream(b"A\x00\x00")))


# --- request framing (server -> sidecar): JSON header + PCM block ---
def test_request_roundtrips_params_and_pcm():
    pcm = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    raw = encode_request(
        pcm,
        voice="charlie",
        transpose=2,
        index_rate=0.5,
        f0_method="rmvpe",
        protect=0.33,
        rms_mix_rate=0.25,
        passes=3,
        gate=True,
        speed=1.1,
    )
    d = read_request(io.BytesIO(raw))
    assert d["voice"] == "charlie"
    assert d["transpose"] == 2
    assert d["index_rate"] == 0.5
    assert d["f0_method"] == "rmvpe"
    assert d["passes"] == 3
    assert d["gate"] is True
    assert d["speed"] == 1.1
    assert d["pcm"] == pcm


def test_request_pcm_may_contain_newline_bytes():
    # The PCM block is length-prefixed, so embedded 0x0a bytes must not be
    # mistaken for the header's line terminator.
    pcm = b"\x0a\x0a\x0a\x0a"
    d = read_request(
        io.BytesIO(
            encode_request(
                pcm,
                voice="charlie",
                transpose=0,
                index_rate=0.5,
                f0_method="rmvpe",
                protect=0.33,
                rms_mix_rate=0.25,
            )
        )
    )
    assert d["pcm"] == pcm


def test_read_request_returns_none_at_eof():
    assert read_request(io.BytesIO(b"")) is None
