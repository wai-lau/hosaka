"""Wire protocol between the server-side RvcEngine client and the isolated
.venv-rvc sidecar. Pure stdlib so it imports cleanly in every venv.

Request  (server -> sidecar): one JSON header line, newline-terminated, then a
length-prefixed source-PCM block:

    {json params}\n
    uint32be N + <N bytes float32 LE PCM @ 24k>

Response (sidecar -> server): tagged length-prefixed frames (same contract as
piper_proto):

    b'A' + uint32be N + <N bytes float32 LE PCM>   converted audio chunk
    b'X' + uint32be N + <N bytes utf-8 message>    synthesis error -> raises
    b'E'                                           end of utterance (success)

A stream that closes before an 'E' marker is a dead sidecar -> RvcProtocolError,
never a silently truncated success.
"""

import json
import struct

_AUDIO = b"A"
_ERROR = b"X"
_END = b"E"
_LEN = struct.Struct(">I")


class RvcSidecarError(RuntimeError):
    """The sidecar reported a synthesis failure (error frame)."""


class RvcProtocolError(RuntimeError):
    """The frame stream was malformed or ended before the end marker."""


def pack_audio(pcm: bytes) -> bytes:
    return _AUDIO + _LEN.pack(len(pcm)) + pcm


def pack_error(message: str) -> bytes:
    m = message.encode("utf-8")
    return _ERROR + _LEN.pack(len(m)) + m


def pack_end() -> bytes:
    return _END


def _read_exact(reader, n: int) -> bytes:
    chunks = []
    remaining = n
    # A raw pipe read(n) may return fewer than n bytes; loop until we have all
    # n or hit a true EOF (empty read).
    while remaining:
        b = reader.read(remaining)
        if not b:
            raise RvcProtocolError(f"stream ended: wanted {n} bytes, short by {remaining}")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


def read_response(reader):
    """Yield converted PCM payloads frame by frame until the end marker.
    Raises RvcSidecarError on an error frame, RvcProtocolError on malformed /
    truncated streams."""
    while True:
        tag = reader.read(1)
        if tag == _END:
            return
        if not tag:
            raise RvcProtocolError("stream closed before end marker")
        if tag == _AUDIO:
            (n,) = _LEN.unpack(_read_exact(reader, 4))
            yield _read_exact(reader, n)
        elif tag == _ERROR:
            (n,) = _LEN.unpack(_read_exact(reader, 4))
            raise RvcSidecarError(_read_exact(reader, n).decode("utf-8", "replace"))
        else:
            raise RvcProtocolError(f"bad frame tag: {tag!r}")


def encode_request(
    pcm: bytes,
    *,
    voice: str,
    transpose: int,
    index_rate: float,
    f0_method: str,
    protect: float,
    rms_mix_rate: float,
) -> bytes:
    header = (
        json.dumps(
            {
                "voice": voice,
                "transpose": transpose,
                "index_rate": index_rate,
                "f0_method": f0_method,
                "protect": protect,
                "rms_mix_rate": rms_mix_rate,
            }
        ).encode("utf-8")
        + b"\n"
    )
    return header + _LEN.pack(len(pcm)) + pcm


def read_request(reader):
    """Read one request (JSON header line + length-prefixed PCM block) from a
    binary reader. Returns a dict with the params plus a `pcm` bytes field, or
    None at clean EOF. Uses the SAME reader for the line and the block, so a
    BufferedReader's read-ahead stays consistent."""
    line = reader.readline()
    if not line:
        return None
    d = json.loads(line)
    (n,) = _LEN.unpack(_read_exact(reader, 4))
    d["pcm"] = _read_exact(reader, n)
    return d
