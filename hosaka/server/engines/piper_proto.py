"""Wire protocol between the server-side PiperEngine client and the isolated
.venv-piper sidecar. Pure stdlib so it imports cleanly in every venv -- the
sidecar runs under .venv-piper, the engine and its tests under the server/dev
venvs.

Request  (server -> sidecar): one JSON object per line, newline-terminated.
Response (sidecar -> server): a stream of tagged, length-prefixed frames:

    b'A' + uint32be N + <N bytes float32 LE PCM>   audio chunk (one sentence)
    b'X' + uint32be N + <N bytes utf-8 message>    synthesis error -> raises
    b'E'                                           end of utterance (success)

A stream that closes before an 'E' marker is a dead sidecar -> PiperProtocolError,
never a silently truncated success (same contract as a mid-stream engine error
on the GPU path).
"""

import json
import struct

_AUDIO = b"A"
_ERROR = b"X"
_END = b"E"
_LEN = struct.Struct(">I")


class PiperSidecarError(RuntimeError):
    """The sidecar reported a synthesis failure (error frame)."""


class PiperProtocolError(RuntimeError):
    """The frame stream was malformed or ended before the end marker."""


def pack_audio(pcm: bytes) -> bytes:
    return _AUDIO + _LEN.pack(len(pcm)) + pcm


def pack_error(message: str) -> bytes:
    m = message.encode("utf-8")
    return _ERROR + _LEN.pack(len(m)) + m


def pack_end() -> bytes:
    return _END


def _read_exact(reader, n: int) -> bytes:
    # A raw pipe read(n) may return fewer than n bytes; loop until we have all
    # n or hit a true EOF (empty read).
    chunks = []
    remaining = n
    while remaining:
        b = reader.read(remaining)
        if not b:
            raise PiperProtocolError(f"stream ended: wanted {n} bytes, short by {remaining}")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


def read_response(reader):
    """Yield PCM payloads (bytes) frame by frame until the end marker.

    Raises PiperSidecarError on an error frame and PiperProtocolError if the
    stream is malformed or closes before the end marker.
    """
    while True:
        tag = reader.read(1)
        if tag == _END:
            return
        if not tag:
            raise PiperProtocolError("stream closed before end marker")
        if tag == _AUDIO:
            (n,) = _LEN.unpack(_read_exact(reader, 4))
            yield _read_exact(reader, n)
        elif tag == _ERROR:
            (n,) = _LEN.unpack(_read_exact(reader, 4))
            raise PiperSidecarError(_read_exact(reader, n).decode("utf-8", "replace"))
        else:
            raise PiperProtocolError(f"bad frame tag: {tag!r}")


def encode_request(
    text: str, *, voice: str, length_scale: float, noise_scale: float, noise_w: float
) -> bytes:
    # Flatten newlines so one request stays one line (the framing is line-based).
    flat = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    payload = {
        "text": flat,
        "voice": voice,
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w,
    }
    return json.dumps(payload).encode("utf-8") + b"\n"


def parse_request(raw: bytes) -> dict:
    return json.loads(raw)
