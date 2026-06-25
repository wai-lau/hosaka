#!/usr/bin/env python
"""Fake RVC sidecar for RvcEngine tests: speaks the wire protocol with no
torch / rvc-python / models. Behavior is keyed off the request `voice` so tests
can drive edge cases over a real subprocess pipe.

  voice == "boom" -> one error frame (sidecar stays alive)
  voice == "die"  -> exit mid-utterance (no end marker)
  voice == "echo" -> error frame carrying the received params (routing proof)
  otherwise       -> echo the received source PCM back as one audio frame, end
                     (proves source accumulation + PCM round-trips both ways)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from hosaka.server.engines.rvc_proto import (  # noqa: E402
    pack_audio,
    pack_end,
    pack_error,
    read_request,
)


def main():
    out = sys.stdout.buffer
    while True:
        req = read_request(sys.stdin.buffer)
        if req is None:
            return
        voice = req.get("voice", "")
        if voice == "echo":
            params = {k: v for k, v in req.items() if k != "pcm"}
            out.write(pack_error(json.dumps(params, sort_keys=True)))
            out.flush()
            continue
        if voice == "boom":
            out.write(pack_error("boom"))
            out.flush()
            continue
        if voice == "die":
            sys.exit(1)
        out.write(pack_audio(req["pcm"]))  # echo the source straight back
        out.write(pack_end())
        out.flush()


if __name__ == "__main__":
    main()
