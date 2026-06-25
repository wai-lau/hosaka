#!/usr/bin/env python
"""Fake Piper sidecar for PiperEngine tests: speaks the wire protocol with no
piper/onnx dependency. Behavior is keyed off the request text so tests can
drive edge cases (errors, mid-utterance death) over a real subprocess pipe.

  text == "BOOM" -> one error frame (sidecar stays alive)
  text == "DIE"  -> exit mid-utterance (no end marker)
  otherwise      -> one 100-sample float32 audio frame per '.' (min 1), then end
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from hosaka.server.engines.piper_proto import (  # noqa: E402
    pack_audio,
    pack_end,
    pack_error,
    parse_request,
)


def main():
    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        req = parse_request(line)
        text = req["text"]
        if text == "ECHOVOICE":
            out.write(pack_error(req.get("voice", "")))  # let tests assert routing
            out.flush()
            continue
        if text == "BOOM":
            out.write(pack_error("boom"))
            out.flush()
            continue
        if text == "DIE":
            sys.exit(1)  # die mid-utterance, no end marker
        for _ in range(max(1, text.count("."))):
            out.write(pack_audio(struct.pack("<100f", *([0.0] * 100))))
        out.write(pack_end())
        out.flush()


if __name__ == "__main__":
    main()
