import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from hosaka.config import SAMPLE_RATE
from hosaka.library import VoiceLibrary
from hosaka.server.engines.chatterbox_engine import ChatterboxEngine
from hosaka.server.engines.kokoro_engine import KokoroEngine

SHORT = "Hello, this is a short first sentence."
RUNS = 5


def first_chunk_ms(stream_iter) -> float:
    t0 = time.perf_counter()
    next(stream_iter)
    return (time.perf_counter() - t0) * 1000.0


def full_gen_ms(engine, text, voice, params):
    t0 = time.perf_counter()
    audio = np.concatenate(list(engine.stream(text, voice, params)))
    return (time.perf_counter() - t0) * 1000.0, audio.size / SAMPLE_RATE


def main(tmp="/tmp/hosaka_bench") -> int:
    Path(tmp).mkdir(exist_ok=True)

    # --- Kokoro: the realtime path. Gate first-chunk < 1000 ms. ---
    kok = KokoroEngine()
    kok.stream(SHORT, "af_heart", {"speed": 1.0}).__next__()  # warm
    k_times = [first_chunk_ms(kok.stream(SHORT, "af_heart", {"speed": 1.0})) for _ in range(RUNS)]
    k_median = statistics.median(k_times)
    print(
        f"kokoro (realtime path): first-chunk ms "
        f"min={min(k_times):.0f} median={k_median:.0f} max={max(k_times):.0f}"
    )

    # --- Chatterbox: quality mode. Report full-generate time; do NOT gate. ---
    seed = np.concatenate(
        list(
            kok.stream(
                "Seed clip for cloning, spoken clearly and calmly.", "af_heart", {"speed": 1.0}
            )
        )
    )
    seed_path = Path(tmp) / "seed.wav"
    sf.write(seed_path, seed, SAMPLE_RATE)
    lib = VoiceLibrary(Path(tmp) / "voices")
    lib.add("bench_seed", seed_path, source="kokoro")

    cb = ChatterboxEngine(lib)
    full_gen_ms(
        cb, SHORT, "bench_seed", {"exaggeration": 0.5, "cfg_weight": 0.4, "temperature": 0.8}
    )  # warm
    cb_runs = [
        full_gen_ms(
            cb, SHORT, "bench_seed", {"exaggeration": 0.5, "cfg_weight": 0.4, "temperature": 0.8}
        )
        for _ in range(RUNS)
    ]
    cb_times = [t for t, _dur in cb_runs]
    print(
        f"chatterbox (quality mode): full-generate ms "
        f"min={min(cb_times):.0f} median={statistics.median(cb_times):.0f} "
        f"max={max(cb_times):.0f} (audio ~{cb_runs[0][1]:.1f}s)"
    )

    # Only the realtime path is gated.
    if k_median > 1000.0:
        print("GATE FAIL: Kokoro realtime first-chunk > 1s on this box.")
        return 1
    print(
        "GATE PASS: Kokoro realtime path within budget. "
        "Chatterbox is the non-realtime quality path (expected ~2-3s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
