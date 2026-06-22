import sys
import time
import statistics
import numpy as np
import soundfile as sf
from pathlib import Path
from hosaka.config import SAMPLE_RATE
from hosaka.library import VoiceLibrary
from hosaka.server.engines.kokoro_engine import KokoroEngine
from hosaka.server.engines.chatterbox_engine import ChatterboxEngine

SHORT = "Hello, this is a short first sentence."
RUNS = 5


def first_chunk_ms(stream_iter) -> float:
    t0 = time.perf_counter()
    next(stream_iter)
    return (time.perf_counter() - t0) * 1000.0


def bench(label, make_iter) -> list[float]:
    make_iter()                              # warm
    times = [first_chunk_ms(make_iter()) for _ in range(RUNS)]
    print(f"{label}: first-chunk ms "
          f"min={min(times):.0f} median={statistics.median(times):.0f} "
          f"max={max(times):.0f}")
    return times


def main(tmp="/tmp/hosaka_bench") -> int:
    Path(tmp).mkdir(exist_ok=True)
    kok = KokoroEngine()
    bench("kokoro", lambda: kok.stream(SHORT, "af_heart", {"speed": 1.0}))

    seed = np.concatenate(list(
        kok.stream("Seed clip for cloning, spoken clearly and calmly.",
                   "af_heart", {"speed": 1.0})))
    seed_path = Path(tmp) / "seed.wav"
    sf.write(seed_path, seed, SAMPLE_RATE)
    lib = VoiceLibrary(Path(tmp) / "voices")
    lib.add("bench_seed", seed_path, source="kokoro")

    cb = ChatterboxEngine(lib)
    cb_times = bench(
        "chatterbox",
        lambda: cb.stream(SHORT, "bench_seed",
                          {"exaggeration": 0.5, "cfg_weight": 0.4,
                           "temperature": 0.8}))

    if statistics.median(cb_times) > 1000.0:
        print("GATE FAIL: Chatterbox median first-chunk > 1s. "
              "Consider fallback: Turbo (loses knobs) or XTTS-v2.")
        return 1
    print("GATE PASS: Chatterbox first-chunk within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
