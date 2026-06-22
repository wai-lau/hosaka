import argparse
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from hosaka.config import BAKE_SEED_TEXT, VOICE_DIR, SAMPLE_RATE
from hosaka.library import VoiceLibrary

_MODEL = "parler-tts/parler-tts-mini-v1.1"


def bake(description: str, voice_id: str, text: str = BAKE_SEED_TEXT) -> str:
    model = ParlerTTSForConditionalGeneration.from_pretrained(_MODEL).to("cuda")
    tok = AutoTokenizer.from_pretrained(_MODEL)
    desc_ids = tok(description, return_tensors="pt").input_ids.to("cuda")
    prompt_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        audio = model.generate(input_ids=desc_ids,
                               prompt_input_ids=prompt_ids)
    wav = audio.cpu().numpy().squeeze().astype(np.float32)

    src_sr = model.config.sampling_rate          # Parler emits 44.1 kHz
    if src_sr != SAMPLE_RATE:
        wav = resample_poly(wav, SAMPLE_RATE, src_sr).astype(np.float32)

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VOICE_DIR / f"{voice_id}.wav"
    sf.write(out_path, wav, SAMPLE_RATE)
    VoiceLibrary(VOICE_DIR).add(voice_id, out_path, source="bake",
                               params={"description": description})
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(prog="hosaka-bake")
    ap.add_argument("description", help="natural-language voice description")
    ap.add_argument("--out", required=True, help="voice id to save as")
    ap.add_argument("--text", default=BAKE_SEED_TEXT,
                    help="sentence to speak for the seed clip")
    args = ap.parse_args()
    path = bake(args.description, args.out, args.text)
    print(f"baked voice '{args.out}' -> {path}")


if __name__ == "__main__":
    main()
