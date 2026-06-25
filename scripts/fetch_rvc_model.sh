#!/usr/bin/env bash
# Fetch RVC assets + the Charlie Morningstar voice into the hosaka data dir.
#
#   hubert_base.pt, rmvpe.pt   canonical RVC assets (lj1995/VoiceConversionWebUI)
#   charlie                    Darkynauta/HazbinHotel "Charlie Morningstar -
#                              Erika Henningsen" (RVC V2, the English voice actor)
#
# Only the trained weights are fetched. Idempotent: skips files already present.
set -euo pipefail

ROOT="${HOME}/.local/share/hosaka/rvc"
HF="https://huggingface.co"
mkdir -p "$ROOT"

fetch() {  # url dest
  if [ -f "$2" ]; then echo "have    ${2#"$ROOT"/}"; return; fi
  echo "fetch   ${2#"$ROOT"/}"
  curl -fL "$1" -o "$2"
}

# --- shared assets ---
fetch "${HF}/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" "${ROOT}/hubert_base.pt"
fetch "${HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" "${ROOT}/rmvpe.pt"

# --- Charlie ---
CDIR="${ROOT}/charlie"
mkdir -p "$CDIR"
if [ -f "${CDIR}/charlie.pth" ] && [ -f "${CDIR}/charlie.index" ]; then
  echo "have    charlie/charlie.{pth,index}"
else
  ZIP="${CDIR}/charlie.zip"
  fetch "${HF}/Darkynauta/HazbinHotel/resolve/main/Charlie%20Morningstar%20-%20Erika%20Henningsen.zip" "$ZIP"
  echo "unzip   charlie"
  unzip -o -j "$ZIP" -d "$CDIR" >/dev/null
  # Normalize whatever the archive named them to charlie.{pth,index}.
  pth="$(find "$CDIR" -maxdepth 1 -name '*.pth' | head -1)"
  idx="$(find "$CDIR" -maxdepth 1 -name '*.index' | head -1)"
  [ -n "$pth" ] || { echo "ERROR: no .pth in zip"; exit 1; }
  [ -n "$idx" ] || { echo "ERROR: no .index in zip"; exit 1; }
  [ "$pth" = "${CDIR}/charlie.pth" ] || mv -f "$pth" "${CDIR}/charlie.pth"
  [ "$idx" = "${CDIR}/charlie.index" ] || mv -f "$idx" "${CDIR}/charlie.index"
  rm -f "$ZIP"
fi

echo "FETCH_RVC_DONE -> ${ROOT}"
