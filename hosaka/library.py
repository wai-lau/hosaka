import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class VoiceEntry:
    id: str
    path: str
    source: str            # "recording" | "bake" | "kokoro"
    params: dict
    created: str


class VoiceLibrary:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.json"

    def _read(self) -> dict:
        if not self.manifest.exists():
            return {}
        return json.loads(self.manifest.read_text())

    def _write(self, data: dict) -> None:
        tmp = self.manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.manifest)

    def add(self, voice_id, wav_path, source, params=None, created="") -> VoiceEntry:
        dest = self.root / f"{voice_id}.wav"
        shutil.copyfile(wav_path, dest)
        entry = VoiceEntry(voice_id, str(dest), source, params or {}, created)
        data = self._read()
        data[voice_id] = asdict(entry)
        self._write(data)
        return entry

    def get(self, voice_id) -> VoiceEntry | None:
        raw = self._read().get(voice_id)
        return VoiceEntry(**raw) if raw else None

    def list(self) -> list[VoiceEntry]:
        return [VoiceEntry(**v) for v in self._read().values()]

    def path_for(self, voice_id) -> Path | None:
        e = self.get(voice_id)
        return Path(e.path) if e else None
