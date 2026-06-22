from pathlib import Path
from hosaka.library import VoiceLibrary


def _make_wav(p: Path):
    p.write_bytes(b"RIFFfakewav")


def test_add_then_get_and_list(tmp_path):
    src = tmp_path / "src.wav"
    _make_wav(src)
    lib = VoiceLibrary(tmp_path / "voices")

    entry = lib.add("calm_brit", src, source="bake",
                    params={"exaggeration": 0.4}, created="2026-06-22")

    assert entry.id == "calm_brit"
    assert Path(entry.path).exists()
    assert lib.get("calm_brit").source == "bake"
    assert [e.id for e in lib.list()] == ["calm_brit"]


def test_persists_across_instances(tmp_path):
    src = tmp_path / "s.wav"
    _make_wav(src)
    root = tmp_path / "voices"
    VoiceLibrary(root).add("v1", src, source="recording")

    reopened = VoiceLibrary(root)
    assert reopened.get("v1") is not None


def test_get_missing_returns_none(tmp_path):
    assert VoiceLibrary(tmp_path / "voices").get("nope") is None
