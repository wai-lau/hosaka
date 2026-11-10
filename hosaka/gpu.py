"""Resolve a configured GPU (NVIDIA UUID) to a torch device string.

Pure matching lives in `pick_device` (tested in .venv-dev); `cuda_device` and
`present_uuid` query torch lazily, so importing this module never pulls torch.
"""

import sys


def _norm(uuid: str) -> str:
    u = str(uuid).strip().lower()
    return u[4:] if u.startswith("gpu-") else u


def pick_device(spec, uuids) -> str:
    """spec: a "GPU-..." UUID or None. uuids: torch-ordered device UUIDs.
    Returns "cuda:N" for the matching card, else "cuda" (torch's default)."""
    if not spec:
        return "cuda"
    want = _norm(spec)
    for i, u in enumerate(uuids):
        if _norm(u) == want:
            return f"cuda:{i}"
    print(f"hosaka: GPU {spec} not found; using default cuda device", file=sys.stderr)
    return "cuda"


def _torch_uuids():
    import torch

    return [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]


def cuda_device(spec) -> str:
    """Torch device string for an in-process engine (Kokoro, Chatterbox)."""
    return pick_device(spec, _torch_uuids()) if spec else "cuda"


def present_uuid(spec):
    """spec if this box has that card, else None -- for CUDA_VISIBLE_DEVICES on a
    sidecar (pinning a missing UUID would hide every GPU from it)."""
    return spec if spec and pick_device(spec, _torch_uuids()) != "cuda" else None
