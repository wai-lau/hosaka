import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    if has_gpu:
        return
    skip = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
