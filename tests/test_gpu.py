from hosaka.gpu import pick_device

FAST = "3772ace9-158c-1130-cc88-f09cacf35b27"
SLOW = "58cfdb6c-f7da-5d61-c081-73c6f81984cd"
UUIDS = [FAST, SLOW]  # torch order (fastest first)


def test_none_spec_is_default_cuda():
    assert pick_device(None, UUIDS) == "cuda"


def test_uuid_maps_to_torch_index_with_or_without_prefix():
    assert pick_device(f"GPU-{SLOW}", UUIDS) == "cuda:1"
    assert pick_device(FAST, UUIDS) == "cuda:0"
    assert pick_device(f"gpu-{SLOW.upper()}", UUIDS) == "cuda:1"


def test_missing_uuid_falls_back_to_default(capsys):
    assert pick_device("GPU-deadbeef", UUIDS) == "cuda"
    assert "not found" in capsys.readouterr().err
