import sys

import torch


def main() -> int:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("device", name, "capability", cap)
    if cap != (12, 0):
        print(f"WARN: expected sm_120 (12, 0), got {cap}")
    x = torch.randn(4096, 4096, device="cuda")
    val = (x @ x).sum().item()  # must not raise "no kernel image"
    print("matmul ok, sum =", val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
