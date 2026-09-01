"""Confirm the GPU actually runs kernels before starting a multi-hour job.

torch.cuda.is_available() returns True on cards whose architecture the installed
build does not support, so it is a false reassurance. This runs a real bf16
matmul, which is what training will do.

    python check_gpu.py
"""

import sys


def main():
    try:
        import torch
    except ImportError:
        print("FAIL  torch is not installed")
        return 1

    print(f"torch        {torch.__version__}")
    print(f"cuda         {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("FAIL  no CUDA device visible")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu          {name}")
    print(f"sm           {cap[0]}.{cap[1]}")
    print(f"vram         {total:.1f} GB")
    print(f"bf16         {torch.cuda.is_bf16_supported()}")

    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        ok = bool((x @ x).sum().isfinite())
    except Exception as exc:
        print(f"\nFAIL  bf16 matmul raised {type(exc).__name__}: {exc}")
        print("      The build does not support this card's architecture.")
        print(f"      sm_{cap[0]}{cap[1]} needs a newer CUDA/PyTorch template:")
        print("      Blackwell (sm_120) requires CUDA 12.8+ and PyTorch 2.7+.")
        return 1

    if not ok:
        print("\nFAIL  matmul produced non-finite values")
        return 1

    print("\nOK    bf16 matmul succeeded — safe to train")
    if not torch.cuda.is_bf16_supported():
        print("WARN  bf16 unsupported on this card; pass --fp16 to train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
