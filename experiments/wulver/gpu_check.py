"""Report what GPU a course job actually gets, and measure its matmul throughput.

The streaming-multiprocessor (SM) count shows which MIG slice this is: a full A100 has 108 SMs,
so a slice with 14 SMs is one seventh of the GPU. The measured bf16 TFLOP/s replaces the speed
assumption in the project plan. Writes results/pilot/gpu_check.json.
"""
import json
import time
from pathlib import Path

import torch


def matmul_tflops(dtype, n=8192, iters=20):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


def main():
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        info.update({
            "device_name": p.name,
            "total_memory_GB": round(p.total_memory / 1e9, 2),
            "streaming_multiprocessors": p.multi_processor_count,
            "fraction_of_full_A100_SMs": round(p.multi_processor_count / 108, 3),
            "compute_capability": f"{p.major}.{p.minor}",
            "cuda_runtime": torch.version.cuda,
            "bf16_matmul_TFLOPs": round(matmul_tflops(torch.bfloat16), 1),
            "tf32_matmul_TFLOPs": round(matmul_tflops(torch.float32), 1),
        })
    out = Path("results/pilot")
    out.mkdir(parents=True, exist_ok=True)
    (out / "gpu_check.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
