"""Check the chunked pure-PyTorch selective scan against the sequential reference
(forward and all input gradients), on CPU in float32, for several sequence lengths
including a non-divisible one. Usage: python -I test_scan.py <scripts/sum dir>"""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch
from sum_common import selective_scan_seq_ref, selective_scan_chunked

torch.manual_seed(0)
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
worst = 0.0
for (b, K, h, N, L) in [(2, 4, 6, 16, 64), (1, 4, 8, 16, 256), (2, 4, 4, 16, 100), (1, 4, 4, 16, 1024)]:
    D = K * h
    u = torch.randn(b, D, L, device=dev, requires_grad=True)
    delta = (torch.randn(b, D, L, device=dev) * 0.5).requires_grad_(True)
    A = (-torch.exp(torch.randn(D, N, device=dev) * 0.5 + 0.5)).requires_grad_(True)
    B = torch.randn(b, K, N, L, device=dev, requires_grad=True)
    C = torch.randn(b, K, N, L, device=dev, requires_grad=True)
    Dsk = torch.randn(D, device=dev, requires_grad=True)
    bias = (torch.randn(D, device=dev) * 0.3).requires_grad_(True)
    gy = torch.randn(b, D, L, device=dev)
    outs = {}
    for name, fn in (("ref", selective_scan_seq_ref), ("chunk", selective_scan_chunked)):
        for t in (u, delta, A, B, C, Dsk, bias):
            t.grad = None
        t0 = time.time()
        y = fn(u, delta, A, B, C, Dsk, z=None, delta_bias=bias, delta_softplus=True)
        (y * gy).sum().backward()
        outs[name] = [y.detach().clone()] + [t.grad.clone() for t in (u, delta, A, B, C, Dsk, bias)]
        print(f"  {name} L={L}: {time.time()-t0:.3f}s")
    names = ["y", "du", "ddelta", "dA", "dB", "dC", "dD", "dbias"]
    for n, r, c in zip(names, outs["ref"], outs["chunk"]):
        rel = float((r - c).abs().max() / (r.abs().max() + 1e-12))
        worst = max(worst, rel)
        print(f"L={L:5d} {n:7s} rel max err {rel:.2e}")
print(f"WORST rel err {worst:.2e}")
assert worst < 1e-4, "chunked scan disagrees with reference"
print("SCAN TEST PASSED")
