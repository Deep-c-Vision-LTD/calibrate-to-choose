"""
Inference cost of a fine-tuned ftblocks model, measured with the protocol of scripts/compile/cost_bench.py
(model 'ours'): the deployed pipeline is ASPPSaliency (DINOv2-base 448 + ASPP readout) with the fine-tuned
weights of blocks 12-N+1..12, the final LayerNorm and the readout loaded into it (no new parameters, no
new ops). Reports:
  * parameter count and state-dict compatibility with ours (strict load, identical key set and shapes)
  * GFLOPs: FlopCounterMode, SDPA math backend, batch 1, 1x3x448x448 (cost_bench convention)
  * end-to-end check: from pixels (full pipeline) vs the training path (cached frozen block output ->
    fine-tuned blocks -> readout) on a few of the fold's test images (CPU, same weights)
  * CPU latency (6 threads, 3 warm-up + 30 timed batch-1 passes, median) ONLY with --latency, to be used
    only when the machine is quiet (no other job on CPU/GPU)
Writes results/ftblocks_cost.json (new file).

  python -I cost_ftblocks.py --model-file cache/ftblocks/models/final_k0_<cfg>.pt [--latency]
"""
import sys
sys.dont_write_bytecode = True
import os, re, json, time, argparse, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import ft_common as FC  # noqa: E402
import ours_lib as L    # noqa: E402
sys.path.insert(0, os.path.join(FC.KB, "scripts", "ours"))
from model_aspp import ASPPSaliency  # noqa: E402


class OursPipe(nn.Module):          # = cost_bench.py OursPipe
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        out_ = self.m.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        tok = out_[:, 1:, :]
        B, N, C = tok.shape; h = int(round(N ** 0.5))
        f = tok.transpose(1, 2).reshape(B, C, h, h)
        s = self.m.head(self.m.aspp(f))
        s = s + F.interpolate(self.m.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(F.interpolate(s, size=(384, 384), mode="bilinear", align_corners=False))


def map_state(sd, n_ft):
    out = {}
    for k, v in sd.items():
        if n_ft == 0:
            out[k] = v                                   # ReadoutHead keys: aspp.*, head.*, center_bias
        elif k.startswith("head."):
            out[k[len("head."):]] = v
        elif k.startswith("blocks."):
            i, rest = k[len("blocks."):].split(".", 1)
            out[f"backbone.encoder.layer.{12 - n_ft + int(i)}.{rest}"] = v
        elif k.startswith("norm."):
            out["backbone.layernorm." + k[len("norm."):]] = v
        elif k.startswith("frozen."):
            continue                                     # frozen copies of pretrained blocks (unchanged weights)
        else:
            raise KeyError(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--n-check", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(FC.RESULTS, "ftblocks_cost.json"))
    a = ap.parse_args()
    FC.set_env(); torch.set_num_threads(a.threads)
    ck = torch.load(a.model_file, map_location="cpu", weights_only=False)
    cfg = ck["config"]; n_ft = int(cfg["N"])
    model = ASPPSaliency(pretrained=True).eval()
    ref_keys = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    n_ref = sum(p.numel() for p in model.parameters())
    new = map_state(ck["state_dict"], n_ft)
    bad = [k for k, v in new.items() if k not in ref_keys or tuple(v.shape) != ref_keys[k]]
    assert not bad, bad[:5]
    full = model.state_dict(); full.update(new)
    model.load_state_dict(full, strict=True)
    n_after = sum(p.numel() for p in model.parameters())
    net = OursPipe(model).eval()
    out = {"model_file": a.model_file, "config": cfg, "params_total": int(n_after), "params_total_ours": int(n_ref),
           "same_param_count_as_ours": n_after == n_ref, "state_dict_keys_replaced": len(new),
           "torch": torch.__version__, "machine": platform.platform(), "threads": a.threads}
    # FLOPs, cost_bench convention
    from torch.utils.flop_counter import FlopCounterMode
    from torch.nn.attention import sdpa_kernel, SDPBackend
    x = torch.randn(1, 3, 448, 448, generator=torch.Generator().manual_seed(0))
    net.requires_grad_(False)
    fc = FlopCounterMode(display=False, depth=3)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), fc:
        net(x)
    out["GFLOPs"] = int(fc.get_total_flops()) / 1e9
    out["flops_by_op_global"] = {str(k): v / 1e9 for k, v in fc.get_flop_counts().get("Global", {}).items()}
    out["flops_convention"] = "cost_bench.py: 2 x MACs of convolution and matmul ops, FlopCounterMode, SDPA math backend, batch 1, 448x448"
    # end-to-end check: pixels -> full pipeline vs cached block output -> fine-tuned blocks -> readout (CPU)
    ids = L.load_ids()
    fold = int(re.search(r"_k(\d)_", os.path.basename(a.model_file)).group(1))
    rec = json.load(open(FC.INNER_SPLITS))["folds"][fold]
    test_ids = rec["test_ids"][:a.n_check]
    head = L.ReadoutHead("full")
    if n_ft == 0:
        tm = head; tm.load_state_dict(ck["state_dict"])
        f16 = np.load(os.path.join(L.CACHE, "feats_dinov2b_448_fp16.npy"), mmap_mode="r")
        getx = lambda j: torch.from_numpy(np.array(f16[j:j + 1])).float()
    else:
        tm = FC.FTModel(head, FC.load_backbone(), n_ft, int(cfg["source"]))
        tm.load_state_dict(ck["state_dict"])
        mm = np.load(FC.ADD_CACHE[int(cfg["source"])], mmap_mode="r")
        getx = lambda j: torch.from_numpy(np.array(mm[j:j + 1])).float()
    tm.eval()
    diffs = []
    with torch.inference_mode():
        for i in test_ids:
            j = ids.index(i)
            p_full = net(FC.preprocess(os.path.join(L.FLAT, "images", f"{i}.jpg")))[0, 0]
            s = tm(getx(j))
            p_cache = torch.sigmoid(F.interpolate(s, size=(384, 384), mode="bilinear", align_corners=False))[0, 0]
            diffs.append(float((p_full - p_cache).abs().max()))
    out["end_to_end_check"] = {"images": test_ids, "max_abs_diff_sigmoid_384": diffs,
                               "note": "full pixel pipeline vs the training path from the cached frozen block output (n_ft = 0: fp16 feature "
                                       "cache, so differences up to fp16 rounding are expected), CPU"}
    if a.latency:
        times = []
        with torch.inference_mode():
            for r in range(a.warmup + a.runs):
                t0 = time.perf_counter(); net(x); times.append(time.perf_counter() - t0)
        t = times[a.warmup:]
        q = np.percentile(t, [25, 50, 75])
        out["cpu_latency_s"] = {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]), "min": float(min(t)),
                                "runs": a.runs, "warmup": a.warmup, "loadavg_at_start": os.getloadavg()}
    else:
        out["cpu_latency_s"] = "not measured (machine not quiet)"
    p = a.out
    json.dump(out, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    print(json.dumps({k: v for k, v in out.items() if k != "flops_by_op_global"}, indent=1))


if __name__ == "__main__":
    main()
