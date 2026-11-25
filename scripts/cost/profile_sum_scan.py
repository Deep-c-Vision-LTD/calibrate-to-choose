"""
Where does SUM's CPU (and MPS) forward time go?  Selective scan vs everything else.

Reuses the cost-bench protocol of scripts/compile/cost_bench.py for --model sum exactly:
SUM built by sum_common.load_sum(..., scan="chunked") from the released weights, batch 1, float32,
fixed input torch.randn(1,3,256,256, generator seed 0) + one-hot domain token 2, torch.inference_mode,
torch.set_num_threads(threads), `warmup` untimed passes then the median of `runs` timed passes
(MPS: torch.mps.synchronize() at the end of each pass). Each mode below is measured that way, in one
process, in this order:

  chunked          the cost-table configuration (pure-PyTorch chunked scan, sum_common.py), unwrapped
  chunked_timed    same, with vm.selective_scan_fn wrapped by perf_counter timers (MPS: synchronize
                   before and after each scan call) -> per-forward scan time and scan fraction
  chunked_sections (CPU only) an instrumented copy of the chunked scan that runs the same ops in the
                   same order through the same sum_common helpers, with timers between sections;
                   checked to give a bit-identical model output
  chunked_mulloop  (CPU only, DIAGNOSTIC) the same chunked scan with torch.cumprod over the chunk axis
                   replaced by a Python loop of torch.mul computing the same running product
  identity         the scan replaced by `return u` (same (b, K*D, L) float32 output, zero work):
                   an upper bound on what a perfect scan kernel could give
  ref_mamba        (CPU only, if available locally) mamba-ssm's selective_scan_ref, the sequential
                   reference recurrence, taken verbatim from the installed copy shipped inside
                   transformers 4.48.2 (kernels/falcon_mamba/selective_scan_with_ln_interface.py;
                   that module imports CUDA extensions at top level, so only the pure-PyTorch
                   function's source is extracted with ast and executed with torch/F/einops)
  ref_project      (CPU only) sum_common.selective_scan_seq_ref, the project's sequential reference
  chunked_end      the cost-table configuration again (drift bracket)

plus one torch.profiler pass (CPU only; 1 warm-up + 3 profiled forwards, chunked scan wrapped in
record_function("sum_selective_scan")) giving self time per aten op inside vs outside the scan.

Usage (OMP_NUM_THREADS=6, run from a temporary directory, python -I):
  python -I profile_sum_scan.py --device cpu --out <raw_cpu.json>
  python -I profile_sum_scan.py --device mps --out <raw_mps.json>        # under the shared MPS lock
  python -I profile_sum_scan.py --merge <raw_cpu.json> [<raw_mps.json>] --out results/sum_cpu_profile.json
Never overwrites an existing --out file.
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse, json, time, platform, warnings, ast, subprocess, resource
warnings.filterwarnings("ignore")

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SUM_DIR = os.path.join(KB, "scripts", "sum")
SUM_CODE = os.path.join(KB, "third_party", "SUM", "code")
SUM_W = os.path.join(KB, "third_party", "SUM", "weights")
# optional: a file that defines mamba-ssm's selective_scan_ref (the measured run used transformers 4.48.2,
# kernels/falcon_mamba/selective_scan_with_ln_interface.py); the reference mode is skipped when it is absent
MAMBA_REF_SRC = os.environ.get("MAMBA_REF_SRC", "")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--ref_warmup", type=int, default=3)
ap.add_argument("--ref_runs", type=int, default=30)
ap.add_argument("--skip_refs", action="store_true")
ap.add_argument("--merge", nargs="+", default=None)
ap.add_argument("--repro", default=None, help="merge: same-session cost_bench.py --model sum CPU output")
ap.add_argument("--notes", default=None, help="merge: JSON file with run notes (quiet check etc.)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if os.path.exists(a.out):
    sys.exit(f"refusing to overwrite existing file: {a.out}")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:      # noqa
        return f"<{e}>"


# =====================================================================================================
# merge mode
# =====================================================================================================
if a.merge:
    raws = [json.load(open(p)) for p in a.merge]
    cpu = next((r for r in raws if r["device"] == "cpu"), None)
    mps = next((r for r in raws if r["device"] == "mps"), None)
    ref = json.load(open(os.path.join(KB, "results", "cost_raw", "sum_cpu.json")))
    ours = json.load(open(os.path.join(KB, "results", "cost_raw", "ours_cpu.json")))
    out = {"description": ("Profile of SUM's batch-1 forward: time inside the selective-scan function vs "
                           "everything else, the identity-scan upper bound, and the sequential reference "
                           "scan; same protocol as scripts/compile/cost_bench.py --model sum."),
           "script": os.path.abspath(__file__),
           "raw_files": a.merge,
           "cost_table_reference": {
               "sum_cpu_latency_s_median": ref["latency_s"]["median"],
               "sum_GFLOPs": ref["GFLOPs"], "sum_scan_GFLOPs_estimate": ref["selective_scan_GFLOPs_estimate"],
               "ours_cpu_latency_s_median": ours["latency_s"]["median"],
               "ours_GFLOPs": ours.get("GFLOPs"),
               "source": "results/cost_raw/sum_cpu.json, results/cost_raw/ours_cpu.json (quiet machine)"}}
    if a.notes:
        out["run_notes"] = json.load(open(a.notes))
    if a.repro:
        rp = json.load(open(a.repro))
        out["same_session_cost_bench_repro"] = {
            "command": rp["command"], "latency_s": {k: rp["latency_s"][k] for k in
                                                    ("median", "q25", "q75", "mean", "min", "max")},
            "ratio_to_cost_table": rp["latency_s"]["median"] / ref["latency_s"]["median"],
            "peak_rss_during_inference_GB": rp.get("peak_rss_during_inference_GB")}
    if cpu:
        c, sm = cpu["modes"], cpu["summary"]
        base = sm["forward_chunked_s_median"]
        idl = sm["forward_identity_scan_s_median"]
        o_cpu = ours["latency_s"]["median"]
        out["headline_cpu"] = {
            "forward_s_median_chunked_scan": base,
            "forward_s_median_chunked_scan_end_of_session": sm["forward_chunked_end_s_median"],
            "scan_s_per_forward_median": sm["scan_s_per_forward_median"],
            "non_scan_s_per_forward_median": sm["non_scan_s_per_forward_median"],
            "scan_fraction_timed_wrapper": sm["scan_fraction_timed_wrapper"],
            "scan_fraction_profiler_self_time": sm.get("scan_fraction_profiler_self_time"),
            "forward_s_median_identity_scan": idl,
            "identity_saving_fraction": sm["identity_saving_fraction"],
            "identity_speedup_x": sm["identity_speedup_x"],
            "torch_cumprod_share_of_forward": c["chunked_sections"]["scan_sections_share_of_forward"]["4b_cumprod_decays"],
            "forward_s_median_mulloop_diagnostic": sm.get("forward_chunked_mulloop_s_median"),
            "forward_s_median_ref_mamba_selective_scan_ref": sm.get("forward_ref_mamba_s_median"),
            "forward_s_median_ref_project_seq": sm.get("forward_ref_project_s_median"),
            "ratio_sum_over_ours_B448_cost_table": ref["latency_s"]["median"] / o_cpu,
            "ratio_sum_identity_scan_over_ours_B448": idl / o_cpu,
            "ratio_sum_non_scan_over_ours_B448": sm["non_scan_s_per_forward_median"] / o_cpu,
            "ratio_sum_mulloop_over_ours_B448": (sm["forward_chunked_mulloop_s_median"] / o_cpu
                                                 if sm.get("forward_chunked_mulloop_s_median") else None),
            "ratio_sum_ref_mamba_over_ours_B448": (sm["forward_ref_mamba_s_median"] / o_cpu
                                                   if sm.get("forward_ref_mamba_s_median") else None),
            "ours_B448_cpu_s_quiet_cost_table": o_cpu}
    if mps:
        ours_m = json.load(open(os.path.join(KB, "results", "cost_raw", "ours_mps.json")))
        sm = mps["summary"]
        out["headline_mps"] = {
            "forward_s_median_chunked_scan": sm["forward_chunked_s_median"],
            "scan_s_per_forward_median_synced_wrapper": sm["scan_s_per_forward_median"],
            "scan_fraction_timed_wrapper": sm["scan_fraction_timed_wrapper"],
            "forward_s_median_identity_scan": sm["forward_identity_scan_s_median"],
            "identity_saving_fraction": sm["identity_saving_fraction"],
            "identity_speedup_x": sm["identity_speedup_x"],
            "sum_mps_cost_table_s": json.load(open(os.path.join(KB, "results", "cost_raw", "sum_mps.json")))["latency_s"]["median"],
            "ours_B448_mps_s_cost_table": ours_m["latency_s"]["median"],
            "note": ("the timed wrapper synchronises the GPU before and after every scan call, which inflates "
                     "that pass (0.49 s vs 0.455 s); the identity difference does not depend on it")}
    if cpu:
        out["cpu"] = cpu
    if mps:
        out["mps"] = mps
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    sys.exit(0)

# =====================================================================================================
# measurement mode
# =====================================================================================================
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(a.threads)
sys.path.insert(0, SUM_DIR)
import sum_common as sc

out = {"device": a.device, "threads": a.threads, "torch_num_threads": torch.get_num_threads(),
       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "warmup": a.warmup, "runs": a.runs,
       "torch": torch.__version__, "python": sys.version.split()[0], "executable": sys.executable,
       "machine": platform.platform(), "command": "python -I " + " ".join(sys.argv),
       "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
       "load_at_start": {"getloadavg": list(os.getloadavg()), "uptime": sh("uptime")}}

model, cfg = sc.load_sum(SUM_CODE, SUM_W, "cpu", scan="chunked")
model.eval()
vm = sys.modules["sum_thirdparty_models.vmamba"]
CHUNKED = sc.selective_scan_chunked
assert vm.selective_scan_fn is CHUNKED
out["params_total"] = sc.n_params(model)
out["config"] = cfg

dev = torch.device(a.device)
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 3, 256, 256, generator=g)
cond = torch.zeros(1, 4); cond[0, 2] = 1
if a.device == "mps":
    model.to(dev)
    torch.mps.synchronize()
xd, cd = x.to(dev), cond.to(dev)


def sync():
    if a.device == "mps":
        torch.mps.synchronize()


def fwd():
    with torch.inference_mode():
        y = model(xd, cd)
    sync()
    return y


def stats(v):
    v = np.asarray(v, float)
    q = np.percentile(v, [25, 50, 75])
    return {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]), "mean": float(v.mean()),
            "min": float(v.min()), "max": float(v.max())}


# ---------------------------------------------------------------- per-call scan timer
CALLS = []          # (shape key, seconds) for the current forward


def make_timed(fn):
    def timed(u, delta, A, B, C, *r, **k):
        sync()
        t0 = time.perf_counter()
        y = fn(u, delta, A, B, C, *r, **k)
        sync()
        CALLS.append(((tuple(u.shape), tuple(B.shape)), time.perf_counter() - t0))
        return y
    return timed


def identity_scan(u, delta, A, B, C, *r, **k):
    return u.float()


def bench(name, scan_fn, warmup, runs, collect_calls=False):
    vm.selective_scan_fn = scan_fn
    times, scan_tot, per_shape = [], [], {}
    try:
        for i in range(warmup):
            CALLS.clear(); fwd()
        for i in range(runs):
            CALLS.clear()
            t0 = time.perf_counter(); y = fwd(); times.append(time.perf_counter() - t0)
            if collect_calls:
                scan_tot.append(sum(t for _, t in CALLS))
                agg = {}
                for key, t in CALLS:
                    agg.setdefault(key, []).append(t)
                for key, ts in agg.items():
                    per_shape.setdefault(key, []).append((len(ts), sum(ts)))
    finally:
        vm.selective_scan_fn = CHUNKED
    r = {"latency_s": stats(times), "all": [float(t) for t in times],
         "load_after": list(os.getloadavg())}
    if collect_calls:
        fr = np.asarray(scan_tot) / np.asarray(times)
        r["scan_s_per_forward"] = stats(scan_tot)
        r["scan_s_all"] = [float(t) for t in scan_tot]
        r["scan_fraction_per_run"] = stats(fr)
        r["scan_fraction_ratio_of_medians"] = float(np.median(scan_tot) / np.median(times))
        r["non_scan_s_per_forward"] = stats(np.asarray(times) - np.asarray(scan_tot))
        r["n_scan_calls_per_forward"] = len(CALLS)
        r["per_shape"] = [{"u_shape": list(k[0]), "B_shape": list(k[1]), "calls_per_forward": v[0][0],
                           "L": k[0][2], "seconds_per_forward_median": float(np.median([s for _, s in v])),
                           "seconds_per_call_median": float(np.median([s / n for n, s in v]))}
                          for k, v in sorted(per_shape.items(), key=lambda kv: -kv[0][0][2])]
    print(f"[{time.strftime('%H:%M:%S')}] {name}: median {r['latency_s']['median']:.4f} s"
          + (f", scan {np.median(scan_tot):.4f} s ({r['scan_fraction_ratio_of_medians']:.3f})"
             if collect_calls else ""), flush=True)
    return r, y


# ---------------------------------------------------------------- instrumented chunked scan (CPU)
SEC = {}


def _tick(key, t0):
    t1 = time.perf_counter()
    SEC[key] = SEC.get(key, 0.0) + (t1 - t0)
    return t1


def scan_layout_instr(a_, bb, cumprod_mode="torch"):
    """sum_common._scan_layout, same ops, with timers. cumprod_mode="mulloop" (diagnostic only)
    replaces torch.cumprod over the chunk axis by the same running product written as a Python loop
    of torch.mul (P_t = P_{t-1} * a_t)."""
    t = time.perf_counter()
    T = a_.shape[0]
    x_ = bb[0]
    xs = [x_]
    for s in range(1, T):
        x_ = torch.addcmul(bb[s], a_[s], x_)
        xs.append(x_)
    X = torch.stack(xs, 0)
    del xs
    t = _tick("4a_within_chunk_recurrence_and_stack", t)
    nc = a_.shape[-1]
    if nc > 1:
        if cumprod_mode == "torch":
            P = torch.cumprod(a_.reshape(T, -1), 0).reshape(a_.shape)
        else:
            ps = [a_[0]]
            for s in range(1, T):
                ps.append(ps[-1] * a_[s])
            P = torch.stack(ps, 0)
            del ps
        t = _tick("4b_cumprod_decays", t)
        Pend = P[-1].unbind(-1)
        Xend = X[-1].unbind(-1)
        c = torch.zeros_like(Xend[0])
        cs = [c]
        for k in range(1, nc):
            c = torch.addcmul(Xend[k - 1], Pend[k - 1], c)
            cs.append(c)
        cst = torch.stack(cs, -1).unsqueeze(0)
        t = _tick("4c_cross_chunk_carry", t)
        X.addcmul_(P, cst)
        del P
        t = _tick("4d_carry_fixup", t)
    return X


def chunked_instr(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                  return_last_state=False, cumprod_mode="torch"):
    """sum_common.selective_scan_chunked + _ChunkScan.forward/_states, same ops in the same order."""
    t = time.perf_counter()
    assert not return_last_state
    u = u.float(); delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    if B.dim() == 3:
        B = B.unsqueeze(1); C = C.unsqueeze(1)
    A = A.float(); B = B.float().contiguous(); C = C.float().contiguous()
    t = _tick("1_prep_softplus_casts", t)
    b, Dd, L = u.shape
    K, N = B.shape[1], B.shape[2]
    h = Dd // K
    T = sc._chunk_len(L)
    pad = (-L) % T
    with torch.no_grad():
        u_, dt_, B_, C_ = u, delta, B, C
        if pad:
            u_ = F.pad(u_, (0, pad)); dt_ = F.pad(dt_, (0, pad))
            B_ = F.pad(B_, (0, pad)); C_ = F.pad(C_, (0, pad))
        dtL = sc._to_layout(dt_.reshape(b, K, h, -1), T)
        dtuL = sc._to_layout((dt_ * u_).reshape(b, K, h, -1), T)
        BL = sc._to_layout(B_, T).unsqueeze(3)
        CL = sc._to_layout(C_, T).unsqueeze(3)
        AL = A.reshape(1, 1, K, h, N, 1)
        t = _tick("2_to_chunk_layout", t)
        a_ = torch.exp(dtL.unsqueeze(4) * AL)
        bb = dtuL.unsqueeze(4) * BL
        t = _tick("3_discretize_materialize_dA_dBu_(L,D,N)", t)
        X = scan_layout_instr(a_, bb, cumprod_mode)
        t = time.perf_counter()
        del bb
        y = torch.einsum("tbkhnc,tbknc->tbkhc", X, CL[:, :, :, 0])
        t = _tick("5_readout_einsum_C", t)
        y = sc._from_layout(y).reshape(b, Dd, L + pad)[..., :L]
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    _tick("6_from_layout_and_D_skip", t)
    return y


# ---------------------------------------------------------------- mamba-ssm reference (verbatim)
def load_mamba_ref():
    info = {"source_file": MAMBA_REF_SRC}
    if not os.path.exists(MAMBA_REF_SRC):
        info["available"] = False
        return None, info
    src = open(MAMBA_REF_SRC).read()
    tree = ast.parse(src)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "selective_scan_ref"), None)
    if node is None:
        info["available"] = False
        return None, info
    seg = ast.get_source_segment(src, node)
    from einops import rearrange, repeat
    ns = {"torch": torch, "F": F, "rearrange": rearrange, "repeat": repeat}
    exec(compile(seg, MAMBA_REF_SRC + "::selective_scan_ref", "exec"), ns)
    info.update({"available": True, "function": "selective_scan_ref", "lines": [node.lineno, node.end_lineno],
                 "header_comment": src.splitlines()[14] if len(src.splitlines()) > 14 else "",
                 "package": "transformers 4.48.2 (user site-packages, Python 3.9), copied from mamba-ssm",
                 "note": ("the module imports mamba_ssm/selective_scan_cuda at top level (not installed), so "
                          "only this pure-PyTorch function's source is extracted with ast and executed")})
    return ns["selective_scan_ref"], info


# =====================================================================================================
print(f"[{time.strftime('%H:%M:%S')}] device={a.device} threads={torch.get_num_threads()} "
      f"load={os.getloadavg()}", flush=True)
modes = {}

r, y_chunk = bench("chunked", CHUNKED, a.warmup, a.runs)
modes["chunked"] = r
y_chunk = y_chunk.float().cpu().clone()
out["output_shape"] = list(y_chunk.shape)

r, y_t = bench("chunked_timed", make_timed(CHUNKED), a.warmup, a.runs, collect_calls=True)
modes["chunked_timed"] = r
modes["chunked_timed"]["output_identical_to_chunked"] = bool(torch.equal(y_t.float().cpu(), y_chunk))

if a.device == "cpu":
    # instrumented sections: equality check first, then the timed protocol
    vm.selective_scan_fn = chunked_instr
    with torch.inference_mode():
        y_i = model(xd, cd).float().cpu()
    vm.selective_scan_fn = CHUNKED
    eq = bool(torch.equal(y_i, y_chunk))
    sec_runs = []

    def instr_timed(*args_, **kw):
        return chunked_instr(*args_, **kw)

    vm.selective_scan_fn = instr_timed
    times = []
    for i in range(a.warmup):
        fwd()
    for i in range(a.runs):
        SEC.clear()
        t0 = time.perf_counter(); fwd(); times.append(time.perf_counter() - t0)
        sec_runs.append(dict(SEC))
    vm.selective_scan_fn = CHUNKED
    keys = sorted(sec_runs[0])
    med_tot = float(np.median(times))
    secs = {k: float(np.median([s[k] for s in sec_runs])) for k in keys}
    scan_sum = float(np.median([sum(s.values()) for s in sec_runs]))
    modes["chunked_sections"] = {
        "output_identical_to_chunked": eq, "latency_s": stats(times), "all": [float(t) for t in times],
        "scan_sections_s_per_forward_median": secs,
        "scan_sections_total_s_median": scan_sum,
        "scan_sections_share_of_forward": {k: v / med_tot for k, v in secs.items()},
        "scan_sections_share_of_scan": {k: v / scan_sum for k, v in secs.items()},
        "load_after": list(os.getloadavg())}
    print(f"[{time.strftime('%H:%M:%S')}] chunked_sections: median {med_tot:.4f} s, identical={eq}", flush=True)
    for k in keys:
        print(f"    {k:45s} {secs[k]:.4f} s  ({secs[k] / med_tot:.3f} of forward)", flush=True)

if a.device == "cpu":
    # diagnostic only (not used for any table row): same chunked scan, torch.cumprod over the chunk
    # axis replaced by a loop of torch.mul computing the same running product
    def mulloop(*args_, **kw):
        return chunked_instr(*args_, cumprod_mode="mulloop", **kw)
    r, y_ml = bench("chunked_mulloop", make_timed(mulloop), a.warmup, a.runs, collect_calls=True)
    y_ml = y_ml.float().cpu()
    r["output_identical_to_chunked"] = bool(torch.equal(y_ml, y_chunk))
    r["model_output_vs_chunked_max_abs_diff"] = float((y_ml - y_chunk).abs().max())
    r["note"] = ("DIAGNOSTIC, not a SUM row: identical chunked scan with torch.cumprod(dim=0) replaced by a "
                 "Python loop of torch.mul computing the same running product")
    modes["chunked_mulloop"] = r

r, y_id = bench("identity", identity_scan, a.warmup, a.runs)
modes["identity"] = r
modes["identity"]["note"] = "scan replaced by `return u` (same output shape, zero work); outputs are not SUM's"

if a.device == "cpu":
    # torch.profiler: self time per op inside vs outside the scan
    from torch.profiler import profile, ProfilerActivity, record_function

    def prof_scan(*args_, **kw):
        with record_function("sum_selective_scan"):
            return CHUNKED(*args_, **kw)

    vm.selective_scan_fn = prof_scan
    fwd()
    NPROF = 3
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for i in range(NPROF):
            fwd()
    vm.selective_scan_fn = CHUNKED
    inside, outside = {}, {}
    tot_in = tot_out = 0.0
    for ev in prof.events():
        if ev.name == "sum_selective_scan":
            continue
        p, ins = ev.cpu_parent, False
        while p is not None:
            if p.name == "sum_selective_scan":
                ins = True; break
            p = p.cpu_parent
        st = ev.self_cpu_time_total / 1e6 / NPROF          # seconds per forward
        d = inside if ins else outside
        d[ev.name] = d.get(ev.name, 0.0) + st
        if ins:
            tot_in += st
        else:
            tot_out += st
    top = lambda d: [{"op": k, "self_s_per_forward": v} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:20]]
    modes["profiler"] = {
        "n_profiled_forwards": NPROF,
        "self_time_inside_scan_s_per_forward": tot_in,
        "self_time_outside_scan_s_per_forward": tot_out,
        "scan_fraction_of_self_time": tot_in / (tot_in + tot_out) if tot_in + tot_out > 0 else None,
        "top_ops_inside_scan": top(inside), "top_ops_outside_scan": top(outside),
        "note": ("CPU self time of every profiled op, attributed by whether it ran under "
                 "record_function('sum_selective_scan'); profiler overhead inflates small-op-heavy code")}
    print(f"[{time.strftime('%H:%M:%S')}] profiler: inside {tot_in:.4f} s, outside {tot_out:.4f} s", flush=True)

    if not a.skip_refs:
        ref_fn, info = load_mamba_ref()
        out["mamba_ref_info"] = info
        if ref_fn is not None:
            vm.selective_scan_fn = ref_fn
            with torch.inference_mode():
                y_r = model(xd, cd).float().cpu()
            vm.selective_scan_fn = CHUNKED
            r, _ = bench("ref_mamba", make_timed(ref_fn), a.ref_warmup, a.ref_runs, collect_calls=True)
            r["model_output_vs_chunked"] = {
                "max_abs_diff": float((y_r - y_chunk).abs().max()),
                "max_rel_diff": float((y_r - y_chunk).abs().max() / y_chunk.abs().max()),
                "output_abs_max": float(y_chunk.abs().max())}
            r["warmup"], r["runs"] = a.ref_warmup, a.ref_runs
            r["note"] = "timed wrapper (perf_counter around each scan call, negligible overhead on CPU)"
            modes["ref_mamba"] = r
        else:
            modes["ref_mamba"] = {"available": False}
        vm.selective_scan_fn = sc.selective_scan_seq_ref
        with torch.inference_mode():
            y_p = model(xd, cd).float().cpu()
        vm.selective_scan_fn = CHUNKED
        r, _ = bench("ref_project", make_timed(sc.selective_scan_seq_ref), a.ref_warmup, a.ref_runs,
                     collect_calls=True)
        r["model_output_vs_chunked"] = {"max_abs_diff": float((y_p - y_chunk).abs().max()),
                                        "max_rel_diff": float((y_p - y_chunk).abs().max() / y_chunk.abs().max())}
        r["warmup"], r["runs"] = a.ref_warmup, a.ref_runs
        modes["ref_project"] = r

r, y_e = bench("chunked_end", CHUNKED, a.warmup, a.runs)
modes["chunked_end"] = r
modes["chunked_end"]["output_identical_to_chunked"] = bool(torch.equal(y_e.float().cpu(), y_chunk))
out["modes"] = modes

# ---------------------------------------------------------------- derived summary
base = modes["chunked"]["latency_s"]["median"]
ct = modes["chunked_timed"]
idm = modes["identity"]["latency_s"]["median"]
summ = {
    "forward_chunked_s_median": base,
    "forward_chunked_end_s_median": modes["chunked_end"]["latency_s"]["median"],
    "scan_s_per_forward_median": ct["scan_s_per_forward"]["median"],
    "non_scan_s_per_forward_median": ct["non_scan_s_per_forward"]["median"],
    "scan_fraction_timed_wrapper": ct["scan_fraction_ratio_of_medians"],
    "forward_identity_scan_s_median": idm,
    "identity_saving_fraction": (base - idm) / base,
    "identity_speedup_x": base / idm,
}
if "profiler" in modes:
    summ["scan_fraction_profiler_self_time"] = modes["profiler"]["scan_fraction_of_self_time"]
for k in ("chunked_mulloop", "ref_mamba", "ref_project"):
    if k in modes and "latency_s" in modes[k]:
        summ[f"forward_{k}_s_median"] = modes[k]["latency_s"]["median"]
        summ[f"scan_fraction_{k}"] = modes[k]["scan_fraction_ratio_of_medians"]
out["summary"] = summ
out["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
out["load_at_end"] = {"getloadavg": list(os.getloadavg()), "uptime": sh("uptime")}
out["ru_maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(summ, indent=1))
