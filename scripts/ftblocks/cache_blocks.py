"""
Cache the frozen output of DINOv2-base blocks 8 and 10 (all 1025 tokens: CLS + 32x32 patches; the
checkpoint has no registers) for every ADD1000 image, computed exactly as scripts/ours/cache_features.py
computes the readout features (CPU, batch 1, 6 threads, PIL RGB bilinear 448, ImageNet mean/std,
interpolate_pos_encoding=True), by reading HF hidden_states[8] and hidden_states[10] of the same forward.
No labels are used.

Why both depths: N = 4 trains blocks 9-12 and starts from block 8; N = 2 trains blocks 11-12 and starts
from block 10. Caching block 10 as well (3.15 GB) means an N = 2 step never pays for a frozen forward of
blocks 9-10, so the N = 2 step time is the true fine-tuning cost; FTModel can still apply frozen blocks
on the fly when only a shallower cache exists (used for Pitt if both N are selected).

Verification (every image, on the fly): the remaining frozen blocks d+1..12 + final LayerNorm applied
to the cached block-d output must reproduce last_hidden_state of the same forward (expected bit-for-bit
on the CPU), and the fp16 cast of its patch tokens must equal the existing cache
cache/ours/feats_dinov2b_448_fp16.npy. After writing, a few images are re-read from disk and
re-checked (CPU, and optionally on MPS to report the device deviation; MPS only under the shared lock).

  python -I cache_blocks.py --set add                      # ADD1000, depths 8,10, fp32 (6.3 GB)
  python -I cache_blocks.py --set add --verify-only --mps   # re-check from disk (+ MPS deviation)
  python -I cache_blocks.py --set pitt --depths 8 --dtype fp16   # ONLY if the optional row triggers
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
import ft_common as FC  # noqa: E402
import ours_lib as L    # noqa: E402


def items(which):
    if which == "add":
        ids = L.load_ids()
        return [os.path.join(L.FLAT, "images", f"{i}.jpg") for i in ids], ids, os.path.join(L.CACHE, "feats_dinov2b_448_fp16.npy"), None
    sub = json.load(open(FC.PITT_SUBSET))
    pp = sub["pool_positions"]
    files = [sub["subset_files"][k] for k in pp]
    return [os.path.join(FC.PITT_DIR, f) for f in files], files, FC.PITT_FINAL16, np.array(pp)


def out_path(which, d, dtype):
    p = (FC.ADD_CACHE if which == "add" else FC.PITT_CACHE)[d]
    want = "fp32" if dtype == "fp32" else "fp16"
    assert p.endswith(f"_{want}.npy"), (p, want)
    return p


@torch.no_grad()
def replay(bb, h, d):
    for layer in bb.encoder.layer[d:]:
        h = layer(h)
    return bb.layernorm(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["add", "pitt"], default="add")
    ap.add_argument("--depths", default="8,10")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default=None, help="default fp32 for add, fp16 for pitt")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--n-verify", type=int, default=6)
    ap.add_argument("--mps", action="store_true", help="also report the MPS deviation of the replay (takes the shared lock)")
    a = ap.parse_args()
    dtype = a.dtype or ("fp32" if a.set == "add" else "fp16")
    depths = [int(x) for x in a.depths.split(",")]
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    paths, names, ref16_path, sub_index = items(a.set)
    n = len(paths)
    bb = FC.load_backbone()
    ref16 = np.load(ref16_path, mmap_mode="r")
    os.makedirs(FC.CFT, exist_ok=True)
    meta_path = os.path.join(FC.CFT, f"{a.set}_blocks_meta.json")
    np_dt = np.float32 if dtype == "fp32" else np.float16
    t0 = time.time()
    if not a.verify_only:
        outs = {}
        for d in depths:
            p = out_path(a.set, d, dtype)
            if os.path.exists(p):
                raise SystemExit(f"{p} exists; refusing to overwrite (move it to ~/.Trash first if a rebuild is intended)")
            outs[d] = np.lib.format.open_memmap(p + ".partial.npy", mode="w+", dtype=np_dt, shape=(n, FC.NTOK, FC.C))
        stats = {d: {"replay_max_abs_vs_last_hidden": 0.0, "fp16_mismatch_elems_vs_existing_cache": 0,
                     "fp16_max_abs_vs_existing_cache": 0.0} for d in depths}
        for j, pth in enumerate(paths):
            x = FC.preprocess(pth)
            out = bb(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True)
            hs = out.hidden_states
            assert len(hs) == 13
            last = out.last_hidden_state
            ref = np.asarray(ref16[j if sub_index is None else sub_index[j]])              # (768,32,32) fp16
            for d in depths:
                h = hs[d]
                outs[d][j] = h[0].numpy().astype(np_dt)
                hin = h if dtype == "fp32" else h.half().float()       # replay from exactly what is stored
                rec = replay(bb, hin, d)
                st = stats[d]
                st["replay_max_abs_vs_last_hidden"] = max(st["replay_max_abs_vs_last_hidden"], float((rec - last).abs().max()))
                tok16 = rec[:, 1:, :].transpose(1, 2).reshape(1, FC.C, 32, 32)[0].half().numpy()
                mis = tok16 != ref
                st["fp16_mismatch_elems_vs_existing_cache"] += int(mis.sum())
                if mis.any():
                    st["fp16_max_abs_vs_existing_cache"] = max(st["fp16_max_abs_vs_existing_cache"],
                                                               float(np.abs(tok16.astype(np.float32) - ref.astype(np.float32)).max()))
            if (j + 1) % 100 == 0 or j == n - 1:
                print(f"  {j + 1}/{n}  {time.time() - t0:.0f}s  " + "  ".join(
                    f"b{d}: replay max {stats[d]['replay_max_abs_vs_last_hidden']:.2e}, fp16 mismatches {stats[d]['fp16_mismatch_elems_vs_existing_cache']}"
                    for d in depths), flush=True)
        for d in depths:
            outs[d].flush(); del outs[d]
            p = out_path(a.set, d, dtype)
            os.replace(p + ".partial.npy", p)
        meta = {"set": a.set, "n": n, "names": names, "depths": depths, "dtype": dtype, "shape": [n, FC.NTOK, FC.C],
                "token_layout": "token 0 = CLS, tokens 1..1024 = 32x32 patches row-major (no registers in dinov2-base)",
                "depth_meaning": "output of block d (1-indexed) = HF Dinov2Model hidden_states[d], before the final LayerNorm",
                "preprocessing": "PIL RGB, bilinear 448x448, /255, ImageNet mean/std, interpolate_pos_encoding=True (= scripts/ours/cache_features.py)",
                "backbone": FC.BACKBONE, "attn_implementation": bb.config._attn_implementation,
                "device": "cpu", "threads": a.threads, "batch": 1,
                "verification_all_images": {str(d): v for d, v in stats.items()},
                "verification_note": "replay = frozen blocks d+1..12 + final LayerNorm applied on the CPU to the stored block-d "
                                     "output; compared with last_hidden_state of the same forward and (fp16 cast of the "
                                     "patch tokens) with the existing readout feature cache",
                "wall_s": time.time() - t0, "torch": torch.__version__, "machine": platform.platform(),
                "command": " ".join(["python", "-I"] + sys.argv)}
        FC.jdump(meta, meta_path)
        print("written", [out_path(a.set, d, dtype) for d in depths], f"{time.time() - t0:.0f}s", flush=True)
    # ---- re-read from disk, re-check a few images (CPU; optionally MPS)
    rng = np.random.default_rng(7)
    sel = sorted(rng.choice(n, size=min(a.n_verify, n), replace=False).tolist())
    rep = {"images": [names[j] for j in sel]}
    lock = None
    if a.mps:
        lock = FC.MPSLock(); lock.acquire()
        bb_mps = FC.load_backbone().to("mps")
    try:
        for d in depths:
            mm = np.load(out_path(a.set, d, dtype), mmap_mode="r")
            r = {"cpu_replay_max_abs_vs_full_forward": 0.0, "cpu_fp16_mismatch_elems": 0, "mps_replay_max_abs_vs_cpu": 0.0,
                 "mps_fp16_mismatch_elems": 0}
            for j in sel:
                h = torch.from_numpy(np.array(mm[j]))[None].float()
                rec = replay(bb, h, d)
                full = bb(pixel_values=FC.preprocess(paths[j]), interpolate_pos_encoding=True).last_hidden_state
                if dtype == "fp32":
                    r["cpu_replay_max_abs_vs_full_forward"] = max(r["cpu_replay_max_abs_vs_full_forward"], float((rec - full).abs().max()))
                ref = np.asarray(ref16[j if sub_index is None else sub_index[j]])
                tok16 = rec[:, 1:, :].transpose(1, 2).reshape(1, FC.C, 32, 32)[0].half().numpy()
                r["cpu_fp16_mismatch_elems"] += int((tok16 != ref).sum())
                if a.mps:
                    with torch.no_grad():
                        rm = replay(bb_mps, h.to("mps"), d).cpu()
                    r["mps_replay_max_abs_vs_cpu"] = max(r["mps_replay_max_abs_vs_cpu"], float((rm - rec).abs().max()))
                    tm = rm[:, 1:, :].transpose(1, 2).reshape(1, FC.C, 32, 32)[0].half().numpy()
                    r["mps_fp16_mismatch_elems"] += int((tm != ref).sum())
                    r["mps_fp16_max_abs_vs_existing_cache"] = max(r.get("mps_fp16_max_abs_vs_existing_cache", 0.0),
                                                                   float(np.abs(tm.astype(np.float32) - ref.astype(np.float32)).max()))
            r["elems_checked_fp16"] = len(sel) * FC.C * 1024
            rep[str(d)] = r
            print(f"[reread b{d}] {r}", flush=True)
    finally:
        if lock:
            lock.release()
    meta = json.load(open(meta_path))
    meta["reread_check"] = rep
    meta["reread_check_time"] = time.ctime()
    FC.jdump(meta, meta_path)
    print("verify done", flush=True)


if __name__ == "__main__":
    main()
