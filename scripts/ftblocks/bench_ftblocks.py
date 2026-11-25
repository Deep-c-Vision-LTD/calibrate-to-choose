"""
Timing bench for the ftblocks plan (under the shared MPS lock, released at the end): seconds per optimiser
step at batch 16 on MPS for N = 0 (base readout on the fp16 cache), N = 2 (blocks 11-12 from the block-10
cache) and N = 4 (blocks 9-12 from the block-8 cache), fp32, with real batches (memmap gather included),
plus the evaluation forward per image; optional bf16-autocast timings for information only (the plan
runs fp32). Writes cache/ftblocks/bench.json.

  python -I bench_ftblocks.py [--steps 30] [--bf16]
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
import ft_common as FC  # noqa: E402
import ours_lib as L    # noqa: E402


def sync():
    torch.mps.synchronize()


def bench_cfg(data, bb, n_ft, steps, warm, dev, amp=None):
    torch.manual_seed(0)
    head = L.ReadoutHead("full")
    if n_ft == 0:
        model, view = head.to(dev), data.view("final16")
    else:
        src = FC.src_depth_for(n_ft)
        model, view = FC.FTModel(head, bb, n_ft, src).to(dev), data.view(src)
    opt, sched, _ = FC.make_optimizer(model, n_ft, 3e-5, 2400)
    rng = np.random.default_rng(0)
    tr = rng.permutation(1000)[:640]
    model.train()
    t_gather, t_step = [], []
    for s in range(warm + steps):
        ids = tr[(s * 16) % 640:(s * 16) % 640 + 16]
        sync(); t0 = time.time()
        x, g = view.batch(ids)
        sync(); t1 = time.time()
        if amp:
            with torch.autocast(device_type="mps", dtype=amp):
                out = model(x)
            loss = FC.kl_cc(out.float(), g)
        else:
            loss = FC.kl_cc(model(x), g)
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None:
            sched.step()
        float(loss.detach()); sync(); t2 = time.time()
        if s >= warm:
            t_gather.append(t1 - t0); t_step.append(t2 - t0)
    # evaluation forward (as raw_preds: eval mode, batches of 50, sigmoid + upsample to 384)
    sync(); t0 = time.time()
    _ = L.raw_preds(model, view, list(range(200)))
    sync(); t_eval = (time.time() - t0) / 200
    mem = torch.mps.driver_allocated_memory() / 2 ** 30
    del model, opt; torch.mps.empty_cache()
    return {"N": n_ft, "amp": str(amp) if amp else "fp32", "s_per_step_mean": float(np.mean(t_step)),
            "s_per_step_median": float(np.median(t_step)), "s_per_step_p90": float(np.percentile(t_step, 90)),
            "gather_s_mean": float(np.mean(t_gather)), "eval_s_per_image": t_eval, "driver_alloc_gib": mem,
            "steps_timed": steps, "warmup_steps": warm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(6)
    dev = "mps"
    lock = FC.MPSLock()
    data = FC.FTData(dev)
    bb = FC.load_backbone()
    lock.acquire()
    t0 = time.time()
    try:
        load = os.getloadavg()
        res = []
        for n in (0, 2, 4):
            r = bench_cfg(data, bb, n, a.steps, a.warm, dev)
            print(r, flush=True); res.append(r)
        if a.bf16:          # information only; the plan runs fp32
            for n in (2, 4):
                try:
                    r = bench_cfg(data, bb, n, a.steps, a.warm, dev, amp=torch.bfloat16)
                except Exception as e:          # noqa
                    r = {"N": n, "amp": "torch.bfloat16", "error": f"{type(e).__name__}: {e}"[:300]}
                    torch.mps.empty_cache()
                print(r, flush=True); res.append(r)
    finally:
        lock.release()
    out = {"results": res, "loadavg_at_start": load, "wall_s": time.time() - t0, "torch": torch.__version__,
           "machine": platform.platform(), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "note": "N=0 is timed with the ftblocks loop (same math as ours_lib.train); s/step includes the batch gather"}
    p = os.path.join(FC.CFT, "bench.json")
    if os.path.exists(p):
        p = os.path.join(FC.CFT, f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
    FC.jdump(out, p)
    print("written", p)


if __name__ == "__main__":
    main()
