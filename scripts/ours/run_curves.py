"""
Matched-budget learning curves for the "ours" track (sample_efficiency.py v3 protocol), all
(n, r) runs of one initialisation in one process (features loaded once).

Fixed test set = the 200 ids of fold 0 (idx[:200], idx = default_rng(0).permutation(1000));
pool = the other 800. For each n in {25,50,100,200,400,800} and r in {0,1,2}:
training ids = default_rng(1000 n + r).choice(pool, n, replace=False); data-order rng =
default_rng(r); batch = min(16, n); 60 epochs; Gaussian finalizer sigma chosen on all n
training images. Init seed torch.manual_seed(1000 n + r). Writes
results/curve_ours_<init>_n<n>_r<r>_{per_image,summary}.json and an aggregate
results/curve_ours_<init>_aggregate.json.

  python -I run_curves.py --init scratch|warm
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ours_lib as L  # noqa: E402

SIZES = [25, 50, 100, 200, 400, 800]
R = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["scratch", "warm"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES, help="subset of n to run in this process (others may run elsewhere)")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if a.aggregate_only:
        write_aggregate(a.init); return
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.set_num_threads(6)
    t0 = time.time()
    data = L.Data(dev)
    F = json.load(open(os.path.join(L.KB, "splits", "kfold_seed0.json")))
    idx = np.random.default_rng(0).permutation(1000)
    te, pool = idx[:200], idx[200:]
    assert [data.ids[j] for j in te] == F["folds"][0]
    for n in a.sizes:
        for r in range(R):
            row = f"curve_ours_{a.init}_n{n}_r{r}"
            out_sum = os.path.join(L.RESULTS, f"{row}_summary.json")
            if os.path.exists(out_sum):
                continue
            tk = time.time()
            tr = np.random.default_rng(1000 * n + r).choice(pool, size=n, replace=False)
            torch.manual_seed(1000 * n + r)
            head = L.ReadoutHead("full")
            winfo = {}
            if a.init == "warm":
                res = head.load_state_dict(L.warm_state(), strict=False)
                winfo = {"missing_keys": list(res.missing_keys)}
            head = head.to(dev)
            L.train(head, data, tr, np.random.default_rng(r), epochs=a.epochs, bs=min(16, n))
            preds = [p.astype(np.float64) for p in L.raw_preds(head, data, tr)]
            sig = L.choose_sigma(preds, [data.gte[i] for i in tr])
            rows = L.score(head, data, te, sig)
            for x in rows:
                x.update({"fold": 0, "seed": r, "n": n, "r": r})
            json.dump(rows, open(os.path.join(L.RESULTS, f"{row}_per_image.json"), "w"))
            st = L.fold_stats(rows, 1)
            summ = {"row": row, "kind": "curve", "init": a.init, "n": n, "r": r, **st,
                    "sigma": sig, "train_ids": [data.ids[j] for j in tr], "test_set": "fold 0 of splits/kfold_seed0.json (200 ids)",
                    "calibration_and_training_data": {"sigma_selection_maps": n, "sigma_selection_maps_are_subset_of_training_maps": True,
                                                      "train_weight_maps": n,
                                                      "centre_prior": "learned 16x16 centre-bias logit map, trained jointly on the same n maps",
                                                      "warm_start_source": ("readout initialised from a UEyes-trained checkpoint (0 ADD1000 maps)" if a.init == "warm" else "random init"),
                                                      "warm_load": winfo},
                    "protocol": {"epochs": a.epochs, "batch": min(16, n), "lr": 3e-4, "optimizer": "AdamW wd 1e-4", "loss": "KL + (1 - CC) at 64x64",
                                 "subset_rule": "default_rng(1000*n + r).choice(pool of 800, n, replace=False) (sample_efficiency.py)",
                                 "init_seed_rule": "torch.manual_seed(1000*n + r); data order default_rng(r)",
                                 "finalizer": "Gaussian sigma from {0,2,...,12} by mean CC on all n training images",
                                 "scoring": "unified: 384x384, GT bilinear to 384, clip at 0 and divide by sum"},
                    "trainable_params": L.n_params(head), "wall_clock_s": time.time() - tk, "device": dev, "torch": torch.__version__,
                    "machine": platform.platform(), "command": " ".join([os.path.basename(sys.executable), "-I"] + sys.argv)}
            json.dump(summ, open(out_sum, "w"), indent=1)
            print(f"[{row}] sigma={sig} CC={st['mean_CC']:.4f} SIM={st['mean_SIM']:.4f} KLD={st['mean_KLD']:.4f} ({time.time() - tk:.0f}s)", flush=True)
    write_aggregate(a.init)


def write_aggregate(init):
    """Aggregate every finished (n, r) run of this init found on disk (runs may come from different processes/devices)."""
    agg = []
    for n in SIZES:
        for r in range(R):
            f = os.path.join(L.RESULTS, f"curve_ours_{init}_n{n}_r{r}_summary.json")
            if os.path.exists(f):
                s = json.load(open(f))
                agg.append({"n": n, "r": r, "CC": s["mean_CC"], "SIM": s["mean_SIM"], "KLD": s["mean_KLD"], "sigma": s.get("sigma"),
                            "device": s.get("device"), "wall_clock_s": s.get("wall_clock_s")})
    by_n = []
    for n in SIZES:
        v = [x for x in agg if x["n"] == n]
        if not v:
            continue
        by_n.append({"n": n, "n_resamples_done": len(v), **{f"mean_{m}": float(np.mean([x[m] for x in v])) for m in ("CC", "SIM", "KLD")},
                     **{f"sd_{m}_ddof0": float(np.std([x[m] for x in v])) for m in ("CC", "SIM", "KLD")}, "runs": v})
    json.dump({"init": init, "complete": len(agg) == len(SIZES) * R, "by_n": by_n,
               "test_set": "fold 0 of splits/kfold_seed0.json (200 ids)",
               "command": " ".join([os.path.basename(sys.executable), "-I"] + sys.argv)},
              open(os.path.join(L.RESULTS, f"curve_ours_{init}_aggregate.json"), "w"), indent=1)
    print("curve aggregate:", [(b["n"], b["n_resamples_done"], round(b["mean_CC"], 4)) for b in by_n], flush=True)

if __name__ == "__main__":
    main()
