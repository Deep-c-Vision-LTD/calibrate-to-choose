"""
Track B, step 2a: per-fold MSDB teachers.

The msdb track did not save its adapted heads, so teacher k is retrained here with exactly the
msdb_adapt recipe (scripts/msdb/train_msdb.py, imported unchanged): mode head (26,463 params:
saliency network + finalizer sigma / centre-bias weight / log priority), released init,
KL + (1 - CC) at 64x64, 60 epochs, batch 16, AdamW lr 3e-4 wd 1e-4, data order default_rng(0)
(= seed 0), centre prior = log mean training density of fold k's 800 training maps, post-hoc
Gaussian sigma chosen at 384 on 200 training images drawn with default_rng(k). Only fold k's 800
training maps are used. Cached ADD1000 scale-mix features (cache/msdb/mix_fp16.npy).

Per fold it saves cache/distill/teachers/fold{k}.pt (state_dict, post-hoc sigma, 256x256 prior),
the teacher's 256x256 log densities for all 1000 ADD1000 images (training images are used as
targets of distill_gt800_teacher800; the 200 test images are only used for the reproduction check
against msdb_adapt seed 0) and a reproduction check.

Several instances may run (e.g. one MPS, one CPU): folds are claimed with mkdir.
  python -I train_teachers.py --dev mps
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, types
import numpy as np, torch
KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.join(KB, "scripts", "msdb"))
import train_msdb as TM                           # noqa: E402  (unchanged msdb track code)
from msdb_common import (load_split, SIGMA_SELECT_N, EVAL, S, logd256_to_prob, metrics,  # noqa: E402
                         write_json)
from scipy.ndimage import gaussian_filter          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_common as DC                        # noqa: E402

TD = os.path.join(KB, "cache", "distill", "teachers")


@torch.no_grad()
def predict_logd(model, data, pos_list, cbd_dev, dev, bs=16):
    out = np.zeros((len(pos_list), 256, 256), np.float32)
    for j in range(0, len(pos_list), bs):
        idx = pos_list[j:j + bs]
        b = TM.to_dev(data.batch(idx), dev)
        out[j:j + len(idx)] = TM.forward(model, "head", b, cbd_dev).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--yield-s", type=float, default=30.0, help="pause after releasing the MPS lock between folds")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(TD, exist_ok=True)
    hp = types.SimpleNamespace(lr=3e-4, wd=1e-4, epochs=60, bs=16, loss="klcc")
    torch.manual_seed(0)
    data = TM.Data("head", "cpu")             # batches are moved to the device in TM.to_dev
    ids, folds, _ = load_split()
    ref = {}
    for r in json.load(open(os.path.join(KB, "results", "msdb_adapt_per_image.json"))):
        if r["seed"] == 0:
            ref[r["id"]] = r
    plock = DC.PoliteLock() if a.dev == "mps" else None
    for k in [int(x) for x in a.folds.split(",")]:
        out_pt = os.path.join(TD, f"fold{k}.pt")
        claim = os.path.join(TD, f".claim_fold{k}")
        if os.path.exists(out_pt):
            continue
        try:
            os.mkdir(claim)
        except FileExistsError:
            print(f"fold {k} claimed by another instance", flush=True); continue
        lock = plock
        if lock:
            lock.acquire()
        t0 = time.time()
        te = folds[k]; tr = [i for j in range(len(folds)) if j != k for i in folds[j]]
        te_pos = [data.pos[i] for i in te]; tr_pos = [data.pos[i] for i in tr]
        cbd = data.prior(tr)                                       # fold-k training maps only
        rng = np.random.default_rng(k)
        sig_pos = rng.choice(np.array(tr_pos), size=min(SIGMA_SELECT_N, len(tr_pos)), replace=False)
        print(f"[teacher] fold {k} on {a.dev}: train {len(tr)} test {len(te)}", flush=True)
        model, sigma, lg = TM.train_one("head", data, tr_pos, cbd, a.dev, hp, 0, sig_pos)
        cbd_dev = cbd.to(a.dev)
        logd = predict_logd(model, data, list(range(1000)), cbd_dev, a.dev)
        if lock:
            lock.release()
        np.save(os.path.join(TD, f"teacher_add1000_logd256_fold{k}.npy"), logd)
        # reproduction check on fold k's test images against msdb_adapt seed 0 (same recipe)
        rows = []
        for p in te_pos:
            p384 = logd256_to_prob(torch.from_numpy(logd[p]), EVAL)[0]
            if sigma > 0:
                p384 = gaussian_filter(p384, sigma)
            m = metrics(p384, data.g384[p].astype(np.float64))
            rows.append({"id": ids[p], **m})
        cc_new = np.array([r["CC"] for r in rows]); cc_ref = np.array([ref[r["id"]]["CC"] for r in rows])
        chk = {"fold": k, "posthoc_sigma": sigma, "msdb_adapt_s0_posthoc_sigma": 6,
               "CC": float(cc_new.mean()), "SIM": float(np.mean([r["SIM"] for r in rows])), "KLD": float(np.mean([r["KLD"] for r in rows])),
               "msdb_adapt_s0_CC": float(cc_ref.mean()), "msdb_adapt_s0_SIM": float(np.mean([ref[r["id"]]["SIM"] for r in rows])),
               "msdb_adapt_s0_KLD": float(np.mean([ref[r["id"]]["KLD"] for r in rows])),
               "per_image_CC_max_abs_diff": float(np.abs(cc_new - cc_ref).max()),
               "per_image_CC_corr": float(np.corrcoef(cc_new, cc_ref)[0, 1])}
        sd = {kk: v.detach().cpu() for kk, v in model.state_dict().items()}
        torch.save({"state_dict": sd, "posthoc_sigma": sigma, "cbd256": cbd.cpu(), "fold": k,
                    "train_ids": tr, "sigma_select_ids": [ids[p] for p in sig_pos], "learned": lg["learned"],
                    "epoch_loss": lg["epoch_loss"], "sigma_table": lg["sigma_table"], "train_s": lg["train_s"],
                    "device": a.dev, "check": chk, "recipe": vars(hp)}, out_pt + ".tmp")
        os.replace(out_pt + ".tmp", out_pt)
        print(f"[teacher] fold {k}: sigma {sigma} test CC {chk['CC']:.4f} (msdb_adapt s0 {chk['msdb_adapt_s0_CC']:.4f}) "
              f"SIM {chk['SIM']:.4f} KLD {chk['KLD']:.4f} max|dCC| {chk['per_image_CC_max_abs_diff']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        if a.dev == "mps":
            torch.mps.empty_cache()
            time.sleep(a.yield_s)                     # let other tracks take the MPS lock between folds
    if all(os.path.exists(os.path.join(TD, f"fold{k}.pt")) for k in range(5)) and not os.path.exists(os.path.join(TD, "ALL_DONE.json")):
        chks = [torch.load(os.path.join(TD, f"fold{k}.pt"), map_location="cpu", weights_only=False)["check"] for k in range(5)]
        write_json({"checks": chks, "mean_CC": float(np.mean([c["CC"] for c in chks])),
                    "msdb_adapt_s0_mean_CC": float(np.mean([c["msdb_adapt_s0_CC"] for c in chks]))},
                   os.path.join(TD, "ALL_DONE.json"))
        print("[teacher] all 5 folds done", flush=True)


if __name__ == "__main__":
    main()
