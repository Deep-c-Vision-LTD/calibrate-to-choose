"""
Parameter counts and measured costs for the DeepGaze MSDB rows (results/msdb_costs.json).

Counts come from the instantiated released model (CPU, no download: all weights are already in
the local torch-hub / CLIP caches). Timings are the ones measured during this track's own runs
(validation pass, feature-cache pass, head training) and are read back from their logs/JSON.

  python -I cost_msdb.py
"""
import json, os, re, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import RESULTS, LOGS, CACHE, write_json

torch.set_num_threads(2)


def n_par(m):
    return int(sum(p.numel() for p in m.parameters()))


def main():
    t0 = time.time()
    from msdb_backbone import load_msdb
    model = load_msdb("cpu", cast_fp32=False)
    fb = model.features
    out = {"model": "DeepGazeMSDB (deepgaze_pytorch, pretrained=True)"}
    out["total_params_instantiated"] = n_par(model)
    out["saliency_network_params"] = n_par(model.saliency_network)
    out["finalizer_params_all_datasets"] = n_par(model.finalizer)
    # backbone pieces
    mods = dict(model.named_modules())
    clip_vis = None; dino = None
    for name, m in mods.items():
        cls = type(m).__name__
        if cls == "ModifiedResNet" and clip_vis is None:
            clip_vis = (name, m)
        if cls == "DinoVisionTransformer" and dino is None:
            dino = (name, m)
    if clip_vis:
        v = clip_vis[1]
        unused = 0
        try:
            blk = v.layer4[2]
            unused = sum(n_par(getattr(blk, a)) for a in ["bn2", "conv3", "bn3"] if hasattr(blk, a))
        except Exception:
            pass
        out["clip_rn50x64_visual_params"] = n_par(v)
        out["clip_rn50x64_visual_params_executed_to_layer4.2.conv2"] = n_par(v) - unused
    if dino:
        d = dino[1]
        unused = n_par(d.blocks[11]) + n_par(d.norm) + (n_par(d.head) if hasattr(d, "head") else 0)
        out["dinov2_vitb14_params"] = n_par(d)
        out["dinov2_vitb14_params_executed_to_block10"] = n_par(d) - unused
    out["features_module_params_incl_scale_weights"] = n_par(fb)
    out["trainable_params_msdb_adapt_head"] = 26463
    out["trainable_params_msdb_adapt_dsp"] = 13
    out["scales"] = {"pixel_per_dva_scales": len(fb.resolutions_pixel_per_dva),
                     "size_scales": len(fb.resolutions_size),
                     "resolutions_pixel_per_dva": [float(x) for x in fb.resolutions_pixel_per_dva],
                     "resolutions_size": [int(x) for x in fb.resolutions_size]}
    # measured timings from this track's own runs
    t = {}
    vj = os.path.join(CACHE, "validation_10img.json")
    if os.path.exists(vj):
        v = json.load(open(vj))
        t["full_model_cpu_float64_buffers_s_per_img_512"] = v.get("cpu_f64buf_s_per_img")
        t["full_model_mps_float32_s_per_img_512"] = v.get("mps_s_per_img")
        t["timing_note_full_model"] = ("10 images, batch 1, 512x512 input, this Mac while other jobs shared the CPU; "
                                       "includes the 10-scale multi-scale backbone and the head")
    cl = os.path.join(LOGS, "msdb_cache_mps.log")
    if os.path.exists(cl):
        m = re.search(r"worker done: (\d+) images in (\d+)s", open(cl).read())
        if m:
            t["feature_cache_pass_mps_total_s"] = int(m.group(2))
            t["feature_cache_pass_mps_s_per_img"] = int(m.group(2)) / int(m.group(1))
    for row in ["msdb_adapt", "msdb_adapt_dsp", "msdb_adapt_kl", "msdb_adapt_grouped", "msdb_adapt_pairsafe"]:
        p = os.path.join(RESULTS, f"{row}_summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            fl = [f for f in s.get("fold_logs", []) if f.get("seed", 0) == 0]
            if fl:
                t[f"{row}_train_s_per_fold_mean"] = sum(f["train_s"] for f in fl) / len(fl)
                t[f"{row}_device"] = s.get("device")
    t["train_note"] = ("head training = 60 epochs over the fold's 800 training images from cached frozen features "
                       "(batch 16, AdamW); excludes the one-off feature-cache pass")
    out["measured"] = t
    out["wall_clock_s"] = time.time() - t0
    out["command"] = "cd scripts/msdb && python -I cost_msdb.py"
    write_json(out, os.path.join(RESULTS, "msdb_costs.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
