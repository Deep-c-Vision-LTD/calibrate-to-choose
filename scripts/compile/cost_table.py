"""
Build results/cost_table.json from the measured per-model benchmarks (results/cost_raw/*.json, written by
cost_bench.py via run_cost.sh) and the training / adaptation wall-clock recorded in each adapted row's summary.

  python -I cost_table.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, time
import numpy as np

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
RAW = os.path.join(RES, "cost_raw")
T0 = time.time()


def j(p):
    return json.load(open(p))


def rj(name):
    p = os.path.join(RES, name)
    return j(p) if os.path.exists(p) else None


MODELS = {
    "ours": {"name": "Ours: frozen DINOv2-base (448) + ASPP readout", "input": "448x448"},
    "iie": {"name": "DeepGaze IIE (released, 4 backbones x 30 readout components)", "input": "384x384"},
    "msdb": {"name": "DeepGaze MSDB (released, CLIP-RN50x64 + DINOv2-B/14, 10 scales)",
             "input": "512x512, pixel_per_dva 21.7, dataset=None"},
    "sum": {"name": "SUM (released, VMamba U-Net, token 2)", "input": "256x256"},
}
msdb_counts = rj("msdb_costs.json") or {}

inference = {}
for m, meta in MODELS.items():
    o = dict(meta)
    for dev in ("cpu", "mps"):
        p = os.path.join(RAW, f"{m}_{dev}.json")
        if not os.path.exists(p):
            o[f"{dev}_status"] = "not measured (file missing)"
            continue
        d = j(p)
        L = d["latency_s"]
        o[f"{dev}_latency_median_s"] = L["median"]
        o[f"{dev}_latency_iqr_s"] = [L["q25"], L["q75"]]
        o[f"{dev}_latency_min_s"] = L["min"]
        o[f"{dev}_runs"] = d["runs"]; o[f"{dev}_warmup"] = d["warmup"]
        if dev == "cpu":
            o["cpu_threads"] = d["threads"]
            o["params_total"] = d["params_total"]
            o["params_backbone_frozen"] = d["params_backbone_frozen"]
            o["params_trainable_in_our_adaptation_row"] = d["params_trainable_in_adaptation"]
            if "params_trainable_in_adaptation_note" in d:
                o["params_trainable_note"] = d["params_trainable_in_adaptation_note"]
            o["GFLOPs"] = d.get("GFLOPs")
            o["GFLOPs_by_op"] = d.get("flops_by_op_global")
            o["cpu_peak_rss_GB"] = d["peak_rss_during_inference_GB"]
            o["cpu_peak_rss_increment_over_loaded_model_GB"] = d["peak_rss_increment_over_loaded_GB"]
            o["cpu_rss_loaded_model_GB"] = d["rss_after_load_GB"]
            if m == "ours":
                o["finalizer_sigma4_scipy_cpu_s"] = d.get("finalizer_sigma4_scipy_cpu_s_median")
                o["cpu_latency_incl_finalizer_median_s"] = d.get("latency_incl_finalizer_s_median")
                mods = d.get("flops_by_module_depth3", {})
                o["GFLOPs_backbone"] = mods.get("OursPipe.m.backbone")
                o["GFLOPs_readout"] = d["GFLOPs"] - mods.get("OursPipe.m.backbone", 0)
            if m == "sum":
                o["selective_scan_GFLOPs_estimate_not_in_total"] = d.get("selective_scan_GFLOPs_estimate")
                o["GFLOPs_incl_scan_estimate"] = d.get("GFLOPs_incl_scan_estimate")
                o["note"] = ("CPU/MPS latency uses a pure-PyTorch chunked selective scan (no CUDA kernel), so it "
                             "overstates SUM's latency on a GPU with the compiled kernel")
            if m == "msdb":
                o["params_executed_estimate"] = (msdb_counts.get("clip_rn50x64_visual_params_executed_to_layer4.2.conv2", 0)
                                                 + msdb_counts.get("dinov2_vitb14_params_executed_to_block10", 0)
                                                 + msdb_counts.get("saliency_network_params", 0)) or None
                o["params_executed_note"] = ("CLIP-RN50x64 visual up to layer4.2.conv2 + DINOv2-B/14 up to block 10 + "
                                             "saliency network (msdb_costs.json); unexecuted layers are instantiated")
                o["float64_buffers_cast_to_float32"] = d.get("n_float64_buffers_cast")
            if m == "iie":
                o["note"] = "IIE computes features on the image downsampled by 2 (192x192 backbone input)"
        else:
            o["mps_allocated_weights_GB"] = d.get("mps_allocated_after_load_GB")
            o["mps_driver_allocated_max_GB"] = d.get("mps_driver_allocated_max_GB")
            if m == "ours":
                o["mps_latency_incl_finalizer_median_s"] = d.get("latency_incl_finalizer_s_median")
    inference[m] = o

ref = inference["ours"]
ratios = {}
for m in ("iie", "msdb", "sum"):
    o = inference[m]; r = {}
    if o.get("GFLOPs") and ref.get("GFLOPs"):
        r["GFLOPs_vs_ours"] = o["GFLOPs"] / ref["GFLOPs"]
    if o.get("cpu_latency_median_s") and ref.get("cpu_latency_median_s"):
        r["cpu_latency_vs_ours"] = o["cpu_latency_median_s"] / ref["cpu_latency_median_s"]
        if ref.get("cpu_latency_incl_finalizer_median_s"):
            r["cpu_latency_vs_ours_incl_finalizer"] = o["cpu_latency_median_s"] / ref["cpu_latency_incl_finalizer_median_s"]
    if o.get("mps_latency_median_s") and ref.get("mps_latency_median_s"):
        r["mps_latency_vs_ours"] = o["mps_latency_median_s"] / ref["mps_latency_median_s"]
        if ref.get("mps_latency_incl_finalizer_median_s"):
            r["mps_latency_vs_ours_incl_finalizer"] = o["mps_latency_median_s"] / ref["mps_latency_incl_finalizer_median_s"]
    if o.get("params_total"):
        r["params_total_vs_ours"] = o["params_total"] / ref["params_total"]
        r["params_trainable_vs_ours"] = o["params_trainable_in_our_adaptation_row"] / ref["params_trainable_in_our_adaptation_row"]
    if o.get("cpu_peak_rss_GB"):
        r["cpu_peak_rss_vs_ours"] = o["cpu_peak_rss_GB"] / ref["cpu_peak_rss_GB"]
    ratios[m] = r

# ------------------------------------------------------------------ training / adaptation wall-clock
train = []


def add_train(row, model, trainable, device, per_fold_s, total_s, what, recipe, note=""):
    pf = [float(x) for x in per_fold_s] if per_fold_s is not None else None
    train.append({"row": row, "model": model, "trainable_params": trainable, "device": device,
                  "train_s_per_fold": pf, "train_s_per_fold_median": float(np.median(pf)) if pf else None,
                  "train_s_total_5_folds": float(np.sum(pf)) if pf else total_s,
                  "row_wall_clock_s_total": total_s, "what_the_time_covers": what, "recipe": recipe, "note": note})


for nm in ["ours_scratch_s0", "ours_scratch_s1", "ours_scratch_s2", "ours_warm_s0", "ours_warm_s1", "ours_warm_s2",
           "ours_grouped", "ours_pairsafe"]:
    s = rj(f"{nm}_summary.json")
    if not s:
        continue
    add_train(nm, "ours", s.get("trainable_params"), s.get("device"),
              [f.get("wall_s") for f in s.get("folds_detail", [])], s.get("wall_clock_s"),
              "per fold: 60-epoch readout training on cached features + sigma selection + test scoring",
              "AdamW 3e-4, wd 1e-4, batch 16, 60 epochs, KL+(1-CC) on 64x64",
              "MPS run held the shared MPS lock (no other MPS job); CPU shared with other tracks")
for nm in ["iie_ft_kl", "iie_ft_klcc", "iie_ft_klcc_e20", "iie_ft_klcc_grouped", "iie_ft_klcc_pairsafe",
           "iie_ft_klcce20_grouped", "iie_ft_klcce20_pairsafe"]:
    s = rj(f"{nm}_summary.json")
    if not s:
        continue
    dev = s.get("device_per_fold", s.get("device"))
    add_train(nm, "iie", s.get("trainable_params_trained"), dev, s.get("t_train_s_per_fold"),
              s.get("wall_clock_s_total"), "per fold: readout training only (evaluation excluded)",
              s.get("recipe", ""), "measured under contention (several CPU pool workers and other tracks running at "
              "once); devices mixed across folds; the _sigma / _nosigma companions share these trainings")
for nm in ["msdb_adapt", "msdb_adapt_kl", "msdb_adapt_dsp", "msdb_adapt_grouped", "msdb_adapt_pairsafe"]:
    s = rj(f"{nm}_summary.json")
    if not s:
        continue
    fl = [f for f in s.get("fold_logs", []) if f.get("seed", 0) == 0]
    add_train(nm + ("_s0" if nm == "msdb_adapt" else ""), "msdb", s.get("trainable_params"), s.get("device"),
              [f.get("train_s") for f in fl], s.get("wall_clock_s"),
              "per fold: head training from cached frozen features (60 epochs); excludes the one-off feature cache",
              f"AdamW {s.get('lr')}, wd {s.get('weight_decay')}, batch {s.get('batch')}, {s.get('epochs')} epochs, "
              f"{s.get('loss')}", "row_wall_clock_s_total covers all seeds/folds of the row and evaluation"
              if nm == "msdb_adapt" else "")
for nm in ["sum_ft_head", "sum_ft_head_klcc"]:
    s = rj(f"{nm}_summary.json")
    if not s:
        continue
    pf = None
    log = s.get("train_log_per_epoch")
    if log:
        acc = {}
        for e in log:
            acc[e["fold"]] = acc.get(e["fold"], 0.0) + float(e.get("seconds", 0))
        pf = [acc[k] for k in sorted(acc)]
    add_train(nm, "sum", s.get("n_trainable_params"), s.get("device"), pf, s.get("wallclock_s_total_incl_loading"),
              "per fold: output-head training on cached decoder features (8 epochs)", json.dumps(s.get("recipe")),
              "head-only stand-in for SUM fine-tuning; full fine-tuning was not run")

one_off = {
    "ours_feature_cache": {"s_total": (j(os.path.join(KB, "cache", "ours", "cache_meta.json")).get("wall_s")
                                       if os.path.exists(os.path.join(KB, "cache", "ours", "cache_meta.json")) else None),
                           "device": "cpu (6 threads)", "images": 1000,
                           "what": "DINOv2-base 448 forward for all 1000 images (patch tokens fp16)"},
    "iie_feature_cache": {"s_total": 2300, "device": "cpu", "images": 1000,
                          "what": "4 IIE backbones on 384 input, features at the 12x12 readout grid (float32)",
                          "source": "logs/iie_cache_features.log ('done 2300s'); CPU shared with other jobs"},
    "msdb_feature_cache": {"s_total": (msdb_counts.get("measured", {}) or {}).get("feature_cache_pass_mps_total_s"),
                           "device": "mps", "images": 1000,
                           "what": "10-scale CLIP-RN50x64 + DINOv2-B/14 backbone, fp16 scale mix (21 GB)",
                           "source": "msdb_costs.json"},
    "sum_decoder_cache": {"s_total": (j(os.path.join(KB, "cache", "sum_dec", "dec_meta.json")).get("seconds_this_run")
                                      if os.path.exists(os.path.join(KB, "cache", "sum_dec", "dec_meta.json")) else None),
                          "device": "cpu (6 threads)", "images": 1000,
                          "what": "SUM encoder + decoder up to the output head (pure-PyTorch scan)",
                          "source": "cache/sum_dec/dec_meta.json"},
}
for k, v in one_off.items():
    if v.get("s_total"):
        v["s_per_image"] = v["s_total"] / v["images"]

out = {
    "description": "Measured costs for the KBS letter (replaces the unmeasured '50x' statement).",
    "machine": "Apple M4 Pro (14 CPU cores, 20 GPU cores), 48 GB RAM, macOS 26.6, torch 2.12.0; no CUDA GPU available, "
               "so 'GPU' numbers are Apple-silicon MPS numbers",
    "conventions": {
        "GFLOPs": "2 x MACs of convolution and matmul ops only (aten.convolution, mm, addmm, bmm, baddbmm; attention via "
                  "the SDPA math backend), torch.utils.flop_counter.FlopCounterMode, one batch-1 forward at the "
                  "model's protocol input size; element-wise ops, normalisation, interpolation, pooling and SUM's "
                  "selective scan excluded (scan estimate listed separately)",
        "latency": "median of timed batch-1 forward passes after 3 warm-up passes (CPU: 30 runs, MSDB 20; MPS: same), "
                   "float32, preprocessed input tensor already in memory (image decoding/resizing excluded); "
                   "CPU with torch.set_num_threads(6); MPS timed with torch.mps.synchronize()",
        "memory": "CPU: peak process RSS sampled every 2 ms during the timed passes (includes weights); MPS: tensor "
                  "allocation after loading (weights) and the maximum driver allocation seen after passes (the "
                  "caching allocator keeps freed blocks, so it approximates the peak working set)",
        "our_finalizer": "our post-hoc Gaussian finalizer (scipy gaussian_filter, sigma 4, 384x384, CPU) is timed "
                         "separately and also added to our latency; DeepGaze IIE/MSDB include their learned blur "
                         "inside the forward pass",
    },
    "inference": inference,
    "ratios_vs_ours": ratios,
    "training_adaptation": train,
    "one_off_feature_cache": one_off,
    "training_caveat": ("Training wall-clock values are read from the row summaries. They were measured while other "
                        "tracks shared the machine (CPU pool workers, one MPS job at a time under the shared lock), on "
                        "mixed devices, so they are indicative only; ours and msdb_adapt ran on MPS with the lock, "
                        "IIE folds ran on CPU or MPS under heavy CPU contention, SUM head rows on CPU."),
    "wall_clock_s": None,
    "command": "python -I scripts/compile/cost_table.py (after scripts/compile/run_cost.sh)",
}
out["wall_clock_s"] = time.time() - T0
json.dump(out, open(os.path.join(RES, "cost_table.json"), "w"), indent=1)
print(json.dumps({m: {k: v for k, v in o.items() if k in ("params_total", "GFLOPs", "cpu_latency_median_s",
                                                            "mps_latency_median_s", "cpu_peak_rss_GB",
                                                            "mps_driver_allocated_max_GB")}
                  for m, o in inference.items()}, indent=1))
print(json.dumps(ratios, indent=1))
for t in train:
    print(t["row"], t["device"], t["train_s_per_fold_median"], t["train_s_total_5_folds"])
