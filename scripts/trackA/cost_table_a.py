"""
Track A cost table: reads results/cost_raw/trackA_<variant><tag>_<dev>_p<k>.json (cost_bench_a.py,
same protocol as cost_bench.py) and writes results/trackA_cost.json.

Because the machine was shared with other tracks, every session re-measured B448 (= the existing
'ours' model of cost_table.json) before and after the variants. Reported per variant:
  measured median latency (contended, as measured);
  contention factor of the session = mean(B448 medians of that session) / B448 in cost_table.json;
  normalised latency = measured / contention factor (an ESTIMATE of the quiet-machine latency,
  assuming the slowdown is the same for all variants);
  ratios vs DeepGaze MSDB (cost_table.json numbers) and vs ours (B448).
GFLOPs and parameters are deterministic and do not depend on contention.

  python -I cost_table_a.py --tag _run1 [--rows-map L448=ours_L448 ...]
"""
import sys
sys.dont_write_bytecode = True
import os, json, glob, argparse, statistics, time

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RAW = os.path.join(KB, "results", "cost_raw")
ROWS = {"B448": "ours_scratch_s0 (existing)", "L448": "ours_L448_s*", "msB": "ours_ms_B_s0", "B644": "ours_B644_s0", "Lms": "ours_L_ms_s0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["_run1"])
    ap.add_argument("--out", default=os.path.join(KB, "results", "trackA_cost.json"))
    a = ap.parse_args()
    ct = json.load(open(os.path.join(KB, "results", "cost_table.json")))["inference"]
    ref = {"cpu": ct["ours"]["cpu_latency_median_s"], "mps": ct["ours"]["mps_latency_median_s"]}
    msdb = {"cpu": ct["msdb"]["cpu_latency_median_s"], "mps": ct["msdb"]["mps_latency_median_s"], "GFLOPs": ct["msdb"]["GFLOPs"]}
    out = {"description": __doc__.strip(), "reference_cost_table": {"ours_B448": {"GFLOPs": ct["ours"]["GFLOPs"], **{f"{d}_latency_median_s": ref[d] for d in ref}},
                                                                      "msdb": {"GFLOPs": msdb["GFLOPs"], "cpu_latency_median_s": msdb["cpu"], "mps_latency_median_s": msdb["mps"],
                                                                               "params_total": ct["msdb"]["params_total"]}},
           "sessions": {}, "variants": {}}
    for tag in a.tags:
        for dev in ("cpu", "mps"):
            files = sorted(glob.glob(os.path.join(RAW, f"trackA_*{tag}_{dev}_p*.json")), key=lambda p: int(p.rsplit("_p", 1)[1][:-5]))
            if not files:
                continue
            recs = [json.load(open(f)) for f in files]
            b = [r["latency_s"]["median"] for r in recs if r["variant"] == "B448"]
            cf = statistics.mean(b) / ref[dev] if b else None
            out["sessions"][f"{tag}_{dev}"] = {"files": [os.path.basename(f) for f in files], "B448_medians_s": b,
                                              "contention_factor_vs_cost_table": cf,
                                              "load_note": "machine shared with the distillation track (CPU jobs) during this session"}
            for r in recs:
                v = r["variant"]
                if v == "B448" and dev == "cpu" and "GFLOPs" in r:
                    pass
                e = out["variants"].setdefault(v, {"row": ROWS[v], "backbone": r["backbone_name"], "input_sizes": r["input_sizes"]})
                if "GFLOPs" in r:
                    e["GFLOPs"] = r["GFLOPs"]; e["GFLOPs_by_op"] = r.get("flops_by_op_global")
                    e["GFLOPs_vs_msdb_fewer_x"] = msdb["GFLOPs"] / r["GFLOPs"]; e["GFLOPs_vs_ours_B448_x"] = r["GFLOPs"] / ct["ours"]["GFLOPs"]
                e["params_total"] = r["params_total"]; e["params_backbone_frozen"] = r["params_backbone_frozen"]
                e["params_trainable_readout"] = r["params_trainable_in_adaptation"]
                key = f"{dev}{tag}"
                lat = r["latency_s"]["median"]
                d = e.setdefault("latency", {}).setdefault(key, [])
                d.append({"median_s": lat, "iqr_s": [r["latency_s"]["q25"], r["latency_s"]["q75"]], "runs": r["runs"], "warmup": r["warmup"],
                          "threads": r["threads"], "normalised_s_estimate": (lat / cf) if cf else None,
                          "peak_rss_GB" if dev == "cpu" else "mps_driver_allocated_max_GB":
                              r.get("peak_rss_during_inference_GB") if dev == "cpu" else r.get("mps_driver_allocated_max_GB")})
    for v, e in out["variants"].items():
        for dev in ("cpu", "mps"):
            meas = [x for k, lst in e.get("latency", {}).items() if k.startswith(dev) for x in lst]
            if not meas:
                continue
            best = min(meas, key=lambda x: x["median_s"])
            norm = statistics.median([x["normalised_s_estimate"] for x in meas if x["normalised_s_estimate"]])
            e[f"{dev}_latency_measured_min_median_s"] = best["median_s"]
            e[f"{dev}_latency_normalised_estimate_s"] = norm
            e[f"{dev}_msdb_slower_x_normalised"] = msdb[dev] / norm
            e[f"{dev}_msdb_slower_x_raw_contended"] = msdb[dev] / best["median_s"]
            e[f"{dev}_vs_ours_B448_x_normalised"] = norm / ref[dev]
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S"); out["command"] = "python -I " + " ".join(sys.argv)
    json.dump(out, open(a.out, "w"), indent=1)
    for v, e in out["variants"].items():
        print(f"{v:5s} GFLOPs {e.get('GFLOPs', float('nan')):8.1f} (MSDB/{e.get('GFLOPs_vs_msdb_fewer_x', 0):.1f}x)  params {e['params_total']/1e6:.1f}M  "
              f"cpu meas {e.get('cpu_latency_measured_min_median_s', float('nan')):.3f} norm {e.get('cpu_latency_normalised_estimate_s', float('nan')):.3f} (MSDB {e.get('cpu_msdb_slower_x_normalised', 0):.1f}x)  "
              f"mps meas {e.get('mps_latency_measured_min_median_s', float('nan')):.3f} norm {e.get('mps_latency_normalised_estimate_s', float('nan')):.3f} (MSDB {e.get('mps_msdb_slower_x_normalised', 0):.1f}x)")
    for s, d in out["sessions"].items():
        print(s, "contention factor", round(d["contention_factor_vs_cost_table"], 2), d["B448_medians_s"])


if __name__ == "__main__":
    main()
