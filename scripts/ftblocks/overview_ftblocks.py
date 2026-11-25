"""
ftblocks overview: collects the pre-registered outputs into results/ftblocks_overview.json (descriptive only;
no selection or tuning happens here).

  * per-fold selected config, the full inner-val table (CC/SIM/KLD, sigma, status, wall-clock per run)
  * pooled (global, NOT nested) choice and the optional-row trigger
  * outer-test per-fold CC/SIM/KLD + sigma + config used + fallback chain for every assembled row
  * wall-clock of every run (cache/ftblocks/runs/*.json)
  * row summaries (mean over folds, fold SD ddof 1) and the paired statistics of results/ftblocks_stats*.json
    (the newest file), with the four headline comparisons pulled out
  * inference cost: the fine-tuned model is the same graph as ours (results/cost_table.json 'ours');
    results/ftblocks_cost.json is included if it exists

  python -I overview_ftblocks.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, glob, time

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(KB, "cache", "ftblocks", "runs")
RES = os.path.join(KB, "results")
ROWS = ["ours_ftblocks_nested", "ours_ftblocks_global", "ours_ftblocks_distill"]
HEADLINE_REFS = ["ours_scratch_s0", "distill_pitt", "msdb_adapt_s0", "sum_ft"]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def run_brief(r):
    t = r.get("train") or {}
    return {"run": r.get("run"), "stage": r.get("stage"), "fold": r.get("fold"), "config": r["config"]["name"], "status": r.get("status", "ok"),
            "n_train": r.get("n_train"), "n_eval": r.get("n_eval"), "sigma": r.get("sigma"), "CC": r.get("mean_CC"), "SIM": r.get("mean_SIM"),
            "KLD": r.get("mean_KLD"), "steps": t.get("steps"), "s_per_step": t.get("s_per_step"), "train_s": t.get("train_s"),
            "eval_s": r.get("eval_s"), "wall_s": r.get("wall_s"), "finished": r.get("finished"), "error": r.get("error")}


def main():
    plan = jl(os.path.join(HERE, "PLAN.json"))
    grid = [c["name"] for c in plan["active_grid"]]
    runs = {os.path.basename(p)[:-5]: jl(p) for p in sorted(glob.glob(os.path.join(RUNS, "*_k[0-9]_*.json")))}
    sel = [jl(os.path.join(RUNS, f"selection_k{k}.json")) for k in range(5)]
    inner_table = []
    for k in range(5):
        for n in grid:
            r = runs.get(f"inner_k{k}_{n}")
            if r:
                inner_table.append({**run_brief(r), "selected": bool(sel[k] and sel[k]["selected"] == n)})
    rows = {}
    for row in ROWS:
        s = jl(os.path.join(RES, f"{row}_summary.json"))
        if s:
            rows[row] = {"status": s.get("status"), "mean_CC": s["mean_CC"], "sd_CC_ddof1": s["sd_CC_ddof1"],
                         "mean_SIM": s["mean_SIM"], "sd_SIM_ddof1": s["sd_SIM_ddof1"], "mean_KLD": s["mean_KLD"],
                         "sd_KLD_ddof1": s["sd_KLD_ddof1"], "pooled_image_mean_CC": s["pooled_image_mean_CC"],
                         "per_fold": [{"fold": d["fold"], "config": d["config"]["name"], "sigma": d["sigma"], "CC": d["CC"], "SIM": d["SIM"],
                                       "KLD": d["KLD"], "wall_s": d["wall_s"], "run": d["run"]} for d in s["folds_detail"]],
                         "chains": s.get("final_chain_per_fold") or s.get("distill_chain_per_fold")}
    st_files = sorted(glob.glob(os.path.join(RES, "ftblocks_stats*.json")), key=os.path.getmtime)
    stats = jl(st_files[-1]) if st_files else None
    headline = None
    if stats:
        T = stats["primary"]["tests"]
        headline = {}
        for ref in HEADLINE_REFS:
            v = T.get(f"ours_ftblocks_nested - {ref}")
            if v is None:
                continue
            # compile_results.compare(a, b) reports a - b; its '*_ref_better' fields count where a (= ftblocks) is better
            ren = {"folds_ref_better": "folds_ftblocks_better", "frac_images_ref_better": "frac_images_ftblocks_better"}
            headline[ref] = {m: {ren.get(kk, kk): v[m].get(kk) for kk in ("mean_diff", "boot95", "wilcoxon_p", "wilcoxon_p_holm", "nb_t",
                                                                          "nb_p", "nb_p_holm", "fold_diffs", "folds_ref_better",
                                                                          "frac_images_ref_better", "dz", "rank_biserial") if kk in v[m]}
                             for m in ("CC", "SIM", "KLD")}
    cost_tab = jl(os.path.join(RES, "cost_table.json"))
    ours_cost = cost_tab["inference"]["ours"] if cost_tab else {}
    out = {"title": plan["title"], "plan_file": os.path.join(HERE, "PLAN.json"), "plan_frozen_utc": plan["frozen_utc"],
           "plan_mode": plan["mode"], "active_grid": grid, "amendment": jl(os.path.join(HERE, "PLAN_AMENDMENT_1.json")),
           "deviations_file": os.path.join(HERE, "PLAN_DEVIATIONS.md"),
           "selection_per_fold": [{"fold": k, "selected": s["selected"], "inner_val_CC": s["inner_val_CC"], "fallback_order": s["fallback_order"],
                                   "selected_minus_control_CC": s["selected_minus_control_CC"]} if s else None for k, s in enumerate(sel)],
           "inner_val_table": inner_table,
           "pooled_selection_NOT_nested": jl(os.path.join(RUNS, "pooled_selection.json")),
           "distill_trigger": jl(os.path.join(RUNS, "distill_trigger.json")),
           "rows": rows, "statistics_file": st_files[-1] if st_files else None,
           "statistics_family": (stats or {}).get("primary", {}).get("pairs"),
           "headline_comparisons_nested_minus_ref": headline,
           "references_describe": (stats or {}).get("references_describe"),
           "secondary_statistics_note": "ours_ftblocks_global is NOT nested (amendment M1); its tests are descriptive only",
           "runs_wall_clock": [run_brief(r) for r in runs.values()],
           "total_run_wall_h": sum((r.get("wall_s") or 0) for r in runs.values()) / 3600,
           "inference_cost": {"same_graph_as_ours": True,
                              "GFLOPs": ours_cost.get("GFLOPs"), "params_total": ours_cost.get("params_total"),
                              "ours_cpu_latency_median_s": ours_cost.get("cpu_latency_median_s"),
                              "ours_mps_latency_median_s": ours_cost.get("mps_latency_median_s"),
                              "source": os.path.join(RES, "cost_table.json") + " (inference.ours)",
                              "ftblocks_measured": jl(os.path.join(RES, "ftblocks_cost.json"))},
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    p = os.path.join(RES, "ftblocks_overview.json")
    json.dump(out, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    print("written", p)


if __name__ == "__main__":
    main()
