"""Parameter breakdown of SUM (released weights). python -I param_count.py --scripts DIR --code DIR --weights DIR --out FILE"""
import argparse, json, sys
ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "out"):
    ap.add_argument(f"--{a}", required=True)
args = ap.parse_args()
sys.path.insert(0, args.scripts)
import torch
import sum_common as sc
torch.set_num_threads(2)
model, cfg = sc.load_sum(args.code, args.weights, "cpu")
net = model.salu_mamba
groups = {"total": 0, "encoder(patch_embed+layers)": 0, "decoder(layers_up)": 0, "head(final_up+final_conv)": 0, "other": 0}
for n, p in model.named_parameters():
    k = p.numel(); groups["total"] += k
    s = n.split("salu_mamba.", 1)[-1]
    if s.startswith("layers_up"): groups["decoder(layers_up)"] += k
    elif s.startswith("final"): groups["head(final_up+final_conv)"] += k
    elif s.startswith("layers") or s.startswith("patch_embed"): groups["encoder(patch_embed+layers)"] += k
    else: groups["other"] += k
groups["config"] = cfg
json.dump(groups, open(args.out, "w"), indent=1); print(json.dumps(groups))
