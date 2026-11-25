"""Tiny end-to-end check of the training/eval path (16 train ids, 2 epochs, 8 test ids)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C, iie_train as T
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.set_num_threads(6)
D = T.Data()
tr = D.folds[1][:16]; te = D.folds[0][:8]
for loss in ("kl384", "klcc64"):
    res = T.train_eval(D, tr, te, T.seeded_orders(tr, 2, 0), loss, 1e-3, 8, dev, tr[:8], tag=loss)
    print(loss, res["sigma"], res["epoch_mean_loss"], res["published_scoring_nosigma"], res["trainable_params"],
          np.mean([r["CC"] for r in res["rows_nosigma"]]))
# published order replication check vs split
o = T.published_orders(D.folds, 2)
print("orders ok", len(o), len(o[0]), len(o[0][0]))
