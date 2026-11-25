"""Compare StackedIIE with the custom FinalMix backward (IIE_FAST_FINAL=1) against the
autograd logsumexp formulation: outputs and parameter gradients for both losses."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import iie_common as C, iie_train as T
import deepgaze_pytorch
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.set_num_threads(6)
D = T.Data()
ids = D.folds[2][:8]
feats = D.batch_feats(ids, dev)
cb = D.prior(D.folds[1])
cbt = torch.from_numpy(cb.astype(np.float32)).to(dev)[None].expand(8, -1, -1)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
st = C.StackedIIE(model).to(dev)
# perturb parameters a bit so the check is not only at the released optimum
with torch.no_grad():
    g_ = torch.Generator().manual_seed(0)
    for p in st.parameters():
        p.add_(0.05 * p.abs().mean() * torch.randn(p.shape, generator=g_).to(dev))
rep = {"device": dev}
for loss in ("kl384", "klcc64"):
    out = {}
    for fast in (False, True):
        C.FAST_FINAL = fast
        for p in st.parameters(): p.grad = None
        l = T.loss_fn(st, feats, cbt, ids, D, loss, dev); l.backward()
        out[fast] = (float(l), torch.cat([p.grad.detach().flatten().double().cpu() for p in st.parameters()]))
        with torch.no_grad():
            out[str(fast) + "logd"] = st(feats, cbt).double().cpu()
    ga, gb = out[False][1], out[True][1]
    rep[loss] = {"loss_ref": out[False][0], "loss_fast": out[True][0],
                 "grad_global_rel_err": float((ga - gb).norm() / ga.norm()),
                 "max_abs_logd": float((out["Falselogd"] - out["Truelogd"]).abs().max())}
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(C.KBS, "logs", f"iie_verify_fastfinal_{dev}.json"), "w"), indent=1)
