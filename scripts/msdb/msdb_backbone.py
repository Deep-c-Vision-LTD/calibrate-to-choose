"""Loading DeepGazeMSDB on CPU or MPS and extracting its per-scale frozen features exactly."""
import torch
from msdb_common import PPD, RO


def load_msdb(dev="cpu", cast_fp32=True):
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeMSDB(pretrained=True)
    model.eval()            # not chained: the package's train() does not return self
    if cast_fp32:
        # the two input normalisers keep float64 mean/std buffers, which MPS cannot hold
        n = 0
        for m in model.modules():
            for name, buf in list(m._buffers.items()):
                if buf is not None and buf.dtype == torch.float64:
                    m._buffers[name] = buf.float(); n += 1
        print(f"cast {n} float64 buffers to float32", flush=True)
    model.to(dev)
    model.eval()
    return model


@torch.no_grad()
def per_scale_features(model, x):
    """x: (1,3,512,512) float32 in [0,255] on the model device.
    Returns the 10 per-scale feature maps (pixel_per_dva scales then size scales), each
    (1,2560,64,64) float32, computed with the package's own _process_* methods."""
    fb = model.features
    shape = [RO, RO]
    out = [fb._process_pixel_per_dva(x, [PPD], t, shape) for t in fb.resolutions_pixel_per_dva]
    out += [fb._process_size(x, s, shape) for s in fb.resolutions_size]
    return out


def mix_features(feats, scale_w):
    """Generalisation-mode scale mix, summed in the same order as the package."""
    acc = None
    for w, f in zip(scale_w, feats):
        acc = w * f if acc is None else acc + w * f
    return acc
