# Third-party code and weights (not redistributed)

Fetched from their official sources and not redistributed here. SUM (MIT), CLIP (MIT) and DINOv2 (Apache-2.0) are
used under their licences. The DeepGaze IIE/MSDB code, weights and MIT1003 centre-bias template
(github.com/matthias-k/DeepGaze) carry no stated licence; obtain them from upstream. This repository contains none of
them, apart from functions and classes in `scripts/iie/iie_common.py`, `scripts/iie_native/native_common.py` and
`scripts/msdb/msdb_common.py` (about 400 lines) that re-implement DeepGaze's readouts, finalizers and mixture so that
its released weights run on cached features; they are adapted from the DeepGaze code and each is marked by an
attribution comment. The SUM loss in `scripts/sum/ft_sum_folds.py` and `scripts/sum/ft_head.py` is adapted from SUM and
marked the same way. `deepgaze_pytorch` downloads its own backbone weights (through torch.hub and its bundled
`efficientnet_pytorch`) under their own licences.

| What | Where it goes | Source |
| --- | --- | --- |
| SUM code (MIT licence) | `third_party/SUM/code/` | `git clone https://github.com/Arhosseini77/SUM.git` at commit `03b6eb88f35a71e9106e895014bdd8ff526d947e` |
| SUM released weights | `third_party/SUM/weights/` (`config.json`, `model.safetensors`) | Hugging Face `safe-models/SUM` (the authors' checkpoint); `model.safetensors` sha256 `8640fde3e29c34c5ef262ad9bb949c2587c9164fe0d94d52357bd3129dd6e195` |
| DeepGaze IIE and MSDB code (no stated licence) | installed by pip (`requirements.txt`) | `deepgaze_pytorch` from github.com/matthias-k/DeepGaze at commit `c87b106e8698497c59998b469c45770e993baca3` |
| DeepGaze IIE and MSDB weights (no stated licence) | torch hub cache (downloaded by `deepgaze_pytorch` on first use) | github.com/matthias-k/DeepGaze releases v1.0.0 (`deepgaze2e.pth`) and v1.2.0 (`deepgazemsdb.pth`) |
| DeepGaze MIT1003 centre-bias template (no stated licence) | `cache/revision/deepgaze_template/centerbias_mit1003.npy` | github.com/matthias-k/DeepGaze release v1.0.0; sha256 `051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9` |
| CLIP code (MIT licence) and its RN50x64 weights (MSDB's CLIP backbone) | code installed by pip (`requirements.txt`); weights downloaded by the `clip` package on first use (`~/.cache/clip/`) | `git+https://github.com/openai/CLIP.git` at commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| DINOv2-B (frozen readout backbone; DINOv2-L for one readout variant) | Hugging Face cache | `facebook/dinov2-base`, `facebook/dinov2-large` (Apache 2.0) |
| Earlier release (optional) | `third_party/dont-scale-fine-tune/` | `git clone https://github.com/Deep-c-Vision-LTD/dont-scale-fine-tune.git`; its stored zero-shot scores are used only by two reproduction checks (`scripts/iie/zeroshot_iie.py`, which skips the check when the file is absent and takes another path from `PUBLISHED_IIE_JSON`; `scripts/sum/validate_zeroshot.py`, via `--stored`) |

`scripts/sum/sum_common.py` loads SUM from the code folder without installing the package and replaces its
CUDA selective-scan kernel with a pure-PyTorch scan, so no `mamba-ssm` build is needed.
