# Data (not redistributed)

Nothing in this folder is part of the release. The scripts look here by default; set `ADD1000_DIR`
(and `PITT_ADS_DIR`, `UEYES_DIR` for the two optional checks) to use another location.

```
data/
  ADD1000/
    flat/images/<id>.jpg     1000 advertisements (ids 1..1000)
    flat/maps/<id>.png       the matching released density maps
    ext/                     the release's own folders (ADD1000_1 ... ADD1000_8B, GroundTruth1000_*); only
                             scripts/ours/make_splits.py reads them (folder-structure test of the proxy groups)
  pitt_ads/imgs/             Pitt Ads images; only the distillation rows of the supplement use them
  UEyes/images/              UEyes images; only scripts/revision/ueyes_overlap.py uses them
```

ADD1000: Liang, Liu and Qian, "Fixation prediction for advertising images: Dataset and benchmark",
J. Vis. Commun. Image Represent. 81 (2021) 103356, released at https://github.com/lscumt/ADD1000. The release
holds images and density maps only (no fixation coordinates) and states no licence: obtain it from the dataset
authors and observe any terms they set. The `flat/` folder is a convenience copy of the release in which an image
and its map share the basename.
