# Agricultural Plastic Cover Mapping from Satellite Imagery

Classifying agricultural plastic cover (greenhouse film, mulch) from
satellite-derived spectral features, across three regions: Kenya, Spain,
and Vietnam. Built for an FAO/Zindi GEO-AI competition.

**Original submission:** Google Earth Engine JavaScript, Random Forest
(`ee.Classifier.smileRandomForest`), scored 0.9375 on held-out validation,
ranked 57th/78 on the competition leaderboard.

**This version:** the same approach ported to Python/scikit-learn —
readable, reproducible, and runnable outside GEE. Scored 0.9340 on an
80/20 stratified validation split, matching the original within a normal
margin of variation for a single train/test split.

## The problem

Agricultural plastic use is expanding globally to support intensive
farming, and is a growing environmental concern at end-of-life. No global
map of plastic-covered land currently exists, largely because
classification approaches built for one region don't reliably transfer to
another. This project builds a baseline classifier and is a first step
toward testing that transfer problem directly (see Future Work).

## Data

Satellite-derived tabular features (Sentinel-2 optical, 50th-percentile
reflectance; Sentinel-1 VV/VH radar backscatter) at labeled points across
three regions, sampled April–May 2023/2024. Binary target: `1` =
agricultural plastic cover, `2` = not.

**Data is not included in this repository** — it's provided under the
Zindi/FAO competition terms and isn't mine to redistribute. To reproduce:
download the competition dataset from Zindi and place the CSVs in a local
`data/` folder (see `data/README.md` for the exact filenames expected).

Raw per-region IDs restart at 1 in each file (Kenya row 1, Spain row 1,
Vietnam row 1 would all be `1`), so IDs are region-prefixed
(`Kenya_1`, `Spain_1`, `VNM_1`) to match the competition's expected
submission format and avoid collisions when regions are pooled.

## Feature engineering

Six spectral indices, computed from the raw reflectance/backscatter bands:

| Index | Formula | What it captures |
|---|---|---|
| NDVI | (nir − red) / (nir + red) | Vegetation vigor; plastic cover suppresses this relative to open cropland |
| NDBI | (swir1 − nir) / (swir1 + nir) | Originally a built-up-surface index; plastic's SWIR/NIR contrast behaves similarly |
| PMLI | (swir2 − red) / (swir2 + red) | Purpose-built in the literature to separate plastic-mulched land from bare soil/vegetation |
| RPGI | blue / (avg(blue,green,nir) − 1) × 100 | Greenhouse-specific index using the blue-band signature |
| PGI | blue × (nir − red) / (avg(blue,green,nir) − 1) × 100 | Combines blue-band signature with vegetation contrast |
| VI | NDBI × NDVI | Interaction term: "built-up-like signal over what should be cropland" |
| PGHI *(experimental)* | swir2 / blue | Tested during development; not more informative than the indices above. Kept in the codebase for reference, not removed from the feature set. |

Radar bands (VV, VH) are kept as raw features rather than folded into an
index — Sentinel-1 backscatter responds to surface roughness/moisture in a
way that's complementary to the optical indices, and isn't affected by
cloud cover.

## Results

80/20 stratified train/validation split, Random Forest (100 trees),
trained on all three regions pooled:

- **Validation accuracy: 0.9340**
- Full classification report (precision/recall/F1 per class) printed by
  `train.py` at run time.

This is a single split, not cross-validated — the number will move
somewhat with a different random seed. Treat it as a point estimate, not
a precise benchmark.

## Repository structure

```
.
├── train.py          # full pipeline: load, engineer features, train, validate, predict
├── requirements.txt
├── data/              # not tracked in git — see data/README.md
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python train.py
```

Expects the competition CSVs in `data/` (see Data section above).
Produces `data/submission.csv` in the competition's expected format.

## Future work

- **Per-region generalization testing.** Currently all three regions are
  pooled into a single model with a single random train/val split. A more
  rigorous test of the "global mapping" claim would be to train on two
  regions and evaluate on the third, entirely unseen region — testing
  whether the learned spectral patterns actually transfer, or whether the
  model is implicitly relying on region-specific signal. Early exploration
  suggests this generalization gap may be substantial for at least one
  region, which would itself help explain why global plastic-cover mapping
  remains an open problem — this needs to be tested properly and written
  up rather than asserted.
- **Cross-validation** instead of a single train/val split, for a more
  stable accuracy estimate.
- **Resolve or remove PGHI** — either find a justified role for it or drop
  it from the feature set once its lack of contribution is confirmed more
  rigorously (e.g. via feature importance / ablation).
- **Model comparison** — gradient-boosted trees (XGBoost/LightGBM) against
  the current Random Forest baseline.
