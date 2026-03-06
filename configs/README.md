# Configs (paper-frozen YAML)

This folder contains **frozen configuration snapshots** used to run and
reproduce the GeoPrior-v3 (GeoPriorSubsNet) experiments reported in the
paper.

The YAML files are designed to be **human-readable** mirrors of the
defaults in `nat.com/config.py`, with small, explicit overrides for each
city.

---

## Files

- `paper_v3_2.yaml`  
  Baseline “paper” configuration: feature schema, temporal windowing,
  model/training defaults, and physics settings.

- `nansha.yaml`  
  City override for **Nansha**. It sets `CITY_NAME` and the expected
  input CSV filenames (e.g., `nansha_final_main_std...with_zsurf.csv`).

- `zhongshan.yaml`  
  City override for **Zhongshan**. Same structure as `nansha.yaml`.

---

## Data file naming (CSV)

The pipeline expects processed city datasets under `data/` using the
template:

`{city}_final_main_std.harmonized.cleaned.{variant}.csv`

For the paper runs, `variant = with_zsurf`, so typical inputs are:

- `data/nansha_final_main_std.harmonized.cleaned.with_zsurf.csv`
- `data/zhongshan_final_main_std.harmonized.cleaned.with_zsurf.csv`

These large CSVs are not stored in GitHub; they are provided in the Code
Ocean capsule and/or via the original data providers.

---

## How configs are used

In practice, you run:

- the **baseline** (`paper_v3_2.yaml`)  
- plus one **city override** (`nansha.yaml` or `zhongshan.yaml`)

The scripts in `nat.com/` load these settings to control:
- Stage-1: harmonisation + tensor packaging
- Stage-2: training (data loss + physics constraints)
- Stage-3/4/5: evaluation, diagnostics, and transfer (when enabled)

If you change a YAML value, keep a copy for reproducibility (or commit a
new frozen config).