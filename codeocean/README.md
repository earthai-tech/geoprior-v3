# GeoPrior-v3 Code Ocean Capsule

This capsule accompanies the manuscript:

**Physics-Informed Deep Learning Reveals Divergent Urban Land Subsidence Regimes**

It is designed to let reviewers run the GeoPrior-v3 workflow from the packaged command-line interface rather than from ad hoc scripts. The capsule focuses on reproducibility, auditability, and ease of testing in a Code Ocean environment.

---

## 1. What this capsule contains

At minimum, the capsule is expected to contain:

```text
.
├── config.py
├── data/
│   ├── zhongshan_full_city.csv
│   └── nansha_full_city.csv            # optional unless Stage-5 is tested
├── nat.com/
│   └── config.json                     # auto-generated / updated by GeoPrior
├── results/
└── README.md
```

### Important notes

- `config.py` at the capsule root is the **reviewer-facing configuration file**.
- GeoPrior uses `nat.com/config.py` and `nat.com/config.json` internally.
- The commands below use `--config ./config.py` so the shipped root config is installed into `nat.com/config.py` automatically before execution.
- All run outputs are written under the capsule-level `results/` directory.
- The full-city reviewer dataset for this capsule is expected at:
  - `data/zhongshan_full_city.csv`
- If a second city dataset is provided for transfer experiments, place it at:
  - `data/nansha_full_city.csv`

> The built-in fetchers in GeoPrior provide small sample datasets for demonstration and smoke tests. They are **not** the intended data source for reproducing the paper workflow in this capsule.

---

## 2. Scientific context

GeoPrior-v3 implements **GeoPriorSubsNet**, a physics-guided spatiotemporal forecasting framework for urban land subsidence and groundwater-related dynamics. In the submitted Nature Communications study, the framework is evaluated in **two contrasting cities: Nansha and Zhongshan**.

For Code Ocean review, the most important executable checks are:

1. **Stage-1 preprocessing**
2. **Stage-3 training**

Optional reviewer checks include:

3. **Stage-2 short tuning demo**
4. **Stage-4 inference/export**
5. **Stage-5 cross-city transfer evaluation**

---

## 3. Installation

Install the published package from PyPI:

```bash
pip install geoprior-v3
```

You may also install a few common extras explicitly if your environment is very minimal:

```bash
pip install geoprior-v3 joblib pandas scikit-learn matplotlib tensorflow
```

---

## 4. Documentation and source code

- Documentation: [GeoPrior-v3 documentation](https://geoprior-v3.readthedocs.io/)
- Source code: [earthai-tech/geoprior-v3](https://github.com/earthai-tech/geoprior-v3)

Tutorial videos:

- App tutorial: [https://youtu.be/JtOpX5lv4iw](https://youtu.be/JtOpX5lv4iw)
- Example simulation run: [https://youtu.be/nCouLQQFpQg](https://youtu.be/nCouLQQFpQg)

![Screenshot of the GeoPrior-V3 GUI showing running flow](https://github.com/earthai-tech/geoprior-v3/blob/main/codeocean/geoprior-v3.app.png)
---

## 5. GeoPrior CLI overview

GeoPrior-v3 exposes a family-based CLI. The canonical form is:

```bash
geoprior run <command> [args]
geoprior build <command> [args]
geoprior plot <command> [args]
```

For this capsule, the main workflow commands are:

```bash
geoprior run stage1-preprocess
geoprior run stage2-train
geoprior run stage3-tune
geoprior run stage4-infer
geoprior run stage5-transfer
```

You can inspect the installed CLI with:

```bash
geoprior --help
geoprior run --help
geoprior run stage1-preprocess --help
geoprior run stage3-tune --help
```

---

## 6. Configuration model used in this capsule

GeoPrior's NATCOM-style configuration treats `config.py` as the **single source of truth** for experiment settings. The library then regenerates `nat.com/config.json` from it automatically when the workflow runs.

In this capsule:

- edit the root-level `config.py` if you need to inspect or modify settings;
- run all commands with `--config ./config.py`;
- GeoPrior will copy that file into `nat.com/config.py` and update `nat.com/config.json` as needed.

### Recommended reviewer settings

For the Zhongshan reviewer dataset, ensure that the shipped `config.py` points to the root `data/` folder and to the full-city CSV rather than a bundled demo fetcher.

Typical settings are:

```python
CITY_NAME = "zhongshan"
MODEL_NAME = "GeoPriorSubsNet"
DATA_DIR = "."
SEARCH_PATHS = ["data/zhongshan_full_city.csv"]
FALLBACK_PATHS = []
```

If you want to test transfer later, add the second city file and adjust or extend the paths accordingly.

---

## 7. Minimal reviewer workflow

### Step A. Confirm the reviewer dataset exists

```bash
ls data
```

Expected primary file:

```text
data/zhongshan_full_city.csv
```

### Step B. Run Stage-1 preprocessing

This builds the cleaned tables, scalers, sequence NPZ files, and a manifest for downstream stages.

```bash
geoprior run stage1-preprocess \
  --config ./config.py
```

### Expected Stage-1 result

A city-specific Stage-1 folder is created under `results/`, containing artifacts such as:

- raw / cleaned / scaled CSV files
- feature encoders and scalers
- split NPZ files for train / validation / test
- `manifest.json`

A typical path is:

```text
results/zhongshan_GeoPriorSubsNet_stage1/
```

with the key handoff file:

```text
results/zhongshan_GeoPriorSubsNet_stage1/manifest.json
```

### Step C. Run Stage-3 tuning

Stage-3 can be run directly from the Stage-1 manifest.

```bash
geoprior run stage3-tune \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json
```

This is the main reviewer-side computational check after Stage-1.

---

## 8. Fast demonstration mode

For a lighter test run, reviewers may reduce training effort by overriding configuration values from the CLI.

### Short Stage-2 demo (5 epochs)

```bash
geoprior run stage2-train \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json \
  --set EPOCHS=5
```

### Short Stage-3 demo (5 epochs per trial)

```bash
geoprior run stage3-tune \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json \
  --set EPOCHS=5
```

If your local `config.py` already defines a small search space or small number of trials, that is usually sufficient for Code Ocean testing.

---

## 9. Stage-4 inference and export

Stage-4 is used after a successful training or tuning workflow to run inference, evaluation, and export. It accepts the same `--config` installation pattern and can also be pointed at a specific Stage-1 manifest.

Basic form:

```bash
geoprior run stage4-infer \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json
```

If you want to inspect all supported inference arguments in the packaged environment:

```bash
geoprior run stage4-infer --help
```

Because Stage-4 forwards additional inference options to the legacy inference backend, the exact export flags may be appended as needed in the capsule.

---

## 10. Stage-5 cross-city transfer evaluation

Stage-5 is for **cross-city transfer** and therefore requires **two cities**.

To test Stage-5 in this capsule, ensure both of the following are present:

```text
data/zhongshan_full_city.csv
data/nansha_full_city.csv
```

Then run the transfer command with an explicit city pair:

```bash
geoprior run stage5-transfer \
  --config ./config.py \
  --city-a zhongshan \
  --city-b nansha
```

To inspect all forwarded transfer options:

```bash
geoprior run stage5-transfer --help
```

If only Zhongshan data is packaged, Stage-5 should be considered optional and may be skipped.

---

## 11. What Stage-1 produces for downstream stages

Stage-1 is the key handshake stage in GeoPrior-v3.

It exports:

- cleaned and scaled tabular files;
- encoded static, dynamic, and future-known features;
- train / validation / test sequence arrays in NPZ form;
- scalers and one-hot encoders;
- a `manifest.json` file describing paths, dimensions, columns, and configuration.

Downstream stages use this manifest instead of rebuilding the full preprocessing pipeline from scratch.

---

## 12. Typical output layout

All outputs are written under `results/`.

A typical reviewer run may create directories such as:

```text
results/
├── zhongshan_GeoPriorSubsNet_stage1/
├── zhongshan_GeoPriorSubsNet_stage2/
├── zhongshan_GeoPriorSubsNet_stage3/
├── zhongshan_GeoPriorSubsNet_stage4/
└── transfer_zhongshan_to_nansha/
```

The exact names can vary slightly with configuration, but `results/` is the central output root for this capsule.

---

## 13. Suggested reviewer command sequence

### Minimal verification

```bash
geoprior run stage1-preprocess --config ./config.py
geoprior run stage3-tune \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json \
  --set EPOCHS=5
```

### Slightly broader verification

```bash
geoprior run stage1-preprocess --config ./config.py
geoprior run stage2-train \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json \
  --set EPOCHS=5
geoprior run stage4-infer \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json
```

### Full two-city transfer check

```bash
geoprior run stage5-transfer \
  --config ./config.py \
  --city-a zhongshan \
  --city-b nansha
```

---

## 14. Troubleshooting

### Problem: GeoPrior falls back to a small bundled sample dataset

Cause:
- the expected full-city CSV was not found from the configured paths.

Fix:
- confirm `data/zhongshan_full_city.csv` exists;
- confirm `config.py` points to that file through `DATA_DIR`, `SEARCH_PATHS`, `BIG_FN`, or equivalent settings;
- re-run the command with `--config ./config.py`.

### Problem: Stage-3 cannot find Stage-1 artifacts

Fix:
- check that Stage-1 finished successfully;
- pass the exact manifest path using `--stage1-manifest`.

### Problem: Stage-5 cannot run

Cause:
- only one city dataset is available.

Fix:
- add the second full-city dataset to `data/`;
- then re-run Stage-5 with `--city-a` and `--city-b`.

---

## 15. Citation

If you use or discuss this capsule, please cite the manuscript as:

```bibtex
@unpublished{kouadio_geopriorsubsnet_nature_2025,
  author  = {Kouadio, Kouao Laurent and Liu, Rong and Jiang, Shiyu and
             Liu, Zhuo and Kouamelan, Serge and Liu, Wenxiang and
             Qing, Zhanhui and Zheng, Zhiwen},
  title   = {Physics-Informed Deep Learning Reveals Divergent Urban Land Subsidence Regimes},
  journal = {Nature Communications},
  note    = {Submitted},
  year    = {2025}
}
```

---

## 16. Summary

This Code Ocean capsule is set up so that reviewers can:

- install `geoprior-v3` from PyPI,
- use the packaged CLI,
- point GeoPrior to the shipped root `config.py`,
- run Stage-1 and Stage-3 as the primary reproducibility checks,
- optionally run a short 5-epoch Stage-2 demo,
- optionally run Stage-4 inference,
- optionally run Stage-5 transfer if both Zhongshan and Nansha full-city datasets are present.

For most reviewers, the two most informative commands are:

```bash
geoprior run stage1-preprocess --config ./config.py
geoprior run stage3-train \
  --config ./config.py \
  --stage1-manifest results/zhongshan_GeoPriorSubsNet_stage1/manifest.json \
  --set EPOCHS=5
```
