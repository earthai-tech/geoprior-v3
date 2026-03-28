# Sensitivity & Ablation Utilities (Supplement S6)

This folder provides two companion scripts to **summarize** and **visualize**
your ablation / sensitivity runs recorded in:

```
<job_root>/ablation_records/ablation_record.jsonl
```

They are designed for **GeoPrior-v3 (v3.2+)** and follow the same conventions
as other `scripts/` utilities (paper style, robust scanning, consistent outputs).

---

## 1) What each script is for

### A) `build-ablation-table`
Builds **tidy tables** (CSV/JSON/TXT) and **paper-ready LaTeX tables** from one
or many `ablation_record*.jsonl` files.

Typical use cases:
- Create **Table S6**: a grid over **(λ_cons × λ_prior)** for MAE/Coverage/Sharpness.
- Create **Table S7**: an ablations/toggles summary table.
- Export a unified CSV for downstream plotting or QA.

### B) `plot-ablations-sensitivity`
Creates **Nature-friendly figures** from the same records.

Typical use cases:
- Heatmaps / tricontours over **(λ_cons × λ_prior)** for MAE, Coverage, Sharpness.
- **Single figure** with multiple metrics (2 cities × M metrics).
- **Pareto trade-off** plots (MAE vs Sharpness colored by Coverage), optional
  Pareto front and density overlay.

---

## 2) Inputs supported

Both scripts accept any of the following as input:

- A **results root** (scan all runs under it):
  ```
  results/
  ```
- A **city root**:
  ```
  results/nansha_GeoPriorSubsNet_stage1/
  ```
- A **job root** (e.g. `sens__.../` or `train_.../`):
  ```
  results/nansha_GeoPriorSubsNet_stage1/sens__pde_both__lc_0__lp_0p1/
  ```
- The **ablation_records folder**:
  ```
  .../ablation_records/
  ```
- A direct **ablation jsonl file**:
  ```
  .../ablation_records/ablation_record.jsonl
  ```

### Record precedence (v3.2+)
- If both exist, the scripts prefer `ablation_record.updated*.jsonl`.
- If duplicates exist, records are deduped (updated wins).

---

## 3) Build tables: `build-ablation-table`

### 3.1 Basic usage (all cities)
```bash
python -m scripts build-ablation-table \
  --root results \
  --out table_ablations_all \
  --formats csv,json
```

### 3.2 One city or one job
```bash
python -m scripts build-ablation-table \
  --root results/nansha_GeoPriorSubsNet_stage1 \
  --out table_ablations_nansha \
  --formats csv,json
```

```bash
python -m scripts build-ablation-table \
  --root results/.../sens__pde_both__lc_0__lp_0p1 \
  --out table_ablations_one_run \
  --formats csv,txt
```

### 3.3 Paper mode (minimal, publication-oriented)
Paper mode produces compact tables focused on:
- deterministic error (MAE or RMSE or MSE),
- Coverage@80,
- Sharpness@80,
with units formatted for paper.

```bash
python -m scripts build-ablation-table \
  --root results \
  --for-paper \
  --err-metric rmse \
  --metric-unit mm \
  --formats csv,tex \
  --sideway \
  --out tableS6_ablations_paper
```

Notes:
- Use `--keep-r2` if you still want R² in paper tables.
- Unit conversion is applied when possible (e.g., `m → mm`, `m² → mm²`).

---

### 3.4 Grouped outputs (S6 + S7)

#### Table S6: λ_cons × λ_prior grids
```bash
python -m scripts build-ablation-table \
  --root results \
  --for-paper \
  --group-cols s6 \
  --s6-metrics mae,coverage80,sharpness80 \
  --s6-agg mean \
  --formats tex,csv \
  --sideway \
  --out tableS6_lambda_sensitivity
```

#### Table S7: ablation toggles summary
```bash
python -m scripts build-ablation-table \
  --root results \
  --for-paper \
  --group-cols s7 \
  --formats tex,csv \
  --out tableS7_ablations
```

#### Both
```bash
python -m scripts build-ablation-table \
  --root results \
  --for-paper \
  --group-cols all \
  --s6-metrics mae,coverage80,sharpness80 \
  --formats tex,csv,json \
  --sideway \
  --out tablesS6S7
```

---

## 4) Plot figures: `plot-ablations-sensitivity`

### 4.1 Single metric heatmap (legacy layout)
Default layout is:
- top row: bars vs λ_prior (physics off/on),
- bottom row: map for physics-on.

```bash
python -m scripts plot-ablations-sensitivity \
  --root results \
  --city-a Nansha --city-b Zhongshan \
  --bar-metric mae \
  --heatmap-metric mae \
  -o scripts/figs/supp_fig_S6_lambda_mae
```

### 4.2 One figure with multiple metrics (recommended for SI)
This produces a **2 × M grid**:
- rows: cities,
- columns: metrics.

Use `tricontour` when grids have missing cells (best looking).

```bash
python -m scripts plot-ablations-sensitivity \
  --root results \
  --city-a Nansha --city-b Zhongshan \
  --heatmap-metrics mae,coverage80,sharpness80 \
  --no-bars true \
  --map-kind tricontour \
  --levels 14 \
  --contour-lines true \
  --cmap cividis \
  --mark-lambda-cons 1.0 \
  --mark-lambda-prior 0.2 \
  -o scripts/figs/supp_fig_S6_lambda_all
```

Useful switches:
- `--cmap <name>`: change colormap (e.g., `viridis`, `magma`, `cividis`).
- `--align-grid true`: align ticks across cities/metrics (default true).
- `--no-colorbar true`: hide colorbars if needed.

---

### 4.3 Pareto trade-off view (Nature-friendly)
Plots:
- x: MAE
- y: Sharpness@80
- color: Coverage@80
Optionally overlays the **non-dominated front**.

```bash
python -m scripts plot-ablations-sensitivity \
  --root results \
  --city-a Nansha --city-b Zhongshan \
  --pareto true \
  --pareto-x mae \
  --pareto-y sharpness80 \
  --pareto-color coverage80 \
  --pareto-front true \
  --cmap cividis \
  --mark-lambda-cons 1.0 \
  --mark-lambda-prior 0.2 \
  -o scripts/figs/supp_fig_S6_pareto
```

#### Add density overlay (useful when many runs overlap)
```bash
python -m scripts plot-ablations-sensitivity \
  --root results \
  --city-a Nansha --city-b Zhongshan \
  --pareto true \
  --pareto-front true \
  --pareto-density true \
  --pareto-density-bins log \
  --pareto-density-gridsize 40 \
  --pareto-density-alpha 0.30 \
  --pareto-density-cmap Greys_r \
  --cmap cividis \
  -o scripts/figs/supp_fig_S6_pareto_density
```

---

## 5) Recommended “Nature SI” workflow

### Minimal, reviewer-proof package
1) **Figure S6 (maps)**: one figure with MAE/Coverage/Sharpness (2 cities × 3 metrics).
2) **Table S6 (grids)**: per-city λ_cons × λ_prior numeric tables for the same metrics.
3) **Optional**: Pareto scatter as a compact “trade-off story”.

### Typical commands
```bash
# Tables
python -m scripts build-ablation-table \
  --root results \
  --for-paper \
  --group-cols s6 \
  --s6-metrics mae,coverage80,sharpness80 \
  --formats tex,csv \
  --sideway \
  --out tableS6_lambda_sensitivity

# Figures (multi-metric maps)
python -m scripts plot-ablations-sensitivity \
  --root results \
  --city-a Nansha --city-b Zhongshan \
  --heatmap-metrics mae,coverage80,sharpness80 \
  --no-bars true \
  --map-kind tricontour \
  --levels 14 \
  --contour-lines true \
  --cmap cividis \
  --mark-lambda-cons 1.0 \
  --mark-lambda-prior 0.2 \
  -o scripts/figs/supp_fig_S6_lambda_all
```

---

## 6) Outputs

### `build-ablation-table`
Writes to `scripts/out/` by default:
- `*.csv`, `*.json`, `*.txt`
- `*.tex` (paper tables; optional `sidewaystable` wrapper)

### `plot-ablations-sensitivity`
Writes to your chosen `-o` path, producing:
- `<stem>.pdf` and `<stem>.png`
- a tidy CSV next to the plot folder (for traceability), e.g.:
  - `tableS6_ablations_used.csv`

---

## 7) Tips / troubleshooting

- If you see no density overlay in Pareto mode:
  - you may have too few points (hexbin is most useful with many runs), or
  - log binning may compress counts; try:
    `--pareto-density-bins linear` and increase `--pareto-density-alpha`.

- For incomplete λ-grids (missing cells), prefer:
  - `--map-kind tricontour` (best) or `--map-kind contour` (if grid is full).

- To mark the chosen operating point (e.g., for the paper default):
  - `--mark-lambda-cons 1.0 --mark-lambda-prior 0.2`

---

## 8) Help
Run:
```bash
python -m scripts build-ablation-table -h
python -m scripts plot-ablations-sensitivity -h
```
