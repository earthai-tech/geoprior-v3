import numpy as np
import pandas as pd

# -------------------------------------------------
# paths
# -------------------------------------------------
nansha_path = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\nansha_03_02_proc.csv"
zhongshan_path = r"E:\nature\results\zhongshan_GeoPriorSubsNet_stage1\zhongshan_03_02_proc.csv"
borehole_path = r"D:\projects\geoprior-v3\nat.com\borehole_pumping_validation_summary.csv"

# -------------------------------------------------
# load
# -------------------------------------------------
nansha = pd.read_csv(nansha_path)
zhongshan = pd.read_csv(zhongshan_path)
bh = pd.read_csv(borehole_path)

# -------------------------------------------------
# keep only coordinates
# -------------------------------------------------
nxy = nansha[["x_m", "y_m"]].dropna().to_numpy(dtype=float)
zxy = zhongshan[["x_m", "y_m"]].dropna().to_numpy(dtype=float)


# -------------------------------------------------
# helper: nearest distance from one point to a city cloud
# -------------------------------------------------
def nearest_distance(
    point_xy: np.ndarray, city_xy: np.ndarray
) -> float:
    d2 = np.sum((city_xy - point_xy) ** 2, axis=1)
    return float(np.sqrt(d2.min()))


# -------------------------------------------------
# classify each borehole
# -------------------------------------------------
rows = []
for _, r in bh.iterrows():
    pt = np.array([float(r["x"]), float(r["y"])], dtype=float)

    d_nansha = nearest_distance(pt, nxy)
    d_zhongshan = nearest_distance(pt, zxy)

    if d_nansha < d_zhongshan:
        city = "nansha"
    elif d_zhongshan < d_nansha:
        city = "zhongshan"
    else:
        city = "tie"

    out = dict(r)
    out["dist_to_nansha_m"] = d_nansha
    out["dist_to_zhongshan_m"] = d_zhongshan
    out["assigned_city"] = city
    rows.append(out)

classified = pd.DataFrame(rows)

print("\nClassified boreholes\n")
print(
    classified[
        [
            "well_id",
            "x",
            "y",
            "dist_to_nansha_m",
            "dist_to_zhongshan_m",
            "assigned_city",
        ]
    ].to_string(index=False)
)

# -------------------------------------------------
# split into two files
# -------------------------------------------------
nansha_bh = classified[
    classified["assigned_city"] == "nansha"
].copy()
zhongshan_bh = classified[
    classified["assigned_city"] == "zhongshan"
].copy()

out_dir = r"D:\projects\geoprior-v3\nat.com"

nansha_out = rf"{out_dir}\boreholes_nansha.csv"
zhongshan_out = rf"{out_dir}\boreholes_zhongshan.csv"
classified_out = rf"{out_dir}\boreholes_classified.csv"

classified.to_csv(classified_out, index=False)
nansha_bh.to_csv(nansha_out, index=False)
zhongshan_bh.to_csv(zhongshan_out, index=False)

print("\nSaved:")
print(classified_out)
print(nansha_out)
print(zhongshan_out)
