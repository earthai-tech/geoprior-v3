"""
Physics diagnostics bridge: from ``evaluate_physics`` to payload inspection
=============================================================================

This lesson bridges two parts of the GeoPrior documentation:

- the **diagnostics** section, where we ask whether training and
  evaluation behave sensibly,
- and the later **physics / model-inspection** sections, where we
  inspect learned fields and mechanistic structure in more detail.

Why this page matters
---------------------
Stage-2 training curves can tell us that optimization was stable, but
they do not by themselves explain what the physics-informed part of the
model actually learned.

That is why GeoPrior exposes a second diagnostic path built around:

- :meth:`GeoPriorSubsNet.evaluate_physics`
- :func:`gather_physics_payload`
- :meth:`GeoPriorSubsNet.export_physics_payload`
- :func:`identifiability_diagnostics_from_payload`

The first method evaluates physics consistency directly from model
inputs. The payload collector then turns those physics diagnostics into
**flat 1D arrays** suitable for scatter plots, histograms, and summary
statistics. The export/load helpers persist the payload for later
inspection and figure generation. 


What the real helpers do
------------------------
``evaluate_physics(..., return_maps=True)`` can expose:

- scalar physics diagnostics such as:
  ``loss_cons``, ``loss_gw``, ``loss_prior``,
  ``epsilon_cons``, ``epsilon_gw``, ``epsilon_prior``
- and optional last-batch maps such as:
  ``R_prior``, ``R_cons``, ``R_gw``, ``K``, ``Ss``, ``tau``,
  ``tau_prior`` / ``tau_closure`` and thickness fields. 

``gather_physics_payload(...)`` then flattens those maps across batches
into arrays such as:

- ``tau``
- ``tau_prior`` / ``tau_closure``
- ``K``
- ``Ss``
- ``Hd`` (and optionally ``H``)
- ``cons_res_vals``
- ``log10_tau``
- ``log10_tau_prior``

and also computes summary metrics like:

- ``eps_prior_rms``
- ``r2_logtau``

with optional closure-consistency metrics when the model exposes a
usable scalar kappa value. 

What this lesson teaches
------------------------
We will:

1. build a compact synthetic physics payload,
2. mimic the payload structure returned by the real collector,
3. inspect the summary metrics and provenance metadata,
4. plot the main bridge diagnostics:
   - timescale consistency,
   - field distributions,
   - consolidation residual spread,
5. compute identifiability-style summaries from the payload.

This page uses a synthetic payload so it remains fully executable
during the documentation build, but the key names and interpretation
are grounded in the real GeoPrior helper contracts. 
"""

# %%
# Imports
# -------
# We use the real identifiability payload helper, and we mimic the
# flat payload layout returned by the real collector.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geoprior.models.subsidence.payloads import (
    identifiability_diagnostics_from_payload,
)

# %%
# Step 1 - Build a compact synthetic physics payload
# --------------------------------------------------
# The real payload collector flattens batch maps into 1D arrays. We
# mimic that exact style here.
#
# The payload keys are chosen to match the documented contract of
# ``gather_physics_payload(...)``. 

rng = np.random.default_rng(29)

nx = 18
ny = 12

xv = np.linspace(0.0, 12_000.0, nx)
yv = np.linspace(0.0, 8_000.0, ny)
X, Y = np.meshgrid(xv, yv)

coord_x = X.ravel()
coord_y = Y.ravel()
n = coord_x.size

xn = (coord_x - coord_x.min()) / (coord_x.max() - coord_x.min())
yn = (coord_y - coord_y.min()) / (coord_y.max() - coord_y.min())

hotspot = np.exp(
    -(
        ((xn - 0.72) ** 2) / 0.022
        + ((yn - 0.34) ** 2) / 0.032
    )
)
ridge = 0.55 * np.exp(
    -(
        ((xn - 0.28) ** 2) / 0.030
        + ((yn - 0.74) ** 2) / 0.050
    )
)
gradient = 0.40 * xn + 0.22 * (1.0 - yn)

# Physical fields (strictly positive)
K = 10 ** (
    -6.1
    + 0.45 * hotspot
    + 0.10 * gradient
    + rng.normal(0.0, 0.05, size=n)
)
Ss = 10 ** (
    -4.7
    + 0.22 * ridge
    + 0.08 * gradient
    + rng.normal(0.0, 0.04, size=n)
)
Hd = (
    18.0
    + 7.0 * gradient
    + 3.0 * ridge
    + rng.normal(0.0, 0.6, size=n)
)
H = Hd * 1.35

# Effective timescales
kappa_b = 1.0
tau_prior = (Hd ** 2) * Ss / (np.pi**2 * kappa_b * K)
tau = tau_prior * np.exp(rng.normal(0.0, 0.22, size=n))

# Residual-like values
cons_res_vals = rng.normal(0.0, 0.018, size=n)
cons_res_scaled = cons_res_vals / np.maximum(
    np.std(cons_res_vals),
    1e-8,
)
cons_scale = np.full(n, np.std(cons_res_vals))

payload = {
    "tau": tau.astype(np.float32),
    "tau_prior": tau_prior.astype(np.float32),
    "tau_closure": tau_prior.astype(np.float32),
    "K": K.astype(np.float32),
    "Ss": Ss.astype(np.float32),
    "Hd": Hd.astype(np.float32),
    "H": H.astype(np.float32),
    "cons_res_vals": cons_res_vals.astype(np.float32),
    "cons_res_scaled": cons_res_scaled.astype(np.float32),
    "cons_scale": cons_scale.astype(np.float32),
    "log10_tau": np.log10(np.clip(tau, 1e-12, None)).astype(np.float32),
    "log10_tau_prior": np.log10(
        np.clip(tau_prior, 1e-12, None)
    ).astype(np.float32),
}

payload["metrics"] = {
    "eps_prior_rms": float(
        np.sqrt(
            np.mean(
                (
                    np.log(np.clip(payload["tau"], 1e-12, None))
                    - np.log(np.clip(payload["tau_prior"], 1e-12, None))
                )
                ** 2
            )
        )
    ),
    "r2_logtau": float(
        np.corrcoef(
            payload["log10_tau_prior"],
            payload["log10_tau"],
        )[0, 1]
        ** 2
    ),
    "eps_cons_scaled_rms": float(
        np.sqrt(np.mean(payload["cons_res_scaled"] ** 2))
    ),
}

print("Payload keys:")
print(sorted(payload.keys()))

print("")
print("Payload metrics")
print(payload["metrics"])

# %%
# Step 2 - Build lightweight provenance metadata
# ----------------------------------------------
# The real export path pairs the payload with metadata from
# ``default_meta_from_model(...)``. That metadata records the model
# name, active PDE modes, kappa interpretation, unit conventions, and
# the tau-closure formula used for interpretation. 

meta = {
    "model_name": "GeoPriorSubsNet",
    "pde_modes_active": ["consolidation", "gw_flow"],
    "kappa_mode": "kb",
    "kappa_value": 1.0,
    "time_units": "year",
    "time_coord_units": "year",
    "units": {
        "H": "m",
        "Hd": "m",
        "Ss": "1/m",
        "K": "m/s",
        "tau": "s",
        "tau_prior": "s",
        "cons_res_vals": "m/s",
        "kappa": "dimensionless",
    },
    "tau_closure_formula": "Hd^2 * Ss / (pi^2 * kappa_b * K)",
}

print("")
print("Metadata")
for k, v in meta.items():
    print(f"{k}: {v}")

# %%
# Why metadata matters
# --------------------
# Physics payloads are easy to misread if the units or closure
# conventions are unclear.
#
# This is why the real export helper saves lightweight provenance along
# with the payload, rather than only the raw arrays. 

# %%
# Step 3 - Build a flat inspection table
# --------------------------------------
# In practice, users often move from the payload dict to a DataFrame for
# quick inspection and plotting.

diag_df = pd.DataFrame(
    {
        "coord_x": coord_x,
        "coord_y": coord_y,
        "tau": payload["tau"],
        "tau_prior": payload["tau_prior"],
        "K": payload["K"],
        "Ss": payload["Ss"],
        "Hd": payload["Hd"],
        "cons_res_vals": payload["cons_res_vals"],
        "log10_tau": payload["log10_tau"],
        "log10_tau_prior": payload["log10_tau_prior"],
    }
)

print("")
print("Inspection DataFrame head")
print(diag_df.head(8).to_string(index=False))

# %%
# Step 4 - Plot timescale consistency
# -----------------------------------
# This is the most important bridge figure.
#
# It answers:
#
# - does the learned effective timescale ``tau`` broadly agree with the
#   closure timescale ``tau_prior``?
#
# The payload collector itself already summarizes this with
# ``eps_prior_rms`` and ``r2_logtau``. The scatter plot shows the same
# relationship visually. 

fig, ax = plt.subplots(figsize=(6.8, 6.2))

ax.scatter(
    diag_df["log10_tau_prior"],
    diag_df["log10_tau"],
    s=20,
)

lo = float(
    min(
        diag_df["log10_tau_prior"].min(),
        diag_df["log10_tau"].min(),
    )
)
hi = float(
    max(
        diag_df["log10_tau_prior"].max(),
        diag_df["log10_tau"].max(),
    )
)

ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
ax.set_xlabel("log10(tau_prior)")
ax.set_ylabel("log10(tau)")
ax.set_title("Timescale consistency")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %%
# How to read this plot
# ---------------------
# Points near the diagonal indicate agreement between:
#
# - the learned effective timescale,
# - and the closure-implied timescale.
#
# A broad but coherent cloud is often acceptable.
#
# A strong systematic shift away from the diagonal can indicate:
#
# - closure mismatch,
# - unit mismatch,
# - or a regime where the learned fields satisfy the data but violate
#   the intended timescale structure.

# %%
# Step 5 - Plot field distributions
# ---------------------------------
# The flat payload is ideal for histograms. This is one reason the real
# collector returns 1D arrays rather than only last-batch maps. It is
# explicitly designed for scatter plots, histograms, and summary stats. 

fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1))

axes[0].hist(np.log10(diag_df["K"]), bins=20)
axes[0].set_xlabel("log10(K)")
axes[0].set_ylabel("Count")
axes[0].set_title("Hydraulic conductivity")

axes[1].hist(np.log10(diag_df["Ss"]), bins=20)
axes[1].set_xlabel("log10(Ss)")
axes[1].set_ylabel("Count")
axes[1].set_title("Specific storage")

axes[2].hist(diag_df["Hd"], bins=20)
axes[2].set_xlabel("Hd")
axes[2].set_ylabel("Count")
axes[2].set_title("Drainage thickness")

for ax in axes:
    ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Why these distributions matter
# ------------------------------
# Before users interpret map-level physics figures, they should first
# inspect the raw field distributions.
#
# This helps answer:
#
# - are the learned fields finite and plausible?
# - are they concentrated in a narrow range or strongly spread?
# - do they show suspicious clipping or extreme tails?

# %%
# Step 6 - Plot consolidation residual spread
# -------------------------------------------
# The payload also carries ``cons_res_vals`` and can optionally carry
# scaled residuals. These are useful bridge diagnostics because they
# connect:
#
# - the Stage-2 epsilon / loss curves,
# - and the later spatial physics figures. 

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))

axes[0].hist(diag_df["cons_res_vals"], bins=24)
axes[0].set_xlabel("cons_res_vals")
axes[0].set_ylabel("Count")
axes[0].set_title("Consolidation residual distribution")
axes[0].grid(True, linestyle=":", alpha=0.5)

axes[1].scatter(
    diag_df["coord_x"],
    diag_df["coord_y"],
    c=diag_df["cons_res_vals"],
    s=24,
)
axes[1].set_xlabel("coord_x")
axes[1].set_ylabel("coord_y")
axes[1].set_title("Residual pattern in space")
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# How to read the residual panels
# -------------------------------
# Histogram
# ~~~~~~~~~
# A residual distribution centered near zero with moderate spread is
# usually easier to trust than one with a large shift or extreme tails.
#
# Spatial scatter
# ~~~~~~~~~~~~~~~
# Even without a formal map utility, a simple spatial scatter is enough
# to reveal whether large residuals cluster in one region.
#
# That kind of clustering often motivates the later, more specialized
# physics-map pages.

# %%
# Step 7 - Compute identifiability-style diagnostics
# --------------------------------------------------
# The payload module also provides an explicit helper for synthetic
# identifiability analysis:
# ``identifiability_diagnostics_from_payload(...)``.
#
# It summarizes:
#
# 1. relative error in tau,
# 2. closure log-residual,
# 3. log-offsets of K, Ss, and Hd versus true and prior values. 

tau_true = float(np.median(payload["tau"]))
K_true = float(np.median(payload["K"]))
Ss_true = float(np.median(payload["Ss"]))
Hd_true = float(np.median(payload["Hd"]))

# Slightly different "prior" values for demonstration.
K_prior = K_true * 0.85
Ss_prior = Ss_true * 1.10
Hd_prior = Hd_true * 0.95

ident = identifiability_diagnostics_from_payload(
    payload,
    tau_true=tau_true,
    K_true=K_true,
    Ss_true=Ss_true,
    Hd_true=Hd_true,
    K_prior=K_prior,
    Ss_prior=Ss_prior,
    Hd_prior=Hd_prior,
)

print("")
print("Identifiability diagnostics")
print(ident)

# %%
# Why this matters
# ----------------
# This helper is not just for SM3-style synthetic studies.
#
# In the documentation flow, it is also useful as a bridge because it
# shows how a flat payload can support:
#
# - consistency checks,
# - offset diagnostics,
# - and identifiability-oriented summaries,
#
# before users move to the later, more specialized figure-generation
# pages. 

# %%
# Step 8 - Build a compact summary table
# --------------------------------------
# A diagnostics bridge page should end with a small summary the user
# could plausibly print or export.

summary = pd.DataFrame(
    [
        {
            "metric": "eps_prior_rms",
            "value": float(payload["metrics"]["eps_prior_rms"]),
        },
        {
            "metric": "r2_logtau",
            "value": float(payload["metrics"]["r2_logtau"]),
        },
        {
            "metric": "eps_cons_scaled_rms",
            "value": float(payload["metrics"]["eps_cons_scaled_rms"]),
        },
        {
            "metric": "median_log10_K",
            "value": float(np.median(np.log10(payload["K"]))),
        },
        {
            "metric": "median_log10_Ss",
            "value": float(np.median(np.log10(payload["Ss"]))),
        },
        {
            "metric": "median_Hd",
            "value": float(np.median(payload["Hd"])),
        },
    ]
)

print("")
print("Physics payload summary")
print(summary.to_string(index=False))

# %%
# Step 9 - Practical interpretation guide
# ---------------------------------------
# This bridge page should be read in the following order:
#
# 1. inspect the summary metrics,
# 2. inspect timescale consistency,
# 3. inspect field distributions,
# 4. inspect residual spread,
# 5. then move on to dedicated physics-map or identifiability pages.
#
# That order matters because it helps users distinguish:
#
# - a model that is numerically stable but physically incoherent,
# - from one that is both stable and mechanistically interpretable.

# %%
# Final takeaway
# --------------
# This page sits between diagnostics and the later physics sections for
# a reason.
#
# It shows how to move from:
#
# - Stage-2 style residual and loss summaries,
# - to flat physics payloads,
# - to interpretable mechanistic diagnostics.
#
# In GeoPrior, that bridge is built by:
#
# - ``evaluate_physics(...)``
# - ``gather_physics_payload(...)``
# - ``export_physics_payload(...)``
# - ``identifiability_diagnostics_from_payload(...)``
#
# Understanding that bridge makes the later model-inspection and
# figure-generation pages much easier to read.