"""
Stage-2 training curves and physics-aware learning dynamics
===========================================================

This lesson teaches how to read **Stage-2 training curves** in GeoPrior.

Why this page matters
---------------------
A model can finish training and still be hard to trust.

Typical Stage-2 questions include:

- Did the model actually converge?
- Did validation improve together with training,
  or did overfitting appear early?
- Are the physics terms stabilizing,
  exploding, or staying inactive?
- Did the model reduce data loss by violating
  the physics constraints?
- Is the chosen stopping point scientifically
  reasonable, not only numerically good?

In GeoPrior, Stage-2 is the public workflow stage for
**training**. The real Stage-2 pipeline writes a CSV
training log, stores a training summary with the best epoch,
and exports grouped history plots for losses and metrics.

What this lesson teaches
------------------------
We will:

1. build a compact synthetic Stage-2 training-history table,
2. inspect the structure of one training log,
3. identify the best validation epoch,
4. plot total and validation loss,
5. compare data loss against physics loss,
6. inspect individual physics components,
7. track prediction metrics over epochs,
8. explain how to read Stage-2 curves responsibly.

This page uses synthetic data so it remains fully executable
during the documentation build, but the lesson structure
matches the real purpose of Stage-2 in GeoPrior:
**physics-guided model training**.
"""

# %%
# Imports
# -------
# We use pandas + matplotlib because this lesson is about reading
# Stage-2 outputs clearly during the documentation build.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Step 1 - Build a compact synthetic Stage-2 history table
# --------------------------------------------------------
# The real Stage-2 workflow saves a CSV training log and a training
# summary JSON. Here we build a synthetic history DataFrame with
# columns that resemble the real training artifacts:
#
# - total loss and validation loss,
# - data loss versus physics loss,
# - individual physics-loss components,
# - predictive metrics such as subsidence and GWL MAE,
# - a few operational schedule variables.

rng = np.random.default_rng(17)

epochs = np.arange(1, 61)

total_loss = []
val_loss = []
data_loss = []
physics_loss = []
physics_loss_scaled = []

consolidation_loss = []
gw_flow_loss = []
prior_loss = []
smooth_loss = []
bounds_loss = []

subs_pred_mae = []
gwl_pred_mae = []
lambda_offset = []
physics_mult = []

for epoch in epochs:
    e = float(epoch)

    # A simple warmup + convergence pattern:
    # - data loss falls quickly,
    # - physics losses fall more slowly,
    # - validation improves, then plateaus slightly.
    data = (
        0.92 * np.exp(-e / 12.0)
        + 0.18
        + rng.normal(0.0, 0.010)
    )

    cons = (
        0.30 * np.exp(-e / 18.0)
        + 0.030
        + rng.normal(0.0, 0.004)
    )

    gw = (
        0.18 * np.exp(-e / 15.0)
        + 0.020
        + rng.normal(0.0, 0.003)
    )

    prior = (
        0.12 * np.exp(-e / 22.0)
        + 0.018
        + rng.normal(0.0, 0.002)
    )

    smooth = (
        0.040 * np.exp(-e / 24.0)
        + 0.006
        + rng.normal(0.0, 0.001)
    )

    bounds = max(
        0.0,
        0.012 * np.exp(-e / 10.0)
        + rng.normal(0.0, 0.0008),
    )

    phys_raw = cons + gw + prior + smooth + bounds
    phys_scaled = 0.55 * phys_raw

    total = data + phys_scaled

    # Validation:
    # early improvement, then a shallow minimum, then mild flattening.
    val = (
        1.05 * np.exp(-e / 13.5)
        + 0.23
        + 0.0022 * max(0.0, e - 34.0)
        + rng.normal(0.0, 0.012)
    )

    # Prediction metrics
    subs_mae = (
        19.0 * np.exp(-e / 14.0)
        + 5.2
        + rng.normal(0.0, 0.18)
    )
    gwl_mae = (
        8.5 * np.exp(-e / 13.0)
        + 1.9
        + rng.normal(0.0, 0.10)
    )

    # A simple schedule-like variable
    if epoch <= 8:
        lam_off = 0.0
    elif epoch <= 18:
        lam_off = 0.12 * (epoch - 8) / 10.0
    else:
        lam_off = 0.12

    phys_mult = min(1.0, epoch / 12.0)

    total_loss.append(total)
    val_loss.append(val)
    data_loss.append(data)
    physics_loss.append(phys_raw)
    physics_loss_scaled.append(phys_scaled)

    consolidation_loss.append(cons)
    gw_flow_loss.append(gw)
    prior_loss.append(prior)
    smooth_loss.append(smooth)
    bounds_loss.append(bounds)

    subs_pred_mae.append(subs_mae)
    gwl_pred_mae.append(gwl_mae)
    lambda_offset.append(lam_off)
    physics_mult.append(phys_mult)

history_df = pd.DataFrame(
    {
        "epoch": epochs,
        "total_loss": total_loss,
        "val_loss": val_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "physics_loss_scaled": physics_loss_scaled,
        "consolidation_loss": consolidation_loss,
        "gw_flow_loss": gw_flow_loss,
        "prior_loss": prior_loss,
        "smooth_loss": smooth_loss,
        "bounds_loss": bounds_loss,
        "subs_pred_mae": subs_pred_mae,
        "gwl_pred_mae": gwl_pred_mae,
        "lambda_offset": lambda_offset,
        "physics_mult": physics_mult,
    }
)

print("Synthetic Stage-2 history shape:", history_df.shape)
print("")
print(history_df.head(10).to_string(index=False))

# %%
# Step 2 - Identify the best validation epoch
# -------------------------------------------
# The real Stage-2 pipeline records the best epoch and a summary of the
# metrics at that epoch. That is the first thing a user should inspect
# after training finishes.

best_row = history_df.loc[history_df["val_loss"].idxmin()].to_dict()
best_epoch = int(best_row["epoch"])

print("")
print("Best validation epoch")
for key in [
    "epoch",
    "val_loss",
    "total_loss",
    "data_loss",
    "physics_loss_scaled",
    "subs_pred_mae",
    "gwl_pred_mae",
]:
    print(f"{key}: {best_row[key]}")

# %%
# Why this step matters
# ---------------------
# The final epoch is not always the best epoch.
#
# In a Stage-2 run, the best validation epoch is usually more important
# than the last logged epoch because:
#
# - validation may plateau,
# - overfitting may begin late,
# - the best checkpoint is often restored from earlier training.

# %%
# Step 3 - Plot total loss and validation loss
# --------------------------------------------
# This is the first diagnostic view most users should inspect.

fig, ax = plt.subplots(figsize=(8.6, 4.8))

ax.plot(
    history_df["epoch"],
    history_df["total_loss"],
    label="Training total loss",
)
ax.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation loss",
)

ax.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best epoch = {best_epoch}",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Stage-2 total and validation loss")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read the loss plot
# -------------------------
# The two main questions are:
#
# - does validation improve with training,
# - and does it level off sensibly?
#
# Useful signs include:
#
# - steady joint descent:
#   training is progressing in a stable way;
# - falling training loss but flat validation:
#   diminishing returns or early overfitting;
# - noisy validation with no stable minimum:
#   brittle optimization or weak signal.

# %%
# Step 4 - Compare data loss against physics loss
# -----------------------------------------------
# A distinctive part of GeoPrior Stage-2 is that training is not only
# about fitting the data. The physics terms matter too.

fig, ax = plt.subplots(figsize=(8.6, 4.8))

ax.plot(
    history_df["epoch"],
    history_df["data_loss"],
    label="Data loss",
)
ax.plot(
    history_df["epoch"],
    history_df["physics_loss_scaled"],
    label="Physics loss (scaled)",
)
ax.plot(
    history_df["epoch"],
    history_df["physics_loss"],
    label="Physics loss (raw)",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss component")
ax.set_title("Stage-2 data versus physics loss")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read the data-versus-physics panel
# -----------------------------------------
# This panel helps answer one of the most important scientific
# questions in Stage-2:
#
# - is the model improving by learning both the data and the physics,
# - or only by reducing the data term?
#
# A healthy pattern is:
#
# - data loss decreases,
# - physics loss also decreases or stabilizes at a small value,
# - neither term dominates pathologically.
#
# A worrying pattern would be:
#
# - data loss falls strongly,
# - physics loss stays high or grows,
# - suggesting the model is fitting observations while violating the
#   intended constraints.

# %%
# Step 5 - Inspect the individual physics components
# --------------------------------------------------
# Looking only at one aggregated physics loss can hide where problems
# come from.

fig, ax = plt.subplots(figsize=(8.8, 5.0))

ax.plot(
    history_df["epoch"],
    history_df["consolidation_loss"],
    label="consolidation_loss",
)
ax.plot(
    history_df["epoch"],
    history_df["gw_flow_loss"],
    label="gw_flow_loss",
)
ax.plot(
    history_df["epoch"],
    history_df["prior_loss"],
    label="prior_loss",
)
ax.plot(
    history_df["epoch"],
    history_df["smooth_loss"],
    label="smooth_loss",
)
ax.plot(
    history_df["epoch"],
    history_df["bounds_loss"],
    label="bounds_loss",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Physics component")
ax.set_title("Stage-2 physics-loss components")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read the physics-component plot
# --------------------------------------
# This panel is useful because different components mean different
# things.
#
# For example:
#
# - high consolidation loss may indicate the dynamic balance is still
#   inconsistent;
# - high groundwater-flow loss may suggest the head pathway is not yet
#   coherent;
# - persistent bounds loss may suggest the model keeps pushing against
#   admissible parameter ranges.
#
# The point is not that every component must go to zero, but that the
# final pattern should remain interpretable and controlled.

# %%
# Step 6 - Track predictive metrics over epochs
# ---------------------------------------------
# Loss alone is not always intuitive. Prediction metrics are easier for
# many users to interpret directly.

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

axes[0].plot(
    history_df["epoch"],
    history_df["subs_pred_mae"],
)
axes[0].axvline(best_epoch, linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("subs_pred_mae")
axes[0].set_title("Subsidence MAE over training")
axes[0].grid(True, linestyle=":", alpha=0.6)

axes[1].plot(
    history_df["epoch"],
    history_df["gwl_pred_mae"],
)
axes[1].axvline(best_epoch, linestyle="--")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("gwl_pred_mae")
axes[1].set_title("Groundwater MAE over training")
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %%
# Why this metric view matters
# ----------------------------
# A model can have complicated internal loss behavior but still show a
# clean predictive trend.
#
# These metric curves help answer:
#
# - are practical forecast errors improving,
# - and do they stabilize near the same epoch as validation loss?
#
# When the metric minimum and validation minimum disagree strongly,
# users should inspect the run more carefully.

# %%
# Step 7 - Inspect schedule-like controls
# ---------------------------------------
# The Stage-2 workflow can use physics-related schedules. A small
# schedule panel often makes the training story easier to understand.

fig, ax = plt.subplots(figsize=(8.4, 4.5))

ax.plot(
    history_df["epoch"],
    history_df["lambda_offset"],
    label="lambda_offset",
)
ax.plot(
    history_df["epoch"],
    history_df["physics_mult"],
    label="physics_mult",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Schedule value")
ax.set_title("Stage-2 schedule controls")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read the schedule panel
# ------------------------------
# This kind of panel helps explain *why* the earlier loss curves look
# the way they do.
#
# For example:
#
# - a warmup phase can delay full physics pressure,
# - a ramp can make the physics loss appear to "turn on" gradually,
# - an offset schedule can explain a small inflection in total loss.
#
# This is especially helpful when a training curve changes character
# partway through the run.

# %%
# Step 8 - Compare the best epoch to the final epoch
# --------------------------------------------------
# A compact comparison table is often more useful than another figure.

final_row = history_df.iloc[-1].to_dict()

compare_df = pd.DataFrame(
    [
        {
            "checkpoint": "best_epoch",
            "epoch": int(best_row["epoch"]),
            "val_loss": float(best_row["val_loss"]),
            "total_loss": float(best_row["total_loss"]),
            "data_loss": float(best_row["data_loss"]),
            "physics_loss_scaled": float(
                best_row["physics_loss_scaled"]
            ),
            "subs_pred_mae": float(best_row["subs_pred_mae"]),
            "gwl_pred_mae": float(best_row["gwl_pred_mae"]),
        },
        {
            "checkpoint": "final_epoch",
            "epoch": int(final_row["epoch"]),
            "val_loss": float(final_row["val_loss"]),
            "total_loss": float(final_row["total_loss"]),
            "data_loss": float(final_row["data_loss"]),
            "physics_loss_scaled": float(
                final_row["physics_loss_scaled"]
            ),
            "subs_pred_mae": float(final_row["subs_pred_mae"]),
            "gwl_pred_mae": float(final_row["gwl_pred_mae"]),
        },
    ]
)

print("")
print("Best-versus-final comparison")
print(compare_df.to_string(index=False))

# %%
# Why this table matters
# ----------------------
# Users should not assume the final epoch is the one they want.
#
# This comparison makes it easy to ask:
#
# - is the final epoch essentially the same as the best one,
# - or did validation degrade after the best checkpoint?
#
# If the difference is tiny, the run looks stable.
# If the difference is large, the user should trust the best checkpoint
# more than the final logged state.

# %%
# Step 9 - Build a compact decision summary
# -----------------------------------------
# The final step should help the user decide what to do next.

val_min = float(history_df["val_loss"].min())
val_last = float(history_df["val_loss"].iloc[-1])
subs_best = float(compare_df.loc[0, "subs_pred_mae"])
subs_last = float(compare_df.loc[1, "subs_pred_mae"])

decision_summary = pd.DataFrame(
    [
        {"item": "best_epoch", "value": best_epoch},
        {"item": "best_val_loss", "value": val_min},
        {"item": "final_val_loss", "value": val_last},
        {
            "item": "val_gap_final_minus_best",
            "value": val_last - val_min,
        },
        {"item": "best_subs_pred_mae", "value": subs_best},
        {"item": "final_subs_pred_mae", "value": subs_last},
        {
            "item": "physics_loss_scaled_at_best",
            "value": float(best_row["physics_loss_scaled"]),
        },
    ]
)

print("")
print("Training decision summary")
print(decision_summary.to_string(index=False))

# %%
# How to interpret the summary
# ----------------------------
# ``val_gap_final_minus_best``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If this is tiny, the run remained stable after the best epoch.
# If it is larger, the model may have passed its useful stopping point.
#
# ``physics_loss_scaled_at_best``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This helps check whether the best checkpoint is also scientifically
# reasonable, not only numerically strong.
#
# ``best_subs_pred_mae`` versus ``final_subs_pred_mae``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This shows whether practical predictive quality stayed stable after
# the best validation point.

# %%
# Step 10 - Practical reading guide
# ---------------------------------
# A sensible Stage-2 interpretation sequence is:
#
# 1. inspect total loss and validation loss,
# 2. inspect data loss versus physics loss,
# 3. inspect the individual physics components,
# 4. inspect predictive metrics,
# 5. compare the best epoch to the final epoch,
# 6. then decide whether the run should be accepted,
#    repeated, or tuned further.
#
# That sequence helps prevent a common mistake:
# trusting a good validation minimum without checking whether the
# physics behavior stayed coherent.

# %%
# Final takeaway
# --------------
# Stage-2 training curves are not only optimization plots.
#
# They are diagnostics of:
#
# - convergence,
# - overfitting,
# - physics consistency,
# - predictive quality,
# - and the scientific credibility of the chosen checkpoint.
#
# That is why the diagnostics gallery should include a dedicated
# Stage-2 training-curves lesson.