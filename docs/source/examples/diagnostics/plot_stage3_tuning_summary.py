"""
Stage-3 tuning summary and best-trial diagnostics
=================================================

This lesson teaches how to read a **Stage-3 hyperparameter tuning
summary** in GeoPrior.

Why this page matters
---------------------
A tuning run can generate many trials, but a user still needs to answer
a small number of practical questions:

- Which trial actually won?
- Was the best score clearly better than the rest, or only marginally?
- Which hyperparameters seem most associated with better performance?
- Did tuning improve steadily, or did the search plateau early?
- Are the best trials cheap, stable, and scientifically reasonable?

In GeoPrior, Stage-3 is the public workflow stage for
**hyperparameter tuning**. The CLI exposes ``stage3-tune`` as the
Stage-3 command, and the CLI package also exposes ``stage3`` as a
public workflow module. 

What this lesson teaches
------------------------
We will:

1. build a compact synthetic tuning-results table,
2. inspect the structure of one Stage-3 trial table,
3. identify the best trial,
4. plot score progression across trials,
5. inspect parameter-versus-score relationships,
6. compare top trials side by side,
7. explain how to read a tuning summary responsibly.

This page uses synthetic data so it is fully executable during the
documentation build, but the lesson logic is aligned with the real
purpose of Stage-3 in GeoPrior: **hyperparameter tuning**. 
"""

# %%
# Imports
# -------
# We use pandas + matplotlib because this lesson is about how to
# interpret tuning artifacts, not about a dedicated public plotting
# helper.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Step 1 - Build a compact synthetic tuning-results table
# -------------------------------------------------------
# Each row represents one hyperparameter trial.
#
# The table includes:
#
# - trial identifier,
# - objective score,
# - representative hyperparameters,
# - some operational columns that help explain the run.
#
# We define score so that **lower is better**, consistent with a
# validation-loss-style objective.

rng = np.random.default_rng(23)

n_trials = 36
trial_ids = np.arange(1, n_trials + 1)

hidden_units_choices = np.array([32, 48, 64, 96, 128])
dropout_choices = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
batch_choices = np.array([32, 64, 96])
lr_choices = np.array([3e-4, 5e-4, 8e-4, 1e-3, 2e-3])
lambda_cons_choices = np.array([0.01, 0.03, 0.05, 0.10, 0.20])
lambda_prior_choices = np.array([0.01, 0.03, 0.05, 0.10, 0.20])

rows: list[dict[str, float | int | str]] = []

for trial_id in trial_ids:
    hidden_units = int(rng.choice(hidden_units_choices))
    dropout_rate = float(rng.choice(dropout_choices))
    batch_size = int(rng.choice(batch_choices))
    learning_rate = float(rng.choice(lr_choices))
    lambda_cons = float(rng.choice(lambda_cons_choices))
    lambda_prior = float(rng.choice(lambda_prior_choices))

    # Synthetic tuning logic:
    # best region is around:
    #   hidden_units ~ 64
    #   dropout ~ 0.10
    #   lr ~ 8e-4 to 1e-3
    #   lambda_cons ~ 0.05 to 0.10
    #   lambda_prior ~ 0.03 to 0.10
    score = 0.58

    score += 0.000020 * (hidden_units - 64) ** 2
    score += 0.90 * abs(dropout_rate - 0.10)
    score += 55.0 * abs(np.log10(learning_rate) - np.log10(8e-4))
    score += 0.80 * abs(lambda_cons - 0.08)
    score += 0.70 * abs(lambda_prior - 0.06)

    # Add mild interaction and noise
    if learning_rate >= 2e-3 and dropout_rate == 0.0:
        score += 0.05
    if hidden_units >= 96 and batch_size == 32:
        score += 0.03

    score += rng.normal(0.0, 0.015)

    # Simulate training cost and stability
    epochs_ran = int(rng.integers(18, 41))
    fit_minutes = (
        4.0
        + 0.03 * hidden_units
        + 0.015 * epochs_ran
        + 0.3 * (batch_size == 32)
        + rng.normal(0.0, 0.25)
    )

    status = "ok"
    if learning_rate >= 2e-3 and lambda_cons >= 0.20:
        status = "unstable"
        score += 0.08

    rows.append(
        {
            "trial_id": int(trial_id),
            "score": float(score),
            "hidden_units": hidden_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "lambda_cons": lambda_cons,
            "lambda_prior": lambda_prior,
            "epochs_ran": epochs_ran,
            "fit_minutes": float(fit_minutes),
            "status": status,
        }
    )

trials_df = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)

print("Tuning-results table shape:", trials_df.shape)
print("")
print(trials_df.head(10).to_string(index=False))

# %%
# Step 2 - Identify the best trial and rank the search
# ----------------------------------------------------
# In a tuning summary, the first useful action is always to rank the
# trials by the optimization objective.

ranked_df = trials_df.sort_values("score", ascending=True).reset_index(drop=True)
best_trial = ranked_df.iloc[0].to_dict()

print("")
print("Best trial")
for key, value in best_trial.items():
    print(f"{key}: {value}")

print("")
print("Top 8 trials")
print(ranked_df.head(8).to_string(index=False))

# %%
# Why this ranking step matters
# -----------------------------
# A tuning summary should always begin with a ranked table because the
# "best trial" is not meaningful in isolation.
#
# Users need to know:
#
# - whether the best score is clearly separated from the others,
# - whether several trials are effectively tied,
# - whether the winner is operationally sensible.

# %%
# Step 3 - Track the incumbent best score over trials
# ---------------------------------------------------
# A very useful Stage-3 diagnostic is the *incumbent best* curve.
#
# It answers:
#
# - did the search keep improving,
# - or did it plateau early?

trials_df["best_so_far"] = trials_df["score"].cummin()

fig, ax = plt.subplots(figsize=(8.6, 4.8))

ax.plot(
    trials_df["trial_id"],
    trials_df["score"],
    marker="o",
    linestyle="",
    label="Trial score",
)
ax.plot(
    trials_df["trial_id"],
    trials_df["best_so_far"],
    label="Best so far",
)

ax.set_xlabel("Trial")
ax.set_ylabel("Objective score (lower is better)")
ax.set_title("Stage-3 search progression")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# How to read the progression plot
# --------------------------------
# The scattered points show the score for each trial.
#
# The "best so far" line shows the search envelope:
#
# - large early drops suggest the search found better regions quickly,
# - long flat plateaus suggest diminishing returns,
# - a late drop may indicate that the search space still contained
#   unexplored strong settings.
#
# This is often the first figure a user should inspect before studying
# the hyperparameters themselves.

# %%
# Step 4 - Inspect parameter-versus-score relationships
# -----------------------------------------------------
# Next we ask:
#
# - which hyperparameters appear associated with better scores?
#
# This is not a causal analysis. It is an interpretation aid.

fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3))

axes[0].scatter(
    trials_df["learning_rate"],
    trials_df["score"],
    s=40,
)
axes[0].set_xlabel("learning_rate")
axes[0].set_ylabel("score")
axes[0].set_title("Score vs learning rate")
axes[0].grid(True, linestyle=":", alpha=0.5)

axes[1].scatter(
    trials_df["dropout_rate"],
    trials_df["score"],
    s=40,
)
axes[1].set_xlabel("dropout_rate")
axes[1].set_ylabel("score")
axes[1].set_title("Score vs dropout")
axes[1].grid(True, linestyle=":", alpha=0.5)

axes[2].scatter(
    trials_df["hidden_units"],
    trials_df["score"],
    s=40,
)
axes[2].set_xlabel("hidden_units")
axes[2].set_ylabel("score")
axes[2].set_title("Score vs hidden units")
axes[2].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# How to read the parameter panels
# --------------------------------
# These plots are meant to reveal *search structure*, not final truth.
#
# Useful signs include:
#
# - a bowl-shaped region:
#   suggests one parameter has a preferred range;
# - a flat spread:
#   suggests weak sensitivity within the explored values;
# - vertical bands with different score spread:
#   suggests interactions with other parameters.
#
# In a real tuning run, these plots often help decide whether the next
# Stage-3 pass should:
#
# - narrow the search around the best region,
# - or expand into a different part of the space.

# %%
# Step 5 - Compare the best trials side by side
# ---------------------------------------------
# A user rarely needs only the single winner. The *top-k* trials are
# often more informative, especially when several settings perform
# similarly.

top_k = 6
top_df = ranked_df.head(top_k).copy()

fig, ax = plt.subplots(figsize=(8.0, 4.6))

ax.bar(
    top_df["trial_id"].astype(str),
    top_df["score"],
)
ax.set_xlabel("Trial ID")
ax.set_ylabel("Objective score")
ax.set_title("Top Stage-3 trials")
ax.grid(True, linestyle=":", alpha=0.5, axis="y")

plt.tight_layout()
plt.show()

# %%
# This simple view helps answer:
#
# - is there one clear winner?
# - or are several trials effectively tied?
#
# If many top trials are very close, the user should prefer the one
# that is more stable, simpler, or cheaper to train.

# %%
# Step 6 - Add an operational comparison table
# --------------------------------------------
# Stage-3 should not be read only as an optimization race. Operational
# properties also matter.

compare_cols = [
    "trial_id",
    "score",
    "hidden_units",
    "dropout_rate",
    "learning_rate",
    "batch_size",
    "lambda_cons",
    "lambda_prior",
    "epochs_ran",
    "fit_minutes",
    "status",
]

print("")
print("Top-trial operational summary")
print(top_df[compare_cols].to_string(index=False))

# %%
# Why this table matters
# ----------------------
# Two trials can have almost identical score but very different
# practical value.
#
# For example:
#
# - one trial may be slightly worse but much faster,
# - one may be best numerically but marked unstable,
# - one may be simpler and easier to reproduce.
#
# That is why a tuning page should teach users not to over-focus on the
# smallest score difference alone.

# %%
# Step 7 - Inspect score by categorical settings
# ----------------------------------------------
# Box-style summaries can make grouped sensitivity easier to read.

group_summary = (
    trials_df.groupby("batch_size")
    .agg(
        mean_score=("score", "mean"),
        best_score=("score", "min"),
        n_trials=("trial_id", "size"),
    )
    .reset_index()
)

print("")
print("Grouped summary by batch_size")
print(group_summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.0, 4.4))

for batch in sorted(trials_df["batch_size"].unique()):
    g = trials_df[trials_df["batch_size"] == batch]
    x = np.full(len(g), batch, dtype=float)
    ax.scatter(x, g["score"], s=36, label=f"batch={batch}")

ax.set_xlabel("batch_size")
ax.set_ylabel("score")
ax.set_title("Grouped score spread by batch size")
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Step 8 - Build a compact "what to do next" summary
# --------------------------------------------------
# A strong tuning page should end with an interpretation table that
# helps decide the next search step.

best_score = float(ranked_df["score"].iloc[0])
p90_score = float(np.quantile(trials_df["score"], 0.90))
median_score = float(np.median(trials_df["score"]))
gap_to_second = float(ranked_df["score"].iloc[1] - ranked_df["score"].iloc[0])

decision_summary = pd.DataFrame(
    [
        {"item": "best_score", "value": best_score},
        {"item": "median_score", "value": median_score},
        {"item": "90th_percentile_score", "value": p90_score},
        {"item": "gap_best_to_second", "value": gap_to_second},
        {
            "item": "n_unstable_trials",
            "value": int((trials_df["status"] != "ok").sum()),
        },
    ]
)

print("")
print("Tuning decision summary")
print(decision_summary.to_string(index=False))

# %%
# How to interpret the summary
# ----------------------------
# ``gap_best_to_second``
# ~~~~~~~~~~~~~~~~~~~~~~
# If this is tiny, the user should treat the top few trials as a near
# tie and choose using stability or simplicity.
#
# ``n_unstable_trials``
# ~~~~~~~~~~~~~~~~~~~~~
# A large value may suggest the search space is too aggressive or the
# training configuration is brittle.
#
# ``median_score`` vs ``best_score``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A wide separation often means tuning was genuinely useful.
# A narrow separation can mean either:
#
# - the baseline search space was already good,
# - or the metric is not very sensitive to the explored knobs.

# %%
# Step 9 - Practical reading guide
# --------------------------------
# A sensible Stage-3 interpretation sequence is:
#
# 1. inspect the incumbent best curve;
# 2. inspect the top-k trials;
# 3. inspect parameter-versus-score structure;
# 4. check operational cost and instability;
# 5. then decide whether to:
#    - accept the best trial,
#    - rerun a narrower local search,
#    - or revise the search space entirely.
#
# That sequence prevents one common mistake:
# choosing the numerically best trial without understanding whether
# the search actually converged on a stable region.

# %%
# Final takeaway
# --------------
# Stage-3 tuning summaries are not just leaderboards.
#
# They are diagnostics of:
#
# - search progress,
# - parameter sensitivity,
# - practical stability,
# - and whether another tuning round is scientifically justified.
#
# That is why the diagnostics gallery should include a dedicated
# Stage-3 tuning-summary lesson.