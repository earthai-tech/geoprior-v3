from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def rs_module():
    import geoprior.cli.run_sensitivity as mod

    return mod


def test_build_grid_cartesian_product(rs_module):
    runs = rs_module.build_grid(
        ["both", "none"],
        [0.0, 0.1],
        [0.0, 1.0],
    )

    assert len(runs) == 8
    assert runs[0] == rs_module.RunSpec(
        pde_mode="both",
        lambda_cons=0.0,
        lambda_prior=0.0,
    )
    assert runs[-1] == rs_module.RunSpec(
        pde_mode="none",
        lambda_cons=0.1,
        lambda_prior=1.0,
    )


def test_maybe_shuffle_is_deterministic(rs_module):
    runs = rs_module.build_grid(["both"], [0.0, 0.1], [0.0, 1.0])

    out1 = rs_module.maybe_shuffle(
        runs,
        shuffle=True,
        seed=123,
    )
    out2 = rs_module.maybe_shuffle(
        runs,
        shuffle=True,
        seed=123,
    )
    out3 = rs_module.maybe_shuffle(
        runs,
        shuffle=True,
        seed=456,
    )

    assert out1 == out2
    assert {r.key() for r in out1} == {r.key() for r in runs}
    assert out1 != runs
    assert out3 != out1


def test_apply_runner_slicing_handles_bounds(rs_module):
    runs = rs_module.build_grid(["both"], [0.0, 0.1], [0.0, 1.0])

    sliced = rs_module.apply_runner_slicing(
        runs,
        start=-5,
        limit=2,
    )
    assert sliced == runs[:2]

    assert rs_module.apply_runner_slicing(
        runs,
        start=1,
        limit=None,
    ) == runs[1:]

    assert rs_module.apply_runner_slicing(
        runs,
        start=0,
        limit=0,
    ) == []


def test_load_completed_keys_from_done_filters_city(
    rs_module,
    tmp_path: Path,
):
    good = tmp_path / "a" / "DONE.json"
    good.parent.mkdir(parents=True)
    good.write_text(
        json.dumps(
            {
                "city": "nansha",
                "pde_mode": "both",
                "lambda_cons": 0.1,
                "lambda_prior": 0.2,
            }
        ),
        encoding="utf-8",
    )

    other = tmp_path / "b" / "DONE.json"
    other.parent.mkdir(parents=True)
    other.write_text(
        json.dumps(
            {
                "city": "zhongshan",
                "pde_mode": "none",
                "lambda_cons": 0.3,
                "lambda_prior": 0.4,
            }
        ),
        encoding="utf-8",
    )

    done = rs_module._load_completed_keys_from_done(
        tmp_path,
        city="nansha",
    )

    assert done == {
        rs_module.RunSpec(
            pde_mode="on",
            lambda_cons=0.1,
            lambda_prior=0.2,
        ).key()
    }


def test_load_completed_keys_reads_jsonl_and_skips_bad_records(
    rs_module,
    tmp_path: Path,
):
    jsonl = (
        tmp_path
        / "ablation_records"
        / "ablation_record.jsonl"
    )
    jsonl.parent.mkdir(parents=True)
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "city": "nansha",
                        "pde_mode": "both",
                        "lambda_cons": 0.5,
                        "lambda_prior": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "city": "zhongshan",
                        "pde_mode": "none",
                        "lambda_cons": 0.1,
                        "lambda_prior": 0.2,
                    }
                ),
                json.dumps({"city": "nansha"}),
                "not-json",
            ]
        ),
        encoding="utf-8",
    )

    done = rs_module._load_completed_keys(
        tmp_path,
        city="nansha",
    )

    assert done == {
        rs_module.RunSpec(
            pde_mode="on",
            lambda_cons=0.5,
            lambda_prior=1.0,
        ).key()
    }


def test_make_env_sets_fast_safe_overrides(rs_module):
    spec = rs_module.RunSpec(
        pde_mode="both",
        lambda_cons=0.2,
        lambda_prior=0.3,
    )

    env = rs_module.make_env(
        {"CITY": "nansha"},
        epochs=7,
        spec=spec,
        strategy="data_first",
        disable_q=True,
        disable_subs_resid=True,
        no_physics_ramp=True,
        physics_warmup_steps=None,
        physics_ramp_steps=None,
        lambda_gw=0.4,
        lambda_smooth=0.5,
        lambda_bounds=0.6,
        lambda_mv=None,
        lambda_q=None,
        no_early_stopping=True,
        fast=True,
        eval_max_batches=9,
        batch_size=16,
    )

    assert env["PDE_MODE_OVERRIDE"] == "both"
    assert env["EPOCHS_OVERRIDE"] == "7"
    assert env["LAMBDA_CONS_OVERRIDE"] == "0.2"
    assert env["LAMBDA_PRIOR_OVERRIDE"] == "0.3"
    assert env["RUN_TAG"] == spec.run_tag()
    assert env["FAST_SENSITIVITY"] == "1"
    assert env["DISABLE_EARLY_STOPPING"] == "1"
    assert env["Q_POLICY_OVERRIDE"] == "always_off"
    assert env["LAMBDA_Q_OVERRIDE"] == "0.0"
    assert env["SUBS_RESID_POLICY_OVERRIDE"] == "always_off"
    assert env["ALLOW_SUBS_RESIDUAL_OVERRIDE"] == "0"
    assert env["PHYSICS_WARMUP_STEPS_OVERRIDE"] == "0"
    assert env["PHYSICS_RAMP_STEPS_OVERRIDE"] == "0"
    assert env["LAMBDA_GW_OVERRIDE"] == "0.4"
    assert env["LAMBDA_SMOOTH_OVERRIDE"] == "0.5"
    assert env["LAMBDA_BOUNDS_OVERRIDE"] == "0.6"
    assert env["LAMBDA_MV_OVERRIDE"] == "0.0"
    assert env["MV_WEIGHT_OVERRIDE"] == "0.0"
    assert env["SENS_EVAL_MAX_BATCHES"] == "9"
    assert env["BATCH_SIZE_OVERRIDE"] == "16"


def test_sensitivity_main_seeds_scan_root_and_restores_env(
    rs_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[list[str]] = []

    monkeypatch.setattr(
        rs_module,
        "_persist_runtime_overrides",
        lambda overrides, config_root="nat.com": {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
            "RESULTS_DIR": str(tmp_path / "results"),
        },
    )
    monkeypatch.setattr(
        rs_module,
        "_install_user_config",
        lambda *args, **kwargs: "installed",
    )
    monkeypatch.setattr(
        rs_module,
        "main",
        lambda argv=None: calls.append(list(argv or [])),
    )

    monkeypatch.setenv("CITY", "old-city")
    monkeypatch.setenv("MODEL_NAME_OVERRIDE", "old-model")
    monkeypatch.setenv("RESULTS_DIR", "old-results")
    monkeypatch.setenv(
        "GEOPRIOR_RESULTS_DIR",
        "old-geoprior-results",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    rs_module.sensitivity_main(
        [
            "--config",
            str(tmp_path / "config.py"),
            "--stage1-manifest",
            str(manifest),
            "--epochs",
            "10",
            "--fast",
        ]
    )

    assert calls == [
        [
            "--epochs",
            "10",
            "--fast",
            "--scan-root",
            str(tmp_path / "results" / "nansha"),
        ]
    ]

    assert Path(manifest).resolve()
    assert rs_module.os.environ["CITY"] == "old-city"
    assert (
        rs_module.os.environ["MODEL_NAME_OVERRIDE"]
        == "old-model"
    )
    assert rs_module.os.environ["RESULTS_DIR"] == "old-results"
    assert (
        rs_module.os.environ["GEOPRIOR_RESULTS_DIR"]
        == "old-geoprior-results"
    )


def test_sensitivity_main_does_not_duplicate_scan_root(
    rs_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[list[str]] = []

    monkeypatch.setattr(
        rs_module,
        "_persist_runtime_overrides",
        lambda overrides, config_root="nat.com": {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
            "RESULTS_DIR": str(tmp_path / "results"),
        },
    )
    monkeypatch.setattr(
        rs_module,
        "main",
        lambda argv=None: calls.append(list(argv or [])),
    )

    rs_module.sensitivity_main(
        [
            "--epochs",
            "5",
            "--scan-root",
            str(tmp_path / "custom-scan"),
        ]
    )

    assert calls == [
        [
            "--epochs",
            "5",
            "--scan-root",
            str(tmp_path / "custom-scan"),
        ]
    ]


def test_main_returns_early_when_no_runs_selected(
    rs_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    train_script = tmp_path / "_sensitivity.py"
    train_script.write_text("print('stub')\n", encoding="utf-8")

    saved: list[dict[str, object]] = []
    built: list[object] = []

    args = Namespace(
        train_script=str(train_script),
        epochs=3,
        pde_modes=["both"],
        lcons=[0.0],
        lprior=[0.0],
        no_early_stopping=False,
        fast=True,
        eval_max_batches=None,
        batch_size=None,
        inprocess=False,
        gold=False,
        n_jobs=1,
        threads=0,
        device="cpu",
        gpu_ids=None,
        gpu_allow_growth=False,
        strategy="data_first",
        disable_q=False,
        disable_subs_resid=False,
        no_physics_ramp=False,
        physics_warmup_steps=None,
        physics_ramp_steps=None,
        lambda_gw=None,
        lambda_smooth=None,
        lambda_bounds=None,
        lambda_mv=None,
        lambda_q=None,
        no_resume=True,
        scan_root=str(tmp_path / "scan"),
        state_file=str(tmp_path / "state.json"),
        start=0,
        limit=0,
        shuffle=False,
        seed=42,
        continue_on_error=False,
        dry_run=False,
    )

    monkeypatch.setattr(rs_module, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(rs_module, "resolve_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(rs_module, "build_grid", lambda *a, **k: built.append(True) or [rs_module.RunSpec("both", 0.0, 0.0)])
    monkeypatch.setattr(rs_module, "maybe_shuffle", lambda runs, shuffle, seed: runs)
    monkeypatch.setattr(rs_module, "_save_state", lambda state_path, city, scan_root, completed_n, last_key: saved.append({"state_path": str(state_path), "city": city, "scan_root": str(scan_root), "completed_n": completed_n, "last_key": last_key}))

    rs_module.main([])

    assert built == [True]
    assert saved == [
        {
            "state_path": str(tmp_path / "state.json"),
            "city": "<unknown>",
            "scan_root": str(tmp_path / "scan"),
            "completed_n": 0,
            "last_key": None,
        }
    ]
