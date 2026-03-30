from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def sm3_offsets_mod():
    return importlib.import_module(
        "geoprior.cli.sm3_log_offsets_diagnostics"
    )


def _write_phys_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        K_prior=np.array([1.0, 2.0]),
        K_eff=np.array([1.5, 2.5]),
        Ss_prior=np.array([0.1, 0.2]),
        Ss_eff=np.array([0.11, 0.21]),
        Hd_prior=np.array([10.0, 12.0]),
        Hd_eff=np.array([10.5, 12.5]),
        tau_prior=np.array([5.0, 6.0]),
        tau_eff=np.array([5.5, 6.5]),
    )
    return path


def test_parse_override_value_handles_common_inputs(
    sm3_offsets_mod,
):
    fn = sm3_offsets_mod._parse_override_value

    assert fn("true") is True
    assert fn("False") is False
    assert fn("none") is None
    assert fn("42") == 42
    assert fn("3.5") == 3.5
    assert fn("[1, 2]") == [1, 2]
    assert fn("{'a': 1}") == {"a": 1}
    assert fn(" hello ") == "hello"
    assert fn("") == ""


def test_parse_set_items_coerces_values(sm3_offsets_mod):
    got = sm3_offsets_mod._parse_set_items(
        [
            "CITY_NAME='zhongshan'",
            "FAST=true",
            "LIMIT=12",
            "SM3_PHYSICS_NPZ=None",
        ]
    )

    assert got == {
        "CITY_NAME": "zhongshan",
        "FAST": True,
        "LIMIT": 12,
        "SM3_PHYSICS_NPZ": None,
    }


@pytest.mark.parametrize(
    "item, match",
    [
        ("CITY_NAME", "Each --set must be KEY=VALUE"),
        ("=x", "Invalid empty key"),
    ],
)
def test_parse_set_items_rejects_invalid_pairs(
    sm3_offsets_mod,
    item,
    match,
):
    with pytest.raises(SystemExit, match=match):
        sm3_offsets_mod._parse_set_items([item])


def test_cli_overrides_merge_known_flags_and_sets(
    sm3_offsets_mod,
):
    parser = sm3_offsets_mod._build_parser()
    args = parser.parse_args(
        [
            "--city",
            "Zhongshan",
            "--model-name",
            "GeoPriorSubsNet",
            "--results-dir",
            "/tmp/results",
            "--outdir",
            "/tmp/out",
            "--physics-npz",
            "/tmp/payload.npz",
            "--set",
            "FAST=true",
        ]
    )

    got = sm3_offsets_mod._cli_overrides(args)

    assert got == {
        "FAST": True,
        "CITY_NAME": "zhongshan",
        "MODEL_NAME": "GeoPriorSubsNet",
        "RESULTS_DIR": "/tmp/results",
        "SM3_OFFSETS_OUTDIR": "/tmp/out",
        "SM3_PHYSICS_NPZ": "/tmp/payload.npz",
    }


def test_install_user_config_copies_file_and_drops_json(
    sm3_offsets_mod,
    tmp_path,
    natcom_config_dict,
    write_natcom_config,
    monkeypatch,
):
    src = tmp_path / "src_config.py"
    src.write_text(
        "CITY_NAME = 'nansha'\n"
        "MODEL_NAME = 'GeoPriorSubsNet'\n",
        encoding="utf-8",
    )

    written = write_natcom_config(natcom_config_dict)
    old_json = Path(written["config_json"])
    assert old_json.exists()

    def _fake_get_config_paths(*, root: str):
        assert root == "nat.com"
        return (
            str(written["config_py"]),
            str(written["config_json"]),
        )

    monkeypatch.setattr(
        sm3_offsets_mod,
        "get_config_paths",
        _fake_get_config_paths,
    )

    out = sm3_offsets_mod._install_user_config(
        str(src),
        config_root="nat.com",
    )

    assert out == str(Path(written["config_py"]).resolve())
    assert Path(out).read_text(encoding="utf-8") == (
        src.read_text(encoding="utf-8")
    )
    assert not old_json.exists()


def test_persist_runtime_overrides_updates_payload(
    sm3_offsets_mod,
    natcom_config_dict,
    write_natcom_config,
    monkeypatch,
):
    written = write_natcom_config(natcom_config_dict)
    base_cfg = dict(natcom_config_dict)

    def _fake_ensure_config_json(*, root: str):
        assert root == "nat.com"
        return base_cfg, str(written["config_json"])

    monkeypatch.setattr(
        sm3_offsets_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    updated = sm3_offsets_mod._persist_runtime_overrides(
        {
            "CITY_NAME": "zhongshan",
            "RESULTS_DIR": "/tmp/results",
        },
        config_root="nat.com",
    )

    payload = json.loads(
        Path(written["config_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert updated["CITY_NAME"] == "zhongshan"
    assert updated["BIG_FN"].startswith("zhongshan_")
    assert payload["city"] == "zhongshan"
    assert (
        payload["model"] == natcom_config_dict["MODEL_NAME"]
    )
    assert payload["config"]["RESULTS_DIR"] == "/tmp/results"
    assert "__meta__" in payload


def test_iter_payload_candidates_prefers_newest_file(
    sm3_offsets_mod,
    tmp_path,
):
    root = tmp_path / "results"
    older = _write_phys_npz(
        root / "a" / "nansha_phys_payload_run_val.npz"
    )
    newer = _write_phys_npz(
        root / "b" / "nansha_tuned_phys_payload_run_val.npz"
    )

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    hits = sm3_offsets_mod._iter_payload_candidates(
        str(root),
        "nansha",
    )

    assert hits
    assert hits[0] == newer.resolve()
    assert older.resolve() in hits


def test_discover_physics_npz_uses_explicit_before_search(
    sm3_offsets_mod,
    tmp_path,
):
    explicit = _write_phys_npz(tmp_path / "explicit.npz")
    args = SimpleNamespace(
        physics_npz=str(explicit),
        city=None,
        results_dir=None,
    )

    got = sm3_offsets_mod._discover_physics_npz({}, args)

    assert got == str(explicit.resolve())


def test_seed_forwarded_args_adds_defaults_from_cfg(
    sm3_offsets_mod,
    tmp_path,
):
    payload = _write_phys_npz(
        tmp_path
        / "results"
        / "nansha_phys_payload_run_val.npz"
    )
    cfg = {
        "CITY_NAME": "nansha",
        "MODEL_NAME": "GeoPriorSubsNet",
        "RESULTS_DIR": str(tmp_path / "results"),
    }
    args = SimpleNamespace(
        physics_npz=None,
        city=None,
        model_name=None,
        results_dir=None,
        outdir=None,
    )

    got = sm3_offsets_mod._seed_forwarded_args([], cfg, args)

    assert got == [
        "--physics-npz",
        str(payload.resolve()),
        "--outdir",
        str(payload.resolve().parent),
        "--city",
        "nansha",
        "--model-name",
        "GeoPriorSubsNet",
    ]


def test_seed_forwarded_args_does_not_duplicate_present_flags(
    sm3_offsets_mod,
    tmp_path,
):
    payload = _write_phys_npz(tmp_path / "payload.npz")
    cfg = {
        "CITY_NAME": "nansha",
        "MODEL_NAME": "GeoPriorSubsNet",
        "RESULTS_DIR": str(tmp_path),
    }
    args = SimpleNamespace(
        physics_npz=str(payload),
        city="zhongshan",
        model_name="OtherModel",
        results_dir=str(tmp_path),
        outdir=str(tmp_path / "out"),
    )

    forwarded = [
        "--physics-npz",
        "already.npz",
        "--outdir=/tmp/x",
        "--city",
        "keep-me",
        "--model-name=KeepModel",
    ]

    got = sm3_offsets_mod._seed_forwarded_args(
        forwarded,
        cfg,
        args,
    )

    assert got == forwarded


def test_run_sm3_offsets_help_prints_combined_help(
    sm3_offsets_mod,
    capsys,
):
    sm3_offsets_mod.run_sm3_offsets(["--help"])

    out = capsys.readouterr().out
    assert "Forwarded legacy arguments include" in out
    assert "--physics-npz PATH" in out


def test_run_sm3_offsets_requires_resolved_payload(
    sm3_offsets_mod,
    monkeypatch,
):
    monkeypatch.setattr(
        sm3_offsets_mod,
        "_persist_runtime_overrides",
        lambda overrides, config_root="nat.com": {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
        },
    )

    with pytest.raises(
        SystemExit,
        match="Could not resolve --physics-npz automatically",
    ):
        sm3_offsets_mod.run_sm3_offsets([])


def test_run_sm3_offsets_calls_legacy_main_and_restores_sys_argv(
    sm3_offsets_mod,
    tmp_path,
    monkeypatch,
):
    payload = _write_phys_npz(
        tmp_path
        / "results"
        / "nansha_phys_payload_run_val.npz"
    )
    cfg = {
        "CITY_NAME": "nansha",
        "MODEL_NAME": "GeoPriorSubsNet",
        "RESULTS_DIR": str(tmp_path / "results"),
    }
    state: dict[str, object] = {}

    monkeypatch.setattr(
        sm3_offsets_mod,
        "_persist_runtime_overrides",
        lambda overrides, config_root="nat.com": dict(cfg),
    )

    def _fake_main():
        state["argv"] = list(sys.argv)

    fake_mod = SimpleNamespace(main=_fake_main)
    monkeypatch.setattr(
        sm3_offsets_mod.importlib,
        "import_module",
        lambda name: fake_mod,
    )

    original = list(sys.argv)
    sm3_offsets_mod.run_sm3_offsets(["--city", "nansha"])

    assert sys.argv == original
    assert state["argv"] == [
        "sm3-offset-diagnostics",
        "--physics-npz",
        str(payload.resolve()),
        "--outdir",
        str(payload.resolve().parent),
        "--city",
        "nansha",
        "--model-name",
        "GeoPriorSubsNet",
    ]


def test_sm3_offsets_main_and_main_alias_delegate(
    sm3_offsets_mod,
    patch_cli_entry,
):
    state = patch_cli_entry(
        sm3_offsets_mod,
        "run_sm3_offsets",
    )

    sm3_offsets_mod.sm3_offsets_main(["--help"])
    sm3_offsets_mod.main(["--help"])

    assert state["calls"] == [
        {"argv": ["--help"], "args": (), "kwargs": {}},
        {"argv": ["--help"], "args": (), "kwargs": {}},
    ]
