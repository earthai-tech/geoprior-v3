from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def sm3_ident_mod():
    return importlib.import_module(
        "geoprior.cli.sm3_synthetic_identifiability"
    )


def test_parse_override_value_handles_common_inputs(
    sm3_ident_mod,
):
    fn = sm3_ident_mod._parse_override_value

    assert fn("true") is True
    assert fn("False") is False
    assert fn("none") is None
    assert fn("42") == 42
    assert fn("3.5") == 3.5
    assert fn("[1, 2]") == [1, 2]
    assert fn("{'a': 1}") == {"a": 1}
    assert fn(" hello ") == "hello"
    assert fn("") == ""


def test_parse_set_items_coerces_values(sm3_ident_mod):
    got = sm3_ident_mod._parse_set_items(
        [
            "IDENTIFIABILITY_REGIME='anchored'",
            "FAST=true",
            "N_REALIZATIONS=12",
            "NOISE_STD=None",
        ]
    )

    assert got == {
        "IDENTIFIABILITY_REGIME": "anchored",
        "FAST": True,
        "N_REALIZATIONS": 12,
        "NOISE_STD": None,
    }


@pytest.mark.parametrize(
    "item, match",
    [
        (
            "IDENTIFIABILITY_REGIME",
            "Each --set must be KEY=VALUE",
        ),
        ("=x", "Invalid empty key"),
    ],
)
def test_parse_set_items_reject_invalid_pairs(
    sm3_ident_mod,
    item,
    match,
):
    with pytest.raises(SystemExit, match=match):
        sm3_ident_mod._parse_set_items([item])


def test_cli_overrides_merge_known_flags_and_sets(
    sm3_ident_mod,
):
    parser = sm3_ident_mod._build_parser()
    args = parser.parse_args(
        [
            "--outdir",
            "/tmp/out",
            "--ident-regime",
            "closure_locked",
            "--set",
            "N_REALIZATIONS=50",
        ]
    )

    got = sm3_ident_mod._cli_overrides(args)

    assert got == {
        "N_REALIZATIONS": 50,
        "IDENTIFIABILITY_REGIME": "closure_locked",
        "SM3_IDENT_OUTDIR": "/tmp/out",
    }


def test_install_user_config_copies_file_and_drops_json(
    sm3_ident_mod,
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
        sm3_ident_mod,
        "get_config_paths",
        _fake_get_config_paths,
    )

    out = sm3_ident_mod._install_user_config(
        str(src),
        config_root="nat.com",
    )

    assert out == str(Path(written["config_py"]).resolve())
    assert Path(out).read_text(encoding="utf-8") == (
        src.read_text(encoding="utf-8")
    )
    assert not old_json.exists()


def test_persist_runtime_overrides_updates_payload(
    sm3_ident_mod,
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
        sm3_ident_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    updated = sm3_ident_mod._persist_runtime_overrides(
        {
            "CITY_NAME": "zhongshan",
            "IDENTIFIABILITY_REGIME": "anchored",
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
    assert payload["config"]["IDENTIFIABILITY_REGIME"] == (
        "anchored"
    )
    assert "__meta__" in payload


def test_default_outdir_appends_sm3_identifiability_and_city(
    sm3_ident_mod,
):
    cfg = {
        "RESULTS_DIR": "/tmp/results",
        "CITY_NAME": "nansha",
    }

    got = sm3_ident_mod._default_outdir(cfg)

    assert Path(got) == (
        Path("/tmp/results")
        / "sm3_identifiability"
        / "nansha"
    )


def test_default_outdir_keeps_explicit_sm3_folder(
    sm3_ident_mod,
):
    cfg = {
        "SM3_IDENT_OUTDIR": "/tmp/outputs/sm3_identifiability",
        "CITY_NAME": "nansha",
    }

    got = sm3_ident_mod._default_outdir(cfg)

    assert (
        Path(got)
        == Path("/tmp/outputs") / "sm3_identifiability"
    )


def test_seed_forwarded_args_adds_defaults_from_cfg(
    sm3_ident_mod,
):
    cfg = {
        "RESULTS_DIR": "/tmp/results",
        "CITY_NAME": "nansha",
        "IDENTIFIABILITY_REGIME": "anchored",
    }
    args = SimpleNamespace(
        outdir=None,
        ident_regime=None,
    )

    got = sm3_ident_mod._seed_forwarded_args([], cfg, args)

    assert got[0] == "--outdir"
    assert Path(got[1]) == (
        Path("/tmp/results")
        / "sm3_identifiability"
        / "nansha"
    )
    assert got[2:] == ["--ident-regime", "anchored"]


def test_seed_forwarded_args_does_not_duplicate_present_flags(
    sm3_ident_mod,
):
    cfg = {
        "RESULTS_DIR": "/tmp/results",
        "CITY_NAME": "nansha",
        "IDENTIFIABILITY_REGIME": "anchored",
    }
    args = SimpleNamespace(
        outdir="/tmp/other",
        ident_regime="closure_locked",
    )
    forwarded = [
        "--outdir=/keep/this",
        "--ident-regime",
        "base",
    ]

    got = sm3_ident_mod._seed_forwarded_args(
        forwarded,
        cfg,
        args,
    )

    assert got == forwarded


def test_run_sm3_identifiability_help_prints_combined_help(
    sm3_ident_mod,
    capsys,
):
    sm3_ident_mod.run_sm3_identifiability(["--help"])

    out = capsys.readouterr().out
    assert "Forwarded legacy arguments include" in out
    assert "--identify {tau,k,both}" in out


def test_run_sm3_identifiability_calls_legacy_main_and_restores_sys_argv(
    sm3_ident_mod,
    monkeypatch,
):
    cfg = {
        "RESULTS_DIR": "/tmp/results",
        "CITY_NAME": "nansha",
        "IDENTIFIABILITY_REGIME": "anchored",
    }
    state: dict[str, object] = {}

    monkeypatch.setattr(
        sm3_ident_mod,
        "_persist_runtime_overrides",
        lambda overrides, config_root="nat.com": dict(cfg),
    )

    def _fake_main():
        state["argv"] = list(sys.argv)

    fake_mod = SimpleNamespace(main=_fake_main)
    monkeypatch.setattr(
        sm3_ident_mod.importlib,
        "import_module",
        lambda name: fake_mod,
    )

    original = list(sys.argv)
    sm3_ident_mod.run_sm3_identifiability([])

    assert sys.argv == original

    argv = state["argv"]
    assert argv[0] == "sm3-identifiability"
    assert argv[1] == "--outdir"
    assert Path(argv[2]) == (
        Path("/tmp/results")
        / "sm3_identifiability"
        / "nansha"
    )
    assert argv[3:] == ["--ident-regime", "anchored"]


def test_run_sm3_identifiability_raises_if_legacy_main_missing(
    sm3_ident_mod,
    monkeypatch,
):
    monkeypatch.setattr(
        sm3_ident_mod,
        "_persist_runtime_overrides",
        lambda overrides, config_root="nat.com": {},
    )
    monkeypatch.setattr(
        sm3_ident_mod.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(
        AttributeError,
        match="Missing 'main' in sm3_synthetic_identifiability_legacy",
    ):
        sm3_ident_mod.run_sm3_identifiability([])


def test_sm3_identifiability_main_and_main_alias_delegate(
    sm3_ident_mod,
    patch_cli_entry,
):
    state = patch_cli_entry(
        sm3_ident_mod,
        "run_sm3_identifiability",
    )

    sm3_ident_mod.sm3_identifiability_main(["--help"])
    sm3_ident_mod.main(["--help"])

    assert state["calls"] == [
        {"argv": ["--help"], "args": (), "kwargs": {}},
        {"argv": ["--help"], "args": (), "kwargs": {}},
    ]
