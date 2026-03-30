from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def stage1_mod():
    return importlib.import_module("geoprior.cli.stage1")


def test_parse_override_value_handles_common_inputs(
    stage1_mod,
):
    fn = stage1_mod._parse_override_value

    assert fn("true") is True
    assert fn("False") is False
    assert fn("none") is None
    assert fn("42") == 42
    assert fn("3.5") == 3.5
    assert fn("[1, 2]") == [1, 2]
    assert fn("{'a': 1}") == {"a": 1}
    assert fn(" hello ") == "hello"
    assert fn("") == ""


def test_parse_set_items_coerces_values(stage1_mod):
    got = stage1_mod._parse_set_items(
        [
            "TIME_STEPS=6",
            "USE_HEAD_PROXY=false",
            "HEAD_COL='head_m'",
            "MV_DELAY_STEPS=None",
        ]
    )

    assert got == {
        "TIME_STEPS": 6,
        "USE_HEAD_PROXY": False,
        "HEAD_COL": "head_m",
        "MV_DELAY_STEPS": None,
    }


@pytest.mark.parametrize(
    "item, match",
    [
        ("TIME_STEPS", "Each --set must be KEY=VALUE"),
        ("=5", "Invalid empty key"),
    ],
)
def test_parse_set_items_rejects_invalid_pairs(
    stage1_mod,
    item,
    match,
):
    with pytest.raises(SystemExit, match=match):
        stage1_mod._parse_set_items([item])


def test_cli_overrides_merge_known_flags_and_sets(
    stage1_mod,
):
    parser = stage1_mod._build_stage1_parser()
    args = parser.parse_args(
        [
            "--city",
            "Zhongshan",
            "--model",
            "GeoPriorSubsNet",
            "--data-dir",
            "/tmp/data",
            "--set",
            "TIME_STEPS=7",
            "--set",
            "USE_HEAD_PROXY=true",
        ]
    )

    got = stage1_mod._cli_overrides(args)

    assert got["CITY_NAME"] == "zhongshan"
    assert got["MODEL_NAME"] == "GeoPriorSubsNet"
    assert got["DATA_DIR"] == "/tmp/data"
    assert got["TIME_STEPS"] == 7
    assert got["USE_HEAD_PROXY"] is True


def test_refresh_config_fields_rebuilds_filenames(
    stage1_mod,
):
    cfg = {
        "CITY_NAME": "Zhongshan",
        "DATASET_VARIANT": "with_zsurf",
        "BIG_FN_TEMPLATE": (
            "{city}_final_main_std.cleaned.{variant}.csv"
        ),
        "SMALL_FN_TEMPLATE": (
            "{city}_2000.cleaned.{variant}.csv"
        ),
    }

    got = stage1_mod._refresh_config_fields(cfg)

    assert got["BIG_FN"] == (
        "zhongshan_final_main_std.cleaned.with_zsurf.csv"
    )
    assert got["SMALL_FN"] == (
        "zhongshan_2000.cleaned.with_zsurf.csv"
    )


def test_install_user_config_copies_file_and_drops_json(
    stage1_mod,
    tmp_path,
    natcom_config_dict,
    write_natcom_config,
    monkeypatch,
):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "user_config.py"
    src.write_text(
        "CITY_NAME = 'nansha'\nMODEL_NAME = 'GeoPriorSubsNet'\n",
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
        stage1_mod,
        "get_config_paths",
        _fake_get_config_paths,
    )

    out = stage1_mod._install_user_config(
        str(src),
        config_root="nat.com",
    )

    assert out == str(Path(written["config_py"]).resolve())
    assert Path(out).read_text(encoding="utf-8") == (
        src.read_text(encoding="utf-8")
    )
    assert not old_json.exists()


def test_install_user_config_raises_for_missing_file(
    stage1_mod,
    tmp_path,
):
    missing = tmp_path / "missing_config.py"

    with pytest.raises(
        FileNotFoundError, match="Config file not found"
    ):
        stage1_mod._install_user_config(str(missing))


def test_persist_runtime_overrides_updates_payload(
    stage1_mod,
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
        stage1_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    updated = stage1_mod._persist_runtime_overrides(
        {
            "CITY_NAME": "zhongshan",
            "DATASET_VARIANT": "with_zsurf",
            "TIME_STEPS": 9,
        },
        config_root="nat.com",
    )

    payload = json.loads(
        Path(written["config_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert updated["CITY_NAME"] == "zhongshan"
    assert updated["TIME_STEPS"] == 9
    assert updated["BIG_FN"].startswith("zhongshan_")
    assert payload["city"] == "zhongshan"
    assert (
        payload["model"] == natcom_config_dict["MODEL_NAME"]
    )
    assert payload["config"]["TIME_STEPS"] == 9
    assert "__meta__" in payload


def test_persist_runtime_overrides_without_changes_returns_cfg(
    stage1_mod,
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
        stage1_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    got = stage1_mod._persist_runtime_overrides(
        None,
        config_root="nat.com",
    )

    assert got["CITY_NAME"] == natcom_config_dict["CITY_NAME"]
    assert got["BIG_FN"] == natcom_config_dict["BIG_FN"]


def test_stage1_main_delegates_to_run_stage1(
    stage1_mod,
    monkeypatch,
):
    called = {}

    def _fake_run_stage1(
        *,
        overrides,
        config_root,
        config_path,
    ):
        called["overrides"] = overrides
        called["config_root"] = config_root
        called["config_path"] = config_path

    monkeypatch.setattr(
        stage1_mod,
        "run_stage1",
        _fake_run_stage1,
    )

    stage1_mod.stage1_main(
        [
            "--config",
            "/tmp/my_config.py",
            "--config-root",
            "nat.alt",
            "--city",
            "nansha",
            "--set",
            "TIME_STEPS=8",
        ]
    )

    assert called == {
        "overrides": {
            "CITY_NAME": "nansha",
            "TIME_STEPS": 8,
        },
        "config_root": "nat.alt",
        "config_path": "/tmp/my_config.py",
    }


def test_main_is_alias_of_stage1_main(
    stage1_mod,
    monkeypatch,
):
    seen = {}

    def _fake_stage1_main(argv=None):
        seen["argv"] = argv

    monkeypatch.setattr(
        stage1_mod,
        "stage1_main",
        _fake_stage1_main,
    )

    stage1_mod.main(["--city", "nansha"])

    assert seen["argv"] == ["--city", "nansha"]
