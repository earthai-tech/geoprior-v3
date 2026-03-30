from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def stage3_mod():
    return importlib.import_module("geoprior.cli.stage3")


def test_parse_override_value_handles_common_inputs(
    stage3_mod,
):
    fn = stage3_mod._parse_override_value

    assert fn("true") is True
    assert fn("False") is False
    assert fn("none") is None
    assert fn("42") == 42
    assert fn("3.5") == 3.5
    assert fn("[1, 2]") == [1, 2]
    assert fn("{'a': 1}") == {"a": 1}
    assert fn(" hello ") == "hello"
    assert fn("") == ""


def test_parse_set_items_coerces_values(stage3_mod):
    got = stage3_mod._parse_set_items(
        [
            "MAX_TRIALS=20",
            "USE_HEAD_PROXY=false",
            "HEAD_COL='head_m'",
            "MV_DELAY_STEPS=None",
        ]
    )

    assert got == {
        "MAX_TRIALS": 20,
        "USE_HEAD_PROXY": False,
        "HEAD_COL": "head_m",
        "MV_DELAY_STEPS": None,
    }


@pytest.mark.parametrize(
    "item, match",
    [
        ("MAX_TRIALS", "Each --set must be KEY=VALUE"),
        ("=5", "Invalid empty key"),
    ],
)
def test_parse_set_items_rejects_invalid_pairs(
    stage3_mod,
    item,
    match,
):
    with pytest.raises(SystemExit, match=match):
        stage3_mod._parse_set_items([item])


def test_cli_overrides_merge_known_flags_and_sets(
    stage3_mod,
):
    parser = stage3_mod._build_stage3_parser()
    args = parser.parse_args(
        [
            "--city",
            "Zhongshan",
            "--model",
            "GeoPriorSubsNet",
            "--data-dir",
            "/tmp/data",
            "--stage1-manifest",
            "/tmp/results/manifest.json",
            "--set",
            "MAX_TRIALS=20",
            "--set",
            "USE_HEAD_PROXY=true",
        ]
    )

    got = stage3_mod._cli_overrides(args)

    assert got["CITY_NAME"] == "zhongshan"
    assert got["MODEL_NAME"] == "GeoPriorSubsNet"
    assert got["DATA_DIR"] == "/tmp/data"
    assert got["MAX_TRIALS"] == 20
    assert got["USE_HEAD_PROXY"] is True
    assert "STAGE1_MANIFEST" not in got


def test_refresh_config_fields_rebuilds_filenames(
    stage3_mod,
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

    got = stage3_mod._refresh_config_fields(cfg)

    assert got["BIG_FN"] == (
        "zhongshan_final_main_std.cleaned.with_zsurf.csv"
    )
    assert got["SMALL_FN"] == (
        "zhongshan_2000.cleaned.with_zsurf.csv"
    )


def test_install_user_config_copies_file_and_drops_json(
    stage3_mod,
    tmp_path,
    natcom_config_dict,
    write_natcom_config,
    monkeypatch,
):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "user_config.py"
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
        stage3_mod,
        "get_config_paths",
        _fake_get_config_paths,
    )

    out = stage3_mod._install_user_config(
        str(src),
        config_root="nat.com",
    )

    assert out == str(Path(written["config_py"]).resolve())
    assert Path(out).read_text(encoding="utf-8") == (
        src.read_text(encoding="utf-8")
    )
    assert not old_json.exists()


def test_install_user_config_raises_for_missing_file(
    stage3_mod,
    tmp_path,
):
    missing = tmp_path / "missing_config.py"

    with pytest.raises(
        FileNotFoundError,
        match="Config file not found",
    ):
        stage3_mod._install_user_config(str(missing))


def test_persist_runtime_overrides_updates_payload(
    stage3_mod,
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
        stage3_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    updated = stage3_mod._persist_runtime_overrides(
        {
            "CITY_NAME": "zhongshan",
            "DATASET_VARIANT": "with_zsurf",
            "MAX_TRIALS": 20,
        },
        config_root="nat.com",
    )

    payload = json.loads(
        Path(written["config_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert updated["CITY_NAME"] == "zhongshan"
    assert updated["MAX_TRIALS"] == 20
    assert updated["BIG_FN"].startswith("zhongshan_")
    assert payload["city"] == "zhongshan"
    assert (
        payload["model"] == natcom_config_dict["MODEL_NAME"]
    )
    assert payload["config"]["MAX_TRIALS"] == 20
    assert "__meta__" in payload


def test_persist_runtime_overrides_without_changes_returns_cfg(
    stage3_mod,
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
        stage3_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    got = stage3_mod._persist_runtime_overrides(
        None,
        config_root="nat.com",
    )

    assert got["CITY_NAME"] == natcom_config_dict["CITY_NAME"]
    assert got["BIG_FN"] == natcom_config_dict["BIG_FN"]


def test_legacy_module_name_defaults_to_private_stage3(
    stage3_mod,
):
    assert stage3_mod._legacy_module_name() == (
        "geoprior.cli._stage3"
    )


def test_run_stage3_sets_env_for_legacy_module_and_restores_it(
    stage3_mod,
    mini_stage1_bundle,
    monkeypatch,
):
    manifest = Path(mini_stage1_bundle["manifest_path"])
    seen = {}

    def _fake_persist_runtime_overrides(
        overrides=None,
        *,
        config_root="nat.com",
    ):
        assert config_root == "nat.test"
        seen["overrides"] = overrides
        return {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
            "DATASET_VARIANT": "with_zsurf",
        }

    def _fake_run_module(mod_name, run_name=None):
        seen["mod_name"] = mod_name
        seen["run_name"] = run_name
        seen["CITY"] = stage3_mod.os.environ.get("CITY")
        seen["MODEL_NAME_OVERRIDE"] = (
            stage3_mod.os.environ.get("MODEL_NAME_OVERRIDE")
        )
        seen["STAGE1_MANIFEST"] = stage3_mod.os.environ.get(
            "STAGE1_MANIFEST"
        )

    monkeypatch.setattr(
        stage3_mod,
        "_persist_runtime_overrides",
        _fake_persist_runtime_overrides,
    )
    monkeypatch.setattr(
        stage3_mod.runpy,
        "run_module",
        _fake_run_module,
    )
    monkeypatch.setenv("CITY", "old-city")
    monkeypatch.delenv(
        "MODEL_NAME_OVERRIDE",
        raising=False,
    )
    monkeypatch.delenv(
        "STAGE1_MANIFEST",
        raising=False,
    )

    stage3_mod.run_stage3(
        {"MAX_TRIALS": 12},
        config_root="nat.test",
        stage1_manifest=str(manifest),
    )

    assert seen["overrides"] == {"MAX_TRIALS": 12}
    assert seen["mod_name"] == "geoprior.cli._stage3"
    assert seen["run_name"] == "__main__"
    assert seen["CITY"] == "nansha"
    assert seen["MODEL_NAME_OVERRIDE"] == "GeoPriorSubsNet"
    assert seen["STAGE1_MANIFEST"] == str(manifest.resolve())
    assert stage3_mod.os.environ.get("CITY") == "old-city"
    assert "MODEL_NAME_OVERRIDE" not in stage3_mod.os.environ
    assert "STAGE1_MANIFEST" not in stage3_mod.os.environ


def test_run_stage3_installs_user_config_when_requested(
    stage3_mod,
    monkeypatch,
):
    seen = {}

    def _fake_install_user_config(
        config_path,
        *,
        config_root="nat.com",
    ):
        seen["config_path"] = config_path
        seen["config_root"] = config_root
        return config_path

    def _fake_persist_runtime_overrides(
        overrides=None,
        *,
        config_root="nat.com",
    ):
        return {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
        }

    def _fake_run_module(mod_name, run_name=None):
        seen["mod_name"] = mod_name
        seen["run_name"] = run_name

    monkeypatch.setattr(
        stage3_mod,
        "_install_user_config",
        _fake_install_user_config,
    )
    monkeypatch.setattr(
        stage3_mod,
        "_persist_runtime_overrides",
        _fake_persist_runtime_overrides,
    )
    monkeypatch.setattr(
        stage3_mod.runpy,
        "run_module",
        _fake_run_module,
    )

    stage3_mod.run_stage3(
        None,
        config_root="nat.alt",
        config_path="/tmp/custom_config.py",
    )

    assert seen["config_path"] == "/tmp/custom_config.py"
    assert seen["config_root"] == "nat.alt"
    assert seen["mod_name"] == "geoprior.cli._stage3"
    assert seen["run_name"] == "__main__"


def test_stage3_main_delegates_to_run_stage3(
    stage3_mod,
    monkeypatch,
):
    called = {}

    def _fake_run_stage3(
        overrides=None,
        *,
        config_root,
        config_path,
        stage1_manifest,
    ):
        called["overrides"] = overrides
        called["config_root"] = config_root
        called["config_path"] = config_path
        called["stage1_manifest"] = stage1_manifest

    monkeypatch.setattr(
        stage3_mod,
        "run_stage3",
        _fake_run_stage3,
    )

    stage3_mod.stage3_main(
        [
            "--config",
            "/tmp/my_config.py",
            "--config-root",
            "nat.alt",
            "--city",
            "nansha",
            "--stage1-manifest",
            "/tmp/results/manifest.json",
            "--set",
            "MAX_TRIALS=80",
        ]
    )

    assert called == {
        "overrides": {
            "CITY_NAME": "nansha",
            "MAX_TRIALS": 80,
        },
        "config_root": "nat.alt",
        "config_path": "/tmp/my_config.py",
        "stage1_manifest": "/tmp/results/manifest.json",
    }


def test_main_is_alias_of_stage3_main(
    stage3_mod,
    monkeypatch,
):
    seen = {}

    def _fake_stage3_main(argv=None):
        seen["argv"] = argv

    monkeypatch.setattr(
        stage3_mod,
        "stage3_main",
        _fake_stage3_main,
    )

    stage3_mod.main(["--city", "nansha"])

    assert seen["argv"] == ["--city", "nansha"]
