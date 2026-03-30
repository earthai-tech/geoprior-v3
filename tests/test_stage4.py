from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def stage4_mod():
    return importlib.import_module("geoprior.cli.stage4")


def test_parse_override_value_handles_common_inputs(
    stage4_mod,
):
    fn = stage4_mod._parse_override_value

    assert fn("true") is True
    assert fn("False") is False
    assert fn("none") is None
    assert fn("42") == 42
    assert fn("3.5") == 3.5
    assert fn("[1, 2]") == [1, 2]
    assert fn("{'a': 1}") == {"a": 1}
    assert fn(" hello ") == "hello"
    assert fn("") == ""


def test_parse_set_items_coerces_values(stage4_mod):
    got = stage4_mod._parse_set_items(
        [
            "BATCH_SIZE=32",
            "USE_HEAD_PROXY=false",
            "HEAD_COL='head_m'",
            "CALIBRATOR=None",
        ]
    )

    assert got == {
        "BATCH_SIZE": 32,
        "USE_HEAD_PROXY": False,
        "HEAD_COL": "head_m",
        "CALIBRATOR": None,
    }


@pytest.mark.parametrize(
    "item, match",
    [
        ("BATCH_SIZE", "Each --set must be KEY=VALUE"),
        ("=5", "Invalid empty key"),
    ],
)
def test_parse_set_items_rejects_invalid_pairs(
    stage4_mod,
    item,
    match,
):
    with pytest.raises(SystemExit, match=match):
        stage4_mod._parse_set_items([item])


def test_cli_overrides_merge_known_flags_and_sets(
    stage4_mod,
):
    parser = stage4_mod._build_stage4_parser()
    args, forwarded = parser.parse_known_args(
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
            "BATCH_SIZE=32",
            "--set",
            "USE_HEAD_PROXY=true",
            "--dataset",
            "test",
            "--model-path",
            "/tmp/model.keras",
            "--no-figs",
        ]
    )

    got = stage4_mod._cli_overrides(args)

    assert got["CITY_NAME"] == "zhongshan"
    assert got["MODEL_NAME"] == "GeoPriorSubsNet"
    assert got["DATA_DIR"] == "/tmp/data"
    assert got["BATCH_SIZE"] == 32
    assert got["USE_HEAD_PROXY"] is True
    assert "STAGE1_MANIFEST" not in got
    assert forwarded == [
        "--dataset",
        "test",
        "--model-path",
        "/tmp/model.keras",
        "--no-figs",
    ]


def test_refresh_config_fields_rebuilds_filenames(
    stage4_mod,
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

    got = stage4_mod._refresh_config_fields(cfg)

    assert got["BIG_FN"] == (
        "zhongshan_final_main_std.cleaned.with_zsurf.csv"
    )
    assert got["SMALL_FN"] == (
        "zhongshan_2000.cleaned.with_zsurf.csv"
    )


def test_install_user_config_copies_file_and_drops_json(
    stage4_mod,
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
        stage4_mod,
        "get_config_paths",
        _fake_get_config_paths,
    )

    out = stage4_mod._install_user_config(
        str(src),
        config_root="nat.com",
    )

    assert out == str(Path(written["config_py"]).resolve())
    assert Path(out).read_text(encoding="utf-8") == (
        src.read_text(encoding="utf-8")
    )
    assert not old_json.exists()


def test_install_user_config_raises_for_missing_file(
    stage4_mod,
    tmp_path,
):
    missing = tmp_path / "missing_config.py"

    with pytest.raises(
        FileNotFoundError,
        match="Config file not found",
    ):
        stage4_mod._install_user_config(str(missing))


def test_persist_runtime_overrides_updates_payload(
    stage4_mod,
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
        stage4_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    updated = stage4_mod._persist_runtime_overrides(
        {
            "CITY_NAME": "zhongshan",
            "DATASET_VARIANT": "with_zsurf",
            "BATCH_SIZE": 32,
        },
        config_root="nat.com",
    )

    payload = json.loads(
        Path(written["config_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert updated["CITY_NAME"] == "zhongshan"
    assert updated["BATCH_SIZE"] == 32
    assert updated["BIG_FN"].startswith("zhongshan_")
    assert payload["city"] == "zhongshan"
    assert (
        payload["model"] == natcom_config_dict["MODEL_NAME"]
    )
    assert payload["config"]["BATCH_SIZE"] == 32
    assert "__meta__" in payload


def test_persist_runtime_overrides_without_changes_returns_cfg(
    stage4_mod,
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
        stage4_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    got = stage4_mod._persist_runtime_overrides(
        None,
        config_root="nat.com",
    )

    assert got["CITY_NAME"] == natcom_config_dict["CITY_NAME"]
    assert got["BIG_FN"] == natcom_config_dict["BIG_FN"]


def test_legacy_module_name_defaults_to_private_stage4(
    stage4_mod,
):
    assert stage4_mod._legacy_module_name() == (
        "geoprior.cli._stage4"
    )


def test_print_help_shows_forwarded_legacy_examples(
    stage4_mod,
    capsys,
):
    stage4_mod._print_help()
    out = capsys.readouterr().out

    assert "Forwarded legacy arguments include" in out
    assert "--model-path PATH" in out
    assert "--dataset {test,val,train,custom}" in out
    assert "--include-gwl" in out


def test_run_stage4_help_path_prints_help_and_skips_dispatch(
    stage4_mod,
    monkeypatch,
):
    seen = {"help": 0}

    def _fake_print_help():
        seen["help"] += 1

    def _boom(*args, **kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(
        stage4_mod,
        "_print_help",
        _fake_print_help,
    )
    monkeypatch.setattr(
        stage4_mod,
        "_persist_runtime_overrides",
        _boom,
    )
    monkeypatch.setattr(
        stage4_mod.importlib,
        "import_module",
        _boom,
    )

    stage4_mod.run_stage4(["--help"])

    assert seen["help"] == 1


def test_run_stage4_sets_env_and_forwards_unknown_args(
    stage4_mod,
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
        seen["overrides"] = overrides
        seen["config_root"] = config_root
        return {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
        }

    def _legacy_main():
        seen["argv_inside"] = list(stage4_mod.sys.argv)
        seen["CITY_inside"] = stage4_mod.os.environ.get(
            "CITY"
        )
        seen["MODEL_inside"] = stage4_mod.os.environ.get(
            "MODEL_NAME_OVERRIDE"
        )
        seen["MANIFEST_inside"] = stage4_mod.os.environ.get(
            "STAGE1_MANIFEST"
        )

    fake_mod = SimpleNamespace(main=_legacy_main)

    monkeypatch.setattr(
        stage4_mod,
        "_persist_runtime_overrides",
        _fake_persist_runtime_overrides,
    )
    monkeypatch.setattr(
        stage4_mod.importlib,
        "import_module",
        lambda name: fake_mod,
    )
    monkeypatch.setattr(
        stage4_mod.sys,
        "argv",
        ["outer-cmd", "--keep"],
    )
    monkeypatch.delenv("CITY", raising=False)
    monkeypatch.delenv(
        "MODEL_NAME_OVERRIDE",
        raising=False,
    )
    monkeypatch.delenv("STAGE1_MANIFEST", raising=False)

    stage4_mod.run_stage4(
        [
            "--config-root",
            "nat.test",
            "--city",
            "nansha",
            "--stage1-manifest",
            str(manifest),
            "--dataset",
            "test",
            "--model-path",
            "/tmp/model.keras",
            "--no-figs",
        ]
    )

    assert seen["overrides"] == {"CITY_NAME": "nansha"}
    assert seen["config_root"] == "nat.test"
    assert seen["argv_inside"] == [
        "stage4-infer",
        "--dataset",
        "test",
        "--model-path",
        "/tmp/model.keras",
        "--no-figs",
    ]
    assert seen["CITY_inside"] == "nansha"
    assert seen["MODEL_inside"] == "GeoPriorSubsNet"
    assert seen["MANIFEST_inside"] == str(manifest)

    # Current Stage-4 behavior restores sys.argv only.
    assert stage4_mod.sys.argv == ["outer-cmd", "--keep"]
    assert stage4_mod.os.environ["CITY"] == "nansha"
    assert (
        stage4_mod.os.environ["MODEL_NAME_OVERRIDE"]
        == "GeoPriorSubsNet"
    )
    assert stage4_mod.os.environ["STAGE1_MANIFEST"] == (
        str(manifest)
    )


def test_run_stage4_installs_user_config_when_requested(
    stage4_mod,
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

    fake_mod = SimpleNamespace(main=lambda: None)

    monkeypatch.setattr(
        stage4_mod,
        "_install_user_config",
        _fake_install_user_config,
    )
    monkeypatch.setattr(
        stage4_mod,
        "_persist_runtime_overrides",
        _fake_persist_runtime_overrides,
    )
    monkeypatch.setattr(
        stage4_mod.importlib,
        "import_module",
        lambda name: fake_mod,
    )

    stage4_mod.run_stage4(
        [
            "--config",
            "/tmp/custom_config.py",
            "--config-root",
            "nat.alt",
            "--dataset",
            "test",
            "--model-path",
            "/tmp/model.keras",
        ]
    )

    assert seen["config_path"] == "/tmp/custom_config.py"
    assert seen["config_root"] == "nat.alt"


def test_run_stage4_raises_when_legacy_main_is_missing(
    stage4_mod,
    monkeypatch,
):
    monkeypatch.setattr(
        stage4_mod,
        "_persist_runtime_overrides",
        lambda overrides=None, *, config_root="nat.com": {
            "CITY_NAME": "nansha",
            "MODEL_NAME": "GeoPriorSubsNet",
        },
    )
    monkeypatch.setattr(
        stage4_mod.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(
        AttributeError,
        match="Missing 'main' in stage4_legacy",
    ):
        stage4_mod.run_stage4(
            [
                "--dataset",
                "test",
                "--model-path",
                "/tmp/model.keras",
            ]
        )


def test_stage4_main_delegates_to_run_stage4(
    stage4_mod,
    monkeypatch,
):
    called = {}

    def _fake_run_stage4(argv=None):
        called["argv"] = argv

    monkeypatch.setattr(
        stage4_mod,
        "run_stage4",
        _fake_run_stage4,
    )

    stage4_mod.stage4_main(
        [
            "--city",
            "nansha",
            "--dataset",
            "test",
            "--model-path",
            "/tmp/model.keras",
        ]
    )

    assert called["argv"] == [
        "--city",
        "nansha",
        "--dataset",
        "test",
        "--model-path",
        "/tmp/model.keras",
    ]


def test_main_is_alias_of_stage4_main(
    stage4_mod,
    monkeypatch,
):
    seen = {}

    def _fake_stage4_main(argv=None):
        seen["argv"] = argv

    monkeypatch.setattr(
        stage4_mod,
        "stage4_main",
        _fake_stage4_main,
    )

    stage4_mod.main(["--city", "nansha"])

    assert seen["argv"] == ["--city", "nansha"]
