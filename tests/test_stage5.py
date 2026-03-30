from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def stage5_mod():
    return importlib.import_module("geoprior.cli.stage5")


def test_parse_override_value_handles_common_inputs(
    stage5_mod,
):
    fn = stage5_mod._parse_override_value

    assert fn("true") is True
    assert fn("False") is False
    assert fn("none") is None
    assert fn("42") == 42
    assert fn("3.5") == 3.5
    assert fn("[1, 2]") == [1, 2]
    assert fn("{'a': 1}") == {"a": 1}
    assert fn(" hello ") == "hello"
    assert fn("") == ""


def test_parse_set_items_coerces_values(stage5_mod):
    got = stage5_mod._parse_set_items(
        [
            "BATCH_SIZE=32",
            "TRANSFER_CITY_A='nansha'",
            "USE_HEAD_PROXY=false",
            "CALIBRATOR=None",
        ]
    )

    assert got == {
        "BATCH_SIZE": 32,
        "TRANSFER_CITY_A": "nansha",
        "USE_HEAD_PROXY": False,
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
    stage5_mod,
    item,
    match,
):
    with pytest.raises(SystemExit, match=match):
        stage5_mod._parse_set_items([item])


def test_cli_overrides_merge_known_flags_and_sets(
    stage5_mod,
):
    parser = stage5_mod._build_stage5_parser()
    args, forwarded = parser.parse_known_args(
        [
            "--city-a",
            "Nansha",
            "--city-b",
            "Zhongshan",
            "--model",
            "GeoPriorSubsNet",
            "--results-dir",
            "/tmp/results",
            "--set",
            "BATCH_SIZE=32",
            "--set",
            "TRANSFER_CITY_A='override'",
            "--splits",
            "val",
            "test",
            "--warm-epochs",
            "3",
        ]
    )

    got = stage5_mod._cli_overrides(args)

    assert got["TRANSFER_CITY_A"] == "nansha"
    assert got["TRANSFER_CITY_B"] == "zhongshan"
    assert got["MODEL_NAME"] == "GeoPriorSubsNet"
    assert got["RESULTS_DIR"] == "/tmp/results"
    assert got["BATCH_SIZE"] == 32
    assert forwarded == [
        "--splits",
        "val",
        "test",
        "--warm-epochs",
        "3",
    ]


def test_refresh_config_fields_rebuilds_filenames(
    stage5_mod,
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

    got = stage5_mod._refresh_config_fields(cfg)

    assert got["BIG_FN"] == (
        "zhongshan_final_main_std.cleaned.with_zsurf.csv"
    )
    assert got["SMALL_FN"] == (
        "zhongshan_2000.cleaned.with_zsurf.csv"
    )


def test_install_user_config_copies_file_and_drops_json(
    stage5_mod,
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
        stage5_mod,
        "get_config_paths",
        _fake_get_config_paths,
    )

    out = stage5_mod._install_user_config(
        str(src),
        config_root="nat.com",
    )

    assert out == str(Path(written["config_py"]).resolve())
    assert Path(out).read_text(encoding="utf-8") == (
        src.read_text(encoding="utf-8")
    )
    assert not old_json.exists()


def test_install_user_config_raises_for_missing_file(
    stage5_mod,
    tmp_path,
):
    missing = tmp_path / "missing_config.py"

    with pytest.raises(
        FileNotFoundError,
        match="Config file not found",
    ):
        stage5_mod._install_user_config(str(missing))


def test_persist_runtime_overrides_updates_payload(
    stage5_mod,
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
        stage5_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    updated = stage5_mod._persist_runtime_overrides(
        {
            "CITY_NAME": "zhongshan",
            "TRANSFER_CITY_A": "nansha",
            "TRANSFER_CITY_B": "zhongshan",
            "RESULTS_DIR": "/tmp/xfer",
        },
        config_root="nat.com",
    )

    payload = json.loads(
        Path(written["config_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert updated["CITY_NAME"] == "zhongshan"
    assert updated["RESULTS_DIR"] == "/tmp/xfer"
    assert updated["BIG_FN"].startswith("zhongshan_")
    assert payload["city"] == "zhongshan"
    assert (
        payload["model"] == natcom_config_dict["MODEL_NAME"]
    )
    assert payload["config"]["TRANSFER_CITY_A"] == "nansha"
    assert payload["config"]["RESULTS_DIR"] == "/tmp/xfer"
    assert "__meta__" in payload


def test_persist_runtime_overrides_without_changes_returns_cfg(
    stage5_mod,
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
        stage5_mod,
        "ensure_config_json",
        _fake_ensure_config_json,
    )

    got = stage5_mod._persist_runtime_overrides(
        None,
        config_root="nat.com",
    )

    assert got["CITY_NAME"] == natcom_config_dict["CITY_NAME"]
    assert got["BIG_FN"] == natcom_config_dict["BIG_FN"]


def test_legacy_module_name_defaults_to_private_stage5(
    stage5_mod,
):
    assert stage5_mod._legacy_module_name() == (
        "geoprior.cli._stage5"
    )


def test_print_help_shows_forwarded_legacy_examples(
    stage5_mod,
    capsys,
):
    stage5_mod._print_help()
    out = capsys.readouterr().out

    assert "Forwarded legacy arguments include" in out
    assert "--city-a CITY" in out
    assert "--model-name NAME" in out
    assert "--source-load {auto,full,weights}" in out
    assert "--continue-on-error" in out


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--city-a", "nansha"], True),
        (["--city-a=nansha"], True),
        (["--model-name=GeoPriorSubsNet"], False),
    ],
)
def test_has_flag_detects_flags(stage5_mod, argv, expected):
    assert stage5_mod._has_flag(argv, "--city-a") is expected


def test_cfg_first_returns_first_non_empty_value(
    stage5_mod,
):
    cfg = {
        "TRANSFER_CITY_A": "",
        "CITY_A": None,
        "SOURCE_CITY": "nansha",
    }

    got = stage5_mod._cfg_first(
        cfg,
        "TRANSFER_CITY_A",
        "CITY_A",
        "SOURCE_CITY",
        default="fallback",
    )

    assert got == "nansha"


def test_seed_forwarded_args_adds_defaults_from_cfg(
    stage5_mod,
):
    cfg = {
        "TRANSFER_CITY_A": "nansha",
        "TRANSFER_CITY_B": "zhongshan",
        "MODEL_NAME": "GeoPriorSubsNet",
        "RESULTS_DIR": "/tmp/results",
    }

    got = stage5_mod._seed_forwarded_args([], cfg)

    assert got == [
        "--city-a",
        "nansha",
        "--city-b",
        "zhongshan",
        "--model-name",
        "GeoPriorSubsNet",
        "--results-dir",
        "/tmp/results",
    ]


def test_seed_forwarded_args_does_not_override_existing_flags(
    stage5_mod,
):
    cfg = {
        "TRANSFER_CITY_A": "nansha",
        "TRANSFER_CITY_B": "zhongshan",
        "MODEL_NAME": "GeoPriorSubsNet",
        "RESULTS_DIR": "/tmp/results",
    }

    got = stage5_mod._seed_forwarded_args(
        [
            "--city-a",
            "alpha",
            "--model-name=CustomModel",
            "--results-dir",
            "/already/there",
            "--splits",
            "val",
        ],
        cfg,
    )

    assert got == [
        "--city-a",
        "alpha",
        "--model-name=CustomModel",
        "--results-dir",
        "/already/there",
        "--splits",
        "val",
        "--city-b",
        "zhongshan",
    ]


def test_seed_forwarded_args_uses_fallback_keys_and_env(
    stage5_mod,
    monkeypatch,
):
    monkeypatch.setenv("RESULTS_DIR", "/env/results")
    cfg = {
        "CITY_A": "nansha",
        "TARGET_CITY": "zhongshan",
        "TRANSFER_MODEL_NAME": "GeoPriorSubsNet",
    }

    got = stage5_mod._seed_forwarded_args([], cfg)

    assert got == [
        "--city-a",
        "nansha",
        "--city-b",
        "zhongshan",
        "--model-name",
        "GeoPriorSubsNet",
        "--results-dir",
        "/env/results",
    ]


def test_run_stage5_help_path_prints_help_and_skips_dispatch(
    stage5_mod,
    monkeypatch,
):
    seen = {"help": 0}

    def _fake_print_help():
        seen["help"] += 1

    def _boom(*args, **kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(
        stage5_mod,
        "_print_help",
        _fake_print_help,
    )
    monkeypatch.setattr(
        stage5_mod,
        "_persist_runtime_overrides",
        _boom,
    )
    monkeypatch.setattr(
        stage5_mod.importlib,
        "import_module",
        _boom,
    )

    stage5_mod.run_stage5(["--help"])

    assert seen["help"] == 1


def test_run_stage5_seeds_forwarded_args_and_restores_sys_argv(
    stage5_mod,
    monkeypatch,
):
    seen = {}

    def _fake_persist_runtime_overrides(
        overrides=None,
        *,
        config_root="nat.com",
    ):
        seen["overrides"] = overrides
        seen["config_root"] = config_root
        return {
            "TRANSFER_CITY_A": "nansha",
            "TRANSFER_CITY_B": "zhongshan",
            "MODEL_NAME": "GeoPriorSubsNet",
            "RESULTS_DIR": "/tmp/results",
        }

    def _legacy_main():
        seen["argv_inside"] = list(stage5_mod.sys.argv)

    fake_mod = SimpleNamespace(main=_legacy_main)

    monkeypatch.setattr(
        stage5_mod,
        "_persist_runtime_overrides",
        _fake_persist_runtime_overrides,
    )
    monkeypatch.setattr(
        stage5_mod.importlib,
        "import_module",
        lambda name: fake_mod,
    )
    monkeypatch.setattr(
        stage5_mod.sys,
        "argv",
        ["outer-cmd", "--keep"],
    )

    stage5_mod.run_stage5(
        [
            "--config-root",
            "nat.alt",
            "--city-a",
            "nansha",
            "--splits",
            "val",
            "test",
            "--warm-epochs",
            "3",
        ]
    )

    assert seen["overrides"] == {"TRANSFER_CITY_A": "nansha"}
    assert seen["config_root"] == "nat.alt"
    assert seen["argv_inside"] == [
        "stage5-transfer",
        "--splits",
        "val",
        "test",
        "--warm-epochs",
        "3",
        "--city-a",
        "nansha",
        "--city-b",
        "zhongshan",
        "--model-name",
        "GeoPriorSubsNet",
        "--results-dir",
        "/tmp/results",
    ]
    assert stage5_mod.sys.argv == ["outer-cmd", "--keep"]


def test_run_stage5_installs_user_config_when_requested(
    stage5_mod,
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
            "TRANSFER_CITY_A": "nansha",
            "TRANSFER_CITY_B": "zhongshan",
            "MODEL_NAME": "GeoPriorSubsNet",
            "RESULTS_DIR": "/tmp/results",
        }

    fake_mod = SimpleNamespace(main=lambda: None)

    monkeypatch.setattr(
        stage5_mod,
        "_install_user_config",
        _fake_install_user_config,
    )
    monkeypatch.setattr(
        stage5_mod,
        "_persist_runtime_overrides",
        _fake_persist_runtime_overrides,
    )
    monkeypatch.setattr(
        stage5_mod.importlib,
        "import_module",
        lambda name: fake_mod,
    )

    stage5_mod.run_stage5(
        [
            "--config",
            "/tmp/custom_config.py",
            "--config-root",
            "nat.alt",
            "--splits",
            "val",
        ]
    )

    assert seen["config_path"] == "/tmp/custom_config.py"
    assert seen["config_root"] == "nat.alt"


def test_run_stage5_raises_when_legacy_main_is_missing(
    stage5_mod,
    monkeypatch,
):
    monkeypatch.setattr(
        stage5_mod,
        "_persist_runtime_overrides",
        lambda overrides=None, *, config_root="nat.com": {
            "TRANSFER_CITY_A": "nansha",
            "TRANSFER_CITY_B": "zhongshan",
            "MODEL_NAME": "GeoPriorSubsNet",
            "RESULTS_DIR": "/tmp/results",
        },
    )
    monkeypatch.setattr(
        stage5_mod.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(
        AttributeError,
        match="Missing 'main' in stage5_legacy",
    ):
        stage5_mod.run_stage5(["--splits", "val"])


def test_stage5_main_delegates_to_run_stage5(
    stage5_mod,
    monkeypatch,
):
    called = {}

    def _fake_run_stage5(argv=None):
        called["argv"] = argv

    monkeypatch.setattr(
        stage5_mod,
        "run_stage5",
        _fake_run_stage5,
    )

    stage5_mod.stage5_main(
        [
            "--city-a",
            "nansha",
            "--city-b",
            "zhongshan",
            "--splits",
            "val",
        ]
    )

    assert called["argv"] == [
        "--city-a",
        "nansha",
        "--city-b",
        "zhongshan",
        "--splits",
        "val",
    ]


def test_main_is_alias_of_stage5_main(
    stage5_mod,
    monkeypatch,
):
    seen = {}

    def _fake_stage5_main(argv=None):
        seen["argv"] = argv

    monkeypatch.setattr(
        stage5_mod,
        "stage5_main",
        _fake_stage5_main,
    )

    stage5_mod.main(["--city-a", "nansha"])

    assert seen["argv"] == ["--city-a", "nansha"]
