# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from: earthai-tech/fusionlab-learn (BSD-3-Clause)
# https://github.com/earthai-tech/fusionlab-learn
#
# Modifications for GeoPrior-v3:
# Copyright (c) 2026-present Laurent Kouadio
# https://lkouadio.com

"""
GeoPrior logging utilities.

Provides `geopriorlog` to configure logging across the GeoPrior-v3
package, supporting YAML/INI configs and a safe default logger.

YAML supports ${LOG_PATH} placeholders. The resolved LOG_PATH is:
1) GEOPRIOR_LOG_PATH env var
2) LOG_PATH env var
3) `default_log_path` from the YAML
4) "~/.geoprior/logs"
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Optional

import yaml


__all__ = [
    "geopriorlog",
    "get_logger",
    "setup_logging",
    "OncePerMessageFilter",
]


class geopriorlog:
    @staticmethod
    def load_configuration(
        config_path: Optional[str] = None,
        use_default_logger: bool = True,
        verbose: bool = False,
    ) -> None:
        if not config_path:
            if use_default_logger:
                geopriorlog.set_default_logger()
            else:
                logging.basicConfig()
            return

        if verbose:
            print(f"[GeoPrior] Logging config: {config_path}")

        if config_path.endswith((".yaml", ".yml")):
            geopriorlog._configure_from_yaml(
                config_path=config_path,
                verbose=verbose,
            )
            return

        if config_path.endswith(".ini"):
            logging.config.fileConfig(
                config_path,
                disable_existing_loggers=False,
            )
            return

        logging.warning(
            "Unsupported logging config format: %s",
            config_path,
        )

    @staticmethod
    def get_geoprior_logger(
        name: str = "",
    ) -> logging.Logger:
        """GeoPrior logger accessor (preferred name)."""
        return geopriorlog.get_logger(name)
    
    @staticmethod
    def _configure_from_yaml(
        config_path: str,
        verbose: bool = False,
    ) -> None:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            msg = f"YAML config file not found: {path}"
            logging.error(msg)
            raise FileNotFoundError(msg)

        if verbose:
            print(f"[GeoPrior] Loading YAML: {path}")

        with path.open("rt", encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f.read())

        _apply_log_path_substitution(config)
        _ensure_handler_dirs(config)

        logging.config.dictConfig(config)

    @staticmethod
    def set_default_logger() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def get_logger(name: str = "") -> logging.Logger:
        return logging.getLogger(name)

    @staticmethod
    def set_logger_output(
        log_filename: str = "geoprior.log",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        file_mode: str = "a",
        format_: str = (
            "%(asctime)s - %(name)s - "
            "%(levelname)s - %(message)s"
        ),
        level: int = logging.INFO,
        logger_name: str = "geoprior",
    ) -> None:
        handler = logging.FileHandler(
            log_filename,
            mode=file_mode,
            encoding="utf-8",
        )
        handler.setLevel(level)
        formatter = logging.Formatter(
            format_,
            datefmt=date_format,
        )
        handler.setFormatter(formatter)

        logger = geopriorlog.get_logger(logger_name)
        logger.setLevel(level)
        logger.addHandler(handler)

        # Deduplicate handlers
        logger.handlers = list({id(h): h for h in logger.handlers}.values())


class OncePerMessageFilter(logging.Filter):
    """Let each distinct log message through exactly once."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg in self._seen:
            return False
        self._seen.add(msg)
        return True


def setup_logging(config_path: str) -> None:
    """
    Load a YAML logging config and apply ${LOG_PATH} substitution.

    Prefer env var GEOPRIOR_LOG_PATH, then LOG_PATH. Falls back
    to `default_log_path` from YAML or "~/.geoprior/logs".
    """
    path = Path(config_path).expanduser().resolve()
    with path.open("rt", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f.read())

    _apply_log_path_substitution(config)
    _ensure_handler_dirs(config)
    logging.config.dictConfig(config)


def _resolve_log_path(config: dict[str, Any]) -> str:
    env = os.getenv("GEOPRIOR_LOG_PATH") or os.getenv("LOG_PATH")
    if env:
        return str(Path(env).expanduser())
    yaml_default = config.get("default_log_path")
    if yaml_default:
        return str(Path(str(yaml_default)).expanduser())
    return "~/.geoprior/logs"


def _apply_log_path_substitution(config: dict[str, Any]) -> None:
    log_path = _resolve_log_path(config)
    handlers = config.get("handlers", {})

    for h in handlers.values():
        filename = h.get("filename")
        if not filename:
            continue
        h["filename"] = str(filename).replace(
            "${LOG_PATH}",
            log_path,
        )


def _ensure_handler_dirs(config: dict[str, Any]) -> None:
    handlers = config.get("handlers", {})
    for h in handlers.values():
        filename = h.get("filename")
        if not filename:
            continue
        p = Path(str(filename)).expanduser()
        if p.parent and str(p.parent) not in (".", ""):
            p.parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "") -> logging.Logger:
    """
    Shortcut for retrieving a named logger.

    Examples
    --------
    >>> from geoprior._geopriorlog import get_logger
    >>> log = get_logger(__name__)
    """
    return geopriorlog.get_logger(name)

if __name__ == "__main__":
    print(Path(__file__).resolve())