# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

"""Reusable CLI helpers for tabular data I/O.

This module centralizes the small but repeated pieces that
many build commands need:

- parse one or many input paths,
- infer or force the input format,
- read CSV / TSV / Excel / Parquet files,
- combine many files into one DataFrame,
- preserve the source path when requested,
- expose reusable argparse arguments.

The goal is to keep data-loading behavior consistent across
GeoPrior build commands so each command only focuses on its
own business logic.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_FORMAT_ALIASES: dict[str, str] = {
    "auto": "auto",
    "csv": "csv",
    "txt": "csv",
    "tsv": "tsv",
    "tab": "tsv",
    "parquet": "parquet",
    "pq": "parquet",
    "xlsx": "excel",
    "xls": "excel",
    "excel": "excel",
    "json": "json",
    "feather": "feather",
    "ftr": "feather",
    "pkl": "pickle",
    "pickle": "pickle",
}

_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".feather": "feather",
    ".ftr": "feather",
    ".pkl": "pickle",
    ".pickle": "pickle",
}


@dataclass(frozen=True)
class TableSource:
    """Parsed input source for one table."""

    path: Path
    fmt: str
    sheet_name: str | int | None = None


def _dedupe_paths(
    paths: Iterable[Path],
) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []

    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)

    return out


def _parse_int_or_keep(
    value: str,
) -> str | int:
    text = str(value).strip()
    if not text:
        return text
    if text.isdigit() or (
        text.startswith("-") and text[1:].isdigit()
    ):
        return int(text)
    return text


def parse_sheet_name(
    value: str | int | None,
) -> str | int | None:
    """Normalize an Excel sheet selector."""
    if value is None:
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return 0

    low = text.lower()
    if low in {"none", "null"}:
        return None

    return _parse_int_or_keep(text)


def parse_header(
    value: str | int | None,
) -> int | None | str:
    """Normalize pandas header argument from CLI text."""
    if value is None:
        return "infer"
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return "infer"

    low = text.lower()
    if low == "infer":
        return "infer"
    if low in {"none", "null"}:
        return None

    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--header must be 'infer', 'none', or an integer."
        ) from exc


def parse_json_mapping(
    value: str | None,
) -> dict[str, Any]:
    """Parse a JSON object from CLI text."""
    if value is None:
        return {}

    text = str(value).strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "--read-kwargs must be valid JSON."
        ) from exc

    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError(
            "--read-kwargs must decode to a JSON object."
        )

    return obj


def parse_path_spec(
    value: str,
    *,
    default_sheet_name: str | int | None = 0,
    fmt: str = "auto",
) -> TableSource:
    """Parse one input path.

    Excel sources may be passed as:

    ``data.xlsx::Sheet1``
    ``data.xlsx::0``
    """
    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty input path is not allowed.")

    sheet_name = default_sheet_name
    path_text = raw

    if "::" in raw:
        path_text, sheet_text = raw.rsplit("::", 1)
        sheet_name = parse_sheet_name(sheet_text)

    path = Path(path_text).expanduser()
    return TableSource(
        path=path,
        fmt=normalize_format(fmt, path=path),
        sheet_name=sheet_name,
    )


def normalize_format(
    fmt: str | None,
    *,
    path: str | Path | None = None,
) -> str:
    """Resolve a canonical tabular format name."""
    text = "auto" if fmt is None else str(fmt).strip().lower()
    if text in _FORMAT_ALIASES:
        canon = _FORMAT_ALIASES[text]
    else:
        choices = ", ".join(sorted(_FORMAT_ALIASES))
        raise ValueError(
            f"Unsupported input format {fmt!r}. "
            f"Choose from: {choices}."
        )

    if canon != "auto":
        return canon

    if path is None:
        return "auto"

    suffix = Path(path).suffix.lower()
    resolved = _EXTENSION_TO_FORMAT.get(suffix)
    if resolved is None:
        raise ValueError(
            f"Could not infer input format from {path!r}. "
            "Use --input-format."
        )
    return resolved


def _expand_one_path(
    value: str,
) -> list[Path]:
    text = str(value).strip()

    if any(ch in text for ch in "*?[]"):
        matches = sorted(Path().glob(text))
        if not matches:
            raise FileNotFoundError(
                f"No files matched pattern: {text}"
            )
        files = [path for path in matches if path.is_file()]
        if not files:
            raise FileNotFoundError(
                f"Pattern matched no files: {text}"
            )
        return files

    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    return [path]


def expand_input_paths(
    paths: Sequence[str | Path],
) -> list[Path]:
    """Expand file paths and simple glob patterns."""
    out: list[Path] = []

    for item in paths:
        out.extend(_expand_one_path(str(item)))

    return _dedupe_paths(out)


def resolve_table_sources(
    paths: Sequence[str | Path],
    *,
    input_format: str = "auto",
    default_sheet_name: str | int | None = 0,
) -> list[TableSource]:
    """Resolve input specs into concrete table sources."""
    out: list[TableSource] = []

    for item in paths:
        raw = str(item).strip()
        if not raw:
            continue

        sheet_name = default_sheet_name
        path_text = raw

        if "::" in raw:
            path_text, sheet_text = raw.rsplit("::", 1)
            sheet_name = parse_sheet_name(sheet_text)

        for path in _expand_one_path(path_text):
            out.append(
                TableSource(
                    path=path,
                    fmt=normalize_format(
                        input_format,
                        path=path,
                    ),
                    sheet_name=sheet_name,
                )
            )

    if not out:
        raise ValueError("No input tables were resolved.")

    return out


def add_data_reader_args(
    parser: argparse.ArgumentParser,
    *,
    nargs: str = "+",
    paths_metavar: str = "PATH",
) -> argparse.ArgumentParser:
    """Attach reusable data-reader arguments to a parser."""
    parser.add_argument(
        "paths",
        nargs=nargs,
        metavar=paths_metavar,
        help=(
            "One or more input files. "
            "Excel files may use PATH::SHEET."
        ),
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=sorted(_FORMAT_ALIASES),
        help=(
            "Force the input format. "
            "Default: infer from file extension."
        ),
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help=(
            "Default Excel sheet name or zero-based index."
        ),
    )
    parser.add_argument(
        "--excel-engine",
        default=None,
        help="Optional pandas Excel engine.",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Delimiter for CSV-like files.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional text encoding.",
    )
    parser.add_argument(
        "--header",
        default="infer",
        help=(
            "Header row for text files. "
            "Use 'infer', 'none', or an integer."
        ),
    )
    parser.add_argument(
        "--usecols",
        nargs="+",
        default=None,
        help="Optional subset of columns to read.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit when reading.",
    )
    parser.add_argument(
        "--source-col",
        default=None,
        help=(
            "Optional column storing the source file for "
            "each loaded row."
        ),
    )
    parser.add_argument(
        "--read-kwargs",
        default=None,
        help=("Extra pandas reader kwargs as JSON."),
    )
    return parser


def reader_options_from_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build reader options from parsed CLI arguments."""
    return {
        "input_format": args.input_format,
        "default_sheet_name": parse_sheet_name(
            getattr(args, "sheet_name", 0)
        ),
        "excel_engine": getattr(args, "excel_engine", None),
        "sep": getattr(args, "sep", ","),
        "encoding": getattr(args, "encoding", None),
        "header": parse_header(
            getattr(args, "header", "infer")
        ),
        "usecols": getattr(args, "usecols", None),
        "nrows": getattr(args, "nrows", None),
        "source_col": getattr(args, "source_col", None),
        "read_kwargs": parse_json_mapping(
            getattr(args, "read_kwargs", None)
        ),
    }


def _read_csv_like(
    path: Path,
    *,
    sep: str,
    encoding: str | None,
    header: int | None | str,
    usecols: Sequence[str] | None,
    nrows: int | None,
    read_kwargs: dict[str, Any],
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        header=header,
        usecols=usecols,
        nrows=nrows,
        **read_kwargs,
    )


def read_table(
    source: TableSource,
    *,
    excel_engine: str | None = None,
    sep: str = ",",
    encoding: str | None = None,
    header: int | None | str = "infer",
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
    read_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read one tabular source into a DataFrame."""
    kwargs = dict(read_kwargs or {})
    fmt = source.fmt

    if fmt == "csv":
        return _read_csv_like(
            source.path,
            sep=sep,
            encoding=encoding,
            header=header,
            usecols=usecols,
            nrows=nrows,
            read_kwargs=kwargs,
        )

    if fmt == "tsv":
        tsv_sep = "\t" if sep == "," else sep
        return _read_csv_like(
            source.path,
            sep=tsv_sep,
            encoding=encoding,
            header=header,
            usecols=usecols,
            nrows=nrows,
            read_kwargs=kwargs,
        )

    if fmt == "excel":
        excel_header = 0 if header == "infer" else header
        return pd.read_excel(
            source.path,
            sheet_name=source.sheet_name,
            engine=excel_engine,
            header=excel_header,
            usecols=usecols,
            nrows=nrows,
            **kwargs,
        )

    if fmt == "parquet":
        return pd.read_parquet(
            source.path,
            columns=usecols,
            **kwargs,
        )

    if fmt == "json":
        return pd.read_json(
            source.path,
            **kwargs,
        )

    if fmt == "feather":
        return pd.read_feather(
            source.path,
            columns=usecols,
            **kwargs,
        )

    if fmt == "pickle":
        obj = pd.read_pickle(source.path, **kwargs)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(
            f"Pickle file did not contain a DataFrame: "
            f"{source.path}"
        )

    raise ValueError(
        f"Unsupported resolved input format: {fmt!r}."
    )


def read_tables(
    paths: Sequence[str | Path],
    *,
    input_format: str = "auto",
    default_sheet_name: str | int | None = 0,
    excel_engine: str | None = None,
    sep: str = ",",
    encoding: str | None = None,
    header: int | None | str = "infer",
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
    source_col: str | None = None,
    read_kwargs: dict[str, Any] | None = None,
) -> list[pd.DataFrame]:
    """Read one or many tables into a list of DataFrames."""
    sources = resolve_table_sources(
        paths,
        input_format=input_format,
        default_sheet_name=default_sheet_name,
    )
    tables: list[pd.DataFrame] = []

    for src in sources:
        frame = read_table(
            src,
            excel_engine=excel_engine,
            sep=sep,
            encoding=encoding,
            header=header,
            usecols=usecols,
            nrows=nrows,
            read_kwargs=read_kwargs,
        )
        if source_col:
            frame = frame.copy()
            frame[source_col] = str(src.path)
        tables.append(frame)

    return tables


def load_dataframe(
    paths: Sequence[str | Path],
    *,
    input_format: str = "auto",
    default_sheet_name: str | int | None = 0,
    excel_engine: str | None = None,
    sep: str = ",",
    encoding: str | None = None,
    header: int | None | str = "infer",
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
    source_col: str | None = None,
    read_kwargs: dict[str, Any] | None = None,
    ignore_index: bool = True,
) -> pd.DataFrame:
    """Load one or many tables as a single DataFrame."""
    tables = read_tables(
        paths,
        input_format=input_format,
        default_sheet_name=default_sheet_name,
        excel_engine=excel_engine,
        sep=sep,
        encoding=encoding,
        header=header,
        usecols=usecols,
        nrows=nrows,
        source_col=source_col,
        read_kwargs=read_kwargs,
    )

    if not tables:
        raise ValueError("No input tables were loaded.")

    if len(tables) == 1:
        return tables[0].reset_index(drop=ignore_index)

    return pd.concat(
        tables,
        axis=0,
        ignore_index=ignore_index,
        sort=False,
    )


def load_dataframe_from_args(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Load one combined DataFrame from parsed CLI args."""
    return load_dataframe(
        args.paths,
        **reader_options_from_args(args),
    )


def normalize_output_format(
    path: str | Path,
) -> str:
    """Infer output format from the file extension."""
    suffix = Path(path).suffix.lower()
    fmt = _EXTENSION_TO_FORMAT.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Unsupported output file extension: {suffix!r}."
        )
    return fmt


def write_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    *,
    excel_sheet_name: str = "Sheet1",
    excel_engine: str | None = None,
    index: bool = False,
) -> Path:
    """Write a DataFrame to a tabular output file."""
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = normalize_output_format(out)

    if fmt == "csv":
        df.to_csv(out, index=index)
    elif fmt == "tsv":
        df.to_csv(out, sep="\t", index=index)
    elif fmt == "parquet":
        df.to_parquet(out, index=index)
    elif fmt == "excel":
        df.to_excel(
            out,
            sheet_name=excel_sheet_name,
            engine=excel_engine,
            index=index,
        )
    elif fmt == "json":
        df.to_json(
            out,
            orient="records",
            indent=2,
        )
    elif fmt == "feather":
        df.reset_index(drop=True).to_feather(out)
    elif fmt == "pickle":
        df.to_pickle(out)
    else:
        raise ValueError(
            f"Unsupported output format: {fmt!r}."
        )

    return out
