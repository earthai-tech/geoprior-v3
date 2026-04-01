import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def test_parse_wide_df_columns_ignores_unit_suffix():
    from geoprior.plot.forecast import _parse_wide_df_columns

    df = pd.DataFrame(
        {
            "subsidence_2024_q50": [1.0],
            "subsidence_2024_actual": [1.1],
            "subsidence_2024_unit": ["mm"],
            "subsidence_q90": [1.2],
        }
    )
    out = _parse_wide_df_columns(df, ["subsidence"])

    assert (
        out["subsidence"]["2024"]["q50"]
        == "subsidence_2024_q50"
    )
    assert "unit" not in out["subsidence"]["2024"]
    assert (
        out["subsidence"]["static"]["q90"] == "subsidence_q90"
    )


def test_get_metrics_from_cols_skips_metadata_suffixes():
    from geoprior.plot.forecast import _get_metrics_from_cols

    cols = [
        "subsidence_actual",
        "subsidence_q10",
        "subsidence_q50",
        "subsidence_unit",
        "sample_idx",
    ]
    metrics = _get_metrics_from_cols(cols, ["subsidence"])

    assert "actual" in metrics
    assert "q10" in metrics
    assert "q50" in metrics
    assert "unit" not in metrics


def test_normalize_year_key_handles_common_inputs():
    from geoprior.plot.forecast import _normalize_year_key

    assert _normalize_year_key(2024) == "2024"
    assert _normalize_year_key(2024.0) == "2024"
    assert _normalize_year_key("2024.0") == "2024"
    assert _normalize_year_key("2024-01-01") == "2024"
