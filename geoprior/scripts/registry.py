# SPDX-License-Identifier: Apache-2.0
# GeoPrior-v3 — https://github.com/earthai-tech/geoprior-v3
# Copyright (c) 2026-present
# Author: LKouadio <https://lkouadio.com>

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptSpec:
    mod: str
    fn: str
    desc: str
    mode: str = "argv"
    family: str = "build"
    public_name: str | None = None
    aliases: tuple[str, ...] = ()
    legacy_names: tuple[str, ...] = ()


def _drop_known_prefix(name: str) -> tuple[str, str]:
    if name.startswith("plot-"):
        return "plot", name[len("plot-") :]
    if name.startswith("build-"):
        return "build", name[len("build-") :]
    if name.startswith("make-"):
        return "build", name[len("make-") :]
    return "build", name


def _spec(
    legacy_name: str,
    mod: str,
    fn: str,
    desc: str,
    *,
    mode: str = "argv",
    family: str | None = None,
    public_name: str | None = None,
    aliases: tuple[str, ...] = (),
) -> ScriptSpec:
    auto_family, auto_public = _drop_known_prefix(legacy_name)
    return ScriptSpec(
        mod=mod,
        fn=fn,
        desc=desc,
        mode=mode,
        family=family or auto_family,
        public_name=public_name or auto_public,
        aliases=aliases,
        legacy_names=(legacy_name,),
    )


SCRIPT_COMMANDS: dict[str, ScriptSpec] = {
    "plot-driver-response": _spec(
        "plot-driver-response",
        "plot_driver_response",
        "plot_driver_response_main",
        "Driver-response figure.",
    ),
    "plot-core-ablation": _spec(
        "plot-core-ablation",
        "plot_core_ablation",
        "plot_fig3_core_ablation_main",
        "Core ablation figure.",
    ),
    "plot-litho-parity": _spec(
        "plot-litho-parity",
        "plot_litho_parity",
        "figS1_lithology_parity_main",
        "Lithology parity figure.",
    ),
    "plot-uncertainty": _spec(
        "plot-uncertainty",
        "plot_uncertainty",
        "plot_fig5_uncertainty_main",
        "Forecast uncertainty figure.",
    ),
    "plot-spatial-forecasts": _spec(
        "plot-spatial-forecasts",
        "plot_spatial_forecasts",
        "plot_fig6_spatial_forecasts_main",
        "Spatial forecast maps.",
    ),
    "plot-physics-sanity": _spec(
        "plot-physics-sanity",
        "plot_physics_sanity",
        "plot_physics_sanity_main",
        "Physics sanity plots.",
    ),
    "plot-physics-maps": _spec(
        "plot-physics-maps",
        "plot_physics_maps",
        "plot_physics_maps_main",
        "Physics maps plots.",
    ),
    "plot-physics-fields": _spec(
        "plot-physics-fields",
        "plot_physics_fields",
        "plot_physics_fields_main",
        "Physics fields plots.",
    ),
    "plot-physics-profiles": _spec(
        "plot-physics-profiles",
        "plot_physics_profiles",
        "figA1_phys_profiles_main",
        "Physics profiles (Appendix).",
    ),
    "plot-uncertainty-extras": _spec(
        "plot-uncertainty-extras",
        "plot_uncertainty_extras",
        "supp_figS5_uncertainty_extras_main",
        "Extra uncertainty panels.",
        aliases=("plot_uncertainty_extras",),
    ),
    "plot-ablations-sensitivity": _spec(
        "plot-ablations-sensitivity",
        "plot_ablations_sensitivity",
        "plot_ablations_sensivity_main",
        "Ablations sensitivity.",
    ),
    "plot-physics-sensitivity": _spec(
        "plot-physics-sensitivity",
        "plot_physics_sensitivity",
        "plot_physics_sensitivity_main",
        "Physics sensitivity.",
    ),
    "plot-sm3-identifiability": _spec(
        "plot-sm3-identifiability",
        "plot_sm3_identifiability",
        "plot_sm3_identifiability_main",
        "SM3 identifiability figure.",
    ),
    "plot-sm3-bounds-ridge-summary": _spec(
        "plot-sm3-bounds-ridge-summary",
        "plot_sm3_bounds_ridge_summary",
        "plot_sm3_bounds_ridge_summary_main",
        "SM3 bounds vs ridge summary.",
        aliases=("plot-sm3-bounds-ridge",),
    ),
    "plot-sm3-log-offsets": _spec(
        "plot-sm3-log-offsets",
        "plot_sm3_log_offsets",
        "plot_sm3_log_offsets_main",
        "SM3 log-offset diagnostics.",
    ),
    "plot-xfer-transferability": _spec(
        "plot-xfer-transferability",
        "plot_xfer_transferability",
        "figSx_xfer_transferability_main",
        "Cross-city transferability.",
    ),
    "plot-xfer-impact": _spec(
        "plot-xfer-impact",
        "plot_xfer_impact",
        "figSx_xfer_impact_main",
        "Transfer impact (retention + risk).",
    ),
    "plot-transfer": _spec(
        "plot-transfer",
        "plot_xfer_transferability",
        "figSx_xfer_transferability_main",
        "Alias of transferability plot.",
        public_name="transfer",
    ),
    "plot-transfer-impact": _spec(
        "plot-transfer-impact",
        "plot_xfer_impact",
        "figSx_xfer_impact_main",
        "Alias of transfer impact plot.",
        public_name="transfer-impact",
    ),
    "plot-geo-cumulative": _spec(
        "plot-geo-cumulative",
        "plot_geo_cumulative",
        "plot_geo_cumulative_main",
        "Cumulative geo curves.",
    ),
    "plot-hotspot-analytics": _spec(
        "plot-hotspot-analytics",
        "plot_hotspot_analytics",
        "plot_hotspot_analytics_main",
        "Hotspot analytics (maps + timeline).",
    ),
    "plot-external-validation": _spec(
        "plot-external-validation",
        "plot_external_validation",
        "plot_external_validation_main",
        "External point-support validation figure.",
    ),
    "compute-brier-exceedance": _spec(
        "compute-brier-exceedance",
        "compute_brier_exceedance",
        "brier_exceedance_main",
        "Compute exceedance Brier table.",
        family="build",
        public_name="brier-exceedance",
    ),
    "summarize-hotspots": _spec(
        "summarize-hotspots",
        "summarize_hotspots",
        "summarize_hotspots_main",
        "Summarize hotspot outputs.",
        family="build",
        public_name="hotspots-summary",
    ),
    "compute-hotspots": _spec(
        "compute-hotspots",
        "compute_hotspots",
        "compute_hotspots_main",
        "Compute hotspot outputs.",
        family="build",
        public_name="hotspots",
    ),
    "extend-forecast": _spec(
        "extend-forecast",
        "extend_forecast",
        "extend_forecast_main",
        "Extend future forecast CSV by extrapolation.",
        family="build",
    ),
    "update-ablation-records": _spec(
        "update-ablation-records",
        "update_ablation_records",
        "update_ablation_records_main",
        "Patch ablation record JSONL with metrics.",
        family="build",
    ),
    "build-model-metrics": _spec(
        "build-model-metrics",
        "build_model_metrics",
        "build_model_metrics_main",
        "Build unified metrics tables (CSV/JSON).",
    ),
    "build-ablation-table": _spec(
        "build-ablation-table",
        "build_ablation_table",
        "build_ablation_table_main",
        "Build ablation table from record JSONL.",
    ),
    "make-boundary": _spec(
        "make-boundary",
        "make_boundary",
        "make_boundary_main",
        "Create boundary polygon from points.",
    ),
    "make-exposure": _spec(
        "make-exposure",
        "make_exposure",
        "make_exposure_main",
        "Create exposure.csv (proxy) from points.",
    ),
    "make-district-grid": _spec(
        "make-district-grid",
        "make_district_grid",
        "make_district_grid_main",
        "Create grid-based district layer.",
    ),
    "tag-clusters-with-zones": _spec(
        "tag-clusters-with-zones",
        "tag_clusters_with_zones",
        "tag_clusters_with_zones_main",
        "Assign hotspot clusters to Zone IDs.",
        family="build",
        public_name="clusters-with-zones",
    ),
}


SCRIPT_GROUPS = (
    (
        "Figures",
        (
            "driver-response",
            "core-ablation",
            "litho-parity",
            "uncertainty",
            "spatial-forecasts",
            "transfer-impact",
            "hotspot-analytics",
        ),
    ),
    (
        "Supplementary",
        (
            "physics-sanity",
            "physics-fields",
            "physics-maps",
            "physics-profiles",
            "uncertainty-extras",
            "ablations-sensitivity",
            "physics-sensitivity",
            "sm3-identifiability",
            "sm3-bounds-ridge-summary",
            "sm3-log-offsets",
            "xfer-transferability",
            "xfer-impact",
            "transfer",
            "geo-cumulative",
        ),
    ),
    (
        "Tables & summaries",
        (
            "brier-exceedance",
            "hotspots",
            "hotspots-summary",
            "update-ablation-records",
            "ablation-table",
            "model-metrics",
            "extend-forecast",
            "boundary",
            "exposure",
            "district-grid",
            "clusters-with-zones",
        ),
    ),
)
