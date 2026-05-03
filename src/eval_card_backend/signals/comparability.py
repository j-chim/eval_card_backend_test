"""Variant- and cross-party-divergence over a (model, benchmark, metric) group.

Both functions return None when the signal is "not applicable" for the group;
distinguishing N/A (NULL on the row) from "applicable & no divergence" (FALSE)
is mandatory — the frontend renders the two differently.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

from eval_card_backend.signals.setup import (
    _coerce_json,
    canonical_json,
    differing_setup_fields,
    normalize_setup,
)


_WHITESPACE_REGEX = re.compile(r"\s+")


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_org_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    cleaned = _WHITESPACE_REGEX.sub(" ", name).strip()
    if not cleaned:
        return None
    return cleaned.lower()


def _display_org_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    cleaned = _WHITESPACE_REGEX.sub(" ", name).strip()
    return cleaned or None


def compute_threshold(metric_config: Any) -> tuple[float, str]:
    """Return (threshold, basis_label). Basis is one of four labels — see notes/01-.

    Inputs come from the registry's `canonical_metrics`. Per-record metric_config
    fields are not consulted (legacy used them; the new design centralises in
    the registry to avoid per-record drift).
    """
    if isinstance(metric_config, dict):
        metric_unit = metric_config.get("metric_unit")
        metric_kind = metric_config.get("metric_kind")
        if metric_unit == "proportion" or metric_kind == "continuous_normalized":
            return 0.05, "proportion_or_continuous_normalized"
        if metric_unit == "percent":
            return 5.0, "percent"
        min_score = metric_config.get("min_score")
        max_score = metric_config.get("max_score")
        if (
            _is_real_number(min_score)
            and _is_real_number(max_score)
            and max_score > min_score
        ):
            return 0.05 * (max_score - min_score), "range_5pct"
    return 0.05, "fallback_default"


def aggregated_setup(rows_for_org: list[dict]) -> dict | None:
    """Lower-median rule: sort by (score, evaluation_id) ascending — None scores
    treated as +inf — pick row at index (n - 1) // 2.

    For odd n this is the median row; for even n the lower of the two middle rows.
    """
    if not rows_for_org:
        return None
    if len(rows_for_org) == 1:
        return rows_for_org[0].get("generation_args")
    sorted_rows = sorted(
        rows_for_org,
        key=lambda r: (
            r["score"] if _is_real_number(r.get("score")) else float("inf"),
            str(r.get("evaluation_id") or ""),
        ),
    )
    n = len(sorted_rows)
    return sorted_rows[(n - 1) // 2].get("generation_args")


def _coerce_rows_gen_args(rows: list[dict], caller: str) -> None:
    for row in rows:
        row["generation_args"] = _coerce_json(row.get("generation_args"), caller=caller)


def compute_variant_divergence_py(
    rows: list[dict], metric_config: Any
) -> dict | None:
    """Spec §6.1 + legacy. Returns None when not applicable.

    Each row dict carries: fact_id, evaluation_id, score, generation_args
    (str | dict), evaluator_relationship, source_organization_name.
    """
    _coerce_rows_gen_args(rows, "compute_variant_divergence_py.gen_args")

    if len(rows) < 2:
        return None

    setups = [normalize_setup(r.get("generation_args")) for r in rows]
    diffs = differing_setup_fields(setups)
    if not diffs:
        return None

    rows_with_score = [r for r in rows if _is_real_number(r.get("score"))]
    if len(rows_with_score) < 2:
        return None

    scores = [r["score"] for r in rows_with_score]
    divergence = max(scores) - min(scores)
    threshold, basis = compute_threshold(metric_config)

    return {
        "has_variant_divergence": divergence > threshold,
        "divergence_magnitude": divergence,
        "threshold_used": threshold,
        "threshold_basis": basis,
        "differing_setup_fields": diffs,
    }


def compute_cross_party_divergence_py(
    rows: list[dict], metric_config: Any
) -> dict | None:
    """Spec §6.2 + legacy. Returns None when fewer than 2 distinct named orgs.

    Per-org score is the median of the org's scored rows; per-org representative
    setup is `aggregated_setup` of those rows (lower-median). differing_setup_fields
    is computed across org-representative setups (not across all rows).
    """
    _coerce_rows_gen_args(rows, "compute_cross_party_divergence_py.gen_args")

    by_org: dict[str, list[dict]] = {}
    org_display: dict[str, str] = {}

    for row in rows:
        if not _is_real_number(row.get("score")):
            continue
        raw_org = row.get("source_organization_name")
        normalized = normalize_org_name(raw_org)
        if not normalized:
            continue
        by_org.setdefault(normalized, []).append(row)
        if normalized not in org_display:
            org_display[normalized] = _display_org_name(raw_org) or normalized

    if len(by_org) < 2:
        return None

    org_scores: dict[str, float] = {}
    org_setups: dict[str, dict | None] = {}
    for normalized, org_rows in by_org.items():
        org_scores[normalized] = statistics.median(r["score"] for r in org_rows)
        org_setups[normalized] = aggregated_setup(org_rows)

    divergence = max(org_scores.values()) - min(org_scores.values())
    threshold, basis = compute_threshold(metric_config)

    setups_for_diff = [
        normalize_setup(s) for s in org_setups.values()
    ]
    diffs = differing_setup_fields(setups_for_diff)

    scores_by_org = {
        org_display[normalized]: score for normalized, score in org_scores.items()
    }

    return {
        "has_cross_party_divergence": divergence > threshold,
        "divergence_magnitude": divergence,
        "threshold_used": threshold,
        "threshold_basis": basis,
        "differing_setup_fields": diffs,
        "organization_count": len(by_org),
        "scores_by_organization": scores_by_org,
    }
