"""Per-benchmark reporting completeness, scored against the operationalised
28-field set (see `registry/completeness_fields.json`).

Inputs:
  - the benchmark card payload (the AutoBenchmarkCards record, or None when
    no card is present).

The signal also wants `eee_eval.source_metadata.*` fields scored — but those
are per-row, not per-benchmark. Operationally we score them against the
benchmark card alone (the per-row source_metadata flows through `fact_results`
columns separately). The 28-field set still references them so the score's
denominator stays at 28 — they simply contribute 0 unless the per-benchmark
materialisation is later extended to include them. Reserved EvalCards fields
(`lifecycle_status`, `preregistration_url`) are scored 0 today (registry
doesn't carry them); they count toward the denominator per spec §4.2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_card_backend.signals.setup import _coerce_json


_COMPLETENESS_FIELDS_PATH = (
    Path(__file__).resolve().parent.parent / "registry" / "completeness_fields.json"
)


def _load_field_set() -> list[dict]:
    data = json.loads(_COMPLETENESS_FIELDS_PATH.read_text(encoding="utf-8"))
    return list(data.get("fields") or [])


COMPLETENESS_FIELD_SET: list[dict] = _load_field_set()


def _resolve_path(record: Any, path: str) -> Any:
    current = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _build_record(card: Any) -> dict:
    """Joined-record shape per legacy `compute_reporting_completeness`. The
    `eee_eval.source_metadata.*` fields are scored 0 here (per-row data flows
    via fact_results, not via this per-benchmark dim — see module docstring).
    """
    return {
        "autobenchmarkcard": card if isinstance(card, dict) else {},
        "eee_eval": {"source_metadata": {}},
        "evalcards": {},
    }


def compute_completeness_py(card: Any) -> dict:
    """Score the 28-field operationalised completeness set against `card`.

    Returns a dict matching `benchmark_completeness.parquet` column shape.
    """
    parsed = _coerce_json(card, caller="compute_completeness_py")
    record = _build_record(parsed)

    field_scores: list[dict] = []
    partial_fields: list[dict] = []

    for field in COMPLETENESS_FIELD_SET:
        path = field["path"]
        coverage = field["coverage"]

        if coverage in ("full", "reserved"):
            value = _resolve_path(record, path)
            score = 1.0 if value is not None else 0.0
            field_scores.append(
                {"field_path": path, "coverage_type": coverage, "score": score}
            )
        elif coverage == "partial":
            subitem_paths: list[str] = list(field.get("subitem_paths") or [])
            total = len(subitem_paths)
            populated = sum(
                1
                for sp in subitem_paths
                if _resolve_path(record, sp) is not None
            )
            score = (populated / total) if total else 0.0
            field_scores.append(
                {"field_path": path, "coverage_type": coverage, "score": score}
            )
            if 0 < score < 1:
                partial_fields.append(
                    {
                        "field_path": path,
                        "score": score,
                        "populated_subitems": populated,
                        "total_subitems": total,
                    }
                )
        else:
            raise ValueError(
                f"Unknown coverage type {coverage!r} for field {path!r}"
            )

    total = len(field_scores)
    populated_count = sum(fs["score"] for fs in field_scores)
    completeness_score = (populated_count / total) if total else 0.0
    missing_required_fields = [fs["field_path"] for fs in field_scores if fs["score"] == 0]

    return {
        "completeness_score": completeness_score,
        "total_fields_evaluated": total,
        "populated_count": populated_count,
        "missing_required_fields": missing_required_fields,
        "partial_fields": partial_fields,
        "field_scores": field_scores,
    }
