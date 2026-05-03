"""Unit tests for the four interpretive signals.

Pure-logic only — no DuckDB / no I/O. Maps roughly onto the spec test cases
referenced in notes/02-: TC-R*, TC-C*, TC-P*, TC-V*, TC-CP*.
"""
from __future__ import annotations

from eval_card_backend.signals.comparability import (
    aggregated_setup,
    compute_cross_party_divergence_py,
    compute_threshold,
    compute_variant_divergence_py,
    normalize_org_name,
)
from eval_card_backend.signals.completeness import compute_completeness_py
from eval_card_backend.signals.reproducibility import (
    compute_repro_missing_py,
    is_agentic_py,
)
from eval_card_backend.signals.setup import (
    canonical_json,
    differing_setup_fields,
    fact_id_py,
    normalize_setup,
    variant_key_py,
)


# ---------- setup helpers ----------

def test_normalize_setup_collapses_float_repr():
    # `.8g` formatting absorbs cosmetic IEEE-float noise but preserves 8-sig-fig
    # differences. 0.7 and 0.70000000001 collapse; 0.7 and 0.7000001 don't.
    a = normalize_setup({"temperature": 0.7, "max_tokens": 100})
    b = normalize_setup({"temperature": 0.70000000001, "max_tokens": "100"})
    assert a == b
    assert a["temperature"] == 0.7
    assert a["max_tokens"] == 100


def test_normalize_setup_text_strip_only():
    out = normalize_setup({"prompt_template": " hello\r\nworld \r"})
    assert out["prompt_template"] == "hello\nworld"


def test_variant_key_stable_under_key_order():
    a = variant_key_py({"temperature": 0.7, "max_tokens": 100})
    b = variant_key_py({"max_tokens": 100, "temperature": 0.7})
    assert a == b


def test_variant_key_handles_none():
    assert variant_key_py(None) == variant_key_py({})


def test_fact_id_determinism():
    assert fact_id_py("eval_x", 0) == fact_id_py("eval_x", 0)
    assert fact_id_py("eval_x", 0) != fact_id_py("eval_x", 1)
    assert fact_id_py(None, 0) is None
    assert fact_id_py("", 0) is None
    assert len(fact_id_py("eval_x", 0)) == 16


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json(None) is None


def test_differing_setup_fields_records_originals():
    out = differing_setup_fields([
        {"temperature": 0.0, "max_tokens": 100},
        {"temperature": 0.7, "max_tokens": 100},
    ])
    assert len(out) == 1
    assert out[0]["field"] == "temperature"
    # values are JSON-encoded for the STRUCT(field VARCHAR, "values" JSON)[] shape
    assert out[0]["values"] == "[0.0, 0.7]"


# ---------- reproducibility ----------

def test_compute_repro_missing_base():
    assert compute_repro_missing_py(False, True, True, False, False) == []
    assert compute_repro_missing_py(False, False, True, False, False) == ["temperature"]
    assert compute_repro_missing_py(False, True, False, False, False) == ["max_tokens"]


def test_compute_repro_missing_agentic():
    assert compute_repro_missing_py(True, True, True, True, True) == []
    assert compute_repro_missing_py(True, True, True, False, False) == [
        "eval_plan", "eval_limits"
    ]


def test_is_agentic_three_rules():
    # Rule 1: card.tasks
    assert is_agentic_py(
        "mmlu",
        {"purpose_and_intended_users": {"tasks": ["agentic"]}},
        None,
    )
    # Rule 2: agentic_eval_config presence
    assert is_agentic_py(
        "mmlu", None, {"agentic_eval_config": {"k": 1}}
    )
    # Rule 3: name regex against canonical id
    assert is_agentic_py("swe-bench-verified", None, None)
    # No rule fires
    assert not is_agentic_py("mmlu", None, None)
    # Robust to JSON-string inputs (DuckDB UDF call shape)
    assert is_agentic_py(
        "mmlu", '{"purpose_and_intended_users": {"tasks": ["agentic"]}}', None
    )


# ---------- completeness ----------

def test_completeness_empty_card():
    out = compute_completeness_py(None)
    assert out["total_fields_evaluated"] == 28
    assert out["populated_count"] == 0.0
    assert out["completeness_score"] == 0.0
    assert all(fs["score"] == 0 for fs in out["field_scores"])


def test_completeness_full_autobenchmarkcard():
    card = {
        "benchmark_details": {"name": "Foo", "overview": "x", "data_type": "qa",
                               "domains": ["a"], "languages": ["en"],
                               "similar_benchmarks": ["s"], "resources": ["r"]},
        "purpose_and_intended_users": {"goal": "g", "audience": ["a"],
                                        "tasks": ["x"], "limitations": "l",
                                        "out_of_scope_uses": ["o"]},
        "methodology": {"methods": ["m"], "metrics": ["x"], "calculation": "c",
                         "interpretation": "i", "baseline_results": "b",
                         "validation": "v"},
        "data": {"source": "s", "size": "z", "format": "f", "annotation": "a"},
        "ethical_and_legal_considerations": {
            "privacy_and_anonymity": "p",
            "data_licensing": "d",
            "consent_procedures": "c",
            "compliance_with_regulations": "r",
        },
    }
    out = compute_completeness_py(card)
    # 22 full + 1 partial (full) + 0 EEE/eval_cards = 23 / 28
    assert out["populated_count"] == 23
    assert abs(out["completeness_score"] - 23 / 28) < 1e-9


def test_completeness_partial_data_section():
    card = {"data": {"source": "s", "size": "z"}}  # 2 of 4 sub-items
    out = compute_completeness_py(card)
    partials = [p for p in out["partial_fields"] if p["field_path"].endswith(".data")]
    assert len(partials) == 1
    assert partials[0]["score"] == 0.5
    assert partials[0]["populated_subitems"] == 2
    assert partials[0]["total_subitems"] == 4


# ---------- comparability ----------

def test_normalize_org_name():
    assert normalize_org_name("Foo  Inc") == "foo inc"
    assert normalize_org_name(None) is None
    assert normalize_org_name("  ") is None


def test_compute_threshold_four_rules():
    assert compute_threshold({"metric_unit": "proportion"}) == (
        0.05, "proportion_or_continuous_normalized"
    )
    assert compute_threshold({"metric_kind": "continuous_normalized"}) == (
        0.05, "proportion_or_continuous_normalized"
    )
    assert compute_threshold({"metric_unit": "percent"}) == (5.0, "percent")
    t, b = compute_threshold({"min_score": 0, "max_score": 100})
    assert t == 5.0 and b == "range_5pct"
    assert compute_threshold({}) == (0.05, "fallback_default")
    assert compute_threshold(None) == (0.05, "fallback_default")


def test_aggregated_setup_lower_median():
    rows = [
        {"score": 0.5, "evaluation_id": "a", "generation_args": {"x": 1}},
        {"score": 0.7, "evaluation_id": "b", "generation_args": {"x": 2}},
        {"score": 0.9, "evaluation_id": "c", "generation_args": {"x": 3}},
    ]
    # n=3 → index 1 → x=2
    assert aggregated_setup(rows) == {"x": 2}
    # n=2 → index 0 (lower of two)
    assert aggregated_setup(rows[:2]) == {"x": 1}


def test_variant_divergence_returns_none_when_inapplicable():
    # < 2 rows
    assert compute_variant_divergence_py([], {}) is None
    # all identical setups
    rows = [
        {"fact_id": str(i), "evaluation_id": str(i), "score": 0.1 * i,
         "generation_args": {"temperature": 0.0},
         "evaluator_relationship": "first_party",
         "source_organization_name": "X"}
        for i in range(3)
    ]
    assert compute_variant_divergence_py(rows, {}) is None
    # < 2 scored rows
    rows[0]["score"] = None
    rows[1]["score"] = None
    rows[2]["generation_args"] = {"temperature": 0.7}
    assert compute_variant_divergence_py(rows, {}) is None


def test_variant_divergence_positive():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {"temperature": 0.0, "max_tokens": 100},
         "evaluator_relationship": "first_party", "source_organization_name": "X"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {"temperature": 0.7, "max_tokens": 100},
         "evaluator_relationship": "first_party", "source_organization_name": "X"},
    ]
    out = compute_variant_divergence_py(rows, {"metric_unit": "proportion"})
    assert out["has_variant_divergence"] is True
    assert out["threshold_basis"] == "proportion_or_continuous_normalized"
    fields = {f["field"] for f in out["differing_setup_fields"]}
    assert fields == {"temperature"}


def test_cross_party_divergence_returns_none_below_min_orgs():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo"},
    ]
    assert compute_cross_party_divergence_py(rows, {}) is None


def test_cross_party_divergence_normalised_orgs():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo Inc"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {}, "evaluator_relationship": "third_party",
         "source_organization_name": "  foo  inc"},
    ]
    # Same normalised org → < 2 distinct → None
    assert compute_cross_party_divergence_py(rows, {}) is None


def test_cross_party_divergence_positive():
    rows = [
        {"fact_id": "1", "evaluation_id": "a", "score": 0.5,
         "generation_args": {}, "evaluator_relationship": "first_party",
         "source_organization_name": "Foo"},
        {"fact_id": "2", "evaluation_id": "b", "score": 0.7,
         "generation_args": {}, "evaluator_relationship": "third_party",
         "source_organization_name": "Bar"},
    ]
    out = compute_cross_party_divergence_py(rows, {"metric_unit": "proportion"})
    assert out["has_cross_party_divergence"] is True
    assert out["organization_count"] == 2
    # Display-name keys preserved
    assert set(out["scores_by_organization"]) == {"Foo", "Bar"}
