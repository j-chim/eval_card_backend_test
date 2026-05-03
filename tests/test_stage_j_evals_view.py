"""Stage J — `evals_view` materialisation."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import duckdb
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _run_through_stage_i(tmp_path, monkeypatch, config: str) -> Path:
    eee_root = FIXTURES / "eee"
    cards_root = FIXTURES / "auto_benchmarkcards"
    reg_root = FIXTURES / "entity_registry"
    warehouse = tmp_path / "warehouse"

    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    out_dir = pipeline.run(
        Settings.from_env(),
        configs=[config],
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
        cache_root=str(tmp_path / "cache"),
    )
    assert out_dir is not None
    return out_dir


def _materialise_views(out_dir: Path):
    from eval_card_backend.canonicalise import stages
    from eval_card_backend.canonicalise.resolver_setup import register_udfs
    from eval_card_backend.sources import registry as registry_src
    from eval_entity_resolver import Resolver

    con = duckdb.connect()
    alias_store = registry_src.load_alias_store(FIXTURES / "entity_registry")
    register_udfs(con, Resolver(alias_store))
    for table in ("fact_results", "benchmarks", "models", "canonical_metrics"):
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_parquet('{out_dir}/{table}.parquet')"
        )
    stages.stage_j_eval_results_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")
    return con


def test_evals_view_columns_match_spec(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    cols = {row[0]: row[1] for row in con.execute("DESCRIBE evals_view").fetchall()}

    expected = {
        "snapshot_id", "evaluation_id", "benchmark_id", "primary_metric_id",
        "evaluation_name", "canonical_display_name",
        "composite_benchmark_key", "composite_benchmark_name",
        "benchmark_family_key", "benchmark_leaf_key", "category",
        "metric_config",
        "models_count", "evaluator_names", "source_types",
        "latest_source_name", "third_party_ratio",
        "missing_generation_config_count",
        "best_model", "worst_model",
        "avg_score", "avg_score_norm", "top_score",
        "has_card", "benchmark_card",
        "is_aggregated", "aggregate_sources",
        "is_summary_score", "summary_eval_ids",
        "tags", "source_data",
        "reproducibility_summary", "provenance_summary", "comparability_summary",
        "instance_data",
        "metrics_count", "metric_names",
        "leaderboard_metrics", "leaderboard_rows",
        "root_metrics", "subtasks", "subtasks_count",
    }
    missing = expected - cols.keys()
    assert not missing, f"missing columns: {missing}"


def test_evaluation_id_round_trips(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    rows = con.execute(
        "SELECT benchmark_id, evaluation_id FROM evals_view"
    ).fetchall()
    for benchmark_id, slug in rows:
        assert unquote(slug) == benchmark_id


def test_primary_metric_picked_deterministically(tmp_path, monkeypatch):
    """Single-metric fixture → primary_metric_id = the only metric."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT primary_metric_id, metrics_count FROM evals_view"
    ).fetchone()
    primary, count = row
    assert primary == "accuracy"
    assert count == 1


def test_leaderboard_metrics_one_per_metric(tmp_path, monkeypatch):
    """leaderboard_metrics[] has one entry per (benchmark, metric)."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT leaderboard_metrics, metrics_count FROM evals_view"
    ).fetchone()
    metrics, count = row
    assert len(metrics) == count
    for m in metrics:
        assert {"column_key", "metric_summary_id", "metric_id",
                "metric_name", "display_name", "canonical_display_name",
                "lower_is_better", "unit", "scope",
                "subtask_key", "subtask_name"} <= set(m.keys())
        assert m["scope"] == "root"


def test_leaderboard_rows_pivoted_values_map(tmp_path, monkeypatch):
    """Each row's `values` is a MAP keyed by metric column_key.
    fixtures_variant: 3 fact rows, all same model+benchmark+metric,
    representative score = median = 0.78."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_views(out)
    row = con.execute("SELECT leaderboard_rows FROM evals_view").fetchone()[0]
    assert row is not None
    assert len(row) >= 1
    first = row[0]
    assert "values" in first
    # values keys are metric column keys (= metric_id today).
    metric_keys = list(first["values"].keys())
    assert len(metric_keys) == 1
    score = first["values"][metric_keys[0]]
    assert abs(score - 0.78) < 1e-9
    assert first["metrics_present"] == 1


def test_avg_score_normalisation_uses_metric_bounds(tmp_path, monkeypatch):
    """avg_score_norm = (avg_score - min) / (max - min). For a 0..1 metric
    avg_score == avg_score_norm."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_variant")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT avg_score, avg_score_norm FROM evals_view"
    ).fetchone()
    avg, norm = row
    assert avg is not None
    assert abs(avg - norm) < 1e-9  # 0..1 metric → identity


def test_best_model_uses_lower_is_better(tmp_path, monkeypatch):
    """best_model.score equals top_score for higher-is-better metrics."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT best_model, top_score FROM evals_view"
    ).fetchone()
    best, top = row
    assert best is not None
    assert best["score"] == top
    assert best["name"] == "GPT-4o"


def test_has_card_and_card_struct_populated(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    row = con.execute(
        "SELECT has_card, benchmark_card.benchmark_details.name "
        "FROM evals_view"
    ).fetchone()
    has_card, name = row
    assert has_card is True
    assert name is not None


def test_tags_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    tags = con.execute("SELECT tags FROM evals_view").fetchone()[0]
    assert set(tags.keys()) == {"domains", "languages", "tasks"}


def test_third_party_ratio_xparty_fixture(tmp_path, monkeypatch):
    """fixtures_xparty: 1 model, 1 benchmark, coverage_cell='both' → ratio=1.0."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    ratio = con.execute(
        "SELECT third_party_ratio FROM evals_view"
    ).fetchone()[0]
    assert ratio == 1.0


def test_provenance_summary_not_inflated_by_array_unnest(tmp_path, monkeypatch):
    """Regression: when a triple has multiple evaluator_relationships and
    multiple reporting_orgs, naive `CROSS JOIN UNNEST` of both arrays
    multiplies row counts by their cross-product. fixtures_xparty has 1
    triple with 2 relationships × 2 orgs — every per-triple SUM/COUNT
    must still equal 1, not 4."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    repro, prov, comp = con.execute(
        "SELECT reproducibility_summary, provenance_summary, comparability_summary "
        "FROM evals_view"
    ).fetchone()

    # One triple in fixtures_xparty — every triple-counting field must equal 1.
    assert repro["results_total"] == 1, (
        f"reproducibility.results_total inflated: {repro['results_total']}"
    )
    assert prov["total_groups"] == 1
    assert prov["total_results"] == 1
    assert comp["total_groups"] == 1
    # The triple is collaborative (one self + one third-party report).
    assert prov["source_type_distribution"]["collaborative"] == 1
    assert sum(prov["source_type_distribution"].values()) == 1
    # Cross-party divergence eligible for this triple (>1 distinct org).
    assert comp["groups_with_cross_party_check"] == 1


def test_is_aggregated_false_for_standalone(tmp_path, monkeypatch):
    """v1: is_aggregated is always FALSE."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    rows = con.execute("SELECT is_aggregated FROM evals_view").fetchall()
    for r in rows:
        assert r[0] is False


def test_primary_key_unique(tmp_path, monkeypatch):
    """(snapshot_id, evaluation_id) uniquely identifies every row."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_xparty")
    con = _materialise_views(out)
    n_rows, n_unique = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT (snapshot_id, evaluation_id)) "
        "FROM evals_view"
    ).fetchone()
    assert n_rows == n_unique


def test_instance_data_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    instance = con.execute("SELECT instance_data FROM evals_view").fetchone()[0]
    assert set(instance.keys()) == {
        "available", "url_count", "sample_urls",
        "models_with_loaded_instances",
    }


def test_metric_config_struct_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views(out)
    cfg = con.execute("SELECT metric_config FROM evals_view").fetchone()[0]
    assert set(cfg.keys()) == {
        "evaluation_description", "lower_is_better", "score_type",
        "min_score", "max_score", "unit",
    }
