"""Stage J — JSON sidecars (manifest, headline, hierarchy)."""
from __future__ import annotations

import json
from pathlib import Path

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


def _materialise_views_and_sidecars(out_dir: Path):
    from eval_card_backend.canonicalise import sidecars, stages
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
    stages.stage_j_models_view(con, "2026-04-30T00:00:00Z")
    stages.stage_j_evals_view(con, "2026-04-30T00:00:00Z")

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    sidecars.write_manifest(con, out_dir, snap)
    sidecars.write_headline(con, out_dir, snap)
    sidecars.write_hierarchy(con, out_dir, snap)
    sidecars.write_comparison_index(con, out_dir, snap)
    return con


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------


def test_manifest_required_keys(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert {
        "generated_at", "config_version", "skipped_configs",
        "model_count", "eval_count", "metric_eval_count",
        "source_config_count", "skipped_config_count",
        "summary_artifacts",
    } <= set(manifest.keys())


def test_manifest_skipped_configs_lists_alphaxiv(tmp_path, monkeypatch):
    """alphaxiv is the canonical IGNORED_CONFIGS member."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "alphaxiv" in manifest["skipped_configs"]
    assert manifest["skipped_config_count"] == len(manifest["skipped_configs"])


def test_manifest_summary_artifact_pointers(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["summary_artifacts"]["corpus_aggregates"] == "headline.json"
    assert manifest["summary_artifacts"]["eval_hierarchy"] == "hierarchy.json"
    assert manifest["summary_artifacts"]["comparison_index"] == "comparison-index.json"


# ---------------------------------------------------------------------------
# headline.json
# ---------------------------------------------------------------------------


def test_headline_top_level_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    h = json.loads((out / "headline.json").read_text())
    for key in ("reproducibility", "completeness", "provenance", "comparability"):
        assert key in h
        assert "overall" in h[key]
        assert "by_category" in h[key]
    assert "developers" in h
    assert "families" in h
    assert "categories" in h


def test_headline_by_category_keys_match_typed_enum(tmp_path, monkeypatch):
    """by_category blocks key on every CategoryType — even when empty."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    h = json.loads((out / "headline.json").read_text())
    expected = {"General", "Reasoning", "Agentic", "Safety", "Knowledge"}
    for signal in ("reproducibility", "completeness", "provenance", "comparability"):
        assert set(h[signal]["by_category"].keys()) == expected


def test_reproducibility_per_field_missingness(tmp_path, monkeypatch):
    """fixtures_clean has full reproducibility coverage → all fields
    have missing_count=0 over the one triple."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    h = json.loads((out / "headline.json").read_text())
    pfm = h["reproducibility"]["overall"]["per_field_missingness"]
    # The active rule covers temperature + max_tokens (base) and eval_plan
    # + eval_limits (agentic-only).
    assert {"temperature", "max_tokens", "eval_plan", "eval_limits"} <= pfm.keys()
    assert pfm["temperature"]["denominator"] == "all_triples"
    assert pfm["eval_plan"]["denominator"] == "agentic_only"
    assert pfm["temperature"]["missing_count"] == 0


def test_developers_list_route_id_set(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    h = json.loads((out / "headline.json").read_text())
    assert len(h["developers"]) >= 1
    for dev in h["developers"]:
        assert "developer" in dev
        assert "route_id" in dev
        assert dev["model_count"] >= 0


def test_categories_list_typed(tmp_path, monkeypatch):
    """Each entry in categories[] uses the typed enum."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    h = json.loads((out / "headline.json").read_text())
    valid = {"General", "Reasoning", "Agentic", "Safety", "Knowledge"}
    for entry in h["categories"]:
        assert entry["category"] in valid


# ---------------------------------------------------------------------------
# hierarchy.json
# ---------------------------------------------------------------------------


def test_hierarchy_stats_conservation(tmp_path, monkeypatch):
    """family_count = composite_count + standalone_benchmark_count."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    s = hi["stats"]
    assert s["family_count"] == s["composite_count"] + s["standalone_benchmark_count"]


def test_hierarchy_families_have_required_keys(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    assert hi["families"], "no families in hierarchy"
    for fam in hi["families"]:
        assert {
            "key", "display_name", "category", "has_card", "tags",
            "evals_count", "eval_summary_ids",
            "composites", "standalone_benchmarks",
        } <= fam.keys()
        # Each family is either composite OR standalone (could be both on
        # mixed data; v1 fixtures only test the standalone path).
        assert isinstance(fam["composites"], list)
        assert isinstance(fam["standalone_benchmarks"], list)


def test_hierarchy_standalone_benchmark_path(tmp_path, monkeypatch):
    """fixtures don't carry parent_benchmark_id → every family is
    standalone, with one benchmark entry under standalone_benchmarks[]."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    fam = hi["families"][0]
    assert len(fam["composites"]) == 0
    assert len(fam["standalone_benchmarks"]) == 1
    leaf = fam["standalone_benchmarks"][0]
    assert leaf["key"] == fam["key"]
    assert "metrics" in leaf
    assert "slices" in leaf  # always present even if empty


def test_metric_count_matches_distinct_benchmark_metric_pairs(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    expected = con.execute(
        "SELECT COUNT(DISTINCT (benchmark_id, metric_id)) "
        "FROM fact_results "
        "WHERE benchmark_id IS NOT NULL AND metric_id IS NOT NULL"
    ).fetchone()[0]
    assert hi["stats"]["metric_count"] == expected


# ---------------------------------------------------------------------------
# comparison-index.json
# ---------------------------------------------------------------------------


def test_comparison_index_top_level_shape(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    assert {
        "generated_at", "config_version", "metric_group_order",
        "evals", "by_model",
    } <= set(ci.keys())
    assert ci["metric_group_order"][0] == "capability"


def test_comparison_index_eval_keyset_covers_eval_results_view(tmp_path, monkeypatch):
    """Every (evaluation_id, metric_summary_id) with a non-NULL score in
    eval_results_view must be reachable in comparison-index. The frontend
    skips evals not in this keyset, so any gap silently empties the grid.
    """
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    expected = {
        (row[0], row[1])
        for row in con.execute(
            """
            SELECT DISTINCT evaluation_id, metric_summary_id
            FROM eval_results_view
            WHERE score IS NOT NULL
              AND evaluation_id IS NOT NULL
              AND metric_summary_id IS NOT NULL
              AND model_route_id IS NOT NULL
            """
        ).fetchall()
    }
    actual = {
        (eval_id, metric["metric_summary_id"])
        for eval_id, eval_entry in ci["evals"].items()
        for metric in eval_entry["metrics"]
    }
    assert expected == actual


def test_comparison_index_eval_entry_required_fields(tmp_path, monkeypatch):
    """ComparisonEvalEntry contract — fields the frontend reads for
    benchmark_family_key, display_name, category, etc."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    assert ci["evals"], "comparison-index has no evals; nothing to validate"
    sample = next(iter(ci["evals"].values()))
    assert {
        "eval_summary_id", "benchmark_family_key", "benchmark_family_name",
        "benchmark_parent_key", "benchmark_parent_name",
        "benchmark_leaf_key", "benchmark_leaf_name",
        "display_name", "category", "is_summary_score",
        "summary_score_for", "summary_eval_ids", "metrics",
    } <= set(sample.keys())


def test_comparison_index_scores_ranked_best_first(tmp_path, monkeypatch):
    """Within a metric, scores[] is ranked best-first respecting
    lower_is_better; ties share a rank (dense-tie ranking)."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    for eval_entry in ci["evals"].values():
        for metric in eval_entry["metrics"]:
            scores = metric["scores"]
            if len(scores) < 2:
                continue
            lib = metric["lower_is_better"]
            for prev, curr in zip(scores, scores[1:]):
                if lib:
                    assert prev["score"] <= curr["score"]
                else:
                    assert prev["score"] >= curr["score"]
                # Rank monotonic; equal-score peers may share a rank.
                if prev["score"] == curr["score"]:
                    assert prev["rank"] == curr["rank"]
                else:
                    assert prev["rank"] < curr["rank"]


def test_comparison_index_metric_group_classified(tmp_path, monkeypatch):
    """Every metric carries a group from the legacy taxonomy (not just a
    flat default). Verifies the kind→group classifier and the regex
    fallback both flow through to the emitted artifact."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    valid = set(ci["metric_group_order"])
    seen_groups = set()
    for eval_entry in ci["evals"].values():
        for metric in eval_entry["metrics"]:
            assert metric["group"] in valid
            assert isinstance(metric["group_order"], int)
            seen_groups.add(metric["group"])
    # Fixtures_clean covers at least one capability metric (accuracy).
    assert "capability" in seen_groups


def test_comparison_index_metrics_ordered_by_group(tmp_path, monkeypatch):
    """Within an eval, metrics sort by (group_order, metric_name) so
    capability tabs surface first across every benchmark."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    for eval_entry in ci["evals"].values():
        metrics = eval_entry["metrics"]
        keys = [(m["group_order"], m["metric_name"] or "") for m in metrics]
        assert keys == sorted(keys)


def test_comparison_index_by_model_inverse_consistent(tmp_path, monkeypatch):
    """For every (route, eval, metric) entry in by_model, the corresponding
    score in evals[eval].metrics[metric].scores must agree on score/rank/total."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    for route, eval_map in ci["by_model"].items():
        for eval_id, metric_map in eval_map.items():
            for metric_summary_id, by_model_entry in metric_map.items():
                metric = next(
                    m for m in ci["evals"][eval_id]["metrics"]
                    if m["metric_summary_id"] == metric_summary_id
                )
                forward = next(
                    s for s in metric["scores"] if s["model_route_id"] == route
                )
                assert forward["score"] == by_model_entry["score"]
                assert forward["rank"]  == by_model_entry["rank"]
                assert forward["total"] == by_model_entry["total"]
