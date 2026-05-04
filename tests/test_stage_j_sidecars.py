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
    for table in (
        "fact_results", "benchmarks", "composites", "families", "models",
        "canonical_metrics",
    ):
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
    # The model-family rollup got renamed `model_families` (was `families`)
    # to disambiguate from the benchmark-family taxonomy in hierarchy.json.
    assert "model_families" in h
    assert "families" not in h
    assert "composites" in h
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


def test_hierarchy_top_level_shape(tmp_path, monkeypatch):
    """New shape: composites[] (primary tree) + families[] (lookup index)."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    assert {"generated_at", "stats", "composites", "families"} <= hi.keys()
    s = hi["stats"]
    assert {
        "composite_count", "family_count", "benchmark_count",
        "slice_count", "metric_count", "metric_rows_scanned",
    } <= s.keys()


def test_hierarchy_composites_have_required_keys(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    assert hi["composites"], "no composites in hierarchy"
    for c in hi["composites"]:
        assert {
            "key", "display_name", "category", "tags",
            "evals_count", "benchmarks",
        } <= c.keys()
        assert isinstance(c["benchmarks"], list)
        for b in c["benchmarks"]:
            assert {
                "key", "display_name", "family_id", "is_slice",
                "tags", "metrics", "slices", "summary_eval_ids",
            } <= b.keys()


def test_hierarchy_families_index_shape(tmp_path, monkeypatch):
    """families[] is a flat lookup: one entry per family_id with the
    list of member benchmark keys. No nested benchmarks."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    for fam in hi["families"]:
        assert {"key", "display_name", "member_benchmark_keys"} <= fam.keys()
        assert isinstance(fam["member_benchmark_keys"], list)
        # No legacy nested fields:
        assert "composites" not in fam
        assert "standalone_benchmarks" not in fam


def test_hierarchy_legacy_field_names_absent(tmp_path, monkeypatch):
    """The legacy benchmark-stem-keyed `families[].composites[]` /
    `families[].standalone_benchmarks[]` shape is gone (AC-7)."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    body = json.dumps(hi)
    assert "standalone_benchmarks" not in body
    assert "benchmark_family_key" not in body


def test_hierarchy_gpqa_diamond_emits_as_benchmark_not_slice(tmp_path):
    """A promoted family member should surface as a benchmark sibling,
    not as a slice under the bare-stem benchmark.
    """
    from eval_card_backend.canonicalise import sidecars

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE composites AS
        SELECT
            TIMESTAMP '2026-04-30 00:00:00' AS snapshot_id,
            'wasp' AS composite_slug,
            'WASP' AS composite_display_name,
            ['gpqa-diamond']::VARCHAR[] AS source_configs,
            2::BIGINT AS evals_count
        """
    )
    con.execute(
        """
        CREATE TABLE families AS
        SELECT
            TIMESTAMP '2026-04-30 00:00:00' AS snapshot_id,
            'gpqa' AS family_id,
            'GPQA family' AS family_display_name,
            ['gpqa', 'gpqa-diamond']::VARCHAR[] AS member_benchmark_keys
        """
    )
    con.execute(
        """
        CREATE TABLE benchmarks AS
        SELECT
            TIMESTAMP '2026-04-30 00:00:00' AS snapshot_id,
            'wasp' AS composite_slug,
            'WASP' AS composite_display_name,
            'gpqa' AS benchmark_id,
            'GPQA' AS display_name,
            'GPQA' AS benchmark_display_name,
            NULL::VARCHAR AS description,
            NULL::VARCHAR AS dataset_repo,
            NULL::VARCHAR AS parent_benchmark_id,
            'gpqa' AS family_id,
            'GPQA family' AS family_display_name,
            FALSE AS is_slice,
            ['reasoning']::VARCHAR[] AS registry_tags,
            NULL::JSON AS registry_metadata,
            'reviewed' AS review_status,
            NULL::VARCHAR AS card_name,
            NULL::VARCHAR AS overview,
            NULL::VARCHAR AS data_type,
            ['reasoning']::VARCHAR[] AS domains,
            []::VARCHAR[] AS languages,
            []::VARCHAR[] AS similar_benchmarks,
            []::VARCHAR[] AS resources,
            NULL::VARCHAR AS goal,
            []::VARCHAR[] AS audience,
            ['qa']::VARCHAR[] AS tasks,
            NULL::VARCHAR AS limitations,
            []::VARCHAR[] AS out_of_scope_uses,
            NULL::VARCHAR AS data_source,
            NULL::VARCHAR AS data_size,
            NULL::VARCHAR AS data_format,
            NULL::VARCHAR AS data_annotation,
            []::VARCHAR[] AS methods,
            []::VARCHAR[] AS card_metrics,
            NULL::VARCHAR AS calculation,
            NULL::VARCHAR AS interpretation,
            NULL::VARCHAR AS baseline_results,
            NULL::VARCHAR AS validation,
            NULL::VARCHAR AS privacy_and_anonymity,
            NULL::VARCHAR AS data_licensing,
            NULL::VARCHAR AS consent_procedures,
            NULL::VARCHAR AS compliance_with_regulations,
            NULL AS possible_risks,
            NULL::JSON AS flagged_fields,
            FALSE AS card_present,
            NULL::VARCHAR AS card_generated_by,
            0 AS card_flagged_count,
            0 AS card_missing_count
        UNION ALL
        SELECT
            TIMESTAMP '2026-04-30 00:00:00',
            'wasp',
            'WASP',
            'gpqa-diamond',
            'GPQA Diamond',
            'GPQA Diamond',
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            'gpqa',
            'GPQA family',
            FALSE,
            ['reasoning']::VARCHAR[],
            NULL::JSON,
            'reviewed',
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            ['reasoning']::VARCHAR[],
            []::VARCHAR[],
            []::VARCHAR[],
            []::VARCHAR[],
            NULL::VARCHAR,
            []::VARCHAR[],
            ['qa']::VARCHAR[],
            NULL::VARCHAR,
            []::VARCHAR[],
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            []::VARCHAR[],
            []::VARCHAR[],
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL::VARCHAR,
            NULL,
            NULL::JSON,
            FALSE,
            NULL::VARCHAR,
            0,
            0
        """
    )
    con.execute(
        """
        CREATE TABLE evals_view AS
        SELECT
            'wasp%2Fgpqa' AS evaluation_id,
            'wasp' AS composite_slug,
            'gpqa' AS benchmark_id,
            'GPQA' AS evaluation_name,
            'Reasoning' AS category,
            1::BIGINT AS models_count,
            NULL AS reproducibility_summary,
            NULL AS provenance_summary,
            NULL AS comparability_summary
        UNION ALL
        SELECT
            'wasp%2Fgpqa-diamond',
            'wasp',
            'gpqa-diamond',
            'GPQA Diamond',
            'Reasoning',
            1::BIGINT,
            NULL,
            NULL,
            NULL
        """
    )
    con.execute(
        """
        CREATE TABLE eval_results_view AS
        SELECT
            'wasp' AS composite_slug,
            'gpqa' AS benchmark_id,
            'accuracy' AS metric_id,
            'Accuracy' AS metric_display_name,
            struct_pack(source_organization_name := 'WASP') AS source_metadata
        UNION ALL
        SELECT
            'wasp',
            'gpqa-diamond',
            'accuracy',
            'Accuracy',
            struct_pack(source_organization_name := 'WASP')
        """
    )
    con.execute(
        """
        CREATE TABLE fact_results (
            composite_slug VARCHAR,
            benchmark_id VARCHAR,
            slice_key VARCHAR,
            slice_name VARCHAR,
            metric_id VARCHAR,
            model_key VARCHAR,
            org_raw VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("wasp", "gpqa", None, None, "accuracy", "m1", "WASP"),
            ("wasp", "gpqa-diamond", None, None, "accuracy", "m1", "WASP"),
        ],
    )
    con.execute(
        "CREATE TABLE canonical_metrics AS "
        "SELECT 'accuracy' AS id, 'Accuracy' AS display_name"
    )

    sidecars.write_hierarchy(
        con,
        tmp_path,
        {"snapshot_id": "2026-04-30T00:00:00Z"},
    )
    hierarchy = json.loads((tmp_path / "hierarchy.json").read_text())

    wasp = next(c for c in hierarchy["composites"] if c["key"] == "wasp")
    assert [b["key"] for b in wasp["benchmarks"]] == ["gpqa", "gpqa-diamond"]
    assert all(
        s["key"] != "gpqa-diamond"
        for b in wasp["benchmarks"]
        for s in b["slices"]
    )
    gpqa_family = next(f for f in hierarchy["families"] if f["key"] == "gpqa")
    assert gpqa_family["member_benchmark_keys"] == ["gpqa", "gpqa-diamond"]


def test_metric_count_matches_distinct_composite_benchmark_metric_triples(
    tmp_path, monkeypatch,
):
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    con = _materialise_views_and_sidecars(out)
    hi = json.loads((out / "hierarchy.json").read_text())
    expected = con.execute(
        "SELECT COUNT(DISTINCT (composite_slug, benchmark_id, metric_id)) "
        "FROM fact_results "
        "WHERE composite_slug IS NOT NULL "
        "  AND benchmark_id IS NOT NULL AND metric_id IS NOT NULL"
    ).fetchone()[0]
    assert hi["stats"]["metric_count"] == expected


def test_manifest_has_composite_count(tmp_path, monkeypatch):
    """AC-9 — manifest.json carries composite_count alongside the
    existing model/eval/metric counts."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "composite_count" in manifest
    assert manifest["composite_count"] >= 1


def test_comparison_index_evaluation_id_format(tmp_path, monkeypatch):
    """AC-10 — evaluation_id is `<composite_slug>/<benchmark_id>` URL-encoded."""
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    for eval_id in ci["evals"]:
        # Decoded form contains a `/` separator.
        from urllib.parse import unquote
        decoded = unquote(eval_id)
        assert "/" in decoded, f"evaluation_id missing /: {eval_id}"


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
    composite_slug / family_id / display_name / category, etc.
    """
    pytest.importorskip("duckdb")
    out = _run_through_stage_i(tmp_path, monkeypatch, "fixtures_clean")
    _materialise_views_and_sidecars(out)
    ci = json.loads((out / "comparison-index.json").read_text())
    assert ci["evals"], "comparison-index has no evals; nothing to validate"
    sample = next(iter(ci["evals"].values()))
    assert {
        "eval_summary_id",
        "composite_slug", "composite_display_name",
        "family_id", "family_display_name", "is_slice",
        "display_name", "category", "is_summary_score",
        "summary_score_for", "summary_eval_ids", "metrics",
    } <= set(sample.keys())
    # Legacy fields removed in the composite/family/slice cutover:
    assert "benchmark_family_key" not in sample
    assert "benchmark_leaf_key" not in sample


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
