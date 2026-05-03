"""End-to-end smoke test: full pipeline over a hand-built fixture.

Builds a tiny EEE/cards/registry fixture in `tmp_path`, runs the canonicalisation
pipeline, asserts the warehouse parquets exist with non-zero rows. Exercises
the full DuckDB stage chain + UDF registration.

Skipped if duckdb isn't installed (it's now a hard dep so this is just paranoia).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_eee_fixture(eee_root: Path) -> None:
    config_dir = eee_root / "data" / "minibench" / "openai" / "gpt-4o"
    config_dir.mkdir(parents=True)
    record = {
        "evaluation_id": "ev_001",
        "schema_version": "0.2.2",
        "retrieved_timestamp": "2026-04-30T00:00:00Z",
        "model_info": {
            "developer": "openai", "name": "GPT-4o", "id": "openai/gpt-4o",
            "inference_platform": "openai-api",
        },
        "source_metadata": {
            "source_name": "OpenAI", "source_type": "documentation",
            "source_organization_name": "OpenAI",
            "evaluator_relationship": "first_party",
        },
        "eval_library": {"name": "minibench", "version": "1.0"},
        "evaluation_results": [
            {
                "evaluation_name": "minibench",
                "source_data": {"dataset_name": "minibench"},
                "metric_config": {
                    "metric_id": "minibench.acc",
                    "metric_name": "accuracy",
                    "evaluation_description": "Accuracy on minibench",
                },
                "score_details": {"score": 0.85},
                "generation_config": {
                    "generation_args": {"temperature": 0.0, "max_tokens": 1024}
                },
            },
            {
                # row with different setup → variant divergence eligible
                "evaluation_name": "minibench",
                "source_data": {"dataset_name": "minibench"},
                "metric_config": {
                    "metric_id": "minibench.acc",
                    "metric_name": "accuracy",
                    "evaluation_description": "Accuracy on minibench",
                },
                "score_details": {"score": 0.50},
                "generation_config": {
                    "generation_args": {"temperature": 0.7, "max_tokens": 1024}
                },
            },
        ],
    }
    (config_dir / "rec.json").write_text(json.dumps(record))


def _write_registry_fixture(reg_root: Path) -> None:
    import pandas as pd

    (reg_root / "aliases").mkdir(parents=True)
    pd.DataFrame(
        [
            {"id": "1", "raw_value": "openai/gpt-4o", "entity_type": "model",
             "canonical_id": "openai/gpt-4o", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "2", "raw_value": "minibench", "entity_type": "benchmark",
             "canonical_id": "minibench", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
            {"id": "3", "raw_value": "Accuracy", "entity_type": "metric",
             "canonical_id": "accuracy", "source_config": None,
             "source_field": None, "status": "active", "strategy": "exact",
             "confidence": 1.0, "notes": None, "created_at": "", "updated_at": ""},
        ]
    ).to_parquet(reg_root / "aliases" / "part-0.parquet")

    (reg_root / "canonical_orgs").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "openai", "display_name": "OpenAI", "parent_org_id": None,
          "website": "https://openai.com", "hf_org": "openai",
          "tags": "[]", "metadata": "{}", "review_status": "reviewed",
          "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_orgs" / "part-0.parquet")

    (reg_root / "canonical_models").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "openai/gpt-4o", "display_name": "GPT-4o", "developer": "OpenAI",
          "org_id": "openai", "family": "GPT-4", "architecture": None,
          "params_billions": None, "parent_model_id": None,
          "tags": "[]", "metadata": "{}", "review_status": "reviewed",
          "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_models" / "part-0.parquet")

    (reg_root / "canonical_benchmarks").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "minibench", "display_name": "MiniBench",
          "description": "tiny benchmark", "dataset_repo": None,
          "parent_benchmark_id": None, "tags": "[]", "metadata": "{}",
          "review_status": "reviewed", "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_benchmarks" / "part-0.parquet")

    (reg_root / "canonical_metrics").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "accuracy", "display_name": "Accuracy",
          "score_type": "continuous", "lower_is_better": False,
          "min_score": 0.0, "max_score": 1.0, "metadata": "{}",
          "review_status": "reviewed", "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "canonical_metrics" / "part-0.parquet")

    (reg_root / "eval_harnesses").mkdir(parents=True)
    pd.DataFrame(
        [{"id": "minibench", "display_name": "MiniBench Harness", "version": "1.0",
          "fork_url": None, "metadata": "{}", "review_status": "reviewed",
          "created_at": "", "updated_at": ""}]
    ).to_parquet(reg_root / "eval_harnesses" / "part-0.parquet")


def _write_cards_fixture(cards_root: Path) -> None:
    (cards_root / "cards").mkdir(parents=True)
    card = {
        "benchmark_card": {
            "benchmark_details": {
                "name": "MiniBench", "overview": "A tiny test benchmark.",
                "data_type": "qa", "domains": ["test"], "languages": ["en"],
                "similar_benchmarks": [], "resources": [],
            },
            "purpose_and_intended_users": {
                "goal": "test the pipeline", "audience": ["devs"],
                "tasks": ["qa"], "limitations": "tiny",
                "out_of_scope_uses": [],
            },
            "data": {"source": "synthetic", "size": "10",
                     "format": "json", "annotation": "manual"},
            "methodology": {
                "methods": ["exact match"], "metrics": ["accuracy"],
                "calculation": "n_correct / n_total",
                "interpretation": "higher is better",
                "baseline_results": "0.5",
                "validation": "manual review",
            },
            "ethical_and_legal_considerations": {
                "privacy_and_anonymity": "n/a", "data_licensing": "MIT",
                "consent_procedures": "n/a", "compliance_with_regulations": "n/a",
            },
        }
    }
    (cards_root / "cards" / "minibench.json").write_text(json.dumps(card))


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")

    eee_root = tmp_path / "eee"
    reg_root = tmp_path / "reg"
    cards_root = tmp_path / "cards"
    warehouse = tmp_path / "warehouse"
    _write_eee_fixture(eee_root)
    _write_registry_fixture(reg_root)
    _write_cards_fixture(cards_root)

    # Force the source loaders into local-only mode by setting the env vars.
    monkeypatch.setenv("EEE_LOCAL_DATASET_DIR", str(eee_root))
    monkeypatch.setenv("BENCHMARK_METADATA_LOCAL_DIR", str(cards_root))
    monkeypatch.delenv("EEE_REFRESH_SNAPSHOT", raising=False)
    monkeypatch.delenv("BENCHMARK_METADATA_REFRESH", raising=False)

    from eval_card_backend.canonicalise import pipeline
    from eval_card_backend.config import Settings

    settings = Settings.from_env()

    out_dir = pipeline.run(
        settings,
        snapshot_id="2026-04-30T00:00:00Z",
        warehouse_dir=str(warehouse),
        registry_local_dir=str(reg_root),
    )

    import duckdb

    con = duckdb.connect()
    fact_path = out_dir / "fact_results.parquet"
    assert fact_path.exists()

    rows = con.execute(
        f"SELECT model_id, benchmark_id, metric_id, score, has_reproducibility_gap, "
        f"has_variant_divergence, distinct_reporting_orgs FROM read_parquet('{fact_path}') "
        f"ORDER BY score"
    ).fetchall()
    assert len(rows) == 2
    # Both rows resolved
    assert all(r[0] == "openai/gpt-4o" for r in rows)
    assert all(r[1] == "minibench" for r in rows)
    assert all(r[2] == "accuracy" for r in rows)
    # Both rows are in the same comparability group with two distinct setups
    # → variant divergence is applicable; magnitude (0.85 - 0.50) > threshold (0.05)
    assert all(r[5] is True for r in rows)
    # Single org → cross-party not applicable
    assert all(r[6] == 1 for r in rows)

    bench = out_dir / "benchmarks.parquet"
    assert bench.exists()
    bench_rows = con.execute(
        f"SELECT benchmark_id, display_name, card_present FROM read_parquet('{bench}')"
    ).fetchall()
    assert bench_rows == [("minibench", "MiniBench", True)]

    completeness = out_dir / "benchmark_completeness.parquet"
    cs = con.execute(
        f"SELECT benchmark_id, completeness_score, populated_count, total_fields_evaluated "
        f"FROM read_parquet('{completeness}')"
    ).fetchall()
    assert len(cs) == 1
    bid, score, populated, total = cs[0]
    assert bid == "minibench"
    assert total == 28
    # 22 full + 1 partial-data full + 0 EEE source_metadata + 0 reserved = 23
    assert populated == 23
    assert abs(score - 23 / 28) < 1e-9

    snap = json.loads((out_dir / "snapshot_meta.json").read_text())
    assert snap["row_counts"]["fact_results"] == 2
