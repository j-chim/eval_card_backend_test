"""Canonicalisation pipeline orchestrator.

End-to-end flow:
  1. Resolve upstream snapshots (EEE, registry, cards) — fetch if missing.
  2. Preflight: hard-fail on empty EEE / empty registry; warn on missing cards.
  3. Build a DuckDB connection, register the resolver + helper UDFs.
  4. Load all three sources into in-memory tables (Stages A–B).
  5. Resolve identity (Stage C), flatten + JOIN dims (D), per-row signals (E),
     group signals (F), dim materialisation (G), completeness (H).
  6. Emit parquets (Stage I) + snapshot_meta.json sidecar.
  7. Print resolver / coercion / row-count summary.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from eval_card_backend.canonicalise import stages, udfs
from eval_card_backend.canonicalise.resolver_setup import register_udfs
from eval_card_backend.config import (
    BENCHMARK_METADATA_DATASET_REPO,
    EEE_DATASET_REPO,
    Settings,
)
from eval_card_backend.signals.setup import (
    log_json_coerce_summary,
    reset_json_coerce_counter,
)
from eval_card_backend.sources import benchmark_cards, eee, registry as registry_src

log = logging.getLogger(__name__)


REGISTRY_DATASET_REPO = "evaleval/entity-registry-data"
DEFAULT_REGISTRY_LOCAL_DIR = ".cache/entity_registry"
DEFAULT_WAREHOUSE_DIR = "warehouse"


def _make_snapshot_id() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _snapshot_dir_name(snapshot_id: str) -> str:
    return snapshot_id.replace(":", "-")


def _hf_revision(repo_id: str, hf_token: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        info = api.dataset_info(repo_id, token=hf_token)
        return info.sha
    except Exception:
        return None


def preflight(
    eee_root: Path | None, registry_root: Path | None, cards_root: Path | None
) -> None:
    errors: list[str] = []

    if eee_root is None or not (eee_root / "data").exists() or not any(
        (eee_root / "data").rglob("*.json")
    ):
        errors.append(
            f"EEE source empty/missing at {eee_root}/data (HF download failed?). "
            f"Set EEE_REFRESH_SNAPSHOT=1 to force re-download."
        )

    aliases = (
        registry_src.aliases_path(registry_root) if registry_root else None
    )
    if aliases is None or not aliases.exists():
        errors.append(
            f"Registry alias store missing under {registry_root}. "
            f"Pull evaleval/entity-registry-data."
        )

    if cards_root is None or not Path(cards_root).exists():
        log.warning(
            "AutoBenchmarkCards source missing at %s — proceeding without "
            "card content. card_present will be false on every benchmark.",
            cards_root,
        )

    if errors:
        for err in errors:
            log.error(err)
        raise SystemExit("Preflight failed; see errors above.")


def run(
    settings: Settings,
    *,
    configs: list[str] | None = None,
    config_limit: int | None = None,
    snapshot_id: str | None = None,
    warehouse_dir: str = DEFAULT_WAREHOUSE_DIR,
    registry_local_dir: str = DEFAULT_REGISTRY_LOCAL_DIR,
    skip_preflight: bool = False,
) -> Path:
    """Run the full pipeline. Returns the snapshot output directory."""

    snapshot_id = snapshot_id or _make_snapshot_id()
    log.info("snapshot_id = %s", snapshot_id)

    # 1. fetch snapshots
    eee_root = eee.ensure_snapshot(
        settings.eee_local_dir, settings.hf_token, settings.refresh_eee
    )
    cards_root = benchmark_cards.ensure_snapshot(
        settings.benchmark_metadata_local_dir,
        settings.hf_token,
        settings.refresh_benchmark_metadata,
    )
    registry_root = registry_src.ensure_snapshot(
        registry_local_dir, settings.hf_token, force_refresh=False
    )

    if not skip_preflight:
        preflight(eee_root, registry_root, cards_root)

    # Resolve config list
    all_configs = eee.discover_configs(eee_root, settings.hf_token)
    if configs is not None:
        wanted = {c for c in configs}
        chosen = [c for c in all_configs if c in wanted]
    else:
        chosen = list(all_configs)
    if config_limit is not None:
        chosen = chosen[:config_limit]
    log.info("running over %d / %d configs: %s", len(chosen), len(all_configs), chosen[:10])

    # Cards (best effort; empty dict if root missing)
    cards = benchmark_cards.load_cards(cards_root) if cards_root else {}

    # 2. DuckDB + UDFs
    con = duckdb.connect()
    udfs.reset_resolver_counters()
    reset_json_coerce_counter()

    alias_store = registry_src.load_alias_store(registry_root)
    from eval_entity_resolver import Resolver

    resolver = Resolver(alias_store)
    register_udfs(con, resolver)

    # 3. Stage A
    log.info("Stage A: loading sources …")
    n_eee = stages.stage_a_load_eee_jsonl(con, eee_root, chosen, settings.hf_token)
    log.info("  loaded %d EEE records", n_eee)
    n_cards = stages.stage_a_load_cards(con, cards)
    log.info("  loaded %d cards", n_cards)
    dim_paths = registry_src.open_dim_paths(registry_root)
    stages.stage_a_load_registry(con, dim_paths)
    log.info("  registry dims loaded: %s", sorted(dim_paths))

    # 4. Stage B
    n_exploded = stages.stage_b_explode(con)
    log.info("Stage B: exploded %d evaluation_results rows", n_exploded)

    # 5. Stage C
    log.info("Stage C: resolving identities …")
    stages.stage_c_resolve(con)

    # 6. Stage D
    log.info("Stage D: flattening + joining dims …")
    stages.stage_d_flatten(con)

    # 7. Stage E
    pre, post = stages.stage_e_per_row_signals(con)
    log.info("Stage E: %d rows in, %d rows out (dropped %d for missing score)",
             pre, post, pre - post)

    # 8. Stage F
    log.info("Stage F: group signals …")
    stages.stage_f_group_signals(con, snapshot_id)

    # 9. Stage G
    log.info("Stage G: dim materialisation …")
    stages.stage_g_dims(con, snapshot_id)

    # 10. Stage H
    log.info("Stage H: benchmark completeness …")
    stages.stage_h_completeness(con, snapshot_id)

    # 11. Stage I — emit parquet
    out_dir = Path(warehouse_dir) / _snapshot_dir_name(snapshot_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Stage I: emitting parquets to %s", out_dir)
    stages.stage_i_emit(con, out_dir)

    # 12. Snapshot meta sidecar
    meta = {
        "snapshot_id": snapshot_id,
        "generated_at": _make_snapshot_id(),
        "configs": chosen,
        "eee_revision": _hf_revision(EEE_DATASET_REPO, settings.hf_token),
        "registry_revision": _hf_revision(REGISTRY_DATASET_REPO, settings.hf_token),
        "cards_revision": _hf_revision(
            BENCHMARK_METADATA_DATASET_REPO, settings.hf_token
        ),
        "tables": [
            "fact_results.parquet",
            "benchmark_completeness.parquet",
            "benchmarks.parquet",
            "models.parquet",
            "canonical_metrics.parquet",
        ],
        "row_counts": {
            "eee_records": n_eee,
            "exploded_results": n_exploded,
            "fact_results_pre_drop": pre,
            "fact_results": post,
            "dropped_rows_no_score": pre - post,
            "cards": n_cards,
        },
    }
    (out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))

    # 13. Summaries
    udfs.log_resolver_summary()
    log_json_coerce_summary()
    _log_canonicalisation_summary(con, post)

    return out_dir


def _log_canonicalisation_summary(con, fact_count: int) -> None:
    log.info("=== canonicalisation summary ===")
    log.info("  fact_results rows: %d", fact_count)
    if fact_count == 0:
        log.warning("  fact_results is EMPTY — investigate before downstream use")
        return
    summary = con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE model_id IS NULL)     AS unresolved_model,
            COUNT(*) FILTER (WHERE benchmark_id IS NULL) AS unresolved_benchmark,
            COUNT(*) FILTER (WHERE metric_id IS NULL)    AS unresolved_metric,
            COUNT(*) FILTER (WHERE org_id IS NULL)       AS unresolved_org,
            COUNT(*) FILTER (WHERE harness_id IS NULL)   AS unresolved_harness,
            COUNT(*) FILTER (WHERE score_scale_anomaly)  AS score_scale_anomalies
        FROM fact_results
        """
    ).fetchone()
    log.info(
        "  unresolved: model=%d benchmark=%d metric=%d org=%d harness=%d",
        *summary[:5],
    )
    log.info("  score_scale_anomalies: %d", summary[5])

    groups = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT model_id, benchmark_id, metric_id
            FROM fact_results
            WHERE model_id IS NOT NULL
              AND benchmark_id IS NOT NULL
              AND metric_id IS NOT NULL
        )
        """
    ).fetchone()[0]
    log.info("  resolved (m,b,metric) groups: %d", groups)

    variant_eligible = con.execute(
        "SELECT COUNT(DISTINCT comparability_group_id) FROM fact_results "
        "WHERE variant_divergence_threshold IS NOT NULL"
    ).fetchone()[0]
    cross_eligible = con.execute(
        "SELECT COUNT(DISTINCT comparability_group_id) FROM fact_results "
        "WHERE cross_party_divergence_threshold IS NOT NULL"
    ).fetchone()[0]
    log.info("  variant-divergence eligible groups: %d", variant_eligible)
    log.info("  cross-party-divergence eligible groups: %d", cross_eligible)

    variant_pos = con.execute(
        "SELECT COUNT(DISTINCT comparability_group_id) FROM fact_results "
        "WHERE has_variant_divergence = TRUE"
    ).fetchone()[0]
    cross_pos = con.execute(
        "SELECT COUNT(DISTINCT comparability_group_id) FROM fact_results "
        "WHERE has_cross_party_divergence = TRUE"
    ).fetchone()[0]
    log.info("  variant-divergence groups: %d", variant_pos)
    log.info("  cross-party-divergence groups: %d", cross_pos)

    bc = con.execute(
        "SELECT COUNT(*), AVG(completeness_score), MIN(completeness_score), "
        "MAX(completeness_score) FROM benchmark_completeness"
    ).fetchone()
    if bc[0]:
        log.info(
            "  benchmark_completeness: n=%d mean=%.3f min=%.3f max=%.3f",
            bc[0], bc[1] or 0.0, bc[2] or 0.0, bc[3] or 0.0,
        )
    no_card = con.execute(
        "SELECT COUNT(*) FROM benchmarks WHERE card_present = FALSE"
    ).fetchone()[0]
    log.info("  benchmarks without AutoBenchmarkCard: %d", no_card)


run_pipeline = run  # alias for callers
