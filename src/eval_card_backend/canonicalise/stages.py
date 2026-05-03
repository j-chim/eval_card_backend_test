"""DuckDB stages for canonicalisation. SQL-heavy; orchestrated by `pipeline.run`.

Each `stage_*` function takes a DuckDB connection (and any extra inputs it
needs) and creates one or more tables on the connection. Tables are wired by
name across stages — see the doc/notes/02-producer-shape.md for the data flow.

Implementation notes (non-spec):
- EEE records are loaded via a producer-side temp JSONL pre-staging (Python
  iterates `loaders.iter_config_results`, dumps each raw record once, with the
  `source_config` already attached). Lets us subset to a few configs easily and
  sidesteps DuckDB filesystem-glob fragility on schema drift.
- Cards are also pre-staged into a JSONL (matches the spec's approach).
- The registry's `canonical_metrics` table doesn't carry `metric_kind`/
  `metric_unit` columns yet; we project NULL for them at load. The legacy
  fact_results columns (`metric_kind`, `metric_unit`) thus default to NULL until
  the registry adds them. Threshold computation falls through to
  `range_5pct`/`fallback_default`.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage A — raw load
# ---------------------------------------------------------------------------


def stage_a_load_eee_jsonl(con, eee_root: Path, configs: Iterable[str], hf_token: str | None) -> int:
    """Stage EEE records to a temp JSONL (one record per line), then load into
    `eee_raw` via `read_json_auto`. Returns the row count.

    `source_config` is attached to each record so downstream stages don't have
    to extract it from the on-disk path.
    """
    from eval_card_backend.sources import eee

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    n = 0
    with tmp:
        for cfg in configs:
            for path in eee.list_json_files(cfg, eee_root, hf_token):
                try:
                    record = eee.read_record(path, eee_root, hf_token)
                except Exception as exc:
                    log.warning("stage A: failed to read %s: %s", path, exc)
                    continue
                if not isinstance(record, dict):
                    continue
                record["source_config"] = cfg
                record["_record_path"] = path
                # Compact serialisation keeps DuckDB inference happy on
                # multi-MB nested objects without forcing pretty-print
                # whitespace through the parser.
                tmp.write(json.dumps(record, ensure_ascii=False, default=str))
                tmp.write("\n")
                n += 1

    if n == 0:
        # Create an empty `eee_raw` table so downstream stages don't error.
        con.execute("CREATE TABLE eee_raw AS SELECT NULL WHERE 0=1")
        return 0

    con.execute(
        f"""
        CREATE TABLE eee_raw AS
        SELECT * FROM read_json_auto(
            '{tmp.name}',
            format = 'newline_delimited',
            union_by_name = true,
            maximum_object_size = 268435456
        )
        """
    )
    return n


def stage_a_load_cards(con, cards: dict, has_resolver_for_cards: bool = True) -> int:
    """Stage AutoBenchmarkCards into `cards_raw_in`, then resolve card keys to
    canonical benchmark_ids into `cards_raw`.
    """
    if not cards:
        # Empty placeholder so LEFT JOINs cleanly miss.
        con.execute(
            "CREATE TABLE cards_raw (card_key VARCHAR, card JSON, "
            "benchmark_id VARCHAR, card_resolution_strategy VARCHAR)"
        )
        return 0

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    with tmp:
        for k, v in cards.items():
            tmp.write(json.dumps({"card_key": k, "card": v}, default=str, ensure_ascii=False))
            tmp.write("\n")

    con.execute(
        f"""
        CREATE TABLE cards_raw_in AS
        SELECT * FROM read_json_auto('{tmp.name}', format = 'newline_delimited',
                                      union_by_name = true,
                                      maximum_object_size = 268435456)
        """
    )

    if has_resolver_for_cards:
        con.execute(
            """
            CREATE TABLE cards_resolved AS
            SELECT
                card_key,
                card,
                resolve_canonical_id(card_key, 'benchmark', NULL) AS benchmark_id,
                resolve_strategy(card_key, 'benchmark', NULL)     AS card_resolution_strategy
            FROM cards_raw_in
            WHERE card_key IS NOT NULL
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE cards_resolved AS
            SELECT card_key, card,
                   CAST(NULL AS VARCHAR) AS benchmark_id,
                   'no_match' AS card_resolution_strategy
            FROM cards_raw_in
            WHERE card_key IS NOT NULL
            """
        )

    # Dedupe per benchmark_id — multiple card_keys can resolve to the same
    # canonical benchmark (e.g. dataset alias and registered name). Without
    # dedup the LEFT JOIN at Stage D / G fans out fact rows and breaks Stage H's
    # scalar-subquery UPDATE for card_missing_count.
    # First-seen-by-card_key wins (deterministic via ORDER BY card_key).
    con.execute(
        """
        CREATE TABLE cards_raw AS
        SELECT card_key, card, benchmark_id, card_resolution_strategy
        FROM (
            SELECT *,
                row_number() OVER (
                    PARTITION BY benchmark_id ORDER BY card_key
                ) AS _rn
            FROM cards_resolved
            WHERE benchmark_id IS NOT NULL
        )
        WHERE _rn = 1

        UNION ALL BY NAME

        -- Cards whose key didn't resolve (benchmark_id IS NULL) are kept as
        -- orphans for the triage path. They never join to fact_results because
        -- benchmark_id IS NULL on both sides.
        SELECT card_key, card, benchmark_id, card_resolution_strategy
        FROM cards_resolved
        WHERE benchmark_id IS NULL
        """
    )

    return con.execute("SELECT count(*) FROM cards_raw").fetchone()[0]


def _read_parquet_path(path: Path) -> str:
    if path.is_dir():
        return str(path / "*.parquet")
    return str(path)


def _table_columns(con, table_path: Path) -> set[str]:
    path = _read_parquet_path(table_path)
    rows = con.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{path}'))"
    ).fetchall()
    return {r[0] for r in rows}


def stage_a_load_registry(con, dim_paths: dict) -> None:
    """Load registry dim tables. Aliases each dim's columns to the spec shape;
    where the registry doesn't carry a column yet, project NULL.
    """
    # canonical_orgs
    if "canonical_orgs" in dim_paths:
        path = _read_parquet_path(dim_paths["canonical_orgs"])
        cols = _table_columns(con, dim_paths["canonical_orgs"])
        # Ensure all columns the spec references exist as NULLs when missing.
        select_parts = []
        for col, ddl in [
            ("id", "VARCHAR"),
            ("display_name", "VARCHAR"),
            ("parent_org_id", "VARCHAR"),
            ("website", "VARCHAR"),
            ("hf_org", "VARCHAR"),
            ("tags", "VARCHAR"),
            ("metadata", "VARCHAR"),
            ("review_status", "VARCHAR"),
        ]:
            if col in cols:
                select_parts.append(col)
            else:
                select_parts.append(f"CAST(NULL AS {ddl}) AS {col}")
        con.execute(
            f"CREATE TABLE canonical_orgs AS SELECT {', '.join(select_parts)} "
            f"FROM read_parquet('{path}')"
        )
    else:
        con.execute(
            "CREATE TABLE canonical_orgs (id VARCHAR, display_name VARCHAR, "
            "parent_org_id VARCHAR, website VARCHAR, hf_org VARCHAR, tags VARCHAR, "
            "metadata VARCHAR, review_status VARCHAR)"
        )

    # canonical_models
    if "canonical_models" in dim_paths:
        path = _read_parquet_path(dim_paths["canonical_models"])
        cols = _table_columns(con, dim_paths["canonical_models"])
        select_parts = []
        for col, ddl in [
            ("id", "VARCHAR"),
            ("display_name", "VARCHAR"),
            ("developer", "VARCHAR"),
            ("org_id", "VARCHAR"),
            ("family", "VARCHAR"),
            ("architecture", "VARCHAR"),
            ("params_billions", "DOUBLE"),
            ("parent_model_id", "VARCHAR"),
            ("tags", "VARCHAR"),
            ("metadata", "VARCHAR"),
            ("review_status", "VARCHAR"),
        ]:
            if col in cols:
                select_parts.append(col)
            else:
                select_parts.append(f"CAST(NULL AS {ddl}) AS {col}")
        # Roadmap fields (released, license, modality, access, context_tokens,
        # context_label) are projected from the registry when present, else NULL.
        for col, ddl in [
            ("released", "DATE"),
            ("license", "VARCHAR"),
            ("modality", "VARCHAR[]"),
            ("access", "VARCHAR"),
            ("context_tokens", "INTEGER"),
            ("context_label", "VARCHAR"),
        ]:
            if col in cols:
                select_parts.append(col)
            else:
                select_parts.append(f"CAST(NULL AS {ddl}) AS {col}")
        con.execute(
            f"CREATE TABLE canonical_models AS SELECT {', '.join(select_parts)} "
            f"FROM read_parquet('{path}')"
        )
    else:
        con.execute(
            "CREATE TABLE canonical_models (id VARCHAR, display_name VARCHAR, "
            "developer VARCHAR, org_id VARCHAR, family VARCHAR, architecture VARCHAR, "
            "params_billions DOUBLE, parent_model_id VARCHAR, tags VARCHAR, "
            "metadata VARCHAR, review_status VARCHAR, "
            "released DATE, license VARCHAR, modality VARCHAR[], "
            "access VARCHAR, context_tokens INTEGER, context_label VARCHAR)"
        )

    # canonical_benchmarks
    if "canonical_benchmarks" in dim_paths:
        path = _read_parquet_path(dim_paths["canonical_benchmarks"])
        cols = _table_columns(con, dim_paths["canonical_benchmarks"])
        select_parts = []
        for col, ddl in [
            ("id", "VARCHAR"),
            ("display_name", "VARCHAR"),
            ("description", "VARCHAR"),
            ("dataset_repo", "VARCHAR"),
            ("parent_benchmark_id", "VARCHAR"),
            ("tags", "VARCHAR"),
            ("metadata", "VARCHAR"),
            ("review_status", "VARCHAR"),
        ]:
            if col in cols:
                select_parts.append(col)
            else:
                select_parts.append(f"CAST(NULL AS {ddl}) AS {col}")
        con.execute(
            f"CREATE TABLE canonical_benchmarks AS SELECT {', '.join(select_parts)} "
            f"FROM read_parquet('{path}')"
        )
    else:
        con.execute(
            "CREATE TABLE canonical_benchmarks (id VARCHAR, display_name VARCHAR, "
            "description VARCHAR, dataset_repo VARCHAR, parent_benchmark_id VARCHAR, "
            "tags VARCHAR, metadata VARCHAR, review_status VARCHAR)"
        )

    # canonical_metrics — registry has `score_type` but spec wants
    # `metric_kind` / `metric_unit`. Map score_type → metric_kind; metric_unit
    # NULL until the registry adds it.
    if "canonical_metrics" in dim_paths:
        path = _read_parquet_path(dim_paths["canonical_metrics"])
        cols = _table_columns(con, dim_paths["canonical_metrics"])
        select_parts = ["id"]
        # display_name
        select_parts.append(
            "display_name" if "display_name" in cols
            else "CAST(NULL AS VARCHAR) AS display_name"
        )
        # metric_kind: prefer existing metric_kind, fall back to score_type
        if "metric_kind" in cols:
            select_parts.append("metric_kind")
        elif "score_type" in cols:
            select_parts.append("score_type AS metric_kind")
        else:
            select_parts.append("CAST(NULL AS VARCHAR) AS metric_kind")
        # metric_unit
        select_parts.append(
            "metric_unit" if "metric_unit" in cols
            else "CAST(NULL AS VARCHAR) AS metric_unit"
        )
        for col, ddl in [
            ("lower_is_better", "BOOLEAN"),
            ("min_score", "DOUBLE"),
            ("max_score", "DOUBLE"),
            ("metadata", "VARCHAR"),
            ("review_status", "VARCHAR"),
        ]:
            if col in cols:
                select_parts.append(col)
            else:
                select_parts.append(f"CAST(NULL AS {ddl}) AS {col}")
        con.execute(
            f"CREATE TABLE canonical_metrics AS SELECT {', '.join(select_parts)} "
            f"FROM read_parquet('{path}')"
        )
    else:
        con.execute(
            "CREATE TABLE canonical_metrics (id VARCHAR, display_name VARCHAR, "
            "metric_kind VARCHAR, metric_unit VARCHAR, "
            "lower_is_better BOOLEAN, min_score DOUBLE, max_score DOUBLE, "
            "metadata VARCHAR, review_status VARCHAR)"
        )

    # eval_harnesses
    if "eval_harnesses" in dim_paths:
        path = _read_parquet_path(dim_paths["eval_harnesses"])
        cols = _table_columns(con, dim_paths["eval_harnesses"])
        select_parts = []
        for col, ddl in [
            ("id", "VARCHAR"),
            ("display_name", "VARCHAR"),
            ("version", "VARCHAR"),
            ("fork_url", "VARCHAR"),
            ("metadata", "VARCHAR"),
            ("review_status", "VARCHAR"),
        ]:
            if col in cols:
                select_parts.append(col)
            else:
                select_parts.append(f"CAST(NULL AS {ddl}) AS {col}")
        con.execute(
            f"CREATE TABLE eval_harnesses AS SELECT {', '.join(select_parts)} "
            f"FROM read_parquet('{path}')"
        )
    else:
        con.execute(
            "CREATE TABLE eval_harnesses (id VARCHAR, display_name VARCHAR, "
            "version VARCHAR, fork_url VARCHAR, metadata VARCHAR, review_status VARCHAR)"
        )


# ---------------------------------------------------------------------------
# Stage B — explode evaluation_results[]
# ---------------------------------------------------------------------------


def _eee_raw_columns(con) -> set[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'eee_raw'"
    ).fetchall()
    return {r[0] for r in rows}


def _maybe_col(eee_cols: set[str], col: str, ddl: str) -> str:
    """Project `e.<col>` if present, else `CAST(NULL AS <ddl>) AS col`."""
    if col in eee_cols:
        return f"e.{col}"
    return f"CAST(NULL AS {ddl}) AS {col}"


def stage_b_explode(con) -> int:
    """One row per (evaluation, result_idx). result_idx is 0-based to match the registry."""
    cols = _eee_raw_columns(con)
    if "evaluation_results" not in cols:
        con.execute(
            "CREATE TABLE results_exploded AS SELECT * FROM eee_raw WHERE 0=1"
        )
        return 0

    # Serialise each evaluation_results[i] to JSON before pulling fields. This
    # avoids "key not in struct" Binder errors when the JSON-inferred struct
    # type doesn't carry every spec field for every record (e.g. the optional
    # `evaluation_result_id`).
    select_clauses = [
        _maybe_col(cols, "evaluation_id", "VARCHAR"),
        _maybe_col(cols, "retrieved_timestamp", "VARCHAR"),
        _maybe_col(cols, "source_metadata", "JSON"),
        _maybe_col(cols, "eval_library", "JSON"),
        _maybe_col(cols, "model_info", "JSON"),
        _maybe_col(cols, "detailed_evaluation_results", "JSON"),
        _maybe_col(cols, "source_config", "VARCHAR"),
        "(idx_1based - 1) AS result_idx",
        "to_json(e.evaluation_results[idx_1based]) AS _er_json",
    ]

    con.execute(
        f"""
        CREATE TABLE results_exploded_raw AS
        SELECT
            {', '.join(select_clauses)}
        FROM eee_raw e,
             range(1, len(e.evaluation_results) + 1) AS t(idx_1based)
        WHERE e.evaluation_results IS NOT NULL
          AND len(e.evaluation_results) > 0
        """
    )

    con.execute(
        """
        CREATE TABLE results_exploded AS
        SELECT
            * EXCLUDE (_er_json),
            json_extract_string(_er_json, '$.evaluation_result_id') AS evaluation_result_id_raw,
            json_extract_string(_er_json, '$.evaluation_name')      AS evaluation_name,
            json_extract(_er_json, '$.source_data')                 AS source_data,
            json_extract(_er_json, '$.metric_config')               AS metric_config,
            json_extract(_er_json, '$.score_details')               AS score_details,
            json_extract(_er_json, '$.generation_config')           AS generation_config
        FROM results_exploded_raw
        """
    )

    con.execute(
        """
        ALTER TABLE results_exploded
        ADD COLUMN evaluation_result_id VARCHAR
        """
    )
    con.execute(
        """
        UPDATE results_exploded
        SET evaluation_result_id = COALESCE(
            evaluation_result_id_raw,
            evaluation_id || '#' || result_idx::VARCHAR
        )
        """
    )
    con.execute("ALTER TABLE results_exploded ADD COLUMN fact_id VARCHAR")
    con.execute(
        "UPDATE results_exploded "
        "SET fact_id = fact_id_udf(evaluation_id, CAST(result_idx AS INTEGER))"
    )

    return con.execute("SELECT count(*) FROM results_exploded").fetchone()[0]


# ---------------------------------------------------------------------------
# Stage C — resolve identity
# ---------------------------------------------------------------------------


def stage_c_resolve(con) -> None:
    # Top-level fields (`source_metadata`, `eval_library`, `model_info`) come
    # from the EEE record root and are reliably struct-typed (or JSON if the
    # record didn't carry them). The per-element nested groups (`metric_config`,
    # etc.) are JSON because Stage B normalises them to dodge schema drift.
    con.execute(
        """
        CREATE TABLE results_resolved AS
        WITH raw AS (
            SELECT
                *,
                json_extract_string(to_json(model_info), '$.id')                  AS _model_raw,
                clean_eval_name_udf(evaluation_name)                              AS _benchmark_raw,
                extract_metric_udf(
                    COALESCE(json_extract_string(metric_config, '$.evaluation_description'),
                             json_extract_string(metric_config, '$.metric_name'),
                             evaluation_name))                                    AS _metric_raw,
                json_extract_string(to_json(source_metadata), '$.source_organization_name') AS _org_raw,
                trim(
                    COALESCE(json_extract_string(to_json(eval_library), '$.name'),    '') || ' ' ||
                    COALESCE(json_extract_string(to_json(eval_library), '$.version'), '')
                )                                                                 AS _harness_raw
            FROM results_exploded
        )
        SELECT
            *,
            _model_raw      AS model_raw,
            _benchmark_raw  AS benchmark_raw,
            _metric_raw     AS metric_raw,
            _org_raw        AS org_raw,
            NULLIF(_harness_raw, '') AS harness_raw,

            resolve_canonical_id(_model_raw,     'model',     source_config) AS model_id,
            resolve_canonical_id(_benchmark_raw, 'benchmark', source_config) AS benchmark_id,
            resolve_canonical_id(_metric_raw,    'metric',    source_config) AS metric_id,
            resolve_canonical_id(_org_raw,       'org',       source_config) AS org_id,
            resolve_canonical_id(NULLIF(_harness_raw, ''), 'harness', source_config) AS harness_id,

            resolve_strategy(_model_raw,     'model',     source_config) AS model_resolution_strategy,
            resolve_strategy(_benchmark_raw, 'benchmark', source_config) AS benchmark_resolution_strategy,
            resolve_strategy(_metric_raw,    'metric',    source_config) AS metric_resolution_strategy,
            resolve_strategy(_org_raw,       'org',       source_config) AS org_resolution_strategy,
            resolve_strategy(NULLIF(_harness_raw, ''), 'harness', source_config) AS harness_resolution_strategy
        FROM raw
        """
    )


# ---------------------------------------------------------------------------
# Stage D — flatten + join canonical dims
# ---------------------------------------------------------------------------


def stage_d_flatten(con) -> None:
    """Flatten + JOIN.

    Nested struct types inferred by `read_json_auto` are heterogeneous across
    EEE records — some configs carry `score_details.uncertainty.standard_error`,
    some don't. Direct dot-notation on a missing struct field is a Binder
    error. Workaround: serialise the nested groups to JSON via `to_json` and
    pull deep values with `json_extract*` (returns NULL on missing path).
    Top-level fields that are reliably present stay as dot-notation reads.
    """
    con.execute(
        """
        CREATE TABLE fact_results_staging AS
        WITH json_view AS (
            SELECT
                *,
                to_json(score_details)                AS _sd_json,
                to_json(generation_config)            AS _gc_json,
                json_extract(to_json(generation_config), '$.generation_args') AS _ga_json,
                to_json(source_metadata)              AS _sm_json,
                to_json(metric_config)                AS _mc_json,
                to_json(detailed_evaluation_results)  AS _det_json,
                to_json(eval_library)                 AS _el_json
            FROM results_resolved
        )
        SELECT
            rr.fact_id,
            rr.evaluation_id, rr.result_idx, rr.evaluation_result_id,

            rr.model_raw,     rr.model_id,
            rr.benchmark_raw, rr.benchmark_id,
            rr.metric_raw,    rr.metric_id,
            rr.org_raw,       rr.org_id,
            rr.harness_raw,   rr.harness_id,

            cb.parent_benchmark_id,
            cm_model.parent_model_id,

            CASE WHEN c.card IS NOT NULL THEN rr.benchmark_id ELSE NULL END AS benchmark_card_id,

            rr.model_resolution_strategy, rr.benchmark_resolution_strategy,
            rr.metric_resolution_strategy, rr.org_resolution_strategy,
            rr.harness_resolution_strategy,

            -- score (top-level scalar — dot OK; nested values via json paths)
            TRY_CAST(json_extract_string(rr._sd_json, '$.score') AS DOUBLE) AS score,
            TRY_CAST(json_extract_string(rr._sd_json, '$.uncertainty.standard_error.value') AS DOUBLE) AS score_se,
            TRY_CAST(json_extract_string(rr._sd_json, '$.uncertainty.confidence_interval.lower') AS DOUBLE) AS score_ci_lower,
            TRY_CAST(json_extract_string(rr._sd_json, '$.uncertainty.confidence_interval.upper') AS DOUBLE) AS score_ci_upper,
            TRY_CAST(json_extract_string(rr._sd_json, '$.uncertainty.confidence_interval.confidence_level') AS DOUBLE) AS score_ci_level,
            TRY_CAST(json_extract_string(rr._sd_json, '$.uncertainty.num_samples') AS INTEGER) AS n_samples,

            -- source / provenance
            json_extract_string(rr._sm_json, '$.evaluator_relationship') AS evaluator_relationship,
            json_extract_string(rr._sm_json, '$.source_type') AS source_type,
            json_extract_string(rr._sm_json, '$.source_organization_url') AS source_organization_url,
            json_extract_string(rr._el_json, '$.name')    AS eval_library_name,
            json_extract_string(rr._el_json, '$.version') AS eval_library_version,

            -- metric meta from canonical_metrics
            cmet.metric_kind                                                              AS metric_kind,
            cmet.metric_unit                                                              AS metric_unit,
            cmet.lower_is_better                                                          AS lower_is_better,
            cmet.min_score                                                                AS min_score,
            cmet.max_score                                                                AS max_score,

            -- generation config — flattened via JSON path (heterogeneous shape)
            TRY_CAST(json_extract_string(rr._ga_json, '$.temperature') AS DOUBLE)         AS temperature,
            TRY_CAST(json_extract_string(rr._ga_json, '$.top_p')       AS DOUBLE)         AS top_p,
            TRY_CAST(json_extract_string(rr._ga_json, '$.top_k')       AS DOUBLE)         AS top_k,
            TRY_CAST(json_extract_string(rr._ga_json, '$.max_tokens')  AS INTEGER)        AS max_tokens,
            json_extract_string(rr._ga_json, '$.prompt_template')                         AS prompt_template,
            TRY_CAST(json_extract_string(rr._ga_json, '$.reasoning')   AS BOOLEAN)        AS reasoning,
            json_extract(rr._ga_json, '$.agentic_eval_config')                            AS agentic_eval_config,
            json_extract(rr._ga_json, '$.eval_plan')                                      AS eval_plan,
            json_extract(rr._ga_json, '$.eval_limits')                                    AS eval_limits,
            json_extract(rr._ga_json, '$.sandbox')                                        AS sandbox,

            CAST(rr._ga_json AS VARCHAR)                                                  AS generation_args_json,

            json_extract(rr._sm_json, '$.additional_details')                             AS source_additional_details,
            json_extract(rr._gc_json, '$.additional_details')                             AS generation_additional_details,
            json_extract(rr._mc_json, '$.additional_details')                             AS metric_additional_details,

            -- instance pointer
            json_extract_string(rr._det_json, '$.file_path')                              AS instance_file_path,
            json_extract_string(rr._det_json, '$.format')                                 AS instance_file_format,
            json_extract_string(rr._det_json, '$.checksum')                               AS instance_checksum,
            json_extract_string(rr._det_json, '$.hash_algorithm')                         AS instance_hash_algorithm,
            TRY_CAST(json_extract_string(rr._det_json, '$.total_rows') AS INTEGER)        AS instance_rows,

            c.card AS card_payload
        FROM json_view rr
        LEFT JOIN canonical_benchmarks cb       ON cb.id = rr.benchmark_id
        LEFT JOIN canonical_models     cm_model ON cm_model.id = rr.model_id
        LEFT JOIN canonical_metrics    cmet     ON cmet.id = rr.metric_id
        LEFT JOIN cards_raw            c        ON c.benchmark_id = rr.benchmark_id
        """
    )


# ---------------------------------------------------------------------------
# Stage E — per-row signals (pass 1)
# ---------------------------------------------------------------------------


def stage_e_per_row_signals(con) -> tuple[int, int]:
    """Returns (row_count_before_drop, row_count_after_drop)."""
    pre = con.execute("SELECT count(*) FROM fact_results_staging").fetchone()[0]
    con.execute(
        """
        CREATE TABLE fact_results_signaled AS
        WITH base AS (
            SELECT
                *,
                temperature           IS NOT NULL  AS has_temperature,
                top_p                 IS NOT NULL  AS has_top_p,
                top_k                 IS NOT NULL  AS has_top_k,
                max_tokens            IS NOT NULL  AS has_max_tokens,
                prompt_template       IS NOT NULL  AS has_prompt_template,
                eval_plan             IS NOT NULL  AS has_eval_plan,
                eval_limits           IS NOT NULL  AS has_eval_limits,
                agentic_eval_config   IS NOT NULL  AS has_agentic_eval_config,
                is_agentic_udf(benchmark_id, to_json(card_payload), generation_args_json) AS is_agentic
            FROM fact_results_staging
            WHERE score IS NOT NULL
        )
        SELECT
            *,
            CASE WHEN is_agentic
                 THEN NOT (has_temperature AND has_max_tokens
                           AND has_eval_plan AND has_eval_limits)
                 ELSE NOT (has_temperature AND has_max_tokens)
            END AS has_reproducibility_gap,

            compute_repro_missing_udf(
                is_agentic, has_temperature, has_max_tokens, has_eval_plan, has_eval_limits
            ) AS repro_missing_fields,

            CASE WHEN is_agentic THEN 4 ELSE 2 END AS repro_required_count,

            (CASE WHEN is_agentic THEN 4 ELSE 2 END
                - len(compute_repro_missing_udf(
                    is_agentic, has_temperature, has_max_tokens,
                    has_eval_plan, has_eval_limits
                ))) AS repro_populated_count,

            COALESCE(
                CASE WHEN evaluator_relationship = 'other' THEN 'unspecified'
                     ELSE evaluator_relationship
                END, 'unspecified'
            ) AS provenance_source_type,

            variant_key_udf(generation_args_json) AS variant_key,

            (metric_unit = 'proportion' AND (score < 0 OR score > 1)) AS score_scale_anomaly,

            -- reserved EvalCards fields (not yet populated)
            CAST(NULL AS VARCHAR) AS lifecycle_status,
            CAST(NULL AS VARCHAR) AS preregistration_url
        FROM base
        """
    )
    post = con.execute("SELECT count(*) FROM fact_results_signaled").fetchone()[0]
    return pre, post


# ---------------------------------------------------------------------------
# Stage F — group signals (pass 2)
# ---------------------------------------------------------------------------


def stage_f_group_signals(con, snapshot_id: str) -> None:
    # F.1 — group-derived provenance
    con.execute(
        """
        CREATE TABLE fact_results_grouped AS
        WITH org_normalized AS (
            SELECT *,
                NULLIF(trim(regexp_replace(lower(org_raw), '\\s+', ' ', 'g')), '')
                  AS org_normalized_key
            FROM fact_results_signaled
            WHERE model_id IS NOT NULL
              AND benchmark_id IS NOT NULL
              AND metric_id IS NOT NULL
        ),
        group_orgs AS (
            SELECT
                model_id, benchmark_id, metric_id,
                COUNT(DISTINCT org_normalized_key)
                  FILTER (WHERE org_normalized_key IS NOT NULL)
                  AS distinct_reporting_orgs
            FROM org_normalized
            GROUP BY 1, 2, 3
        )
        SELECT
            o.*,
            go.distinct_reporting_orgs,
            substr(md5(o.model_id || '|' || o.benchmark_id || '|' || o.metric_id), 1, 16)
              AS comparability_group_id,
            go.distinct_reporting_orgs > 1 AS is_multi_source,
            (o.provenance_source_type = 'first_party' AND go.distinct_reporting_orgs = 1)
              AS first_party_only
        FROM org_normalized o
        JOIN group_orgs go USING (model_id, benchmark_id, metric_id)
        """
    )

    # F.2 — variant + cross-party divergence
    con.execute(
        """
        CREATE TABLE fact_results_grouped_annotated AS
        WITH group_payloads AS (
            SELECT
                model_id, benchmark_id, metric_id,
                array_agg(struct_pack(
                    fact_id                  := fact_id,
                    evaluation_id            := evaluation_id,
                    score                    := score,
                    generation_args          := generation_args_json,
                    evaluator_relationship   := evaluator_relationship,
                    source_organization_name := org_raw
                )) AS group_rows,
                any_value(struct_pack(
                    metric_kind := metric_kind,
                    metric_unit := metric_unit,
                    min_score   := min_score,
                    max_score   := max_score
                )) AS metric_config
            FROM fact_results_grouped
            GROUP BY 1, 2, 3
        ),
        group_annotations AS (
            SELECT
                model_id, benchmark_id, metric_id,
                compute_variant_divergence_udf(group_rows, metric_config)      AS variant,
                compute_cross_party_divergence_udf(group_rows, metric_config)  AS cross_party
            FROM group_payloads
        )
        SELECT
            fr.*,
            ga.variant.has_variant_divergence       AS has_variant_divergence,
            ga.variant.divergence_magnitude         AS variant_divergence_magnitude,
            ga.variant.threshold_used               AS variant_divergence_threshold,
            ga.variant.threshold_basis              AS variant_threshold_basis,
            ga.variant.differing_setup_fields       AS variant_differing_fields,

            ga.cross_party.has_cross_party_divergence  AS has_cross_party_divergence,
            ga.cross_party.divergence_magnitude        AS cross_party_divergence_magnitude,
            ga.cross_party.threshold_used              AS cross_party_divergence_threshold,
            ga.cross_party.threshold_basis             AS cross_party_threshold_basis,
            ga.cross_party.differing_setup_fields      AS cross_party_differing_fields,
            ga.cross_party.organization_count          AS cross_party_org_count,
            ga.cross_party.scores_by_organization      AS scores_by_organization
        FROM fact_results_grouped fr
        LEFT JOIN group_annotations ga USING (model_id, benchmark_id, metric_id)
        """
    )

    # F.4 — final fact_results: union resolved-with-group-signals + unresolved passthrough
    con.execute(
        f"""
        CREATE TABLE fact_results AS
        SELECT
            TIMESTAMP '{snapshot_id_to_sql(snapshot_id)}' AS snapshot_id,
            * EXCLUDE (card_payload, org_normalized_key, generation_args_json)
        FROM fact_results_grouped_annotated

        UNION ALL BY NAME

        SELECT
            TIMESTAMP '{snapshot_id_to_sql(snapshot_id)}' AS snapshot_id,
            fr.* EXCLUDE (card_payload, generation_args_json),
            CAST(NULL AS INTEGER)                              AS distinct_reporting_orgs,
            CAST(NULL AS VARCHAR)                              AS comparability_group_id,
            CAST(NULL AS BOOLEAN)                              AS is_multi_source,
            CAST(NULL AS BOOLEAN)                              AS first_party_only,
            CAST(NULL AS BOOLEAN)                              AS has_variant_divergence,
            CAST(NULL AS DOUBLE)                               AS variant_divergence_magnitude,
            CAST(NULL AS DOUBLE)                               AS variant_divergence_threshold,
            CAST(NULL AS VARCHAR)                              AS variant_threshold_basis,
            CAST(NULL AS STRUCT(field VARCHAR, "values" JSON)[]) AS variant_differing_fields,
            CAST(NULL AS BOOLEAN)                              AS has_cross_party_divergence,
            CAST(NULL AS DOUBLE)                               AS cross_party_divergence_magnitude,
            CAST(NULL AS DOUBLE)                               AS cross_party_divergence_threshold,
            CAST(NULL AS VARCHAR)                              AS cross_party_threshold_basis,
            CAST(NULL AS STRUCT(field VARCHAR, "values" JSON)[]) AS cross_party_differing_fields,
            CAST(NULL AS INTEGER)                              AS cross_party_org_count,
            CAST(NULL AS MAP(VARCHAR, DOUBLE))                 AS scores_by_organization
        FROM fact_results_signaled fr
        WHERE fr.model_id IS NULL OR fr.benchmark_id IS NULL OR fr.metric_id IS NULL
        """
    )


def snapshot_id_to_sql(snapshot_id: str) -> str:
    """DuckDB's TIMESTAMP literal doesn't accept the trailing 'Z'. Strip it
    and the parser does the right thing.
    """
    return snapshot_id[:-1] if snapshot_id.endswith("Z") else snapshot_id


# ---------------------------------------------------------------------------
# Stage G — dim tables (benchmarks, models)
# ---------------------------------------------------------------------------


def stage_g_dims(con, snapshot_id: str) -> None:
    sid = snapshot_id_to_sql(snapshot_id)

    # benchmarks.parquet — accesses card subfields via JSON path so missing
    # struct fields don't raise; the card schema is heterogeneous across the
    # cards corpus (some carry _generated_by / flagged_fields, some don't).
    con.execute(
        f"""
        CREATE TABLE benchmarks AS
        WITH cards_json AS (
            SELECT card_key, benchmark_id, to_json(card) AS card_j FROM cards_raw
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            cb.id AS benchmark_id,

            cb.display_name,
            cb.description,
            cb.dataset_repo,
            cb.parent_benchmark_id,
            TRY_CAST(from_json(cb.tags, '["VARCHAR"]') AS VARCHAR[]) AS registry_tags,
            cb.metadata AS registry_metadata,
            cb.review_status,

            json_extract_string(c.card_j, '$.benchmark_details.name')      AS card_name,
            json_extract_string(c.card_j, '$.benchmark_details.overview')  AS overview,
            json_extract_string(c.card_j, '$.benchmark_details.data_type') AS data_type,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.domains'),     '["VARCHAR"]') AS VARCHAR[]) AS domains,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.languages'),   '["VARCHAR"]') AS VARCHAR[]) AS languages,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.similar_benchmarks'), '["VARCHAR"]') AS VARCHAR[]) AS similar_benchmarks,
            TRY_CAST(from_json(json_extract(c.card_j, '$.benchmark_details.resources'),   '["VARCHAR"]') AS VARCHAR[]) AS resources,

            json_extract_string(c.card_j, '$.purpose_and_intended_users.goal')       AS goal,
            TRY_CAST(from_json(json_extract(c.card_j, '$.purpose_and_intended_users.audience'), '["VARCHAR"]') AS VARCHAR[]) AS audience,
            TRY_CAST(from_json(json_extract(c.card_j, '$.purpose_and_intended_users.tasks'),    '["VARCHAR"]') AS VARCHAR[]) AS tasks,
            json_extract_string(c.card_j, '$.purpose_and_intended_users.limitations') AS limitations,
            TRY_CAST(from_json(json_extract(c.card_j, '$.purpose_and_intended_users.out_of_scope_uses'), '["VARCHAR"]') AS VARCHAR[]) AS out_of_scope_uses,

            json_extract_string(c.card_j, '$.data.source')     AS data_source,
            json_extract_string(c.card_j, '$.data.size')       AS data_size,
            json_extract_string(c.card_j, '$.data.format')     AS data_format,
            json_extract_string(c.card_j, '$.data.annotation') AS data_annotation,

            TRY_CAST(from_json(json_extract(c.card_j, '$.methodology.methods'), '["VARCHAR"]') AS VARCHAR[]) AS methods,
            TRY_CAST(from_json(json_extract(c.card_j, '$.methodology.metrics'), '["VARCHAR"]') AS VARCHAR[]) AS card_metrics,
            json_extract_string(c.card_j, '$.methodology.calculation')      AS calculation,
            json_extract_string(c.card_j, '$.methodology.interpretation')   AS interpretation,
            json_extract_string(c.card_j, '$.methodology.baseline_results') AS baseline_results,
            json_extract_string(c.card_j, '$.methodology.validation')       AS validation,

            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.privacy_and_anonymity')        AS privacy_and_anonymity,
            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.data_licensing')               AS data_licensing,
            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.consent_procedures')           AS consent_procedures,
            json_extract_string(c.card_j, '$.ethical_and_legal_considerations.compliance_with_regulations')  AS compliance_with_regulations,

            json_extract(c.card_j, '$.possible_risks') AS possible_risks,
            json_extract(c.card_j, '$.flagged_fields') AS flagged_fields,

            (c.card_j IS NOT NULL) AS card_present,
            json_extract_string(c.card_j, '$._generated_by') AS card_generated_by,
            COALESCE(len(json_keys(json_extract(c.card_j, '$.flagged_fields'))), 0) AS card_flagged_count,
            CAST(NULL AS INTEGER) AS card_missing_count

        FROM canonical_benchmarks cb
        LEFT JOIN cards_json c ON c.benchmark_id = cb.id
        WHERE cb.id IN (SELECT DISTINCT benchmark_id FROM fact_results
                        WHERE benchmark_id IS NOT NULL)
        """
    )

    # models.parquet
    con.execute(
        f"""
        CREATE TABLE models AS
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            cm.id AS model_id,

            cm.display_name,
            cm.developer,
            cm.org_id,
            cm.family,
            cm.architecture,
            cm.params_billions,
            cm.parent_model_id,
            TRY_CAST(from_json(cm.tags, '["VARCHAR"]') AS VARCHAR[]) AS registry_tags,
            cm.metadata AS registry_metadata,
            cm.review_status,

            co.display_name        AS org_display_name,
            co.website             AS org_website,
            co.hf_org              AS org_hf_org,
            co.parent_org_id       AS org_parent_id,

            cm.released,
            cm.context_tokens,
            cm.context_label,
            cm.modality,
            cm.access,
            cm.license

        FROM canonical_models cm
        LEFT JOIN canonical_orgs co ON co.id = cm.org_id
        WHERE cm.id IN (SELECT DISTINCT model_id FROM fact_results
                        WHERE model_id IS NOT NULL)
        """
    )


# ---------------------------------------------------------------------------
# Stage H — benchmark_completeness
# ---------------------------------------------------------------------------


def stage_h_completeness(con, snapshot_id: str) -> None:
    sid = snapshot_id_to_sql(snapshot_id)
    con.execute(
        f"""
        CREATE TABLE benchmark_completeness AS
        WITH scored AS (
            SELECT
                cb.id AS benchmark_id,
                compute_completeness_udf(to_json(c.card)) AS comp
            FROM canonical_benchmarks cb
            LEFT JOIN cards_raw c ON c.benchmark_id = cb.id
            WHERE cb.id IN (SELECT DISTINCT benchmark_id FROM fact_results
                            WHERE benchmark_id IS NOT NULL)
        )
        SELECT
            TIMESTAMP '{sid}' AS snapshot_id,
            benchmark_id,
            comp.completeness_score        AS completeness_score,
            comp.total_fields_evaluated    AS total_fields_evaluated,
            comp.populated_count           AS populated_count,
            comp.missing_required_fields   AS missing_required_fields,
            comp.partial_fields            AS partial_fields,
            comp.field_scores              AS field_scores
        FROM scored
        """
    )

    con.execute(
        """
        UPDATE benchmarks AS b
        SET card_missing_count = (
            SELECT len(bc.missing_required_fields)
            FROM benchmark_completeness bc
            WHERE bc.snapshot_id = b.snapshot_id
              AND bc.benchmark_id = b.benchmark_id
        )
        """
    )


# ---------------------------------------------------------------------------
# Stage I — emit Parquet
# ---------------------------------------------------------------------------


def stage_i_emit(con, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, sort_key in [
        ("fact_results", "(model_id, benchmark_id, metric_id)"),
        ("benchmark_completeness", "(benchmark_id)"),
        ("benchmarks", "(benchmark_id)"),
        ("models", "(model_id)"),
        ("canonical_metrics", "(id)"),
    ]:
        path = out_dir / f"{table}.parquet"
        con.execute(
            f"""
            COPY (SELECT * FROM {table} ORDER BY {sort_key} NULLS LAST)
            TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
