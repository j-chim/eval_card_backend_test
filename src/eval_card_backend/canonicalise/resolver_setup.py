"""Register the resolver and all helper UDFs against a DuckDB connection.

Single entry point: `register_udfs(con, resolver)`. Idempotent — re-registering
a UDF on an existing connection raises, so callers should use a fresh
connection per pipeline run.
"""
from __future__ import annotations

from eval_entity_resolver.eee import clean_eval_name, extract_metric

from eval_card_backend.canonicalise import udfs


def register_udfs(con, resolver) -> None:
    resolve_canonical_id_py, resolve_strategy_py = udfs.make_resolver_udfs(resolver)

    con.create_function(
        "resolve_canonical_id", resolve_canonical_id_py,
        ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "resolve_strategy", resolve_strategy_py,
        ["VARCHAR", "VARCHAR", "VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "clean_eval_name_udf",
        lambda v: clean_eval_name(v) if isinstance(v, str) else None,
        ["VARCHAR"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "extract_metric_udf",
        lambda v: extract_metric(v) if isinstance(v, str) else None,
        ["VARCHAR"], "VARCHAR",
        null_handling="special",
    )

    con.create_function(
        "fact_id_udf", udfs.fact_id_py,
        ["VARCHAR", "INTEGER"], "VARCHAR",
        null_handling="special",
    )
    con.create_function(
        "variant_key_udf",
        lambda ga: udfs.variant_key_py(ga),
        ["JSON"], "VARCHAR",
        null_handling="special",
    )

    con.create_function(
        "is_agentic_udf",
        lambda bid, card, ga: udfs.is_agentic_py(bid, card, ga),
        ["VARCHAR", "JSON", "JSON"], "BOOLEAN",
        null_handling="special",
    )
    con.create_function(
        "compute_repro_missing_udf",
        lambda is_ag, ht, hm, hep, hel: udfs.compute_repro_missing_py(
            bool(is_ag), bool(ht), bool(hm), bool(hep), bool(hel)
        ),
        ["BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN"],
        "VARCHAR[]",
        null_handling="special",
    )

    con.create_function(
        "canonical_json_udf",
        lambda v: udfs.canonical_json(v),
        ["JSON"], "VARCHAR",
        null_handling="special",
    )

    completeness_struct_type = (
        "STRUCT(completeness_score DOUBLE, total_fields_evaluated INTEGER, "
        "populated_count DOUBLE, missing_required_fields VARCHAR[], "
        'partial_fields STRUCT(field_path VARCHAR, score DOUBLE, '
        "populated_subitems INTEGER, total_subitems INTEGER)[], "
        'field_scores STRUCT(field_path VARCHAR, coverage_type VARCHAR, '
        "score DOUBLE)[])"
    )
    con.create_function(
        "compute_completeness_udf",
        lambda card: udfs.compute_completeness_py(card),
        ["JSON"], completeness_struct_type,
        null_handling="special",
    )

    # Group-level signal UDFs. `differing_setup_fields` is a list of
    # STRUCT(field, "values"); the UDF body produces those structs directly.
    group_row_type = (
        'STRUCT(fact_id VARCHAR, evaluation_id VARCHAR, score DOUBLE, '
        "generation_args VARCHAR, evaluator_relationship VARCHAR, "
        "source_organization_name VARCHAR)"
    )
    metric_cfg_type = (
        "STRUCT(metric_kind VARCHAR, metric_unit VARCHAR, "
        "min_score DOUBLE, max_score DOUBLE)"
    )
    variant_out_type = (
        "STRUCT(has_variant_divergence BOOLEAN, divergence_magnitude DOUBLE, "
        "threshold_used DOUBLE, threshold_basis VARCHAR, "
        'differing_setup_fields STRUCT(field VARCHAR, "values" JSON)[])'
    )
    cross_out_type = (
        "STRUCT(has_cross_party_divergence BOOLEAN, divergence_magnitude DOUBLE, "
        "threshold_used DOUBLE, threshold_basis VARCHAR, "
        'differing_setup_fields STRUCT(field VARCHAR, "values" JSON)[], '
        "organization_count INTEGER, "
        "scores_by_organization MAP(VARCHAR, DOUBLE))"
    )
    con.create_function(
        "compute_variant_divergence_udf",
        udfs.compute_variant_divergence_udf_body,
        [f"{group_row_type}[]", metric_cfg_type], variant_out_type,
        null_handling="special",
    )
    con.create_function(
        "compute_cross_party_divergence_udf",
        udfs.compute_cross_party_divergence_udf_body,
        [f"{group_row_type}[]", metric_cfg_type], cross_out_type,
        null_handling="special",
    )
