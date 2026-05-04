"""Stage J — JSON sidecars for the view layer.

Four small documents the frontend reads alongside the view parquets:

- `manifest.json` — corpus-level scalars (model_count, eval_count, …).
- `headline.json` — corpus signal aggregates with stratified by-category
  blocks. Drives the home-page corpus signal strip.
- `hierarchy.json` — six-level rollout tree (families → composites →
  benchmarks → metrics). Drives the home-page rollout strip + family
  detail page.
- `comparison-index.json` — per-(eval, metric) leaderboards plus an inverse
  model→peer index. Backs the model-detail grid view; without it the grid
  renders empty regardless of how many cells the model has.

These are emitted by Python serialisation rather than DuckDB COPY because
they're scalar/JSON-shaped, not columnar, and the frontend reads them as
JSON. Stage cache integration deliberately skips them — sidecars are
cheap to re-derive from the cached canonical + view parquets.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from eval_card_backend.config import IGNORED_CONFIGS
from eval_card_backend.signals.reproducibility import (
    AGENTIC_REPRODUCIBILITY_FIELDS,
    BASE_REPRODUCIBILITY_FIELDS,
)
from eval_card_backend.slugs import url_encode


CONFIG_VERSION = 1
SIGNAL_VERSION = "1.0"


def write_manifest(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Tiny scalar manifest. Loaded once per process by the consumer."""
    counts = con.execute(
        """
        SELECT
            -- Use model_key so unresolved models (registry NULL) still
            -- count toward the corpus headline numbers.
            COUNT(DISTINCT model_key) FILTER (WHERE model_key IS NOT NULL)
                AS model_count,
            COUNT(DISTINCT (composite_slug, benchmark_id))
                FILTER (WHERE composite_slug IS NOT NULL AND benchmark_id IS NOT NULL)
                AS eval_count,
            COUNT(DISTINCT (model_key, composite_slug, benchmark_id, metric_id))
                FILTER (WHERE model_key      IS NOT NULL
                        AND   composite_slug IS NOT NULL
                        AND   benchmark_id   IS NOT NULL
                        AND   metric_id      IS NOT NULL)
                AS metric_eval_count
        FROM fact_results
        """
    ).fetchone()
    model_count, eval_count, metric_eval_count = counts
    # Read composite_count from the composites dim — that's the
    # filtered "live" set the frontend renders. Counting from
    # fact_results would double-count fully-unresolved composites
    # which never make it into the dim or hierarchy.json.
    composite_count = con.execute(
        "SELECT COUNT(*) FROM composites"
    ).fetchone()[0]

    skipped = sorted(IGNORED_CONFIGS)
    payload = {
        "generated_at":          snapshot_meta["snapshot_id"],
        "config_version":        CONFIG_VERSION,
        "skipped_configs":       skipped,
        "model_count":           int(model_count or 0),
        "eval_count":            int(eval_count or 0),
        "metric_eval_count":     int(metric_eval_count or 0),
        "composite_count":       int(composite_count or 0),
        "source_config_count":   len(snapshot_meta.get("configs") or []),
        "skipped_config_count":  len(skipped),
        # Upstream-input pins — answer "is this snapshot reading stale
        # registry/EEE data?" without a follow-up HF query. Each entry
        # is `{repo_id, sha, last_modified}`; values are None on lookup
        # failure (best-effort, never fatal — see _hf_dataset_snapshot).
        "upstream_pins":         snapshot_meta.get("upstream_pins") or {},
        "summary_artifacts": {
            "corpus_aggregates": "headline.json",
            "eval_hierarchy":    "hierarchy.json",
            "comparison_index":  "comparison-index.json",
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# headline.json
# ---------------------------------------------------------------------------


def _category_filter_clause(category: str | None, alias: str = "erv") -> str:
    """Return ` AND <alias>.category = '<cat>' ` (or empty when None).

    Single-quote escaping is sufficient — the CategoryType enum is
    closed (no operator-supplied input reaches this string).
    """
    if category is None:
        return ""
    safe = category.replace("'", "''")
    return f"AND {alias}.category = '{safe}'"


def _reproducibility_block(con, category: str | None) -> dict:
    """Triple-level rollups for reproducibility.

    Per-field missingness scans `fact_results` for every fact row's
    `repro_missing_fields[]`, then BOOL_OR's per-triple — a triple is
    flagged as missing field f if *any* of its rows had f in the gap.
    Base fields are denominated against all triples; agentic fields
    against agentic triples only (a triple is agentic if any row in it
    is_agentic).
    """
    # Category filtering needs eval_results_view's category column,
    # since fact_results doesn't carry one. Skip the JOIN entirely on
    # `category=None` to avoid a wasted scan on the corpus-overall path.
    if category is None:
        cat_join = ""
        cat_where = ""
    else:
        cat_join = (
            "JOIN eval_results_view erv "
            "  ON erv.model_key    = fr.model_key "
            " AND erv.benchmark_id = fr.benchmark_id "
            " AND erv.metric_id    = fr.metric_id"
        )
        cat_where = _category_filter_clause(category, alias="erv")

    field_flags_sql_parts = []
    for f in BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS:
        field_flags_sql_parts.append(
            f"BOOL_OR(array_contains(fr.repro_missing_fields, '{f}')) "
            f"AS missing_{f}"
        )
    field_flags_sql = ",\n            ".join(field_flags_sql_parts)

    triple_rollup_sql = f"""
        WITH triple_rollups AS (
            SELECT
                fr.model_key, fr.benchmark_id, fr.metric_id,
                BOOL_OR(fr.has_reproducibility_gap) AS triple_has_gap,
                BOOL_OR(fr.is_agentic)              AS triple_agentic,
                {field_flags_sql}
            FROM fact_results fr
            {cat_join}
            WHERE fr.model_key    IS NOT NULL
              AND fr.benchmark_id IS NOT NULL
              AND fr.metric_id    IS NOT NULL
              {cat_where}
            GROUP BY 1, 2, 3
        )
        SELECT
            COUNT(*)                           AS total_triples,
            SUM(CASE WHEN triple_has_gap THEN 1 ELSE 0 END) AS triples_with_gap,
            AVG(CASE WHEN triple_has_gap THEN 1.0 ELSE 0.0 END)
                                                AS gap_rate,
            SUM(CASE WHEN triple_agentic THEN 1 ELSE 0 END) AS agentic_triples,
            {",".join(
                f"SUM(CASE WHEN missing_{f} THEN 1 ELSE 0 END) AS n_missing_{f}"
                for f in BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS
            )}
        FROM triple_rollups
        """
    row = con.execute(triple_rollup_sql).fetchone()
    columns = [d[0] for d in con.description]
    rec = dict(zip(columns, row))

    total = int(rec["total_triples"] or 0)
    triples_with_gap = int(rec["triples_with_gap"] or 0)
    gap_rate = rec["gap_rate"]
    agentic = int(rec["agentic_triples"] or 0)

    per_field: dict[str, dict] = {}
    for f in BASE_REPRODUCIBILITY_FIELDS:
        n = int(rec[f"n_missing_{f}"] or 0)
        per_field[f] = {
            "missing_count":     n,
            "missing_rate":      (n / total) if total else None,
            "denominator":       "all_triples",
            "denominator_count": total,
        }
    for f in AGENTIC_REPRODUCIBILITY_FIELDS:
        n = int(rec[f"n_missing_{f}"] or 0)
        per_field[f] = {
            "missing_count":     n,
            "missing_rate":      (n / agentic) if agentic else None,
            "denominator":       "agentic_only",
            "denominator_count": agentic,
        }

    return {
        "total_triples":                       total,
        "triples_with_reproducibility_gap":    triples_with_gap,
        "reproducibility_gap_rate":            gap_rate,
        "agentic_triples":                     agentic,
        "per_field_missingness":               per_field,
    }


def _completeness_block(con, category: str | None) -> dict:
    """Triple-level completeness rollup.

    A triple's completeness is the AVG of its fact rows' completeness
    scores; the corpus-level rate is AVG over triples (not over rows).
    """
    cat_clause = _category_filter_clause(category)
    row = con.execute(
        f"""
        WITH triple_rollups AS (
            SELECT
                erv.model_key, erv.benchmark_id, erv.metric_id,
                erv.completeness_score AS triple_avg_completeness
            FROM eval_results_view erv
            WHERE 1 = 1 {cat_clause}
        )
        SELECT
            COUNT(*)                                AS total_triples,
            AVG(triple_avg_completeness)            AS completeness_avg,
            MIN(triple_avg_completeness)            AS completeness_min,
            MAX(triple_avg_completeness)            AS completeness_max
        FROM triple_rollups
        """
    ).fetchone()
    total, avg, mn, mx = row
    return {
        "total_triples":   int(total or 0),
        "completeness_avg": avg,
        "completeness_min": mn,
        "completeness_max": mx,
    }


def _provenance_block(con, category: str | None) -> dict:
    """Per-triple provenance rollup. coverage_cell + has_third_party drive
    the 4-way source-type distribution."""
    cat_clause = _category_filter_clause(category)
    row = con.execute(
        f"""
        SELECT
            COUNT(*)                                            AS total_triples,
            SUM(CASE WHEN erv.is_multi_source THEN 1 ELSE 0 END)     AS multi_source_triples,
            SUM(CASE WHEN erv.first_party_only THEN 1 ELSE 0 END)    AS first_party_only_triples,
            SUM(CASE WHEN erv.coverage_cell = 'self'                          THEN 1 ELSE 0 END) AS pst_first_party,
            SUM(CASE WHEN erv.coverage_cell = 'third' AND erv.has_third_party THEN 1 ELSE 0 END) AS pst_third_party,
            SUM(CASE WHEN erv.coverage_cell = 'both'                          THEN 1 ELSE 0 END) AS pst_collaborative,
            SUM(CASE WHEN erv.coverage_cell = 'third' AND NOT erv.has_third_party THEN 1 ELSE 0 END) AS pst_unspecified
        FROM eval_results_view erv
        WHERE 1 = 1 {cat_clause}
        """
    ).fetchone()
    (total, multi, first_only, pst_fp, pst_tp, pst_co, pst_un) = row
    return {
        "total_triples":            int(total or 0),
        "multi_source_triples":     int(multi or 0),
        "first_party_only_triples": int(first_only or 0),
        "source_type_distribution": {
            "first_party":   int(pst_fp or 0),
            "third_party":   int(pst_tp or 0),
            "collaborative": int(pst_co or 0),
            "unspecified":   int(pst_un or 0),
        },
    }


def _comparability_block(con, category: str | None) -> dict:
    cat_clause = _category_filter_clause(category)
    row = con.execute(
        f"""
        SELECT
            COUNT(*)                                                  AS total_triples,
            SUM(CASE WHEN erv.has_variant_divergence THEN 1 ELSE 0 END)     AS variant_divergent,
            SUM(CASE WHEN erv.has_cross_party_divergence THEN 1 ELSE 0 END) AS cross_party_divergent,
            SUM(CASE WHEN erv.has_variant_divergence IS NOT NULL THEN 1 ELSE 0 END)
                AS variant_eligible,
            SUM(CASE WHEN erv.has_cross_party_divergence IS NOT NULL THEN 1 ELSE 0 END)
                AS cross_party_eligible
        FROM eval_results_view erv
        WHERE 1 = 1 {cat_clause}
        """
    ).fetchone()
    (total, var_div, cross_div, var_elig, cross_elig) = row
    return {
        "total_triples":               int(total or 0),
        "variant_divergent_count":     int(var_div or 0),
        "cross_party_divergent_count": int(cross_div or 0),
        "groups_with_variant_check":     int(var_elig or 0),
        "groups_with_cross_party_check": int(cross_elig or 0),
    }


def _developers_list(con) -> list[dict]:
    rows = con.execute(
        """
        SELECT
            developer,
            COUNT(DISTINCT model_key)                                    AS model_count,
            COUNT(DISTINCT benchmark_id) FILTER (WHERE benchmark_id IS NOT NULL) AS benchmark_count,
            COUNT(*)                                                     AS evaluation_count
        FROM models_view
        LEFT JOIN UNNEST(benchmark_names) b(benchmark_id) ON TRUE
        WHERE developer IS NOT NULL
        GROUP BY developer
        ORDER BY evaluation_count DESC, developer ASC
        """
    ).fetchall()
    cols = [d[0] for d in con.description]
    out: list[dict] = []
    for r in rows:
        rec = dict(zip(cols, r))
        out.append({
            "developer":        rec["developer"],
            "route_id":         url_encode(rec["developer"]),
            "model_count":      int(rec["model_count"] or 0),
            "benchmark_count":  int(rec["benchmark_count"] or 0),
            "evaluation_count": int(rec["evaluation_count"] or 0),
            "popular_evals":    [],   # placeholder — frontend tolerates empty
        })
    return out


def _model_families_list(con) -> list[dict]:
    """Per-model-family rollup (developer + variant lineage groupings
    derived from `models_view.model_family_id`). Distinct from the
    benchmark-family hierarchy under `families` in hierarchy.json —
    name kept distinct to avoid the historical conflation.
    """
    rows = con.execute(
        """
        SELECT
            model_family_id   AS family_key,
            ANY_VALUE(model_family_name) AS display_name,
            COUNT(DISTINCT model_key)    AS model_count,
            SUM(evaluations_count)       AS eval_count
        FROM models_view
        WHERE model_family_id IS NOT NULL
        GROUP BY model_family_id
        ORDER BY eval_count DESC NULLS LAST, family_key ASC
        """
    ).fetchall()
    return [
        {
            "family_key":   r[0],
            "display_name": r[1] or r[0],
            "model_count":  int(r[2] or 0),
            "eval_count":   int(r[3] or 0),
        }
        for r in rows
    ]


def _composites_list(con) -> list[dict]:
    """Per-composite rollup. Counts of distinct benchmarks + models +
    triples per composite_slug. Drives the homepage composite strip.
    """
    rows = con.execute(
        """
        SELECT
            composite_slug,
            ANY_VALUE(composite_display_name)            AS display_name,
            COUNT(DISTINCT benchmark_id)                 AS benchmark_count,
            COUNT(DISTINCT model_key)                    AS model_count,
            COUNT(*)                                     AS evaluation_count
        FROM eval_results_view
        WHERE composite_slug IS NOT NULL
        GROUP BY composite_slug
        ORDER BY evaluation_count DESC, composite_slug ASC
        """
    ).fetchall()
    return [
        {
            "composite_slug":   r[0],
            "display_name":     r[1] or r[0],
            "benchmark_count":  int(r[2] or 0),
            "model_count":      int(r[3] or 0),
            "evaluation_count": int(r[4] or 0),
        }
        for r in rows
    ]


def _categories_list(con) -> list[dict]:
    rows = con.execute(
        """
        SELECT
            category,
            COUNT(DISTINCT model_key)               AS model_count,
            COUNT(DISTINCT (benchmark_id, metric_id)) AS eval_count
        FROM eval_results_view
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY eval_count DESC, category ASC
        """
    ).fetchall()
    return [
        {"category": r[0], "model_count": int(r[1] or 0), "eval_count": int(r[2] or 0)}
        for r in rows
    ]


def write_headline(con, out_dir: Path, snapshot_meta: dict) -> Path:
    from eval_card_backend.categorisation import categories as enum_categories

    cat_enum = enum_categories()

    def _stratified(builder):
        return {
            "overall":     builder(con, None),
            "by_category": {c: builder(con, c) for c in cat_enum},
        }

    payload = {
        "generated_at":              snapshot_meta["snapshot_id"],
        "signal_version":            SIGNAL_VERSION,
        "stratification_dimensions": ["category"],
        "reproducibility": _stratified(_reproducibility_block),
        "completeness":    _stratified(_completeness_block),
        "provenance":      _stratified(_provenance_block),
        "comparability":   _stratified(_comparability_block),
        "developers":      _developers_list(con),
        # Renamed from `families` so the model-family rollup doesn't
        # collide with the benchmark-family hierarchy in hierarchy.json
        # (the two are unrelated taxonomies that historically shared a
        # name in the legacy producer).
        "model_families":  _model_families_list(con),
        "composites":      _composites_list(con),
        "categories":      _categories_list(con),
    }
    path = out_dir / "headline.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# ---------------------------------------------------------------------------
# hierarchy.json
# ---------------------------------------------------------------------------


def write_hierarchy(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Composite/family/slice tree.

    Two top-level arrays:
      - `composites[]`: one entry per composite_slug, each with its
        list of benchmarks (and per-benchmark slices).
      - `families[]`: lookup index — one entry per family_id with the
        list of member benchmark keys.

    Composites are the primary tree. A benchmark reported in N
    composites appears N times across `composites[].benchmarks[]`,
    each occurrence with its own slices for that composite. The
    families[] index is what the frontend uses on a benchmark-detail
    page to render "related benchmarks in family X" links.
    """
    composites = _hierarchy_composites(con)
    families = _hierarchy_families_index(con)
    stats = _hierarchy_stats(con, composites, families)
    payload = {
        "generated_at": snapshot_meta["snapshot_id"],
        "stats":        stats,
        "composites":   composites,
        "families":     families,
    }
    path = out_dir / "hierarchy.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _hierarchy_stats(
    con, composites: list[dict], families: list[dict]
) -> dict:
    """Snapshot-level rollup counts.

    `metric_rows_scanned` is the raw fact-row count (each row = one
    reported result from one source; a triple measured by multiple
    organisations or split across slices contributes multiple rows). The
    homepage's "Reported results" stat reads this — distinct-triple
    counting would dedup away cross-source reports and read as ~3x
    lower than the user expects.
    """
    row = con.execute(
        """
        SELECT
            (SELECT COUNT(DISTINCT (benchmark_id, slice_key))
                 FROM fact_results
                 WHERE benchmark_id IS NOT NULL
                   AND slice_key    IS NOT NULL)                      AS slice_count,
            (SELECT COUNT(DISTINCT (composite_slug, benchmark_id, metric_id))
                 FROM fact_results
                 WHERE composite_slug IS NOT NULL
                   AND benchmark_id   IS NOT NULL
                   AND metric_id      IS NOT NULL)                    AS metric_count,
            (SELECT COUNT(*) FROM fact_results)                       AS metric_rows_scanned,
            (SELECT COUNT(DISTINCT benchmark_id)
                 FROM benchmarks WHERE benchmark_id IS NOT NULL)      AS benchmark_count
        """
    ).fetchone()
    return {
        "composite_count":     len(composites),
        "family_count":        len(families),
        "benchmark_count":     int(row[3] or 0),
        "slice_count":         int(row[0] or 0),
        "metric_count":        int(row[1] or 0),
        "metric_rows_scanned": int(row[2] or 0),
    }


def _hierarchy_composites(con) -> list[dict]:
    """Build the composites[] tree from the benchmarks dim + per-
    benchmark detail (metrics, slices, signal summaries from
    evals_view). evals_count comes from the composites dim — sum of
    per-(composite, benchmark, metric) triples — not a sum of
    per-benchmark models_count, which would double-count cross-
    benchmark models.
    """
    composite_evals_count = dict(
        con.execute(
            "SELECT composite_slug, evals_count FROM composites"
        ).fetchall()
    )

    rows = con.execute(
        """
        SELECT
            b.composite_slug,
            b.composite_display_name,
            b.benchmark_id,
            b.display_name,
            b.family_id,
            b.is_slice,
            b.parent_benchmark_id,
            b.card_present,
            b.domains, b.languages, b.tasks,
            ev.evaluation_id, ev.evaluation_name, ev.category,
            ev.models_count,
            ev.reproducibility_summary, ev.provenance_summary,
            ev.comparability_summary
        FROM benchmarks b
        LEFT JOIN evals_view ev
          ON ev.composite_slug = b.composite_slug
         AND ev.benchmark_id   = b.benchmark_id
        ORDER BY b.composite_slug, b.benchmark_id
        """
    ).fetchall()
    cols = [d[0] for d in con.description]
    benchmarks = [dict(zip(cols, r)) for r in rows]

    # Bucket by composite_slug. Within each composite, group rows into
    # root benchmarks (is_slice=FALSE) and slice rows attached to their
    # root via parent_benchmark_id.
    from collections import defaultdict
    by_composite: dict[str, list[dict]] = defaultdict(list)
    for b in benchmarks:
        by_composite[b["composite_slug"]].append(b)

    out: list[dict] = []
    for slug in sorted(by_composite):
        members = by_composite[slug]
        display_name = next(
            (m["composite_display_name"] for m in members
             if m["composite_display_name"]),
            slug,
        )
        evals_count = int(composite_evals_count.get(slug) or 0)
        # Group slices under their root benchmark.
        roots = [m for m in members if not m["is_slice"]]
        slices_by_root: dict[str, list[dict]] = defaultdict(list)
        for m in members:
            if m["is_slice"]:
                root_id = m["parent_benchmark_id"] or m["benchmark_id"]
                slices_by_root[root_id].append(m)
            elif (m["parent_benchmark_id"] is not None
                  and m["parent_benchmark_id"] == m["benchmark_id"]):
                # Self-parented bare-stem (e.g. gaia → gaia): include
                # the bare stem as a slice of itself, marked is_bare_stem.
                slices_by_root[m["benchmark_id"]].append({
                    **m, "_is_bare_stem": True,
                })

        bench_records = [
            _hierarchy_composite_benchmark(con, slug, root, slices_by_root.get(root["benchmark_id"], []))
            for root in roots
        ]
        bench_records.sort(key=lambda r: r["key"])

        # Composite-level rollups: union of tags and counts across roots.
        domains = sorted({t for r in bench_records for t in r["tags"]["domains"]})
        languages = sorted({t for r in bench_records for t in r["tags"]["languages"]})
        tasks = sorted({t for r in bench_records for t in r["tags"]["tasks"]})
        category = _composite_category(roots)
        repro = _aggregate_reproducibility(roots)
        prov = _aggregate_provenance(roots)
        comp = _aggregate_comparability(roots)

        out.append({
            "key":          slug,
            "display_name": display_name,
            "category":     category,
            "tags":         {"domains": domains, "languages": languages, "tasks": tasks},
            "evals_count":  evals_count,
            "benchmarks":   bench_records,
            "reproducibility_summary": repro,
            "provenance_summary":      prov,
            "comparability_summary":   comp,
        })
    return out


def _composite_category(members: list[dict]) -> str:
    """Mode-most-common category across the composite's root benchmarks."""
    from collections import Counter

    cats = [m["category"] for m in members if m.get("category")]
    if not cats:
        return "General"
    counts = Counter(cats)
    top_count = max(counts.values())
    candidates = [c for c, n in counts.items() if n == top_count]
    if len(candidates) == 1:
        return candidates[0]
    for m in sorted(members, key=lambda x: x["benchmark_id"]):
        if m.get("category") in candidates:
            return m["category"]
    return candidates[0]


def _hierarchy_composite_benchmark(
    con, composite_slug: str, root: dict, slice_rows: list[dict]
) -> dict:
    """Build one benchmark sub-record under composites[].benchmarks[].

    Slices are the within-benchmark cuts for this (composite, benchmark)
    pair. Each slice carries its own metric list. The bare-stem slice
    (when one exists, e.g. `gaia` inside the `gaia` benchmark) is
    flagged with is_bare_stem=true so the frontend can render it as
    "Overall" / "Main".
    """
    benchmark_id = root["benchmark_id"]
    metrics_rows = con.execute(
        """
        SELECT metric_id,
               ANY_VALUE(metric_display_name) AS display_name,
               ARRAY_AGG(DISTINCT source_metadata.source_organization_name)
                   FILTER (WHERE source_metadata.source_organization_name IS NOT NULL)
                   AS sources
        FROM eval_results_view
        WHERE composite_slug = ? AND benchmark_id = ?
        GROUP BY metric_id
        ORDER BY metric_id
        """,
        [composite_slug, benchmark_id],
    ).fetchall()

    eval_ids_row = con.execute(
        "SELECT ARRAY_AGG(DISTINCT evaluation_id ORDER BY evaluation_id) "
        "FROM evals_view WHERE composite_slug = ? AND benchmark_id = ?",
        [composite_slug, benchmark_id],
    ).fetchone()

    return {
        "key":          benchmark_id,
        "display_name": root.get("display_name") or benchmark_id,
        "family_id":    root.get("family_id") or benchmark_id,
        "is_slice":     False,
        "has_card":     bool(root.get("card_present"))
                        if root.get("card_present") is not None else False,
        "tags": {
            "domains":   list(root.get("domains") or []),
            "languages": list(root.get("languages") or []),
            "tasks":     list(root.get("tasks") or []),
        },
        "metrics": [
            {
                "key":          m[0],
                "display_name": m[1] or m[0],
                "sources":      m[2] or [],
            }
            for m in metrics_rows
        ],
        "slices":   _hierarchy_composite_slices(
                        con, composite_slug, benchmark_id, slice_rows),
        "summary_eval_ids":        eval_ids_row[0] if eval_ids_row and eval_ids_row[0] else [],
        "reproducibility_summary": root.get("reproducibility_summary"),
        "provenance_summary":      root.get("provenance_summary"),
        "comparability_summary":   root.get("comparability_summary"),
    }


def _hierarchy_composite_slices(
    con, composite_slug: str, benchmark_id: str, slice_rows: list[dict]
) -> list[dict]:
    """Per-(composite, benchmark) slices[]: within-benchmark cuts.

    Two slice mechanisms are unioned:

      1. **Sibling-benchmark slices** (`slice_rows`) — separate
         canonical benchmarks parented to the same root. GAIA's
         `gaia-level-1/2/3` and the self-parented `gaia` "Overall"
         row, CapArena's `caparena-vs-X`, GPQA's `gpqa-diamond`. These
         are full (composite, benchmark) rows in the dim with their
         own metrics/orgs in `eval_results_view`; the slice key is
         the slice benchmark's id and `is_bare_stem=true` when it
         equals the root.
      2. **slice_key cuts** (within-benchmark) — MMLU subjects style,
         where multiple raw evaluation_names collapse to one
         canonical id and Stage C keeps the raw on each fact row's
         `slice_key`. Read from `fact_results` keyed on `slice_key`.

    The two never collide in practice (a benchmark uses one mechanism
    or the other), but a UNION at the output level is the safe contract.
    """
    out: list[dict] = []

    # (1) Sibling-benchmark slices from the benchmarks dim.
    for sr in slice_rows:
        slice_id = sr["benchmark_id"]
        rows = con.execute(
            """
            SELECT
                erv.metric_id,
                ANY_VALUE(erv.metric_display_name) AS metric_display,
                ARRAY_AGG(DISTINCT erv.source_metadata.source_organization_name)
                    FILTER (WHERE erv.source_metadata.source_organization_name IS NOT NULL)
                    AS sources
            FROM eval_results_view erv
            WHERE erv.composite_slug = ? AND erv.benchmark_id = ?
            GROUP BY erv.metric_id
            ORDER BY erv.metric_id
            """,
            [composite_slug, slice_id],
        ).fetchall()
        out.append({
            "key":          slice_id,
            "display_name": sr.get("display_name") or slice_id,
            "is_bare_stem": bool(sr.get("_is_bare_stem"))
                            or slice_id == benchmark_id,
            "metrics": [
                {
                    "key":          r[0],
                    "display_name": r[1] or r[0],
                    "sources":      list(r[2] or []),
                }
                for r in rows
            ],
        })

    # (2) Within-benchmark slice_key cuts (MMLU subjects, etc.).
    rows = con.execute(
        """
        WITH per_slice_metric AS (
            SELECT
                fr.slice_key,
                fr.metric_id,
                MIN(fr.slice_name)                       AS slice_name_rep,
                ANY_VALUE(cmet.display_name)             AS metric_display,
                ARRAY_AGG(DISTINCT fr.org_raw)
                    FILTER (WHERE fr.org_raw IS NOT NULL) AS sources
            FROM fact_results fr
            LEFT JOIN canonical_metrics cmet ON cmet.id = fr.metric_id
            WHERE fr.composite_slug = ?
              AND fr.benchmark_id = ?
              AND fr.slice_key IS NOT NULL
              AND fr.metric_id IS NOT NULL
            GROUP BY fr.slice_key, fr.metric_id
        )
        SELECT
            slice_key,
            MIN(slice_name_rep) AS slice_display_name,
            ARRAY_AGG(struct_pack(
                metric_id      := metric_id,
                metric_display := metric_display,
                sources        := sources
            ) ORDER BY metric_id) AS metrics
        FROM per_slice_metric
        GROUP BY slice_key
        ORDER BY slice_key
        """,
        [composite_slug, benchmark_id],
    ).fetchall()
    sibling_keys = {s["key"] for s in out}
    for row in rows:
        slice_key = row[0]
        # Skip if a sibling-benchmark slice already covers this id (the
        # bare-stem `gaia` slice_key would otherwise duplicate the
        # `gaia` self-parented sibling slice).
        if slice_key in sibling_keys:
            continue
        out.append({
            "key":          slice_key,
            "display_name": row[1] or slice_key,
            "is_bare_stem": slice_key == benchmark_id,
            "metrics": [
                {
                    "key":          m["metric_id"],
                    "display_name": m["metric_display"] or m["metric_id"],
                    "sources":      list(m["sources"] or []),
                }
                for m in row[2]
            ],
        })

    out.sort(key=lambda s: s["key"])
    return out


def _hierarchy_families_index(con) -> list[dict]:
    """Lightweight family lookup index. One entry per family_id with the
    member benchmark keys. No per-composite info — that lives under
    composites[].
    """
    rows = con.execute(
        """
        SELECT family_id, family_display_name, member_benchmark_keys
        FROM families
        ORDER BY family_id
        """
    ).fetchall()
    return [
        {
            "key":                   r[0],
            "display_name":          r[1] or r[0],
            "member_benchmark_keys": list(r[2] or []),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Phantom / display-name helpers (used by Stage G phantom synthesis and
# the composite hierarchy aggregations).
# ---------------------------------------------------------------------------


# Acronyms preserved verbatim when title-casing a stem fallback display name.
# Keep small — only well-known names where the registry-style display would
# otherwise lose mixed-case (CapArena, GAIA, GPQA, MMLU, etc.).
_KNOWN_ACRONYMS: dict[str, str] = {
    "gaia":     "GAIA",
    "gpqa":     "GPQA",
    "mmlu":     "MMLU",
    "bfcl":     "BFCL",
    "helm":     "HELM",
    "bbh":      "BBH",
    "bbq":      "BBQ",
    "ifeval":   "IFEval",
    "musr":     "MuSR",
    "caparena": "CapArena",
    "videomme": "VideoMME",
    "math":     "MATH",
    "aime":     "AIME",
    "lcb":      "LCB",
    "ace":      "ACE",
    "mmmu":     "MMMU",
    "arc":      "ARC",
    "agi":      "AGI",
    "swe":      "SWE",
}


def _title_case_stem(stem: str) -> str:
    if stem in _KNOWN_ACRONYMS:
        return _KNOWN_ACRONYMS[stem]
    parts = [_KNOWN_ACRONYMS.get(p, p.capitalize()) for p in stem.split("-") if p]
    return " ".join(parts) if parts else stem


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return ""
    # If the prefix cuts mid-word in any of the originals (next char in
    # that string is alphanumeric), back off to the last word boundary so
    # we don't return ragged stubs like "ARC-AGI v".
    if any(
        len(s) > len(prefix) and s[len(prefix)].isalnum()
        for s in strings
    ):
        last_ws = max(
            (i for i, ch in enumerate(prefix) if ch.isspace()),
            default=-1,
        )
        prefix = prefix[:last_ws] if last_ws >= 0 else ""
    prefix = prefix.rstrip(" \t-_:/")
    # Drop trailing one-letter alphabetic tokens (e.g. "Videomme W" →
    # "Videomme") — they're artefacts of the LCP. Numeric tokens stay
    # ("RewardBench 2" should not collapse to "RewardBench").
    while True:
        tokens = prefix.split()
        if (
            len(tokens) >= 2
            and len(tokens[-1]) == 1
            and tokens[-1].isalpha()
        ):
            prefix = " ".join(tokens[:-1])
        else:
            break
    return prefix.rstrip(" \t-_:/")


def _aggregate_reproducibility(members: list[dict]) -> dict | None:
    """Sum results_total + has_reproducibility_gap_count; recompute
    populated_ratio_avg as a results-weighted mean (spec §4)."""
    total = 0
    gap_count = 0
    weighted_sum = 0.0
    weight = 0
    seen = False
    for m in members:
        s = m.get("reproducibility_summary")
        if not s:
            continue
        seen = True
        n = int(s.get("results_total") or 0)
        total += n
        gap_count += int(s.get("has_reproducibility_gap_count") or 0)
        ratio = s.get("populated_ratio_avg")
        if ratio is not None and n > 0:
            weighted_sum += float(ratio) * n
            weight += n
    if not seen:
        return None
    return {
        "results_total":                 total,
        "has_reproducibility_gap_count": gap_count,
        "populated_ratio_avg":           (weighted_sum / weight) if weight else None,
    }


def _aggregate_provenance(members: list[dict]) -> dict | None:
    out = {
        "total_results":           0,
        "total_groups":            0,
        "multi_source_groups":     0,
        "first_party_only_groups": 0,
        "source_type_distribution": {
            "first_party":   0,
            "third_party":   0,
            "collaborative": 0,
            "unspecified":   0,
        },
    }
    seen = False
    for m in members:
        s = m.get("provenance_summary")
        if not s:
            continue
        seen = True
        out["total_results"] += int(s.get("total_results") or 0)
        out["total_groups"] += int(s.get("total_groups") or 0)
        out["multi_source_groups"] += int(s.get("multi_source_groups") or 0)
        out["first_party_only_groups"] += int(s.get("first_party_only_groups") or 0)
        dist = s.get("source_type_distribution") or {}
        for k in out["source_type_distribution"]:
            out["source_type_distribution"][k] += int(dist.get(k) or 0)
    return out if seen else None


def _aggregate_comparability(members: list[dict]) -> dict | None:
    out = {
        "total_groups":                  0,
        "groups_with_variant_check":     0,
        "groups_with_cross_party_check": 0,
        "variant_divergent_count":       0,
        "cross_party_divergent_count":   0,
    }
    seen = False
    for m in members:
        s = m.get("comparability_summary")
        if not s:
            continue
        seen = True
        for k in out:
            out[k] += int(s.get(k) or 0)
    return out if seen else None


# ---------------------------------------------------------------------------
# comparison-index.json
# ---------------------------------------------------------------------------


# Tab-strip ordering that the frontend's plotbox expects. Capability surfaces
# first (the actual task score), then capability-adjacent groups, then
# instrumental groups, with "other" as the fallback bucket.
_METRIC_GROUP_ORDER = (
    "capability",
    "robustness",
    "efficiency",
    "cost",
    "latency",
    "rank",
    "other",
)
_METRIC_GROUP_INDEX = {group: i for i, group in enumerate(_METRIC_GROUP_ORDER)}


# Authoritative mapping when `metric_kind` is populated on the fact row
# (registry preferred, EEE_meta fallback, regex inference last — see
# `metric_meta_hotfix._infer_metric_kind_from_name`). Mirrors the legacy
# producer's table so cutover doesn't reshuffle the tab strip.
_METRIC_KIND_TO_GROUP: dict[str, str] = {
    "accuracy":   "capability",
    "elo":        "capability",
    "score":      "capability",
    "pass":       "capability",
    "f1":         "capability",
    "win_rate":   "capability",
    "winrate":    "capability",
    "cost":       "cost",
    "latency":    "latency",
    "throughput": "latency",
    "time":       "latency",
    "rank":       "rank",
    "difference": "robustness",
}

# Order matters: first matching pattern wins, listed most-specific first
# so e.g. "Latency Standard Deviation" lands in latency rather than
# robustness. Used when `metric_kind` is absent — covers the long tail of
# metric names the hotfix UDF couldn't classify upstream.
_METRIC_NAME_GROUP_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("cost",       re.compile(r"\b(?:cost|usd|dollar|price)\b", re.IGNORECASE)),
    ("latency",    re.compile(
        r"\b(?:latency|throughput|elapsed|wall[\s_]?time|"
        r"tokens?[\s_/]?(?:per|sec|s)\b|p\d{2,3}|percentile)\b",
        re.IGNORECASE,
    )),
    ("rank",       re.compile(r"\brank\b", re.IGNORECASE)),
    ("robustness", re.compile(
        r"\b(?:sensitivity|delta|stddev|standard[\s_]?deviation|"
        r"variance|robustness)\b",
        re.IGNORECASE,
    )),
    ("efficiency", re.compile(r"\b(?:attempts|retries|tries)\b", re.IGNORECASE)),
    ("capability", re.compile(
        r"\b(?:accuracy|acc|elo|score|pass@\d+|win[\s_]?rate|f1|"
        r"exact[\s_]?match|em|bleu|rouge(?:-\d+)?|recall|precision|"
        r"mrr|ndcg|coverage|correct|harmlessness)\b",
        re.IGNORECASE,
    )),
)


def _classify_metric_group(metric_kind: str | None, metric_name: str | None) -> str:
    """Return the tab-strip bucket for a metric.

    Precedence: `metric_kind` (registry > EEE > inferred) → name regex →
    `"other"`. Caller is responsible for passing the metric_kind sourced
    from `fact_results`, where the hotfix UDF has already applied that
    precedence at canonicalisation time.
    """
    if metric_kind:
        kind = metric_kind.strip().lower()
        if kind in _METRIC_KIND_TO_GROUP:
            return _METRIC_KIND_TO_GROUP[kind]
    if metric_name:
        for group, pattern in _METRIC_NAME_GROUP_RULES:
            if pattern.search(metric_name):
                return group
    return "other"


def write_comparison_index(con, out_dir: Path, snapshot_meta: dict) -> Path:
    """Per-(eval, metric) leaderboards + inverse model→peer index.

    Backs the grid view on the model detail page. The frontend's
    `plotboxUnits` skips any eval not present here (`comparisonIndex.evals[id]
    ?? continue`), so this artifact's keyset must cover every evaluation_id
    in `eval_results_view`.

    `eval_results_view` already collapses fact rows to one row per
    `(model_key, benchmark_id, metric_id)` triple — the legacy producer's
    submission-tail logic doesn't apply, so every leaderboard row carries
    `submission_count=1, submission_axis="default"`. If/when the view layer
    starts preserving multiple submissions per triple, this is the single
    place that needs to learn about it.
    """
    # `metric_kind` is per-metric within a benchmark; pre-aggregate from
    # fact_results once rather than carry it on every cell row. Mirrors the
    # MAX-FILTER pattern stage I uses when packing it into metric_config.
    rows = con.execute(
        """
        WITH metric_kinds AS (
            SELECT
                benchmark_id,
                metric_id,
                MAX(metric_kind) FILTER (WHERE metric_kind IS NOT NULL) AS metric_kind
            FROM fact_results
            WHERE benchmark_id IS NOT NULL AND metric_id IS NOT NULL
            GROUP BY benchmark_id, metric_id
        )
        SELECT
            erv.evaluation_id,
            erv.metric_summary_id,
            erv.metric_id,
            erv.metric_display_name,
            erv.metric_unit,
            erv.lower_is_better,
            erv.score,
            erv.model_key,
            erv.model_id,
            erv.model_route_id,
            mv.model_family_id,
            mv.model_family_name,
            mv.developer,
            mk.metric_kind
        FROM eval_results_view erv
        LEFT JOIN models_view mv
          ON mv.model_id = erv.model_id
        LEFT JOIN metric_kinds mk
          ON mk.benchmark_id = erv.benchmark_id
         AND mk.metric_id    = erv.metric_id
        WHERE erv.score             IS NOT NULL
          AND erv.evaluation_id     IS NOT NULL
          AND erv.metric_summary_id IS NOT NULL
          AND erv.model_route_id    IS NOT NULL
        """
    ).fetchall()
    cols = [d[0] for d in con.description]

    eval_meta_rows = con.execute(
        """
        SELECT
            evaluation_id,
            evaluation_name,
            canonical_display_name,
            composite_slug,
            composite_display_name,
            family_id,
            family_display_name,
            is_slice,
            category,
            is_summary_score,
            summary_eval_ids
        FROM evals_view
        """
    ).fetchall()
    eval_meta_cols = [d[0] for d in con.description]
    eval_meta = {
        r[0]: dict(zip(eval_meta_cols, r))
        for r in eval_meta_rows
        if r[0] is not None
    }

    # Group cells by (evaluation_id, metric_summary_id) — this is the
    # leaderboard key.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        rec = dict(zip(cols, r))
        grouped[(rec["evaluation_id"], rec["metric_summary_id"])].append(rec)

    eval_metric_buckets: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for (eval_id, metric_summary_id), peer_rows in grouped.items():
        first = peer_rows[0]
        lower_is_better = bool(first["lower_is_better"])

        # Two-pass stable sort: route id ascending tiebreak, then score in
        # the metric's preferred direction. Mirrors the legacy producer's
        # ordering so existing UI ranks don't shift on cutover.
        peer_rows.sort(key=lambda r: r["model_route_id"])
        peer_rows.sort(
            key=lambda r: r["score"], reverse=not lower_is_better
        )

        total = len(peer_rows)
        scores_out: list[dict] = []
        position = 0
        previous_score = None
        for idx, rec in enumerate(peer_rows, start=1):
            sc = rec["score"]
            # Dense-tie ranking: position only advances when score changes,
            # so peers at the same score share a rank. Matches legacy.
            if previous_score is None or sc != previous_score:
                position = idx
                previous_score = sc

            scores_out.append({
                "model_route_id":    rec["model_route_id"],
                "model_family_id":   rec["model_family_id"]
                                       or rec["model_id"]
                                       or rec["model_key"],
                "model_family_name": rec["model_family_name"] or "",
                "developer":         rec["developer"] or "",
                "variant_key":       "default",
                "score":             sc,
                "rank":              position,
                "total":             total,
                "submission_count":  1,
                "submission_axis":   "default",
            })
            by_model[rec["model_route_id"]][eval_id][metric_summary_id] = {
                "score":            sc,
                "rank":             position,
                "total":            total,
                "submission_count": 1,
                "submission_axis":  "default",
            }

        group = _classify_metric_group(
            first.get("metric_kind"), first.get("metric_display_name")
        )
        eval_metric_buckets[eval_id].append({
            "metric_summary_id": metric_summary_id,
            "metric_name":       first["metric_display_name"] or "",
            "metric_id":         first["metric_id"],
            "metric_key":        first["metric_id"],
            "group":             group,
            "group_order":       _METRIC_GROUP_INDEX[group],
            "lower_is_better":   lower_is_better,
            "unit":              first["metric_unit"],
            "scores":            scores_out,
        })

    evals_out: dict[str, dict] = {}
    for eval_id, metrics in eval_metric_buckets.items():
        meta = eval_meta.get(eval_id, {})
        # Capability tabs surface first (so the actual task score is the
        # default tab on the histogram strip), then the rest of the group
        # taxonomy. Within-group ordering stays alphabetical for determinism.
        metrics.sort(
            key=lambda m: (m["group_order"], m["metric_name"] or "", m["metric_summary_id"])
        )
        evals_out[eval_id] = {
            "eval_summary_id":         eval_id,
            "composite_slug":          meta.get("composite_slug"),
            "composite_display_name":  meta.get("composite_display_name"),
            "family_id":               meta.get("family_id"),
            "family_display_name":     meta.get("family_display_name"),
            "is_slice":                bool(meta.get("is_slice")),
            "display_name":            meta.get("canonical_display_name")
                                         or meta.get("evaluation_name"),
            "category":                meta.get("category") or "General",
            "is_summary_score":        bool(meta.get("is_summary_score")),
            "summary_score_for":       None,
            "summary_eval_ids":        list(meta.get("summary_eval_ids") or []),
            "metrics":                 metrics,
        }

    payload = {
        "generated_at":       snapshot_meta["snapshot_id"],
        "config_version":     CONFIG_VERSION,
        "metric_group_order": list(_METRIC_GROUP_ORDER),
        "evals":              evals_out,
        "by_model":           {
            route: {ev: dict(metrics) for ev, metrics in evs.items()}
            for route, evs in by_model.items()
        },
    }
    path = out_dir / "comparison-index.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(obj):
    """Coerce DuckDB-native types JSON can't serialise."""
    import datetime
    import decimal

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type not serialisable: {type(obj).__name__}")
