"""Sidecars — slice surfacing in `hierarchy.json`.

`write_hierarchy` populates two slice-derived fields:

  - `stats.slice_count`: distinct (benchmark_id, slice_key) pairs in
    `fact_results` where slice_key is non-NULL. Drives the homepage strip.
  - `families[].standalone_benchmarks[].slices[]` (and the analogous
    composite path): per-benchmark list of `{key, display_name, metrics[]}`.
    Drives the family-table per-row slice count + the family-detail tree.

These tests build minimal synthetic `fact_results` + `canonical_metrics`
tables and call the sidecar helpers directly. The end-to-end fixture
corpus is single-raw across all benchmarks, so it can't exercise the
populated-slice path; that's why this test stays at the helper level.
"""
from __future__ import annotations

import duckdb
import pytest

from eval_card_backend.canonicalise.sidecars import (
    _hierarchy_benchmark_slices,
    _hierarchy_stats,
)


_FACT_DDL = """
CREATE TABLE fact_results (
    benchmark_id  VARCHAR,
    slice_key     VARCHAR,
    slice_name    VARCHAR,
    metric_id     VARCHAR,
    model_key     VARCHAR,
    org_raw       VARCHAR
)
"""

_BENCH_DDL = """
CREATE TABLE benchmarks (
    benchmark_id        VARCHAR,
    parent_benchmark_id VARCHAR
)
"""

_METRICS_DDL = """
CREATE TABLE canonical_metrics (
    id           VARCHAR,
    display_name VARCHAR
)
"""


def _seed_minimal_tables(con):
    """Slice-count SQL also reads `benchmarks` (for family/composite
    counts). Provide a one-row stand-in so the sidecar query runs even
    though those non-slice counts aren't what we're asserting on."""
    con.execute(_FACT_DDL)
    con.execute(_BENCH_DDL)
    con.execute(_METRICS_DDL)
    con.execute("INSERT INTO benchmarks VALUES ('mmlu', NULL)")


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# slice_count
# ---------------------------------------------------------------------------


def test_slice_count_zero_when_no_slices(con):
    """Single-raw corpus → every fact row has slice_key NULL → no slices."""
    _seed_minimal_tables(con)
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mmlu", None, None, "accuracy", "openai/gpt-4o", "OpenAI"),
            ("mmlu", None, None, "accuracy", "anthropic/claude", "Anthropic"),
        ],
    )

    stats = _hierarchy_stats(con)
    assert stats["slice_count"] == 0


def test_slice_count_counts_distinct_benchmark_slice_pairs(con):
    """slice_count is COUNT DISTINCT over (benchmark_id, slice_key);
    duplicate (benchmark, slice) rows from different models / orgs
    don't inflate it."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO benchmarks VALUES ('mmlu-pro', NULL)")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            # 3 distinct (benchmark, slice) pairs total.
            ("mmlu",     "anatomy",   "Anatomy",   "accuracy", "m1", "o1"),
            ("mmlu",     "anatomy",   "Anatomy",   "accuracy", "m2", "o1"),  # dup
            ("mmlu",     "astronomy", "Astronomy", "accuracy", "m1", "o1"),
            ("mmlu-pro", "physics",   "Physics",   "accuracy", "m1", "o1"),
            # NULL slice_key shouldn't count.
            ("mmlu",     None,        None,        "accuracy", "m3", "o2"),
        ],
    )

    stats = _hierarchy_stats(con)
    assert stats["slice_count"] == 3


# ---------------------------------------------------------------------------
# per-benchmark slices[]
# ---------------------------------------------------------------------------


def test_slices_empty_when_no_slice_rows(con):
    """Standalone single-raw benchmark → empty slices[]."""
    _seed_minimal_tables(con)
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mmlu", None, None, "accuracy", "m1", "OpenAI"),
        ],
    )
    assert _hierarchy_benchmark_slices(con, "mmlu") == []


def test_slices_populated_with_per_slice_metrics(con):
    """Each slice carries its own metric list. Multiple metric rows
    per (slice, metric) pair de-dup at the metric level; orgs land in
    `sources[]` deduplicated."""
    _seed_minimal_tables(con)
    con.executemany(
        "INSERT INTO canonical_metrics VALUES (?, ?)",
        [("accuracy", "Accuracy"), ("exact-match", "Exact Match")],
    )
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mmlu", "anatomy",   "Anatomy",   "accuracy",    "m1", "OpenAI"),
            ("mmlu", "anatomy",   "Anatomy",   "accuracy",    "m2", "Scale AI"),
            ("mmlu", "anatomy",   "Anatomy",   "exact-match", "m1", "OpenAI"),
            ("mmlu", "astronomy", "Astronomy", "accuracy",    "m1", "OpenAI"),
        ],
    )

    slices = _hierarchy_benchmark_slices(con, "mmlu")
    assert len(slices) == 2

    by_key = {s["key"]: s for s in slices}
    anatomy = by_key["anatomy"]
    assert anatomy["display_name"] == "Anatomy"
    metric_keys = sorted(m["key"] for m in anatomy["metrics"])
    assert metric_keys == ["accuracy", "exact-match"]
    accuracy = next(m for m in anatomy["metrics"] if m["key"] == "accuracy")
    assert accuracy["display_name"] == "Accuracy"
    assert sorted(accuracy["sources"]) == ["OpenAI", "Scale AI"]

    astronomy = by_key["astronomy"]
    assert astronomy["display_name"] == "Astronomy"
    assert [m["key"] for m in astronomy["metrics"]] == ["accuracy"]
    assert astronomy["metrics"][0]["sources"] == ["OpenAI"]


def test_slices_excludes_null_slice_rows(con):
    """A benchmark can have a mix of slice and non-slice rows (e.g. a
    rolled-up MMLU score row alongside per-subject rows where slice_key
    is also "mmlu"). NULL slice_key rows are dropped from slices[];
    they belong to the parent benchmark's metrics[] aggregation, not a
    sub-slice."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mmlu", "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
            ("mmlu", None,      None,      "accuracy", "m1", "OpenAI"),
        ],
    )

    slices = _hierarchy_benchmark_slices(con, "mmlu")
    assert [s["key"] for s in slices] == ["anatomy"]


def test_slices_filtered_by_benchmark_id(con):
    """Calling for one benchmark only returns its slices, not another
    multi-slice benchmark in the same fact_results."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO benchmarks VALUES ('mmlu-pro', NULL)")
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mmlu",     "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
            ("mmlu-pro", "physics", "Physics", "accuracy", "m1", "OpenAI"),
        ],
    )
    assert [s["key"] for s in _hierarchy_benchmark_slices(con, "mmlu")] == ["anatomy"]
    assert [s["key"] for s in _hierarchy_benchmark_slices(con, "mmlu-pro")] == ["physics"]


def test_slice_display_name_picks_deterministic_representative(con):
    """When the same slice_key has multiple slice_name casings across
    rows (e.g. 'MMLU' + 'mmlu' folded into slice_key='mmlu'), MIN picks
    the lex-earliest so re-runs are byte-stable."""
    _seed_minimal_tables(con)
    con.execute("INSERT INTO canonical_metrics VALUES ('accuracy', 'Accuracy')")
    con.executemany(
        "INSERT INTO fact_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("mmlu", "mmlu", "MMLU", "accuracy", "m1", "OpenAI"),
            ("mmlu", "mmlu", "mmlu", "accuracy", "m2", "OpenAI"),
            ("mmlu", "mmlu", "Mmlu", "accuracy", "m3", "OpenAI"),
            # also include a real second slice so this benchmark has
            # multi-slice fan-out at all (Stage C wouldn't have written
            # slice_key='mmlu' for a single-raw benchmark).
            ("mmlu", "anatomy", "Anatomy", "accuracy", "m1", "OpenAI"),
        ],
    )

    slices = {s["key"]: s for s in _hierarchy_benchmark_slices(con, "mmlu")}
    # 'MMLU' < 'Mmlu' < 'mmlu' under default ASCII comparison.
    assert slices["mmlu"]["display_name"] == "MMLU"
