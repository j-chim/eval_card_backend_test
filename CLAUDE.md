# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

`eval-card-backend` materialises evaluation artifacts from
`evaleval/EEE_datastore` and `evaleval/auto-benchmarkcards`, computes the
four interpretive signals (reproducibility / completeness / provenance /
comparability), and emits Parquet for DuckDB consumption by the Eval
Cards frontend.

This repo is **net-new**, replacing the legacy pipeline at
`../eval_cards_backend_pipeline` (which was over-bloated and is being
retired). Do not import from the legacy repo. Refer to it only for
historical context (e.g., the production reproducibility-fields rule).

## Where the design lives

Iterating design notes are in `notes/`, numbered for stable references:

- `notes/01-schema-from-frontend.md` — canonical Parquet schema, derived
  from the frontend prototype at `../Eval Cards New Design/`. Glossary,
  per-table DDL, controlled vocabularies, query examples.
- `notes/02-producer-shape.md` — ingestion pipeline (DuckDB + resolver
  in-process). Stages, UDFs, edge cases, module layout, CLI.

The read side (Next.js HF Space with `@duckdb/node-api`) is out of scope
for this repo. Consumers refer to `01-` for the per-view queries.

When making schema or pipeline changes, update the notes alongside the
code. The notes are the spec.

## Architecture pointers

- **Identity is the registry's job.** `evalcard-registry` (separate
  repo) resolves all model / benchmark / metric / org / harness names.
  This producer never reinvents identity normalisation. We import the
  `eval-entity-resolver` package and call its resolver in-process; we do
  **not** read the registry's internal `eval_results` table.
- **Storage:** DuckDB on Parquet. Wide fact table (`fact_results.parquet`)
  joined to small dim tables. Snapshots are append-only; `snapshot_id` is
  an ISO timestamp column on every table.
- **Source layers (HF datasets):**
  `evaleval/EEE_datastore` (raw eval records),
  `evaleval/auto-benchmarkcards` (benchmark prose),
  `evaleval/entity-registry-data` (canonical IDs + alias store).

## Conventions

- Tests use `LOCAL_MODE` style — no HF credentials needed for local dev.
- Identity normalisation logic does NOT live here. Bug or pathology in
  resolver output? Fix it in `evalcard-registry`, not here.
- Don't add alias YAMLs to this repo. `registry/` is for operationalised
  config (completeness field set, agentic name regex) only.
- Raw + canonical pair on every entity column. NULL canonical means
  unresolved; raw is always populated. Never drop a row for being
  unresolved — surface it.

## Dependencies — TODO when registry stabilises

Currently `eval-entity-resolver` is wired as a **uv workspace / local
path dependency** to `../evalcard-registry/packages/eval-entity-resolver`
because the package isn't published yet and we want resolver edits to
flow through immediately.

**Switch to a git URL dep pinned to a commit when the registry stabilises
and is properly pushed:**

```toml
# Example future dep declaration in pyproject.toml
"eval-entity-resolver @ git+https://github.com/evaleval/evalcard-registry@<sha>#subdirectory=packages/eval-entity-resolver"
```

Pinning to a commit gives reproducible resolver behaviour per snapshot
without forcing publishing infrastructure. Revisit when:
- The registry's resolver has a stable contract (no breaking changes
  expected within a release cadence).
- Multiple consumers depend on it (worth publishing as a wheel then).

## When working on this

Run tests with `uv run pytest`. Use `uv run eval-card-backend` for the
CLI. Add new functionality through the existing module structure
(`sources/` for IO, `canonicalise/` for transforms, `signals/` for pure
logic).

If the registry's schema changes, update `notes/01-` and `notes/02-`
first, then the implementation.
