# Eval Cards — DuckDB+Parquet schema design

> **Status:** canonical iterating design. Last revised 2026-05-02 to adopt
> evalcard-registry as the source of truth for entity identity.
>
> **Sources:**
> - Frontend prototype: `../Eval Cards New Design/`
> - Entity registry (identity layer):
>   `../evalcard-registry/`, deployed as HF dataset `evaleval/entity-registry-data`
> - EEE / AutoBenchmarkCards flattened schema:
>   `../eval_cards_backend_pipeline/eval_schemas_fields.csv`
> - Interpretive signals spec:
>   `../eval_cards_backend_pipeline/interpretive_signals_spec_LATEST.md`

---

## Design principles

1. **The registry owns entity identity.** Models, benchmarks, metrics, orgs,
   and harnesses are resolved to canonical IDs by the
   `eval-entity-resolver` package (from `evalcard-registry/packages/`). This
   producer never reinvents identity normalisation. Aliases, fuzzy matching,
   parent-child relationships, and quantization-variant collapsing are all
   the registry's job.
2. **One row = one score.** EEE's `evaluation_results[]` is an array. Each
   atomic row in the warehouse is one `(eee_eval × evaluation_result)`.
3. **Wide fact table, joined to small canonical dims.** DuckDB on Parquet is
   columnar; unused columns are free at read. The fact table is wide; the
   small `canonical_*` dim tables (replicated from the registry) carry
   display strings and reach via JOIN.
4. **Both raw and canonical, always.** Every entity that goes through the
   registry gets two columns on the fact row: `<entity>_raw` (always
   populated, never null) and `<entity>_id` (canonical, NULL when resolver
   returned no_match). Backfilling improved resolution = re-run
   canonicalisation; raw values preserve recoverability.
5. **Snapshots are columns, not directories.** `snapshot_id` is an ISO
   timestamp column on every table. Append-only across snapshots.
6. **Missing = absent.** No NULL placeholder rows for unreported `(model,
   benchmark, metric)` combinations. Frontend renders "not reported" by
   subtracting present rows from the expected catalog.
7. **Annotations live as columns on the fact row.** Three of the four
   interpretive signals (reproducibility, provenance, comparability) are
   per-row; their outputs are columns on `fact_results`. Reporting
   completeness is per-benchmark and gets a small side table.
8. **Two-pass canonicalisation.** Pass 1: explode + resolve identity +
   per-row signals. Pass 2: cross-row signals (multi-source, divergence)
   over `(model_id, benchmark_id, metric_id)` groups.
9. **No signal/data versioning yet.** Computed fresh from current data on
   every snapshot. Recompute = overwrite or new snapshot.

---

## Glossary

| term | definition |
|---|---|
| **Registry** | The `evalcard-registry` service, a separate codebase that resolves raw EEE strings to stable canonical IDs and maintains entity dim tables. Source of truth for identity. |
| **Canonical ID** | A stable, human-readable slug minted by the registry — e.g. `meta-llama/Llama-3.1-405B-Instruct`, `mmlu`, `pass-at-1`. Used as a foreign key from `fact_results` to canonical dim tables. |
| **Resolver** | The `eval-entity-resolver` Python package, importable from `evalcard-registry/packages/`. Wraps the registry's three-strategy resolution chain (exact → normalized → fuzzy). |
| **Triple** | The natural unit for per-row signals: `(model_id, benchmark_id, metric_id)`. Used for grouping in cross-row signal computation. |
| **Snapshot** | A canonicalisation run. Identified by ISO timestamp `snapshot_id`. Carried as a column on every table. |
| **AutoBenchmarkCards record** | The structured benchmark metadata document maintained at `evaleval/auto-benchmarkcards`. Optional — not every benchmark has one. |
| **EEE record** | The raw evaluation document at `evaleval/EEE_datastore`. Carries an array of `evaluation_results[]`, each one resolving to one fact row. |

---

## Source layers (upstream)

Three input layers feed canonicalisation:

| layer | source | purpose |
|---|---|---|
| **EEE evaluations** | `evaleval/EEE_datastore` HF dataset | per-result raw fields: `generation_config`, `source_metadata`, `score_details`, `eval_library`, etc. |
| **Entity registry** | `evaleval/entity-registry-data` HF dataset | canonical identity: `aliases.parquet` (for the resolver), `canonical_orgs.parquet`, `canonical_models.parquet`, `canonical_benchmarks.parquet`, `canonical_metrics.parquet`, `eval_harnesses.parquet` |
| **AutoBenchmarkCards** | `evaleval/auto-benchmarkcards` HF dataset | benchmark prose metadata (description, methodology, possible_risks, etc.). Optional per benchmark. |

The producer fetches all three via `huggingface_hub.snapshot_download`, then
loads them into a DuckDB session. Identity resolution is done **in-process**
by importing the resolver package and calling
`Resolver.resolve(raw_string, entity_type, source_config)` for each entity
reference per row — `source_config` is the EEE config name extracted from
the on-disk file path, used by the registry's source-config-scoped aliases
to disambiguate same-string-different-canonical-id cases. See `02-producer-shape.md`.

---

## Storage layout

```
warehouse/<snapshot_id>/
├── fact_results.parquet              # primary fact table — one row per score
├── benchmark_completeness.parquet    # per-benchmark completeness signal
├── benchmarks.parquet                # benchmark dim: canonical_benchmarks ⨝ auto-benchmarkcards
├── models.parquet                    # model dim: canonical_models ⨝ canonical_orgs
├── canonical_metrics.parquet         # registry-mirrored metric dim (joined at query time)
└── snapshot_meta.json                # upstream HF dataset revisions; reproducibility hook
```

The dim tables (`benchmarks.parquet`, `models.parquet`) are **JOIN
materialisations** of registry tables + benchmark cards / org info. They
exist so the frontend reads a single Parquet for each rendering surface
instead of joining at query time. We re-emit them on every snapshot.

`<snapshot_id>` is an ISO timestamp generated at the start of the
canonicalisation run. Output dir is `warehouse/<snapshot_id>/` so multiple
runs don't clobber. Every row also carries `snapshot_id` as a column for
queries spanning snapshots.

Instance-level data (`eee_instance_level_eval`) is **not ingested** in v1
— the fact row carries pointer columns for lazy-load.

---

## `fact_results.parquet` — DDL

The primary fact table. One row per atomic score. Identity columns come in
raw/canonical pairs.

```sql
CREATE TABLE fact_results (
  -- ===== identity =====
  snapshot_id              TIMESTAMP,    -- ISO timestamp of canonicalisation run
  evaluation_id            VARCHAR,      -- eee_eval.evaluation_id (run-level)
  result_idx               INTEGER,      -- index within evaluation_results[]
  evaluation_result_id     VARCHAR,      -- canonical when present, else 'evaluation_id#result_idx'
  fact_id                  VARCHAR,      -- sha256(evaluation_id || ':' || result_idx)[:16]; matches registry's eval_results.id formula
  variant_key              VARCHAR,      -- sha256(canonical_setup_json)[:16]; identical setups → identical key

  -- ===== entities: raw (best-effort populated) + canonical (NULL when unresolved) =====
  -- Raw columns are nullable: EEE doesn't always carry every entity name. Producer
  -- populates whatever's available. Empty/missing raws naturally resolve to no_match.
  model_raw                VARCHAR,
  model_id                 VARCHAR,         -- → canonical_models.id (registry)
  benchmark_raw            VARCHAR,
  benchmark_id             VARCHAR,         -- → canonical_benchmarks.id (registry)
  metric_raw               VARCHAR,
  metric_id                VARCHAR,         -- → canonical_metrics.id (registry)
  org_raw                  VARCHAR,         -- source_organization_name from EEE
  org_id                   VARCHAR,         -- → canonical_orgs.id (registry)
  harness_raw              VARCHAR,         -- eval_library.name + version (concat)
  harness_id               VARCHAR,         -- → eval_harnesses.id (registry)

  -- registry-derived parents / cross-references (also nullable)
  parent_benchmark_id      VARCHAR,         -- e.g. helm-lite as parent of helm-lite-gsm8k
  parent_model_id          VARCHAR,         -- e.g. llama-3.1 as parent of llama-3.1-405b-instruct
  benchmark_card_id        VARCHAR,         -- → benchmarks.benchmark_id (when card exists)

  -- resolution audit (optional but cheap to carry)
  model_resolution_strategy        VARCHAR,    -- 'exact'|'normalized'|'fuzzy'|'no_match'
  benchmark_resolution_strategy    VARCHAR,
  metric_resolution_strategy       VARCHAR,
  org_resolution_strategy          VARCHAR,
  harness_resolution_strategy      VARCHAR,

  -- ===== metric meta (denormalised from canonical_metrics) =====
  -- Denormalised so each fact row carries enough context for solo rendering
  -- (frontend leaderboards format/sort using these without joining the metrics dim).
  metric_kind              VARCHAR,         -- normalised family: 'accuracy','f1','elo','pass_rate',...
  metric_unit              VARCHAR,         -- 'proportion'|'percent'|'points'|'ms'|'tokens'|...
  lower_is_better          BOOLEAN,
  min_score                DOUBLE,
  max_score                DOUBLE,

  -- ===== score + uncertainty (from score_details) =====
  score                    DOUBLE,
  score_se                 DOUBLE,        -- standard error
  score_ci_lower           DOUBLE,
  score_ci_upper           DOUBLE,
  score_ci_level           DOUBLE,        -- e.g. 0.95
  n_samples                INTEGER,

  -- ===== source / provenance (from source_metadata) =====
  evaluator_relationship   VARCHAR,        -- 'first_party'|'third_party'|'collaborative'|'other'
  source_type              VARCHAR,        -- 'documentation'|'evaluation_run'
  source_organization_url  VARCHAR,
  eval_library_name        VARCHAR,
  eval_library_version     VARCHAR,

  -- ===== generation config — raw values (from generation_args) =====
  temperature              DOUBLE,
  top_p                    DOUBLE,
  top_k                    DOUBLE,
  max_tokens               INTEGER,
  prompt_template          VARCHAR,
  reasoning                BOOLEAN,
  agentic_eval_config      JSON,
  eval_plan                JSON,
  eval_limits              JSON,
  sandbox                  JSON,

  -- ===== presence flags (redundant with NULL but explicit) =====
  has_temperature          BOOLEAN,
  has_top_p                BOOLEAN,
  has_top_k                BOOLEAN,
  has_max_tokens           BOOLEAN,
  has_prompt_template      BOOLEAN,
  has_eval_plan            BOOLEAN,
  has_eval_limits          BOOLEAN,
  has_agentic_eval_config  BOOLEAN,

  -- ===== signal: reproducibility (per-row, computed pass 1) =====
  is_agentic                  BOOLEAN,    -- from benchmark card tags or agentic_eval_config presence
  has_reproducibility_gap     BOOLEAN,    -- !(all required has_* flags true)
  repro_missing_fields        VARCHAR[],
  repro_required_count        INTEGER,
  repro_populated_count       INTEGER,

  -- ===== signal: provenance (per-row + group-derived in pass 2) =====
  provenance_source_type      VARCHAR,    -- normally 'first_party'|'third_party'|'collaborative'|'unspecified'.
                                          -- Per legacy: 'other'/null collapse to 'unspecified'; any other unexpected
                                          -- value passes through verbatim (forward-compatible w/ future enum additions).
  distinct_reporting_orgs     INTEGER,    -- group pass over (model_id, benchmark_id, metric_id)
  is_multi_source             BOOLEAN,
  first_party_only            BOOLEAN,

  -- ===== signal: comparability (group pass over (model_id, benchmark_id, metric_id)) =====
  comparability_group_id            VARCHAR,    -- md5(model_id || '|' || benchmark_id || '|' || metric_id)
  has_variant_divergence            BOOLEAN,
  variant_divergence_magnitude      DOUBLE,
  variant_divergence_threshold      DOUBLE,
  variant_threshold_basis           VARCHAR,    -- 'proportion_or_continuous_normalized' | 'percent' | 'range_5pct' | 'fallback_default'
  variant_differing_fields          STRUCT(field VARCHAR, values JSON)[],   -- per legacy: each entry carries the distinct original values
  has_cross_party_divergence        BOOLEAN,
  cross_party_divergence_magnitude  DOUBLE,
  cross_party_divergence_threshold  DOUBLE,
  cross_party_threshold_basis       VARCHAR,
  cross_party_differing_fields      STRUCT(field VARCHAR, values JSON)[],
  cross_party_org_count             INTEGER,    -- distinct named orgs (after whitespace+case normalisation)
  scores_by_organization            MAP(VARCHAR, DOUBLE),  -- per-org median (display-name keys)
  score_scale_anomaly               BOOLEAN,    -- true when metric_unit='proportion' but score ∉ [0,1]

  -- ===== reserved EvalCards fields (NULL until something populates them) =====
  lifecycle_status         VARCHAR,        -- 'draft'/'stable'/'deprecated' or similar; benchmark/eval lifecycle
  preregistration_url      VARCHAR,        -- URL or identifier where the eval was preregistered

  -- ===== pointer to instance-level data (lazy-load from source) =====
  instance_file_path       VARCHAR,
  instance_file_format     VARCHAR,
  instance_checksum        VARCHAR,
  instance_hash_algorithm  VARCHAR,
  instance_rows            INTEGER,

  -- ===== freeform extras =====
  source_additional_details      JSON,
  generation_additional_details  JSON,
  metric_additional_details      JSON,

  PRIMARY KEY (snapshot_id, fact_id)  -- documentation only; Parquet doesn't enforce
);
```

**Display rendering rule (frontend):** for each entity, prefer the
canonical's `display_name` from the dim JOIN; fall back to `<entity>_raw`
when `<entity>_id IS NULL`. SQL pattern:

```sql
COALESCE(cm.display_name, fr.model_raw) AS model_display
```

**Triage query for unresolved coverage:**

```sql
SELECT entity_type, raw, COUNT(*) FROM (
  SELECT 'model'     AS entity_type, model_raw     AS raw FROM fact_results WHERE model_id     IS NULL
  UNION ALL
  SELECT 'benchmark',                benchmark_raw       FROM fact_results WHERE benchmark_id IS NULL
  UNION ALL
  SELECT 'metric',                   metric_raw          FROM fact_results WHERE metric_id    IS NULL
  UNION ALL
  SELECT 'org',                      org_raw             FROM fact_results WHERE org_id       IS NULL
)
GROUP BY 1, 2
ORDER BY 3 DESC;
```

Row counts here drive the registry's alias-coverage roadmap.

**Note on `evaluator_relationship` collapse for the coverage matrix:**

| canonical | UI-collapsed |
|---|---|
| `first_party` | `first_party` |
| `third_party` | `third_party` |
| `collaborative` | `collaborative` (count as both for divergence; render as "mixed/both" tone) |
| `other` / null | `unspecified` |

Upstream data restrictions currently produce only `first_party` and
`third_party`. Schema and renderers must accommodate all four.

**Known deviation from spec §5.4:** spec text says "for audit
computations, collaborative reports count as both first-party and
third-party sources." The legacy implementation (and this producer)
keeps `collaborative` as a distinct fourth bucket and does **not**
double-count for `is_multi_source` / `first_party_only`. Inheriting the
legacy behaviour intentionally — the audit-layer interpretation is
deferred until the data actually exercises it. Frontend coverage-matrix
tone for `collaborative` ≠ schema double-counting.

### Fact-row computation rules

These rules are carried in the producer (`02-`) but are spec-binding for
fact_results semantics:

**`fact_id`** = first 16 hex chars of `sha256(evaluation_id || ':' || result_idx)`.
Matches the registry's `eval_results.id` formula so cross-system
references work even though we don't read `eval_results` directly.
Implemented as a Python UDF (DuckDB's built-in `md5` is the wrong hash).

**`variant_key`** = first 16 hex chars of `sha256(setup_canonical_json)`,
where `setup_canonical_json` is the JSON-canonicalised, per-field
*normalised* setup dict. Two rows with semantically identical setups get
the same `variant_key`; cosmetic differences (whitespace, line endings,
float-repr noise) collapse.

**Setup-field normalisation** (applies to **both** `variant_key` hashing
and `_differing_setup_fields` divergence detection — same function in
both call sites, no drift):

| field | rule |
|---|---|
| `temperature`, `top_p`, `top_k` | `float(f"{float(v):.8g}")` — absorbs float-repr noise. NULL stays NULL. |
| `max_tokens` | cast to `int`. NULL stays NULL. |
| `prompt_template` | strip outer whitespace + normalise `\r\n`/`\r` → `\n`. **Preserve case, internal whitespace, all other content.** |
| `reasoning` | coerce `'true'`/`'1'`/`'yes'` → `True`; `'false'`/`'0'`/`'no'`/`''` → `False`; NULL stays NULL (≠ `False`). |
| `agentic_eval_config` | recursive `sort_keys=True, separators=(',',':'), default=str`. |

The full canonical setup dict is built from these seven
`GENERATION_ARGS_COMPARISON_FIELDS` then JSON-dumped with the same
`sort_keys=True, separators=(',',':'), ensure_ascii=False, default=str`
rules.

**Drop rows where `score_details.score` is missing or NULL.** A fact
without a score isn't useful (no leaderboard ranking, no signal
computation). Producer logs a count of dropped rows per snapshot in the
end-of-run summary. Identity-resolution failures don't drop the row
(NULL canonical, raw preserved); a missing score does.

**`score_scale_anomaly`** = `metric_unit = 'proportion' AND (score < 0 OR score > 1)`.
Per legacy. Surfaces probable metric-unit mislabelling without rejecting
the row; comparability still computes on the as-given scale.

---

## `benchmark_completeness.parquet` — DDL

One row per `(snapshot_id, benchmark_id)`. Materialises the per-benchmark
reporting-completeness signal (spec §4) so the frontend doesn't re-run the
field-by-field scan. Keys on the **canonical** `benchmark_id` from the
registry; rows where benchmark resolution failed have no corresponding
completeness entry.

```sql
CREATE TABLE benchmark_completeness (
  snapshot_id              TIMESTAMP,
  benchmark_id             VARCHAR,    -- canonical
  completeness_score       DOUBLE,     -- 0..1
  total_fields_evaluated   INTEGER,
  populated_count          DOUBLE,     -- partial fields contribute fractionally
  missing_required_fields  VARCHAR[],
  partial_fields           STRUCT(
                              field_path        VARCHAR,
                              score             DOUBLE,
                              populated_subitems INTEGER,
                              total_subitems    INTEGER
                            )[],
  field_scores             STRUCT(
                              field_path     VARCHAR,
                              coverage_type  VARCHAR,    -- 'full'|'partial'|'reserved'
                              score          DOUBLE
                            )[],
  PRIMARY KEY (snapshot_id, benchmark_id)
);
```

### Operationalised completeness field set (28 fields)

Ported from legacy `eval_cards_backend_pipeline/registry/completeness_fields.json`
(2026-04-26 snapshot). Stored as `registry/completeness_fields.json` in
`eval_card_backend`; loaded once at producer startup.

**22 full-coverage** AutoBenchmarkCards fields (each is 1 if populated, 0
otherwise):

```
autobenchmarkcard.benchmark_details.{name, overview, data_type, domains,
                                       languages, similar_benchmarks, resources}
autobenchmarkcard.purpose_and_intended_users.{goal, audience, tasks,
                                                limitations, out_of_scope_uses}
autobenchmarkcard.methodology.{methods, metrics, calculation, interpretation,
                                 baseline_results, validation}
autobenchmarkcard.ethical_and_legal_considerations.{privacy_and_anonymity,
                                                       data_licensing, consent_procedures,
                                                       compliance_with_regulations}
```

**1 partial-coverage** (4 sub-items, fractional score):

```
autobenchmarkcard.data → {source, size, format, annotation}
```

**3 full-coverage** EEE source-metadata fields:

```
eee_eval.source_metadata.{source_type, source_organization_name,
                           evaluator_relationship}
```

**2 reserved** EvalCards fields (count toward denominator even when
unpopulated):

```
evalcards.lifecycle_status
evalcards.preregistration_url
```

**Total: 28 fields.** Per spec §4.2: `completeness_score = sum(field_scores) / 28`.
Recompute when a benchmark's card OR any of its EEE rows changes.

---

## `benchmarks.parquet` — DDL

The benchmark dim. JOIN materialisation of `canonical_benchmarks` (registry)
and the AutoBenchmarkCards record (when present). One row per canonical
benchmark seen in the snapshot's facts. Benchmarks that resolved but have
no AutoBenchmarkCard get a row with `card_present = false` and most prose
fields NULL.

```sql
CREATE TABLE benchmarks (
  snapshot_id            TIMESTAMP,
  benchmark_id           VARCHAR,        -- canonical, FK from fact_results.benchmark_id

  -- from canonical_benchmarks (registry)
  display_name           VARCHAR,
  description            VARCHAR,        -- short, registry-curated
  dataset_repo           VARCHAR,
  parent_benchmark_id    VARCHAR,        -- e.g. mmlu_pro → mmlu (suite/family relationship)
  registry_tags          VARCHAR[],
  registry_metadata      JSON,
  review_status          VARCHAR,

  -- from AutoBenchmarkCards (optional; benchmark_details.*)
  card_name              VARCHAR,
  overview               VARCHAR,
  data_type              VARCHAR,
  domains                VARCHAR[],
  languages              VARCHAR[],
  similar_benchmarks     VARCHAR[],
  resources              VARCHAR[],

  -- purpose_and_intended_users
  goal                   VARCHAR,
  audience               VARCHAR[],
  tasks                  VARCHAR[],          -- includes 'agentic'/'tool_use' tags driving is_agentic
  limitations            VARCHAR,
  out_of_scope_uses      VARCHAR[],

  -- data
  data_source            VARCHAR,
  data_size              VARCHAR,
  data_format            VARCHAR,
  data_annotation        VARCHAR,

  -- methodology
  methods                VARCHAR[],
  card_metrics           VARCHAR[],          -- benchmark-card claimed metrics
  calculation            VARCHAR,
  interpretation         VARCHAR,
  baseline_results       VARCHAR,
  validation             VARCHAR,

  -- ethical_and_legal
  privacy_and_anonymity        VARCHAR,
  data_licensing               VARCHAR,
  consent_procedures           VARCHAR,
  compliance_with_regulations  VARCHAR,

  -- IBM Risk Atlas annotations (from card.possible_risks)
  possible_risks         STRUCT(
                            category    VARCHAR,
                            description VARCHAR,
                            type        VARCHAR,
                            concern     VARCHAR,
                            url         VARCHAR,
                            taxonomy    VARCHAR
                          )[],

  -- factreasoner output: per-field flags ('<section>.<field>' → message).
  -- JSON, not MAP — read_json_auto infers a fixed STRUCT shape from cards
  -- which doesn't fit dynamic field-path keys. Producer writes via to_json().
  flagged_fields         JSON,

  -- card-presence + quality
  card_present           BOOLEAN,
  card_generated_by      VARCHAR,
  card_flagged_count     INTEGER,
  card_missing_count     INTEGER,

  PRIMARY KEY (snapshot_id, benchmark_id)
);
```

The frontend's benchmark-detail page reads this single row plus its
matching `benchmark_completeness` row.

---

## `models.parquet` — DDL

The model dim. JOIN materialisation of `canonical_models` (registry) and
`canonical_orgs` (registry, for developer/org display). One row per
canonical model seen in the snapshot's facts.

```sql
CREATE TABLE models (
  snapshot_id            TIMESTAMP,
  model_id               VARCHAR,        -- canonical, FK from fact_results.model_id

  -- from canonical_models (registry)
  display_name           VARCHAR,
  developer              VARCHAR,        -- denormalised from canonical_orgs.display_name
  org_id                 VARCHAR,        -- → canonical_orgs.id
  family                 VARCHAR,        -- registry family ('Llama 3.1')
  architecture           VARCHAR,
  params_billions        DOUBLE,
  parent_model_id        VARCHAR,
  registry_tags          VARCHAR[],
  registry_metadata      JSON,           -- may carry release_date / context / modality / license / access if registry curates them
  review_status          VARCHAR,

  -- canonical_orgs context (denormalised for one-table reads)
  org_display_name       VARCHAR,
  org_website            VARCHAR,
  org_hf_org             VARCHAR,
  org_parent_id          VARCHAR,

  -- frontend-needed extras (currently NULL unless registry.metadata carries them; see open question)
  released               DATE,
  context_tokens         INTEGER,
  context_label          VARCHAR,
  modality               VARCHAR[],
  access                 VARCHAR,        -- 'open'|'open-research'|'proprietary'
  license                VARCHAR,

  PRIMARY KEY (snapshot_id, model_id)
);
```

Most rich frontend fields (`released`, `license`, `modality`, `access`,
`context_tokens`) are absent from the registry today. The registry will
populate them in a future iteration **as top-level columns on
`canonical_models`** (not buried in `metadata` JSON). Until then, the
columns exist on `models.parquet` and default to NULL; the producer's
SELECT will switch from `NULL::DATE AS released` to `cm.released AS
released` once the registry's table has them. Schema shape is locked so
the registry knows what to fill.

### Controlled values for model fields

Pinned now so the producer, the registry, and the frontend agree on shape:

| field | type | values | notes |
|---|---|---|---|
| `access` | VARCHAR | `'open'` / `'open-research'` / `'proprietary'` | Lowercase enum. Frontend renders to display (`"Open weights"` / `"Open weights · research"` / `"Proprietary (API)"`). Models index filter compares lowercase. |
| `modality` | VARCHAR[] | subset of `['text', 'vision', 'audio', 'video', 'code']` | Lowercase, single-token elements. Renderer joins with `" · "` and titlecases. |
| `license` | VARCHAR | freeform | SPDX identifier when applicable (`"Apache-2.0"`, `"MIT"`); otherwise the source label as-is (`"Llama 3.1 Community"`, `"Mistral Research License (non-commercial)"`, `"Proprietary (API)"`). No controlled vocab. |
| `released` | DATE | ISO calendar date | Day precision. If upstream has only month/year, use the first day of the month and document. |
| `context_tokens` | INTEGER | raw token count | e.g. `128000`, `200000`, `2000000`. Used for sorting / filtering. |
| `context_label` | VARCHAR | display string | e.g. `"128K tokens"`, `"2M tokens"`. Frontend prefers this for display. |
| `params_billions` | DOUBLE | numeric, in billions | Comes from `canonical_models.params_billions` directly. e.g. `405`, `123`, `0.5`. |
| `params` | VARCHAR | freeform | When numeric isn't enough — `"671B (37B active)"` for MoE, `"Undisclosed"`, `"Undisclosed (MoE)"`. Optional; renderer falls back to `params_billions || 'B'` when null. |

When a field is undisclosed (developer didn't say), prefer NULL over a
sentinel like `"Undisclosed"`. The frontend's renderer already detects
NULL and renders the muted "Undisclosed" pill.

---

## How each frontend view is served

Single-table reads dominate facts; benchmark/model dims are joined when
display strings are needed.

**Home — coverage matrix (top-N models × benchmarks):**

```sql
SELECT fr.model_id, m.display_name AS model_name,
       fr.benchmark_id, b.display_name AS benchmark_name,
       BOOL_OR(fr.evaluator_relationship = 'first_party')   AS has_first,
       BOOL_OR(fr.evaluator_relationship IN ('third_party','collaborative')) AS has_third,
       COUNT(*) AS metric_count
FROM fact_results fr
LEFT JOIN models     m USING (snapshot_id, model_id)
LEFT JOIN benchmarks b USING (snapshot_id, benchmark_id)
WHERE fr.snapshot_id = (SELECT MAX(snapshot_id) FROM fact_results)
  AND fr.model_id IS NOT NULL AND fr.benchmark_id IS NOT NULL
GROUP BY 1, 2, 3, 4;
```

(The matrix only renders resolved cells. Unresolved rows feed the triage
query, not the matrix.)

**Models index:**

```sql
SELECT m.*,
       COUNT(DISTINCT fr.benchmark_id) AS benchmark_count,
       COUNT(*) AS metric_count
FROM models m
LEFT JOIN fact_results fr USING (snapshot_id, model_id)
WHERE m.snapshot_id = (SELECT MAX(snapshot_id) FROM models)
GROUP BY m.snapshot_id, m.model_id   -- (full m.* expansion in real query)
ORDER BY benchmark_count DESC;
```

**Benchmark detail leaderboard (single metric):**

```sql
SELECT fr.*, COALESCE(m.display_name, fr.model_raw) AS model_display
FROM fact_results fr
LEFT JOIN models m USING (snapshot_id, model_id)
WHERE fr.snapshot_id  = (SELECT MAX(snapshot_id) FROM fact_results)
  AND fr.benchmark_id = 'helm-lite-gsm8k'
  AND fr.metric_id    = 'exact-match'
ORDER BY fr.score DESC;       -- direction by canonical_metrics.lower_is_better
```

**Model detail (Variant A — full per-row signals):**

```sql
SELECT fr.*,
       COALESCE(b.display_name, fr.benchmark_raw) AS benchmark_display,
       cm.display_name                            AS metric_display,
       cm.lower_is_better,
       cm.min_score, cm.max_score
FROM fact_results fr
LEFT JOIN benchmarks       b  USING (snapshot_id, benchmark_id)
LEFT JOIN canonical_metrics cm ON cm.id = fr.metric_id    -- registry mirror
WHERE fr.snapshot_id = (SELECT MAX(snapshot_id) FROM fact_results)
  AND fr.model_id    = 'meta-llama/Llama-3.1-405B-Instruct'
ORDER BY fr.benchmark_id, fr.evaluator_relationship;
```

(For `canonical_metrics`, either materialise it as a fourth Parquet in our
warehouse or `read_parquet` directly from the registry's local cache. Same
for `canonical_orgs` if needed beyond what's in `models.parquet`.)

**Benchmark detail page — pull the card + completeness:**

```sql
SELECT b.*, bc.completeness_score, bc.missing_required_fields
FROM benchmarks b
LEFT JOIN benchmark_completeness bc USING (snapshot_id, benchmark_id)
WHERE b.snapshot_id = (SELECT MAX(snapshot_id) FROM benchmarks)
  AND b.benchmark_id = 'swebench-verified';
```

**Triage / unresolved entities:** see the UNION query in the
`fact_results` section above.

---

## Reproducibility base fields (decided)

The active reproducibility check uses the **current production rule**:
`temperature` and `max_tokens` only. The spec's `top_p` and
`prompt_template` are not in the active base set, but `fact_results` stores
their raw values + `has_*` flags so restoring the full 4-field rule is one
CTAS, not a re-canonicalisation pass.

```python
SPEC_BASE_REPRODUCIBILITY_FIELDS = ('temperature', 'top_p', 'max_tokens', 'prompt_template')
BASE_REPRODUCIBILITY_FIELDS      = ('temperature', 'max_tokens')   # ACTIVE
AGENTIC_REPRODUCIBILITY_FIELDS   = ('eval_plan', 'eval_limits')
```

Swap-back recipe (single CTAS over `fact_results`):

```sql
CREATE TABLE fact_results_new AS
SELECT
  * REPLACE (
    NOT (has_temperature AND has_top_p AND has_max_tokens AND has_prompt_template
         AND (NOT is_agentic OR (has_eval_plan AND has_eval_limits)))    AS has_reproducibility_gap,
    /* repro_missing_fields, _required_count, _populated_count similarly recomputed */
  )
FROM fact_results;
```

---

## `is_agentic` rule (carried forward)

Replicate the existing three-rule union (registry doesn't classify
agentic-ness; this stays in our pipeline):

```python
import re

# Match legacy normalisation exactly (legacy `helpers/benchmark_identity.py:6`):
# collapse ALL non-alphanumerics to '_'. Catches benchmark_ids with spaces, dots,
# hyphens, slashes, etc. — whatever shape the input takes.
_NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')

def _normalise_benchmark_slug(s: str) -> str:
    if not s:
        return ''
    return _NON_ALNUM_RE.sub('_', s.lower()).strip('_')

def is_agentic(benchmark_id, benchmark_card, generation_args):
    # 1. tasks-literal from card
    tasks = (benchmark_card or {}).get('purpose_and_intended_users', {}).get('tasks') or []
    if any(t.strip().lower() in {'agentic', 'tool_use', 'multi_step_agent'}
           for t in tasks if isinstance(t, str)):
        return True
    # 2. agentic_eval_config presence
    if (generation_args or {}).get('agentic_eval_config') is not None:
        return True
    # 3. hardcoded benchmark-name regex (legacy workaround). Normalise first
    # (collapse non-alnum) then search; matches 'swe-bench-verified' /
    # 'swe.bench verified' / 'swebench_verified' / etc.
    if AGENTIC_NAME_REGEX.search(_normalise_benchmark_slug(benchmark_id)):
        return True
    return False

# Pattern matches the underscore-collapsed normalised form.
AGENTIC_NAME_REGEX = re.compile(
    r'(appworld|swe_bench|tau_bench|browsecomp|agent|livecodebench|terminal_bench)'
)
```

**Input scope:** the legacy applies this regex to the *raw* benchmark
name. Our pipeline applies it to the *canonical* `benchmark_id` (because
identity resolution happens before signal computation). Both forms are
fed through the same `_normalise_benchmark_slug` so the regex match is
robust regardless of slug convention. If a benchmark resolves no_match
(`benchmark_id` is NULL), this rule contributes nothing — the agentic
classification falls back to rules 1 and 2 only.

Refinement is upstream work — better tagging in AutoBenchmarkCards or
adding agentic tags to `canonical_benchmarks.tags` in the registry.

---

## Comparability — thresholds, aggregations, divergence rules

### Threshold rules (4-way)

`compute_threshold(metric_config)` returns `(threshold, basis)` where
`basis` is one of four labels recorded on each fact row's
`*_threshold_basis` column:

| condition | threshold | basis label |
|---|---|---|
| `metric_unit == 'proportion'` OR `metric_kind == 'continuous_normalized'` | `0.05` | `proportion_or_continuous_normalized` |
| `metric_unit == 'percent'` | `5.0` (percentage points) | `percent` |
| `min_score` and `max_score` both real numbers, `max > min` | `0.05 * (max_score - min_score)` | `range_5pct` |
| else | `0.05` (absolute fallback) | `fallback_default` |

Inputs come from `canonical_metrics` (registry-provided), not per-record
`metric_config`. The registry decides metric ranges; per-record values
may be inconsistent.

### `aggregated_setup` (cross-party representative pick)

Per spec §6.2.2: when an org has multiple rows in a `(model, benchmark,
metric)` group, pick **one** representative setup deterministically.

Rule: sort the org's rows by `(score, evaluation_id)` ascending, with
NULL scores treated as `+∞` (so they sort to the end). Take the row at
index `(n - 1) // 2`. For odd `n` this is the median row; for even `n`
this is the lower of the two middle rows.

Why deterministic: snapshots must be reproducible across pipeline runs
regardless of input order; `evaluation_id` breaks ties.

### Variant divergence — return None when

Per spec §6.1 + legacy implementation:

- `< 2` rows in the group, OR
- all rows have identical setups across the seven comparison fields
  (after normalisation), OR
- `< 2` rows have non-null scores after exclusions.

In those cases `has_variant_divergence` is NULL on every row in the
group (signal not applicable, distinct from "applicable and no
divergence").

### Cross-party divergence — return None when

Per spec §6.2 + legacy:

- Fewer than 2 distinct **named** orgs after applying
  `normalize_org_name` (whitespace collapse + lowercase). Display casing
  is preserved separately for `scores_by_organization` keys.
- Rows with NULL `score` are excluded from the org's median; if an org
  has no scored rows, it drops out.

### `_differing_setup_fields` output shape

Returns a list of `{field, values}` dicts (mirrored as
`STRUCT(field VARCHAR, values JSON)[]` in fact_results columns). For
each comparison field where the canonical-form set has size > 1, record
the original distinct values (in first-seen order, deduped by canonical
form). The original values are preserved — not just field names — so
downstream UI can render "max_tokens varies: 2048, 4096, 8192" without
recomputing.

---

## What we explicitly dropped from prior drafts

These appeared earlier and are NOT in the canonical design:

- **`policy_notes` table.** Conditionally generated downstream from card
  fields. Lives in render-time logic.
- **`signals_definitions` table.** The signal display copy is frontend copy.
- **Separate `risk_annotations` table.** Folded into `benchmarks.possible_risks`.
- **`sources` lookup table.** No separate dim; provenance fields denormalised
  on fact rows.
- **`metric_paths` registry.** The registry's `canonical_metrics` + the
  `(benchmark_id, metric_id)` pair on fact rows already address this.
- **Identity normalisation logic in our pipeline.** All goes to the
  registry; we just call the resolver.
- **`samples` polymorphic tables / `eee_instance_level` table.** Not
  ingested in v1; pointer columns on facts enable lazy-load.
- **Per-signal annotation side tables.** Annotations are columns on
  `fact_results`.

---

## Open questions / next iterations

- **Frontend-needed model fields populated** — schema is now locked
  (controlled values above). Registry needs to start populating
  `released`, `license`, `modality`, `access`, `context_tokens`. Until
  then NULL. Roadmap item, not a blocker.
- **`canonical_metrics` materialisation** — replicate into our warehouse
  for self-contained reads, or `read_parquet` from the registry's local
  cache at query time? Registry tables are small; copying is cheap.
- **Producer mechanics** — see `02-producer-shape.md`.

### Resolved this iteration

- **`collaborative` rendering** → render as "mixed/both" tone on the
  coverage matrix; schema accommodates all 4 evaluator_relationship states.
- **Signal versioning / backfill semantics** → deferred; recompute fresh
  from data each snapshot.
- **Instance-level data** → not ingested. Pointer columns on facts.
- **Comparability thresholds** → hardcoded constants matching spec.
- **`is_agentic`** → carried forward, three-rule union with hardcoded regex.
- **Reproducibility base fields** → match production (`temperature` +
  `max_tokens`); raw + has_* for all 4 fields stored.
- **Identity canonicalisation** → registry's job; no producer-side
  normalisation. Raw + canonical pair per entity on fact rows.
- **Unresolved entity handling** → keep the row, NULL canonical_id, raw
  always populated. Triage via SQL on the fact table; no side files.
- **Model-field controlled values** → fixed (see "Controlled values for
  model fields" above). Registry populates incrementally; columns default
  NULL until then.
