# Eval Cards — producer shape

> **Status:** draft, revised 2026-05-02 to consume the registry as the
> identity layer. Translates `01-schema-from-frontend.md` into a concrete
> ingestion pipeline. Reads three upstream HF datasets; emits five
> Parquet outputs + a `snapshot_meta.json` sidecar.

---

## Implementation choice: DuckDB end-to-end + resolver in-process

The pipeline is **DuckDB SQL orchestrated from Python**, with the
`eval-entity-resolver` package imported in-process for entity resolution.

- DuckDB reads JSON natively (`read_json_auto`) and writes Parquet via
  `COPY ... TO ... (FORMAT PARQUET)`.
- `eval-entity-resolver` lives at `../evalcard-registry/packages/eval-entity-resolver/`.
  Wire it in as a **uv workspace / local path dependency** for now —
  resolver edits flow through immediately while the registry iterates.
  When the registry stabilises and gets properly published, switch to a
  git-URL pinned dep (see `CLAUDE.md` for the migration TODO).
- Python registers the resolver as a DuckDB scalar UDF so canonical IDs
  can be filled in pure SQL during canonicalisation.

No Polars, no Pandas in the hot path (Pandas stays a dev dependency for
ad-hoc inspection).

---

## High-level flow

```
                ┌──────────────────────┐
                │  sources/eee.py      │  ← already wired
                │  sources/benchmark_  │
                │  cards.py            │
                │  sources/registry.py │  ← NEW (mirrors the others)
                └──────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │ Stage A · raw load     │  duckdb.read_json_auto()
              │ Stage B · explode      │  unnest evaluation_results[]
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────────┐
              │ Stage C · resolve      │  Resolver UDFs fill canonical IDs
              │ Stage D · flatten +    │  Project nested → flat columns,
              │           join cards   │   LEFT JOIN canonical_* dims
              └──────────┬─────────────┘
                         ▼  fact_results.staging
              ┌────────────────────────┐
              │ Stage E · per-row      │  has_*, repro, provenance source
              │ Stage F · group sigs   │  multi-source, divergence (window)
              └──────────┬─────────────┘
                         ▼  fact_results
              ┌────────────────────────┐
              │ Stage G · dims         │  benchmarks (cb ⨝ cards),
              │                        │   models (cm ⨝ orgs)
              │ Stage H · completeness │  per-benchmark scoring
              └──────────┬─────────────┘
                         ▼
              ┌────────────────────────┐
              │ Stage I · emit Parquet │  COPY each table, ZSTD
              └────────────────────────┘
```

One DuckDB connection holds all intermediate tables in memory; resolver
state is held as a Python `AliasStore` shared by the UDFs.

---

## Inputs

Three HF dataset sources. Two already wired in
`src/eval_card_backend/sources/`; the third (registry) mirrors them.

| source | upstream | local cache | provides |
|---|---|---|---|
| **EEE evaluations** | `evaleval/EEE_datastore` | `.cache/eee_datastore/data/...` | per-record raw fields |
| **AutoBenchmarkCards** | `evaleval/auto-benchmarkcards` | `.cache/auto_benchmarkcards/cards/...` | benchmark prose metadata (optional per benchmark) |
| **Entity registry (NEW)** | `evaleval/entity-registry-data` | `.cache/entity_registry/*.parquet` | canonical_orgs, canonical_models, canonical_benchmarks, canonical_metrics, eval_harnesses, aliases, eval_results, sync_runs |

A new `sources/registry.py` mirrors the existing source modules. It uses
`huggingface_hub.snapshot_download(repo_id='evaleval/entity-registry-data',
repo_type='dataset')` and exposes:

```python
def ensure_snapshot(local_dir: str, hf_token: str | None,
                    force_refresh: bool) -> Path: ...

def load_alias_store(root: Path) -> AliasStore: ...   # for Resolver init

def open_dim_paths(root: Path) -> dict[str, Path]:
    """Return {'canonical_models': Path(...), ...} for DuckDB read_parquet."""
```

The `aliases.parquet` file feeds the Resolver; the `canonical_*.parquet`
files are read directly by DuckDB during dim materialisation.

We do **not** read `eval_results.parquet` from the registry. We resolve in
process so registry alias-store improvements flow through immediately —
see `01-schema-from-frontend.md` "Design principles."

---

## Outputs

```
warehouse/<snapshot_dir_name>/         # snapshot_id with ':' → '-' for Win-fs safety
├── fact_results.parquet
├── benchmark_completeness.parquet
├── benchmarks.parquet
├── models.parquet
├── canonical_metrics.parquet          # registry mirror; query-time JOIN target
└── snapshot_meta.json                 # upstream HF revisions + table list
```

`snapshot_id` (the **column** value inside every parquet) is the ISO
timestamp of canonicalisation start
(`datetime.utcnow().isoformat(timespec='seconds') + 'Z'`). The
**directory name** is the same string with `:` replaced by `-` for
filesystem compatibility — see Stage I.

---

## Stage 0 — pipeline init (UDFs registered before any stage)

UDFs are registered against the DuckDB connection up front so they're
available everywhere:

```python
import logging
from collections import Counter, defaultdict
from duckdb.typing import VARCHAR, INTEGER, BOOLEAN, DOUBLE
from eval_entity_resolver import Resolver, AliasStore
from eval_entity_resolver.eee import clean_eval_name, extract_metric

# DuckDB Python-UDF type notes:
# - For composite types (LIST, STRUCT, MAP) and JSON, pass type string forms:
#     "VARCHAR[]"
#     "STRUCT(field VARCHAR, \"values\" JSON)"
#     "JSON"     (DuckDB serialises JSON as VARCHAR when handed to Python)
# - JSON params are delivered to the UDF body as Python str (the serialised
#   JSON), NOT as a parsed dict. UDFs that need a dict parse internally:
#     parsed = json.loads(arg) if isinstance(arg, str) else arg
#   This keeps the UDF callable from both DuckDB and Python tests.

log = logging.getLogger(__name__)
con = duckdb.connect()

# Resolver setup
alias_store = registry.load_alias_store(registry_root)
resolver    = Resolver(alias_store)

# Counters for end-of-run summary. The registry will not have every
# open-source / community fine-tuned entity, so no_match is EXPECTED at
# scale — we never log per-row on miss. Exceptions (genuine resolver
# bugs) are rate-limited so a systematic edge-case doesn't flood the
# console.
miss_counter      = Counter()                  # (entity_type) -> count
miss_examples     = defaultdict(Counter)       # (entity_type) -> Counter[raw_value]
exception_seen    = set()                      # (entity_type, exception_class) — log first only
exception_counter = Counter()                  # (entity_type, exception_class) -> count

def resolve_canonical_id_py(raw, entity_type, source_config):
    if not raw or not raw.strip():
        return None
    try:
        result = resolver.resolve(raw, entity_type, source_config)
    except Exception as e:
        key = (entity_type, type(e).__name__)
        exception_counter[key] += 1
        if key not in exception_seen:
            exception_seen.add(key)
            log.warning("resolver raised %s on %s (first occurrence): raw=%r config=%r err=%s",
                        type(e).__name__, entity_type, raw, source_config, e)
        return None
    if result.canonical_id is None:
        miss_counter[entity_type] += 1
        miss_examples[entity_type][raw] += 1   # bounded by distinct raw strings
    return result.canonical_id

def resolve_strategy_py(raw, entity_type, source_config):
    if not raw or not raw.strip():
        return 'no_match'
    try:
        return resolver.resolve(raw, entity_type, source_config).strategy
    except Exception:
        return 'no_match'

def log_resolver_summary(top_n: int = 10) -> None:
    """Print at end of run. Call from pipeline.py once Stage I completes."""
    log.info("=== resolver summary ===")
    for entity_type, count in miss_counter.most_common():
        examples = miss_examples[entity_type].most_common(top_n)
        sample_str = ", ".join(f"{raw!r}×{n}" for raw, n in examples)
        log.info("  %s: %d no_match across %d distinct raws — top: %s",
                 entity_type, count, len(miss_examples[entity_type]), sample_str)
    if exception_counter:
        log.warning("--- resolver exceptions ---")
        for (entity_type, exc), count in exception_counter.most_common():
            log.warning("  %s/%s: %d occurrences", entity_type, exc, count)
    else:
        log.info("(no resolver exceptions)")

con.create_function("resolve_canonical_id", resolve_canonical_id_py,
                    [VARCHAR, VARCHAR, VARCHAR], VARCHAR)
con.create_function("resolve_strategy",     resolve_strategy_py,
                    [VARCHAR, VARCHAR, VARCHAR], VARCHAR)
con.create_function("clean_eval_name_udf",  clean_eval_name,   [VARCHAR], VARCHAR)
con.create_function("extract_metric_udf",   extract_metric,    [VARCHAR], VARCHAR)

# Helper UDFs (composite/JSON types as strings; see typing notes above)
con.create_function("is_agentic_udf",            is_agentic_py,
                    [VARCHAR, "JSON", "JSON"], BOOLEAN)
con.create_function("compute_repro_missing_udf", compute_repro_missing_py,
                    [BOOLEAN, BOOLEAN, BOOLEAN, BOOLEAN, BOOLEAN], "VARCHAR[]")
con.create_function("canonical_json_udf",        canonical_json,
                    ["JSON"], VARCHAR)
con.create_function("compute_completeness_udf",  compute_completeness_py,
                    ["JSON"],
                    "STRUCT(completeness_score DOUBLE, total_fields_evaluated INTEGER, "
                    "populated_count DOUBLE, missing_required_fields VARCHAR[], "
                    "partial_fields STRUCT(field_path VARCHAR, score DOUBLE, "
                    "                       populated_subitems INTEGER, total_subitems INTEGER)[], "
                    "field_scores STRUCT(field_path VARCHAR, coverage_type VARCHAR, "
                    "                     score DOUBLE)[])")

# Group-level signal UDFs (used by Stage F.2)
_GROUP_ROW_TYPE  = ("STRUCT(fact_id VARCHAR, evaluation_id VARCHAR, score DOUBLE, "
                    "        generation_args JSON, evaluator_relationship VARCHAR, "
                    "        source_organization_name VARCHAR)")
_METRIC_CFG_TYPE = ("STRUCT(metric_kind VARCHAR, metric_unit VARCHAR, "
                    "        min_score DOUBLE, max_score DOUBLE)")
_VARIANT_OUT_TYPE = ("STRUCT(has_variant_divergence BOOLEAN, divergence_magnitude DOUBLE, "
                    "        threshold_used DOUBLE, threshold_basis VARCHAR, "
                    "        differing_setup_fields STRUCT(field VARCHAR, \"values\" JSON)[])")
_CROSS_OUT_TYPE   = ("STRUCT(has_cross_party_divergence BOOLEAN, divergence_magnitude DOUBLE, "
                    "        threshold_used DOUBLE, threshold_basis VARCHAR, "
                    "        differing_setup_fields STRUCT(field VARCHAR, \"values\" JSON)[], "
                    "        organization_count INTEGER, "
                    "        scores_by_organization MAP(VARCHAR, DOUBLE))")
con.create_function("compute_variant_divergence_udf",     compute_variant_divergence_py,
                    [f"{_GROUP_ROW_TYPE}[]", _METRIC_CFG_TYPE], _VARIANT_OUT_TYPE)
con.create_function("compute_cross_party_divergence_udf", compute_cross_party_divergence_py,
                    [f"{_GROUP_ROW_TYPE}[]", _METRIC_CFG_TYPE], _CROSS_OUT_TYPE)

# Identity / setup helpers (variant_key + fact_id; defined in next subsection)
con.create_function("variant_key_udf", variant_key_py, ["JSON"],          VARCHAR)
con.create_function("fact_id_udf",     fact_id_py,     [VARCHAR, INTEGER], VARCHAR)
```

**Note on doc-vs-code ordering:** the registration block above references
several helpers (`variant_key_py`, `fact_id_py`, `compute_repro_missing_py`,
`compute_completeness_py`, `compute_variant_divergence_py`,
`compute_cross_party_divergence_py`, `canonical_json`, `is_agentic_py`)
that are defined in the next subsection ("Setup field normalisation…")
or imported from `signals/`. In the actual `canonicalise/resolver_setup.py`
module, definitions and imports happen *before* `create_function` calls.
The doc orders them this way for narrative clarity (resolver story
first, supporting helpers second).

**Logging philosophy.** The registry is expected to miss many open-source
and community fine-tuned entities — `no_match` is the *normal* state for
a substantial fraction of rows, not an error. So:

- Per-row miss → silent. Counter increments only.
- Per-row exception → log first occurrence per `(entity_type,
  exception_class)`, then count quietly. No flood even if 10⁵ rows hit
  the same edge case.
- End of run → `log_resolver_summary()` prints miss counts per entity
  type plus the top-N unresolved raw strings — that's the actionable
  signal for what to add to the registry next.
  `log_json_coerce_summary()` (defined alongside `_coerce_json`) prints
  counts of malformed-JSON occurrences per call site, so corrupt
  `generation_args` aren't silently lumped with empty setups.
  `pipeline.run()` calls both summaries after Stage I completes.
- Triage SQL on `fact_results` (UNION-by-entity-type from `01-`) gives
  the same data, queryable post-hoc.

Production may want a `--strict` flag that converts resolver exceptions
to fail-fast (instead of degrade-and-count). Revisit when we know the
failure modes.

### Setup field normalisation + variant_key + fact_id

These helpers are the binding implementations of the rules documented in
`01-` ("Fact-row computation rules"). Same `normalize_setup` is used in
Stage E (per-row `variant_key`) **and** in Stage F (cross-row
`_differing_setup_fields`) — single source of truth, no drift.

```python
import hashlib, json, re

# Stable canonical-JSON for *any* JSON-able value (used for
# agentic_eval_config, eval_plan, eval_limits, etc. in divergence detection).
def canonical_json(obj) -> str | None:
    if obj is None: return None
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, default=str)

# Per-field normalisation rules (see 01-: "Fact-row computation rules" table)
_LINE_ENDING_RE = re.compile(r'\r\n|\r')

def _norm_num(v):
    if v is None: return None
    try:    return float(f"{float(v):.8g}")
    except (ValueError, TypeError): return v

def _norm_int(v):
    if v is None: return None
    try:    return int(v)
    except (ValueError, TypeError): return v

def _norm_text(v):
    if not isinstance(v, str): return v
    return _LINE_ENDING_RE.sub('\n', v).strip()

def _norm_bool(v):
    if v is None or isinstance(v, bool): return v
    if isinstance(v, (int, float)):      return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('true',  '1', 'yes', 't'):       return True
        if s in ('false', '0', 'no',  'f', ''):   return False
    return v

GENERATION_ARGS_COMPARISON_FIELDS = (
    'temperature', 'top_p', 'top_k', 'max_tokens',
    'prompt_template', 'reasoning', 'agentic_eval_config',
)

_malformed_json_counter = Counter()   # (caller_name) -> count, for end-of-run summary

def _coerce_json(arg, caller: str = ''):
    """JSON UDF params arrive as serialised strings from DuckDB.
    Parse to dict on entry; pass-through if already dict.
    On malformed input, increment a counter (logged at end of run) and return None.

    **Every UDF declared with a "JSON" param type must call this on entry** —
    otherwise the body sees a str and silently misbehaves (e.g. `(...).get(...)`
    on a string raises AttributeError, then UDF wrapper returns None and the
    row's signal silently degrades).
    """
    if isinstance(arg, str):
        try:
            return json.loads(arg)
        except (ValueError, TypeError):
            _malformed_json_counter[caller] += 1
            return None
    return arg

def log_json_coerce_summary() -> None:
    if _malformed_json_counter:
        log.warning("--- malformed JSON coercion ---")
        for caller, count in _malformed_json_counter.most_common():
            log.warning("  %s: %d rows had malformed JSON (treated as missing)", caller, count)

def normalize_setup(generation_args) -> dict:
    """Normalised dict for the seven comparison fields. Used for variant_key
    AND for cross-row divergence detection — same function in both call sites.
    Robust to receiving either a dict or a JSON string."""
    ga = _coerce_json(generation_args)
    ga = ga if isinstance(ga, dict) else {}
    return {
        'temperature':         _norm_num(ga.get('temperature')),
        'top_p':               _norm_num(ga.get('top_p')),
        'top_k':               _norm_num(ga.get('top_k')),
        'max_tokens':          _norm_int(ga.get('max_tokens')),
        'prompt_template':     _norm_text(ga.get('prompt_template')),
        'reasoning':           _norm_bool(ga.get('reasoning')),
        'agentic_eval_config': ga.get('agentic_eval_config'),  # canonical_json handles dict normalisation
    }

def setup_canonical_json(generation_args) -> str:
    return canonical_json(normalize_setup(generation_args))

def variant_key_py(generation_args) -> str:
    """16-hex-char prefix of sha256(setup_canonical_json)."""
    s = setup_canonical_json(generation_args)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

def fact_id_py(evaluation_id: str | None, result_idx: int) -> str | None:
    """16-hex-char prefix of sha256('<evaluation_id>:<result_idx>').
    Matches the registry's eval_results.id formula for cross-system referencing.
    Returns None on empty evaluation_id."""
    if not evaluation_id:
        return None
    payload = f"{evaluation_id}:{result_idx}".encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]

# (variant_key_udf and fact_id_udf are registered in the consolidated
# create_function block above.)
```

**Numeric-coercion deviation from legacy.** Legacy `canonicalize_setup_value`
(`signals.py:296-302`) is purely `json.dumps(... default=str)`; it doesn't
coerce numeric types. So legacy treats `top_k = 50` and `top_k = 50.0` as
distinct canonical strings (`"50"` vs `"50.0"`), and `0.7` vs `0.7000001`
as distinct. The new pipeline runs `_norm_num` / `_norm_int` first, so
those collapse. This is an intentional improvement (cosmetic float-repr
noise shouldn't fragment variant_key) but unmarked-in-legacy. If a future
audit needs strict legacy parity, swap the per-field normalisation to a
no-op.

**Malformed-JSON deviation from legacy.** Legacy signal functions assumed
their `generation_args` / `benchmark_card` arguments were already-parsed
Python dicts. A malformed dict would never reach them — JSON parse errors
would raise upstream and abort the pipeline run. The new pipeline routes
JSON-typed UDF params as serialised strings (DuckDB convention) and
parses on entry via `_coerce_json`; corrupt JSON returns `None` (treated
as "no setup recorded"), increments `_malformed_json_counter[caller]`,
and `log_json_coerce_summary()` reports counts at end of run. This trades
fail-fast (legacy) for fail-soft + visible-in-summary (new). Behavioural
implications:
- A row with corrupt `generation_args_json` gets `variant_key` ==
  hash-of-empty-setup (same as a row with no setup), potentially
  grouping unrelated rows.
- The same row's reproducibility signal sees missing fields and flags
  `has_reproducibility_gap = TRUE` (vs legacy crashing).
- The `--strict` mode (deferred — see Open questions) will eventually
  flip this back to fail-fast for production runs.

---

## Stage Pre — preflight checks

Fail fast before doing any DuckDB work. Empty / unreachable upstreams
silently produce all-NULL snapshots otherwise — the worst kind of
regression.

```python
def preflight(settings, eee_root, registry_root, cards_root) -> None:
    errors = []

    # EEE: required, must have data
    if not Path(eee_root, 'data').exists() or not any((eee_root / 'data').rglob('*.json')):
        errors.append(
            f"EEE source empty/missing at {eee_root}/data (HF download failed?). "
            f"Set EEE_REFRESH_SNAPSHOT=1 to force re-download."
        )

    # Registry: required, alias store must be loadable
    aliases_path = Path(registry_root, 'aliases', 'part-0.parquet')
    if not aliases_path.exists():
        errors.append(
            f"Registry alias store missing at {aliases_path}. "
            f"Set BENCHMARK_METADATA_REFRESH=1 / pull evaleval/entity-registry-data."
        )
    elif pd.read_parquet(aliases_path).empty:
        errors.append("Registry alias store is empty (cold start?). "
                      "Seed the registry before running canonicalisation.")

    # Cards: best-effort. Missing → warn, don't fail.
    if cards_root is None or not Path(cards_root).exists():
        log.warning("AutoBenchmarkCards source missing at %s — proceeding without "
                    "card content. card_present will be false on every benchmark.",
                    cards_root)

    if errors:
        for err in errors:
            log.error(err)
        raise SystemExit("Preflight failed; see errors above.")
```

EEE and registry are required (no useful fact rows without them). Cards
are best-effort — a snapshot with thin card coverage is still useful
provided the producer flags it loudly.

---

## Stage A — raw load

```python
# EEE — extract source_config from the on-disk path
# Layout: <local_dir>/data/<config>/<dev>/<model>/<uuid>.json
eee_glob = f"{settings.eee_local_dir}/data/**/*.json"
con.execute(f"""
  CREATE TABLE eee_raw AS
  SELECT *,
    regexp_extract(filename, 'data/([^/]+)/', 1) AS source_config
  FROM read_json_auto(
    '{eee_glob}',
    filename = true,
    union_by_name = true,
    maximum_object_size = 268435456
  )
""")
```

`source_config` is **not** a JSON field on the EEE record — it's encoded
only in the file path. We expose `filename` via DuckDB's `read_json_auto`
parameter and extract the config segment with regex. From here on
`source_config` is just a column flowing through every stage; it's passed
to the resolver UDF whenever entity resolution is config-scoped.

```python
# AutoBenchmarkCards (variable on-disk shape; loader normalises).
# Write to a temp JSONL so DuckDB infers STRUCT types — lets us use dot
# notation when reading card subfields (consistent with EEE access pattern).
import tempfile, json
cards_records = benchmark_cards.load_cards(cards_root)   # {normalised_name: card}
with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    for k, v in cards_records.items():
        f.write(json.dumps({"card_key": k, "card": v}) + "\n")
    cards_jsonl = f.name

con.execute(f"""
  CREATE TABLE cards_raw_in AS
  SELECT * FROM read_json_auto('{cards_jsonl}', union_by_name = true)
""")

# Resolve each card's name to a canonical benchmark_id via the resolver.
# AutoBenchmarkCards keys come from filenames or flat-map keys, normalised
# by the cards loader's own _normalize_key(). The registry's canonical
# benchmark_ids are minted by the registry's own seed/draft process — they
# may not be syntactically identical. The resolver bridges the two.
con.execute("""
  CREATE TABLE cards_raw AS
  SELECT
    card_key,
    card,
    resolve_canonical_id(card_key, 'benchmark', NULL) AS benchmark_id,
    resolve_strategy(card_key, 'benchmark', NULL)     AS card_resolution_strategy
  FROM cards_raw_in
  WHERE card_key IS NOT NULL
""")
```

The `cards_raw` table now has `(card_key, card, benchmark_id,
card_resolution_strategy)`. Cards whose key the resolver can't match get
`benchmark_id = NULL` and become orphan rows — visible via
`SELECT card_key FROM cards_raw WHERE benchmark_id IS NULL`. Add an alias
in the registry to fix.

```python
# Registry — read parquets directly via DuckDB
reg = registry.open_dim_paths(registry_root)
con.execute(f"CREATE TABLE canonical_models     AS SELECT * FROM read_parquet('{reg['canonical_models']}')")
con.execute(f"CREATE TABLE canonical_benchmarks AS SELECT * FROM read_parquet('{reg['canonical_benchmarks']}')")
con.execute(f"CREATE TABLE canonical_metrics    AS SELECT * FROM read_parquet('{reg['canonical_metrics']}')")
con.execute(f"CREATE TABLE canonical_orgs       AS SELECT * FROM read_parquet('{reg['canonical_orgs']}')")
con.execute(f"CREATE TABLE eval_harnesses       AS SELECT * FROM read_parquet('{reg['eval_harnesses']}')")
```

`union_by_name=true` is critical for EEE — schema drift across files
otherwise crashes the inference. AutoBenchmarkCards is loaded via the
existing Python helper because the on-disk shape varies (flat map vs
one-file-per-card); it's small enough that pandas-staged load is fine.

We do **not** read `eval_results.parquet` from the registry — that table
is the registry's internal sync state, not a public artifact. We resolve
entities ourselves via the resolver UDF (Stage C), against the same
alias store the registry uses. Improvements to that alias store flow
through immediately on the next canonicalisation run.

---

## Stage B — explode `evaluation_results[]`

DuckDB's UNNEST doesn't expose the array index directly. Pinned pattern:
generate the index range via `range()`, index into the array, and select
the per-element STRUCT columns by dot notation. `result_idx` is **1-based**
in the SQL but stored **0-based** on the fact row (matches the registry's
`eval_results.result_index`).

```sql
CREATE TABLE results_exploded AS
SELECT
  e.evaluation_id,
  e.retrieved_timestamp,
  e.source_metadata,
  e.eval_library,
  e.model_info,
  e.detailed_evaluation_results,
  e.source_config,                  -- from Stage A path extraction

  (idx_1based - 1)                                       AS result_idx,         -- 0-based, matches registry
  e.evaluation_results[idx_1based].evaluation_result_id  AS evaluation_result_id_raw,
  e.evaluation_results[idx_1based].evaluation_name       AS evaluation_name,
  e.evaluation_results[idx_1based].source_data           AS source_data,
  e.evaluation_results[idx_1based].metric_config         AS metric_config,
  e.evaluation_results[idx_1based].score_details         AS score_details,
  e.evaluation_results[idx_1based].generation_config     AS generation_config

FROM eee_raw e,
     range(1, len(e.evaluation_results) + 1) AS t(idx_1based)
WHERE e.evaluation_results IS NOT NULL
  AND len(e.evaluation_results) > 0;
```

Synthesise the deterministic IDs:

```sql
ALTER TABLE results_exploded ADD COLUMN evaluation_result_id VARCHAR;
UPDATE results_exploded
SET evaluation_result_id = COALESCE(
  evaluation_result_id_raw,
  evaluation_id || '#' || result_idx::VARCHAR
);

-- fact_id via Python UDF (sha256, NOT md5 — matches registry formula).
ALTER TABLE results_exploded ADD COLUMN fact_id VARCHAR;
UPDATE results_exploded
SET fact_id = fact_id_udf(evaluation_id, result_idx);
```

**`fact_id` is sha256-derived** to match the registry's
`eval_results.id` formula. `fact_id_udf` is the Python UDF registered at
Stage 0 (`hashlib.sha256(...).hexdigest()[:16]`). DuckDB's built-in
`md5` is the wrong hash and would silently produce non-matching ids.

---

## Stage C — resolve identity (the registry call site)

UDFs are already registered at Stage 0. Stage C is one CTAS that extracts
raw entity strings and resolves all five entity types per row. The
`source_config` column flows in from Stage A (extracted from the EEE file
path) and gets passed to the resolver — the registry supports source-
config-scoped aliases, so the same raw string can resolve to different
canonical IDs depending on EEE config context.

JSON access throughout: dot notation on STRUCT-typed columns inferred by
`read_json_auto`. Missing nested fields return NULL cleanly; no
operator-chain casts needed.

```sql
CREATE TABLE results_resolved AS
WITH raw AS (
  SELECT
    *,
    -- compute once, reuse below
    model_info.id                                            AS _model_raw,
    clean_eval_name_udf(evaluation_name)                     AS _benchmark_raw,
    extract_metric_udf(
      COALESCE(metric_config.evaluation_description,
               metric_config.metric_name,
               evaluation_name))                             AS _metric_raw,
    source_metadata.source_organization_name                 AS _org_raw,
    -- harness_raw: whatever EEE has. Don't pre-parse name/version —
    -- the resolver's normalize + fuzzy strategies swallow noise.
    trim(
      COALESCE(eval_library.name,    '') || ' ' ||
      COALESCE(eval_library.version, '')
    )                                                         AS _harness_raw
  FROM results_exploded
)
SELECT
  *,
  _model_raw      AS model_raw,
  _benchmark_raw  AS benchmark_raw,
  _metric_raw     AS metric_raw,
  _org_raw        AS org_raw,
  _harness_raw    AS harness_raw,

  -- canonical IDs (NULL for no_match; source_config flows from Stage A)
  resolve_canonical_id(_model_raw,     'model',     source_config) AS model_id,
  resolve_canonical_id(_benchmark_raw, 'benchmark', source_config) AS benchmark_id,
  resolve_canonical_id(_metric_raw,    'metric',    source_config) AS metric_id,
  resolve_canonical_id(_org_raw,       'org',       source_config) AS org_id,
  resolve_canonical_id(_harness_raw,   'harness',   source_config) AS harness_id,

  -- resolution strategy (audit)
  resolve_strategy(_model_raw,     'model',     source_config) AS model_resolution_strategy,
  resolve_strategy(_benchmark_raw, 'benchmark', source_config) AS benchmark_resolution_strategy,
  resolve_strategy(_metric_raw,    'metric',    source_config) AS metric_resolution_strategy,
  resolve_strategy(_org_raw,       'org',       source_config) AS org_resolution_strategy,
  resolve_strategy(_harness_raw,   'harness',   source_config) AS harness_resolution_strategy

FROM raw;
```

---

## Stage D — flatten + join canonical dims

Pull nested fields up to top-level columns; LEFT JOIN benchmark cards (by
canonical `benchmark_id`), `canonical_benchmarks`, `canonical_models`,
**`canonical_metrics`** (for metric meta), and the cards table. Stage D
projects everything Stage E and Stage F will need — once we've left this
stage, only the listed columns are available downstream.

```sql
CREATE TABLE fact_results_staging AS
SELECT
  rr.fact_id,
  rr.evaluation_id, rr.result_idx, rr.evaluation_result_id,

  -- raw + canonical pairs
  rr.model_raw,     rr.model_id,
  rr.benchmark_raw, rr.benchmark_id,
  rr.metric_raw,    rr.metric_id,
  rr.org_raw,       rr.org_id,
  rr.harness_raw,   rr.harness_id,

  -- registry-derived parents (from canonical_* JOINs)
  cb.parent_benchmark_id,
  cm_model.parent_model_id,

  -- benchmark_card_id: canonical benchmark_id when its card row is present
  CASE WHEN c.card IS NOT NULL THEN rr.benchmark_id ELSE NULL END AS benchmark_card_id,

  -- resolution strategies
  rr.model_resolution_strategy, rr.benchmark_resolution_strategy,
  rr.metric_resolution_strategy, rr.org_resolution_strategy,
  rr.harness_resolution_strategy,

  -- score
  rr.score_details.score                                                          AS score,
  rr.score_details.uncertainty.standard_error.value                               AS score_se,
  rr.score_details.uncertainty.confidence_interval.lower                          AS score_ci_lower,
  rr.score_details.uncertainty.confidence_interval.upper                          AS score_ci_upper,
  rr.score_details.uncertainty.confidence_interval.confidence_level               AS score_ci_level,
  rr.score_details.uncertainty.num_samples                                        AS n_samples,

  -- source / provenance
  rr.source_metadata.evaluator_relationship                                       AS evaluator_relationship,
  rr.source_metadata.source_type                                                  AS source_type,
  rr.source_metadata.source_organization_url                                      AS source_organization_url,
  rr.eval_library.name                                                            AS eval_library_name,
  rr.eval_library.version                                                         AS eval_library_version,

  -- metric meta (joined from canonical_metrics — needed by Stage E score_scale_anomaly
  -- and by Stage F threshold computation)
  cmet.metric_kind                                                                AS metric_kind,
  cmet.metric_unit                                                                AS metric_unit,
  cmet.lower_is_better                                                            AS lower_is_better,
  cmet.min_score                                                                  AS min_score,
  cmet.max_score                                                                  AS max_score,

  -- generation config — flattened
  rr.generation_config.generation_args.temperature                                AS temperature,
  rr.generation_config.generation_args.top_p                                      AS top_p,
  rr.generation_config.generation_args.top_k                                      AS top_k,
  rr.generation_config.generation_args.max_tokens                                 AS max_tokens,
  rr.generation_config.generation_args.prompt_template                            AS prompt_template,
  rr.generation_config.generation_args.reasoning                                  AS reasoning,
  -- complex blobs: keep as JSON for downstream comparison via canonical_json_udf
  to_json(rr.generation_config.generation_args.agentic_eval_config)               AS agentic_eval_config,
  to_json(rr.generation_config.generation_args.eval_plan)                         AS eval_plan,
  to_json(rr.generation_config.generation_args.eval_limits)                       AS eval_limits,
  to_json(rr.generation_config.generation_args.sandbox)                           AS sandbox,

  -- generation_args as a JSON blob — preserved so Stage E UDFs can pass it as a
  -- single argument instead of reconstructing from individual flattened columns.
  to_json(rr.generation_config.generation_args)                                   AS generation_args_json,

  -- additional_details JSON pass-through (declared in 01-'s fact_results DDL)
  to_json(rr.source_metadata.additional_details)                                  AS source_additional_details,
  to_json(rr.generation_config.additional_details)                                AS generation_additional_details,
  to_json(rr.metric_config.additional_details)                                    AS metric_additional_details,

  -- instance pointer
  rr.detailed_evaluation_results.file_path                                        AS instance_file_path,
  rr.detailed_evaluation_results.format                                           AS instance_file_format,
  rr.detailed_evaluation_results.checksum                                         AS instance_checksum,
  rr.detailed_evaluation_results.hash_algorithm                                   AS instance_hash_algorithm,
  rr.detailed_evaluation_results.total_rows                                       AS instance_rows,

  -- card payload (used for is_agentic UDF in Stage E; dropped from final fact_results)
  c.card AS card_payload

FROM results_resolved rr
LEFT JOIN canonical_benchmarks cb       ON cb.id = rr.benchmark_id
LEFT JOIN canonical_models     cm_model ON cm_model.id = rr.model_id
LEFT JOIN canonical_metrics    cmet     ON cmet.id = rr.metric_id
LEFT JOIN cards_raw            c        ON c.benchmark_id = rr.benchmark_id;
-- cards_raw.benchmark_id was produced by Stage A via resolver lookup,
-- so this JOIN bridges card keys to canonical benchmark IDs cleanly.
```

Variable-shape blobs (`agentic_eval_config`, `eval_plan`, `eval_limits`,
`sandbox`) are converted to JSON via `to_json` so the downstream
`canonical_json_udf` (Stage F) gets a parseable serialisation. The
top-level fields stay STRUCT-typed for cheap dot-notation access.

---

## Stage E — per-row signals (pass 1)

Computed entirely from the row's own fields. No cross-row queries yet.
**Drops rows with NULL score** here (per `01-` "Fact-row computation
rules") — a fact without a score isn't useful and can't participate in
any signal computation.

`generation_args_json`, `card_payload`, and `metric_unit` are all
projected by Stage D, so Stage E references columns by name without
needing a deeper struct path or an additional JOIN.

```sql
CREATE TABLE fact_results_signaled AS
WITH base AS (
  SELECT *,
    -- presence flags (raw values are nullable; this just makes intent explicit)
    temperature           IS NOT NULL  AS has_temperature,
    top_p                 IS NOT NULL  AS has_top_p,
    top_k                 IS NOT NULL  AS has_top_k,
    max_tokens            IS NOT NULL  AS has_max_tokens,
    prompt_template       IS NOT NULL  AS has_prompt_template,
    eval_plan             IS NOT NULL  AS has_eval_plan,
    eval_limits           IS NOT NULL  AS has_eval_limits,
    agentic_eval_config   IS NOT NULL  AS has_agentic_eval_config,

    -- is_agentic via Python UDF (three-rule union). UDF is registered with
    -- "JSON" types for both card and generation_args, so we coerce card_payload
    -- (a STRUCT from read_json_auto) to a JSON string via to_json. The UDF
    -- calls _coerce_json internally to parse the string back to a dict.
    is_agentic_udf(benchmark_id, to_json(card_payload), generation_args_json) AS is_agentic
  FROM fact_results_staging
  WHERE score IS NOT NULL    -- drop rows with no score; counter logged at end of run
)
SELECT *,
  -- reproducibility (active production rule: temperature + max_tokens;
  -- agentic adds eval_plan + eval_limits)
  CASE WHEN is_agentic
       THEN NOT (has_temperature AND has_max_tokens AND has_eval_plan AND has_eval_limits)
       ELSE NOT (has_temperature AND has_max_tokens)
  END AS has_reproducibility_gap,

  -- repro_missing_fields — list of names of required-but-missing fields
  compute_repro_missing_udf(
    is_agentic, has_temperature, has_max_tokens, has_eval_plan, has_eval_limits
  ) AS repro_missing_fields,

  -- repro_required_count: 2 (base) or 4 (agentic)
  CASE WHEN is_agentic THEN 4 ELSE 2 END AS repro_required_count,

  -- repro_populated_count: required - missing
  (CASE WHEN is_agentic THEN 4 ELSE 2 END
     - len(compute_repro_missing_udf(
         is_agentic, has_temperature, has_max_tokens, has_eval_plan, has_eval_limits
       ))) AS repro_populated_count,

  -- provenance source-type collapse. Spec §5: only null and 'other' collapse to
  -- 'unspecified'. Any other unexpected value passes through verbatim (matches
  -- legacy signals.py:419-422 — pre-empts a future 'audit'-style enum value).
  COALESCE(
    CASE WHEN evaluator_relationship = 'other' THEN 'unspecified'
         ELSE evaluator_relationship
    END, 'unspecified'
  ) AS provenance_source_type,

  -- variant_key: stable hash of the normalised setup dict (Stage 0 helper)
  variant_key_udf(generation_args_json) AS variant_key,

  -- score_scale_anomaly: surfaces probable metric-unit mislabelling.
  -- metric_unit is projected by Stage D from canonical_metrics.
  (metric_unit = 'proportion' AND (score < 0 OR score > 1)) AS score_scale_anomaly

FROM base;
```

The Python wrapper around this CTAS counts `len(staging) - len(signaled)`
and includes it in the end-of-run summary as `dropped_rows_no_score`.

UDFs used here are all registered at Stage 0: `is_agentic_udf`,
`compute_repro_missing_udf`, `variant_key_udf`.

---

## Stage F — group signals (pass 2)

Cross-row signals over `(model_id, benchmark_id, metric_id)` groups. Only
groups where all three IDs are non-null are eligible — unresolved rows
can't participate in cross-row provenance/comparability checks.

**Approach:** simple group facts (`is_multi_source`, `first_party_only`,
`distinct_reporting_orgs`) are computed in SQL window functions.
Variant- and cross-party-divergence are computed in **Python UDFs** that
take the full per-group row payload — the spec rules (§6.1, §6.2) involve
deterministic tie-breaking, NULL-as-+∞ ordering, lower-median picks, and
preserve-original-values logic that's much cleaner in Python than nested
SQL CTEs.

### F.1 — group-derived provenance (pure SQL)

**Two intentional deviations from legacy `compute_provenance`
(`signals.py:384-435`)**, neither breaking spec §5:

- **Implemented in SQL window functions, not Python.** Legacy returned a
  per-row dict via `compute_provenance(group_rows)`; the new pipeline
  derives the same per-row fields (`is_multi_source`, `first_party_only`,
  `distinct_reporting_orgs`) directly via a CTE + `COUNT(DISTINCT)`
  aggregate joined back. No Python UDF needed for provenance — the
  whole-row derivation is expressible in SQL once you pre-aggregate.
- **Column renamed `distinct_reporting_organizations` → `distinct_reporting_orgs`.**
  Cosmetic shortening; both refer to the same count of normalised named
  orgs in the group. Spec §5 uses the long name; producer DDL/SQL uses
  the short one. Frontend / read API callers should be aware.



DuckDB does **not** support `COUNT(DISTINCT …) OVER (PARTITION BY …)` —
DISTINCT inside window aggregates raises an error. Pattern: pre-aggregate
distinct counts per `(model_id, benchmark_id, metric_id)` in a CTE, then
JOIN back.

```sql
CREATE TABLE fact_results_grouped AS
WITH org_normalized AS (
  -- normalize_org_name (whitespace collapse + lowercase) for set membership.
  -- Display casing preserved separately by the cross-party UDF.
  SELECT *,
    nullif(trim(regexp_replace(lower(org_raw), '\s+', ' ', 'g')), '') AS org_normalized_key
  FROM fact_results_signaled
  WHERE model_id IS NOT NULL
    AND benchmark_id IS NOT NULL
    AND metric_id IS NOT NULL
),
group_orgs AS (
  -- Pre-aggregate distinct named orgs per group (replacement for COUNT DISTINCT OVER).
  SELECT
    model_id, benchmark_id, metric_id,
    COUNT(DISTINCT org_normalized_key) FILTER (WHERE org_normalized_key IS NOT NULL)
      AS distinct_reporting_orgs
  FROM org_normalized
  GROUP BY 1, 2, 3
)
SELECT
  o.*,
  go.distinct_reporting_orgs,
  -- Group key
  substr(md5(o.model_id || '|' || o.benchmark_id || '|' || o.metric_id), 1, 16)
    AS comparability_group_id,
  go.distinct_reporting_orgs > 1 AS is_multi_source,
  (o.provenance_source_type = 'first_party' AND go.distinct_reporting_orgs = 1)
    AS first_party_only
FROM org_normalized o
JOIN group_orgs    go USING (model_id, benchmark_id, metric_id);
```

Rows with any NULL identity key are excluded from `fact_results_grouped`
and carried forward separately to the final `fact_results` table with
all group-derived signal columns NULL (see F.4 below).

### F.2 — variant + cross-party divergence (Python UDFs)

Aggregate per-group payloads and pass them to Python UDFs that mirror the
legacy `compute_variant_divergence` and `compute_cross_party_divergence`
behaviour (`eval_cards_backend_pipeline/scripts/signals.py`).

The CTAS structure is `CREATE TABLE x AS WITH ... SELECT ...` — DuckDB
requires the `WITH` clause inside the CTAS, not before it. Per `01-`
("return-None semantics"), divergence flags stay **NULL** when the
underlying signal is not applicable; do not collapse to FALSE.

```sql
CREATE TABLE fact_results_grouped_annotated AS
WITH group_payloads AS (
  SELECT
    model_id, benchmark_id, metric_id,
    -- Rows in the group as a list of structs the Python UDF consumes.
    -- generation_args_json was projected by Stage D — pass it as VARCHAR;
    -- the UDF parses it.
    array_agg(struct_pack(
      fact_id                  := fact_id,
      evaluation_id            := evaluation_id,
      score                    := score,
      generation_args          := generation_args_json,
      evaluator_relationship   := evaluator_relationship,
      source_organization_name := org_raw
    )) AS group_rows,                  -- avoid 'rows' (DuckDB context-reserved)
    -- Per-group metric meta (already projected by Stage D)
    any_value(struct_pack(
      metric_kind   := metric_kind,
      metric_unit   := metric_unit,
      min_score     := min_score,
      max_score     := max_score
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
-- Broadcast group-level annotations back to each row in the group.
-- Divergence flags stay NULL when UDF returned NULL — frontend distinguishes
-- N/A (NULL) vs applicable-and-not-divergent (FALSE).
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
LEFT JOIN group_annotations ga USING (model_id, benchmark_id, metric_id);
```

The two UDFs (`compute_variant_divergence_udf`,
`compute_cross_party_divergence_udf`) are Python wrappers around the
legacy implementations. Their I/O contracts are specified in the
"Helper UDF contracts" section below. **Both must be registered at
Stage 0** (see Stage 0 helpers below — they are added there now).

### F.3 — return-None semantics on the row

When the underlying signal returns None (per the rules in `01-`:
"Variant/Cross-party divergence — return None when…"), the per-row
columns are NULL across the board for that row's group. Frontend renders
"signal not applicable" rather than "no divergence." Distinguish via
`variant_divergence_threshold IS NULL` (NULL → not applicable) vs
`has_variant_divergence = FALSE` (applicable, no divergence flagged).

### F.4 — union annotated rows with unresolved-key passthrough

Rows excluded from F.1 (any of `model_id` / `benchmark_id` / `metric_id`
is NULL) are still kept in `fact_results` per the "don't drop records"
rule, with all group-derived signal columns NULL.

**Both branches `EXCLUDE` the working columns that aren't part of the
published `fact_results` schema:**
- `card_payload` (used only by Stage E's `is_agentic_udf`)
- `org_normalized_key` (added in F.1 for set-membership counting)
- `generation_args_json` (used as a single-arg blob for `is_agentic_udf` and
  `variant_key_udf`; not declared in the fact_results DDL — the seven
  `GENERATION_ARGS_COMPARISON_FIELDS` are projected as flat columns instead)

Without the EXCLUDE these would leak into the output Parquet.

```sql
CREATE TABLE fact_results AS
SELECT
  '<snapshot_id_literal>'::TIMESTAMP AS snapshot_id,
  * EXCLUDE (card_payload, org_normalized_key, generation_args_json)
FROM fact_results_grouped_annotated

UNION ALL BY NAME

SELECT
  '<snapshot_id_literal>'::TIMESTAMP AS snapshot_id,
  fr.* EXCLUDE (card_payload, generation_args_json),  -- fact_results_signaled has no org_normalized_key
  -- group-derived columns NULL (all-NULL passthrough)
  NULL::INTEGER                                    AS distinct_reporting_orgs,
  NULL::VARCHAR                                    AS comparability_group_id,
  NULL::BOOLEAN                                    AS is_multi_source,
  NULL::BOOLEAN                                    AS first_party_only,
  NULL::BOOLEAN                                    AS has_variant_divergence,
  NULL::DOUBLE                                     AS variant_divergence_magnitude,
  NULL::DOUBLE                                     AS variant_divergence_threshold,
  NULL::VARCHAR                                    AS variant_threshold_basis,
  NULL::STRUCT(field VARCHAR, "values" JSON)[]    AS variant_differing_fields,
  NULL::BOOLEAN                                    AS has_cross_party_divergence,
  NULL::DOUBLE                                     AS cross_party_divergence_magnitude,
  NULL::DOUBLE                                     AS cross_party_divergence_threshold,
  NULL::VARCHAR                                    AS cross_party_threshold_basis,
  NULL::STRUCT(field VARCHAR, "values" JSON)[]    AS cross_party_differing_fields,
  NULL::INTEGER                                    AS cross_party_org_count,
  NULL::MAP(VARCHAR, DOUBLE)                       AS scores_by_organization
FROM fact_results_signaled fr
WHERE fr.model_id IS NULL OR fr.benchmark_id IS NULL OR fr.metric_id IS NULL;
```

`UNION ALL BY NAME` (DuckDB-specific) reconciles column order across the
two branches; columns missing from one side fill with NULL. `EXCLUDE` is
DuckDB's syntactic-sugar for projecting all columns except the listed
ones. The unresolved branch surfaces in the triage SQL from `01-`
(the `WHERE *_id IS NULL` UNION).

---

## Stage G — dims (`benchmarks`, `models`)

```sql
-- benchmarks.parquet — JOIN canonical_benchmarks ⨝ AutoBenchmarkCards.
-- cards_raw.card is STRUCT-typed (loaded via temp JSONL in Stage A), so
-- card subfields are accessed with dot notation.
CREATE TABLE benchmarks AS
SELECT
  '<snapshot_id_literal>'::TIMESTAMP AS snapshot_id,
  cb.id AS benchmark_id,

  -- from canonical_benchmarks
  cb.display_name,
  cb.description,
  cb.dataset_repo,
  cb.parent_benchmark_id,
  cb.tags                  AS registry_tags,    -- already a list (JSON-decoded by read_parquet)
  to_json(cb.metadata)     AS registry_metadata,
  cb.review_status,

  -- from AutoBenchmarkCards (via cards_raw join)
  c.card.benchmark_details.name                                  AS card_name,
  c.card.benchmark_details.overview                              AS overview,
  c.card.benchmark_details.data_type                             AS data_type,
  c.card.benchmark_details.domains                               AS domains,
  c.card.benchmark_details.languages                             AS languages,
  c.card.benchmark_details.similar_benchmarks                    AS similar_benchmarks,
  c.card.benchmark_details.resources                             AS resources,

  c.card.purpose_and_intended_users.goal                         AS goal,
  c.card.purpose_and_intended_users.audience                     AS audience,
  c.card.purpose_and_intended_users.tasks                        AS tasks,
  c.card.purpose_and_intended_users.limitations                  AS limitations,
  c.card.purpose_and_intended_users.out_of_scope_uses            AS out_of_scope_uses,

  c.card.data.source                                             AS data_source,
  c.card.data.size                                               AS data_size,
  c.card.data.format                                             AS data_format,
  c.card.data.annotation                                         AS data_annotation,

  c.card.methodology.methods                                     AS methods,
  c.card.methodology.metrics                                     AS card_metrics,
  c.card.methodology.calculation                                 AS calculation,
  c.card.methodology.interpretation                              AS interpretation,
  c.card.methodology.baseline_results                            AS baseline_results,
  c.card.methodology.validation                                  AS validation,

  c.card.ethical_and_legal_considerations.privacy_and_anonymity        AS privacy_and_anonymity,
  c.card.ethical_and_legal_considerations.data_licensing               AS data_licensing,
  c.card.ethical_and_legal_considerations.consent_procedures           AS consent_procedures,
  c.card.ethical_and_legal_considerations.compliance_with_regulations  AS compliance_with_regulations,

  c.card.possible_risks                                          AS possible_risks,
  -- flagged_fields is a {<section>.<field>: message} map. read_json_auto
  -- of the cards JSONL infers its top-level type as STRUCT (one field per
  -- observed key across all card files). We coerce it to JSON for the
  -- output column to preserve the dynamic-key shape; map_keys would not
  -- work directly because the inferred STRUCT has fixed fields.
  to_json(c.card.flagged_fields)                                 AS flagged_fields,

  c.card IS NOT NULL                                             AS card_present,
  -- _generated_by carried as a top-level field in the card
  c.card._generated_by                                           AS card_generated_by,
  -- card_flagged_count: number of fields the factreasoner worker flagged.
  -- factreasoner only adds a key to flagged_fields when a flag fires, so
  -- the top-level key count equals the flag count.
  COALESCE(len(json_keys(to_json(c.card.flagged_fields))), 0)    AS card_flagged_count,
  -- card_missing_count: count of operationalised completeness fields scoring 0
  -- for THIS benchmark. Pull from benchmark_completeness (computed in Stage H).
  -- Since Stage H runs after Stage G, the count is left-joined here from a
  -- second pass — see "Stage G post-fixup" below for the UPDATE.
  NULL::INTEGER                                                  AS card_missing_count

FROM canonical_benchmarks cb
LEFT JOIN cards_raw c ON c.benchmark_id = cb.id
WHERE cb.id IN (SELECT DISTINCT benchmark_id FROM fact_results WHERE benchmark_id IS NOT NULL);

-- models.parquet — canonical_models ⨝ canonical_orgs
CREATE TABLE models AS
SELECT
  '<snapshot_id_literal>'::TIMESTAMP AS snapshot_id,
  cm.id AS model_id,

  -- from canonical_models (top-level columns)
  cm.display_name,
  cm.developer,
  cm.org_id,
  cm.family,
  cm.architecture,
  cm.params_billions,
  cm.parent_model_id,
  cm.tags                AS registry_tags,
  to_json(cm.metadata)   AS registry_metadata,
  cm.review_status,

  -- denormalised from canonical_orgs (one-table reads for the index)
  co.display_name        AS org_display_name,
  co.website             AS org_website,
  co.hf_org              AS org_hf_org,
  co.parent_org_id       AS org_parent_id,

  -- Frontend-needed extras: registry will populate these as TOP-LEVEL columns
  -- on canonical_models in a future iteration. Until then, write NULL —
  -- producer's SELECT switches from `NULL::DATE` to `cm.released` once the
  -- registry's table has the column.
  NULL::DATE         AS released,         -- TODO: cm.released
  NULL::INTEGER      AS context_tokens,   -- TODO: cm.context_tokens
  NULL::VARCHAR      AS context_label,    -- TODO: cm.context_label
  NULL::VARCHAR[]    AS modality,         -- TODO: cm.modality
  NULL::VARCHAR      AS access,           -- TODO: cm.access
  NULL::VARCHAR      AS license           -- TODO: cm.license

FROM canonical_models cm
LEFT JOIN canonical_orgs co ON co.id = cm.org_id
WHERE cm.id IN (SELECT DISTINCT model_id FROM fact_results WHERE model_id IS NOT NULL);
```

When the registry adds `released`, `license`, `modality`, `access`,
`context_tokens` as top-level columns on `canonical_models`, swap
`NULL::TYPE` for `cm.<field>` — single-line edit per field, no schema
shape change downstream.

Both dim tables filter to entities that actually appear in the snapshot's
facts — no orphan rows.

---

## Stage H — `benchmark_completeness`

Per-benchmark scoring against the operationalised field set
(`registry/completeness_fields.json` — port forward from the legacy
pipeline's same-named file).

DuckDB doesn't support `CROSS JOIN LATERAL <function-call>` for
projecting struct-returning UDFs as named columns. The clean pattern is
to apply the UDF once in a CTE producing a STRUCT column, then
dereference fields in the outer SELECT.

```sql
CREATE TABLE benchmark_completeness AS
WITH scored AS (
  SELECT
    cb.id AS benchmark_id,
    -- to_json() coerces the card STRUCT to a JSON string the UDF parses internally
    compute_completeness_udf(to_json(c.card)) AS comp
  FROM canonical_benchmarks cb
  LEFT JOIN cards_raw c ON c.benchmark_id = cb.id
  WHERE cb.id IN (SELECT DISTINCT benchmark_id FROM fact_results
                  WHERE benchmark_id IS NOT NULL)
)
SELECT
  '<snapshot_id_literal>'::TIMESTAMP   AS snapshot_id,
  benchmark_id,
  comp.completeness_score              AS completeness_score,
  comp.total_fields_evaluated          AS total_fields_evaluated,
  comp.populated_count                 AS populated_count,
  comp.missing_required_fields         AS missing_required_fields,
  comp.partial_fields                  AS partial_fields,
  comp.field_scores                    AS field_scores
FROM scored;
```

`compute_completeness_udf` (Stage 0) takes the card as a JSON string and
returns the per-benchmark annotation block from spec §4.2 + the
28-field set from `01-`'s "Operationalised completeness field set".

### Stage H post-fixup — backfill `card_missing_count` on `benchmarks`

Stage G wrote `NULL::INTEGER AS card_missing_count` because completeness
hadn't yet been computed. Now that `benchmark_completeness` is built,
update the dim:

```sql
UPDATE benchmarks b
SET card_missing_count = (
  SELECT len(bc.missing_required_fields)
  FROM benchmark_completeness bc
  WHERE bc.snapshot_id = b.snapshot_id
    AND bc.benchmark_id = b.benchmark_id
);
```

Length of `missing_required_fields` is the canonical "fields that scored
0" count per spec §4.2.

---

## Stage I — emit Parquet + snapshot meta

```python
# Sanitise colons for filesystem safety (Windows rejects them in paths).
# The snapshot_id *column* value inside the parquet files stays canonical ISO
# with colons; only the directory name is sanitised.
snapshot_dir_name = snapshot_id.replace(':', '-')   # 2026-04-30T12:00:00Z → 2026-04-30T12-00-00Z
out = Path(f"warehouse/{snapshot_dir_name}")
out.mkdir(parents=True, exist_ok=True)

for table, sort_key in [
    ("fact_results",            "(model_id, benchmark_id, metric_id)"),
    ("benchmark_completeness",  "(benchmark_id)"),
    ("benchmarks",              "(benchmark_id)"),
    ("models",                  "(model_id)"),
    ("canonical_metrics",       "(id)"),                # registry mirror
]:
    con.execute(f"""
      COPY (SELECT * FROM {table} ORDER BY {sort_key} NULLS LAST)
      TO '{out / table}.parquet'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
```

`NULLS LAST` matters because unresolved fact rows (NULL canonical IDs) sort
to the bottom — they shouldn't intersperse with resolved rows in row
groups, since most queries filter `WHERE *_id IS NOT NULL`.

`canonical_metrics.parquet` is a copy of the registry table (loaded at
Stage A). Including it in the snapshot makes downstream consumers
self-contained — the read API doesn't need to also resolve the registry's
local cache directory.

### Snapshot metadata sidecar

Alongside the Parquet outputs, write a small `snapshot_meta.json` that
records the upstream HF dataset revisions the snapshot was built from.
Cheap reproducibility hook; lets us recreate any historical snapshot from
inputs.

```python
import json
from huggingface_hub import HfApi

def hf_revision(repo_id: str, hf_token: str | None) -> str | None:
    api = HfApi(token=hf_token)
    try:
        info = api.dataset_info(repo_id, token=hf_token)
        return info.sha
    except Exception:
        return None

meta = {
    "snapshot_id":      snapshot_id,
    "generated_at":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "eee_revision":     hf_revision("evaleval/EEE_datastore",         settings.hf_token),
    "registry_revision":hf_revision("evaleval/entity-registry-data",  settings.hf_token),
    "cards_revision":   hf_revision("evaleval/auto-benchmarkcards",   settings.hf_token),
    "tables": [
        "fact_results.parquet",
        "benchmark_completeness.parquet",
        "benchmarks.parquet",
        "models.parquet",
        "canonical_metrics.parquet",
    ],
}
(out / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))
```

When inputs are loaded from local cache (e.g. `EEE_LOCAL_DATASET_DIR`
points at a stale download), `hf_revision()` still returns the upstream
sha — we record what the canonical input was at canonicalisation time,
even if our local copy is older. If reproducing a snapshot exactly, pin
each upstream to its recorded sha before re-running.

---

## Module layout

Drop into the existing package:

```
src/eval_card_backend/
├── cli.py                       # extend: add `canonicalise` subcommand
├── config.py
├── sources/
│   ├── eee.py
│   ├── benchmark_cards.py
│   └── registry.py              ← NEW (mirrors the others)
├── canonicalise/                ← NEW
│   ├── __init__.py
│   ├── pipeline.py              # orchestrator: run_pipeline(settings, snapshot_id)
│   ├── stages.py                # stage_a_load, ..., stage_i_emit
│   ├── udfs.py                  # is_agentic_py, compute_repro_missing,
│   │                            #   compute_completeness_udf, source_config_for
│   ├── resolver_setup.py        # builds Resolver, registers DuckDB UDFs
│   └── thresholds.py            # spec constants
├── signals/                     ← NEW (pure logic, I/O-free, easy to unit-test)
│   ├── reproducibility.py
│   ├── provenance.py
│   └── comparability.py
└── registry/                    ← NEW (checked-in static data)
    ├── completeness_fields.json
    └── agentic_name_regex.json
```

**Identity normalisation lives in `eval-entity-resolver` (external dep), not
here.** No alias YAMLs in `eval_card_backend/registry/`. Local
`registry/` is just operationalised configs that don't belong upstream
(completeness field set, agentic regex).

---

## CLI

```bash
# fetch latest of all three sources, canonicalise, emit
uv run eval-card-backend canonicalise

# pin a snapshot id
uv run eval-card-backend canonicalise --snapshot-id 2026-04-30T12:00:00Z

# limit to subset of EEE configs (smoke test)
uv run eval-card-backend canonicalise --configs cnn_dailymail,xsum

# custom warehouse root
uv run eval-card-backend canonicalise --warehouse ./warehouse-test
```

The existing default behaviour (`uv run eval-card-backend` reports loaded
inputs) stays under no-subcommand invocation.

---

## Edge cases

- **Resolver returns `no_match`.** Row is kept; canonical column is NULL;
  raw column carries the original string (always populated). Cross-row
  signals exclude rows with NULL keys (can't form a group). Counter
  increments; not logged per-row.
- **Card missing for a benchmark.** `cards_raw.benchmark_id` is NULL via
  resolver miss, OR card_payload is NULL via LEFT JOIN. `is_agentic`
  falls back to rules 2 and 3 (config presence, regex). Completeness
  scores low for that benchmark. Frontend renders "no card".
- **Same `evaluation_result_id` from upstream merges.** Fall back to
  `evaluation_id#result_idx` (always unique within a record). The
  registry-aligned `fact_id` (`sha256(evaluation_id:result_idx)[:16]`,
  via `fact_id_udf`) is the canonical PK.
- **Score absent (`score_details.score` is NULL).** Drop the row in
  Stage E (`WHERE score IS NOT NULL`); counter `dropped_rows_no_score`
  increments and is reported at end of run. Identity-resolution failures
  do NOT drop the row; missing score does — a fact without a score
  can't participate in any signal computation.
- **Registry schema drift.** If `evaleval/entity-registry-data` adds a new
  canonical_* table, our load is forward-compatible — we just don't read
  the new table until we wire it. If existing schemas change, our reads
  break loudly; the `snapshot_meta.json` records the registry HF revision
  used so reproduction is deterministic.
- **Registry has `cm.released` (or other model-roadmap field) populated.**
  Producer's `models.parquet` SELECT currently writes `NULL::TYPE`;
  swap each `NULL::TYPE` for `cm.<field>` in Stage G. One-line edit per
  field, no other changes. Tracked in `01-` model-fields TODO.
- **Multiple EEE records for the same fact_id.** Detect via
  `COUNT(*) > 1 GROUP BY snapshot_id, fact_id`; log; keep the row with
  the latest `retrieved_timestamp`. Don't fail the pipeline.
- **`canonical_metrics.{min_score, max_score}` not populated.** Threshold
  falls through to `fallback_default = 0.05` (per legacy
  `compute_threshold`). `*_threshold_basis` records `'fallback_default'`
  so the choice is auditable.
- **Empty `evaluation_results[]`.** `len(...) > 0` filter in Stage B
  excludes the record. No fact rows produced for that EEE record.
- **Card has malformed JSON / missing keys.** `read_json_auto` of the
  cards JSONL surfaces the error during Stage A load. Producer fails fast
  per Stage Pre's strict EEE/registry rule (cards is best-effort: if the
  whole cards source is missing, warn-and-proceed; if present-but-broken,
  fail loudly so the cards pipeline is fixed first).

---

## Helper UDF contracts

All Python UDFs live in `src/eval_card_backend/canonicalise/udfs.py`. Pure
logic with no I/O (resolver state is the one exception, held in module
scope after Stage 0 init). Spec-binding I/O contracts:

### Identity / setup helpers

```python
fact_id_py(evaluation_id: str | None, result_idx: int) -> str | None
  # First 16 hex chars of sha256(f'{evaluation_id}:{result_idx}'.encode()).
  # Returns None if evaluation_id is falsy.

variant_key_py(generation_args: dict | None) -> str
  # First 16 hex chars of sha256(setup_canonical_json(generation_args).encode()).
  # Even when generation_args is None or {}, returns a stable hash of the
  # all-NULL setup dict (so 'no setup recorded' rows still group together).

canonical_json(obj: Any) -> str | None
  # json.dumps(obj, sort_keys=True, separators=(',',':'),
  #            ensure_ascii=False, default=str), or None if obj is None.

normalize_setup(generation_args: dict | None) -> dict
  # The per-field-normalised dict over GENERATION_ARGS_COMPARISON_FIELDS.
  # Always returns a dict with all 7 keys; values may be None.

resolve_canonical_id_py(raw, entity_type, source_config) -> str | None
  # Stage 0 wrapper around eval_entity_resolver.Resolver.resolve.
  # Returns canonical_id or None on no_match / exception.
  # Side effect: increments miss_counter / exception_counter (see Stage 0).

resolve_strategy_py(raw, entity_type, source_config) -> str
  # Returns 'exact' | 'normalized' | 'fuzzy' | 'no_match'.

is_agentic_py(benchmark_id: str | None,
              benchmark_card: str | dict | None,
              generation_args: str | dict | None) -> bool
  # Body MUST start with:
  #   benchmark_card  = _coerce_json(benchmark_card,  'is_agentic_py.card')
  #   generation_args = _coerce_json(generation_args, 'is_agentic_py.gen_args')
  # because DuckDB delivers JSON-typed params as VARCHAR strings.
  # Three-rule union (see 01-: "is_agentic rule"):
  #   1. card.purpose_and_intended_users.tasks ∩ {agentic, tool_use, multi_step_agent}
  #   2. generation_args.agentic_eval_config is not None
  #   3. AGENTIC_NAME_REGEX matches normalised benchmark_id
```

### Signal computation

```python
compute_repro_missing_py(
    is_agentic: bool,
    has_temperature: bool, has_max_tokens: bool,
    has_eval_plan: bool,   has_eval_limits: bool,
) -> list[str]
  # Returns the names of the active reproducibility-required fields that
  # are absent. Field set per 01-: BASE = (temperature, max_tokens);
  # AGENTIC adds (eval_plan, eval_limits) when is_agentic.

compute_completeness_py(card: str | dict | None) -> dict
  # Body MUST start with: card = _coerce_json(card, 'compute_completeness_py')
  # Walks the 28-field operationalised set (loaded once from
  # registry/completeness_fields.json at module import). Returns dict
  # matching benchmark_completeness.parquet column shape:
  #   {completeness_score: float,        # sum(field_scores) / total_fields_evaluated
  #    total_fields_evaluated: int,      # always 28
  #    populated_count: float,           # sum(fs.score for fs in field_scores)
  #    missing_required_fields: list[str],
  #    partial_fields: list[{field_path, score, populated_subitems, total_subitems}],
  #    field_scores:   list[{field_path, coverage_type, score}]}
  # When card is None, all card-sourced fields score 0; EEE source_metadata
  # fields and reserved fields are still scored against their data sources.
  #
  # Note: legacy compute_reporting_completeness (signals.py:165-245) does NOT
  # return `populated_count` — this is a deliberate extension so the dim
  # column in benchmark_completeness.parquet can be read without recomputing
  # the partial-aware sum.

compute_variant_divergence_py(rows: list[dict], metric_config: dict) -> dict | None
  # Spec §6.1.2 + legacy. Returns None when:
  #   (a) len(rows) < 2;
  #   (b) all rows have identical setup (after normalize_setup);
  #   (c) <2 rows have non-null score after exclusions.
  # Otherwise returns:
  #   {has_variant_divergence: bool,
  #    divergence_magnitude:    float,
  #    threshold_used:          float,
  #    threshold_basis:         str,    # one of 4 labels (see 01-)
  #    differing_setup_fields:  list[{field, values}]}   # original distinct values
  #
  # Legacy `compute_variant_divergence` (signals.py:438-489) additionally
  # returned: `scores_in_group`, `triple_count_in_group`,
  # `group_variant_breakdown`, `score_scale_anomaly`, `group_id`,
  # `signal_version`. Intentionally dropped from the new shape:
  #   - `scores_in_group`, `triple_count_in_group`: no fact_results column
  #     consumes these; reproducible from the raw rows by regrouping.
  #   - `group_variant_breakdown`: superseded by the per-row `variant_key`
  #     column, which gives the same information at higher fidelity.
  #   - `score_scale_anomaly`: now a per-row column (Stage E).
  #   - `group_id`: replaced by `comparability_group_id` per row.
  #   - `signal_version`: deferred per `01-` no-versioning-yet rule.
  #
  # Body MUST start with:
  #   for row in rows:
  #       row['generation_args'] = _coerce_json(row.get('generation_args'),
  #                                              'compute_variant_divergence_py.gen_args')
  #
  # Each row dict carries: {fact_id, evaluation_id, score, generation_args,
  #                          evaluator_relationship, source_organization_name}.

compute_cross_party_divergence_py(rows: list[dict], metric_config: dict) -> dict | None
  # Spec §6.2.2 + legacy. Returns None when <2 distinct named orgs (after
  # normalize_org_name). Per-org score is statistics.median of org's
  # scored rows; per-org representative setup is `aggregated_setup` of
  # the org's rows (see 01-: "aggregated_setup" rule). differing_setup_fields
  # is computed *across org-representative setups*, not across all rows.
  #
  # Body MUST start with the same _coerce_json sweep as the variant UDF:
  #   for row in rows:
  #       row['generation_args'] = _coerce_json(row.get('generation_args'),
  #                                              'compute_cross_party_divergence_py.gen_args')
  #
  # Returns:
  #   {has_cross_party_divergence: bool,
  #    divergence_magnitude:        float,
  #    threshold_used:              float,
  #    threshold_basis:             str,
  #    scores_by_organization:      dict[org_display_name → median_score],
  #    differing_setup_fields:      list[{field, values}],
  #    organization_count:          int}
  #
  # Legacy `compute_cross_party_divergence` (signals.py:492-557) additionally
  # returned: `group_id`, `group_variant_breakdown`, `signal_version`.
  # Intentionally dropped, parallel to the variant UDF rationale:
  #   - `group_id`: replaced by the per-row `comparability_group_id` column.
  #   - `group_variant_breakdown`: superseded by per-row `variant_key`.
  #   - `signal_version`: deferred per `01-` no-versioning-yet rule.
```

All `compute_*` functions are imported from `signals/` (the I/O-free
package). The UDF wrappers in `canonicalise/udfs.py` are thin
adapters that handle DuckDB type coercion. Behavioural parity with
legacy is verified via the test fixtures below.

---

## Smoke tests / data validation

Calibrate thresholds after the first full canonicalisation run (per
prior decision — pre-baked thresholds will warn too aggressively before
we know the production-data shape). For now, log the following metrics
and have the producer print them at end of run; tighten into FAIL
conditions later:

**Metrics to track per run** (logged by `log_resolver_summary()` and a
new `log_canonicalisation_summary()`):

```
- fact_results row count
- dropped_rows_no_score count
- per-entity-type unresolved counts + top-N raw strings
- distinct (model_id, benchmark_id, metric_id) groups
- groups eligible for variant_divergence  (>= 2 rows, differing setup, >=2 scored)
- groups eligible for cross_party_divergence (>= 2 named orgs, >=2 scored)
- groups with has_variant_divergence = TRUE
- groups with has_cross_party_divergence = TRUE
- benchmark_completeness mean / median / min
- benchmarks without AutoBenchmarkCard (card_present = false)
- score_scale_anomaly count
- resolver exception counter (per entity_type, exception_class)
```

**Provisional FAIL conditions** (tighten once telemetry exists):

- `fact_results` row count == 0 (catastrophic)
- Required output table missing or empty schema
- `canonical_metrics` dim is empty after registry load (registry catastrophically broken)

Everything else is currently WARN-only. After the first run we'll
calibrate per-entity unresolved-rate ceilings, expected eligibility
fractions, etc.

---

## Test fixtures

Hand-built tiny fixtures under `tests/fixtures/`. Each fixture file is
hand-curated to exercise a specific edge case; no real-data extraction
in v1.

```
tests/fixtures/
├── eee/                              # 5–7 evaluation records
│   ├── 01-resolves-cleanly.json      # all 5 entities resolve via 'exact'; non-agentic
│   ├── 02-no-match-model.json        # community fine-tune; model_id no_match
│   ├── 03-agentic-via-card.json      # SWE-bench-style; benchmark card has tasks=['agentic']
│   ├── 04-agentic-via-config.json    # generation_args.agentic_eval_config present
│   ├── 05-variant-divergence.json    # 3 rows same triple, max_tokens varies, scores diverge
│   ├── 06-cross-party-divergence.json # same triple, 2 orgs (case/whitespace variant), scores diverge
│   └── 07-no-score.json              # score_details.score is null; should be dropped
├── auto_benchmarkcards/
│   └── cards/
│       ├── mmlu.json                 # full card; powers fixture 01
│       ├── swebench-verified.json    # agentic tasks tag
│       └── (no card for fixture 02's benchmark — exercise card-missing path)
├── entity_registry/
│   ├── canonical_orgs.parquet        # 4 rows: openai, meta, google, scale-ai
│   ├── canonical_models.parquet      # 6 rows including parent_model_id chain
│   ├── canonical_benchmarks.parquet  # 5 rows including parent_benchmark_id chain
│   ├── canonical_metrics.parquet     # 4 rows covering proportion/percent/range/fallback
│   ├── eval_harnesses.parquet        # 2 rows
│   └── aliases.parquet               # ~30 aliases; deliberately missing entries for fixture 02
└── expected/
    ├── fact_results.json             # expected output rows (subset of columns)
    ├── benchmarks.json
    └── benchmark_completeness.json
```

**Test layers:**

1. **Unit** (`tests/unit/`) — `signals/` functions against hand-built input
   dicts. Spec test cases TC-R1..R5, TC-C1..C4, TC-P1..P4, TC-V1..V2,
   TC-CP1..CP3 mapped to test functions. **TC-R6 ("benchmark has no EEE
   record → signal returns null")** is *not* a unit test on the
   reproducibility signal in this pipeline: the signal is a per-row fact
   column, and a benchmark with no EEE rows produces no fact rows — the
   "missing = absent" rule (`01-` design principle 6) covers it. The
   frontend renders absence; nothing to assert at the signal layer. No DuckDB,
   no parquet.
2. **UDF round-trip** (`tests/udf/`) — register UDFs against an in-memory
   DuckDB; verify SQL `SELECT compute_*_udf(...)` outputs match the
   underlying Python function for a battery of inputs.
3. **End-to-end** (`tests/e2e/`) — run `pipeline.run(settings)` against
   `tests/fixtures/`, assert resulting Parquet shapes match
   `expected/*.json`. One assertion per fixture × signal column.

Identity-resolution unit tests live in `eval-entity-resolver` (the
upstream package), **not** here. We test integration: that we call the
resolver with the right `entity_type` and `source_config`, and that
`no_match` flows through correctly into NULL canonical IDs + raw preservation.

---

## Performance expectations

For the prototype-scale corpus (~10⁴ rows):

- Stage A–B: a few seconds (JSON parsing dominant).
- Stage C: resolver UDF calls add ~ms per row × 5 entity types × N rows.
  At 10⁴ rows × 5 ≈ 50k resolver calls, well under 60s. Resolver is
  in-memory dict + fuzzy matching; no I/O.
- Stage D–F: low single seconds.
- Stage G–I: low single seconds; dims and Parquet writes are cheap.

Whole pipeline target: **under 90s** on a laptop. Re-evaluate at 10⁶+ rows.

---

## Open questions / next iterations

- **Smoke-test thresholds** — current FAIL set is provisional (just
  catastrophic-empty checks). Calibrate per-entity unresolved-rate
  ceilings, eligibility fractions, etc. after first full run produces
  baseline metrics.
- **`--strict` mode for resolver exceptions** — production may want to
  fail-fast on resolver exceptions instead of count-and-degrade. Add
  flag when we have failure-mode telemetry from a few runs.
- **Read-side deployment.** Out of scope for this doc. Frontend Space is
  Next.js + Docker SDK with `@duckdb/node-api` reading the warehouse
  parquets; consumers refer to `01-` for the per-view queries.

### Resolved this iteration

- **Card-key alignment** → resolve card keys through the same resolver
  in Stage A. `cards_raw` carries a `benchmark_id` column from the start;
  downstream JOINs use that, not the syntactic card_key.
- **`source_config` extraction** → not a UDF; a column extracted from the
  EEE file path during raw load (`regexp_extract(filename, 'data/([^/]+)/', 1)`).
  Flows through every stage as a normal column.
- **Variant-divergence on JSON setup fields** → `canonical_json_udf`
  before any distinct comparison. Stable serialisation by sorted keys +
  compact separators. **NULL handling**: DuckDB's `list_distinct` strips
  NULLs, so the Python UDFs (`compute_variant_divergence_py`,
  `compute_cross_party_divergence_py`) handle NULL-as-distinct in
  Python — not via DuckDB's `list_distinct`. The legacy
  `_differing_setup_fields` (`signals.py:362`) does the right thing:
  iterates each setup, canonicalises with `default=str` (which produces
  `"null"` for None), and uses set-membership on canonical strings. NULL
  values produce the literal canonical string `"null"`, distinguishable
  from a populated value.
- **Identity resolution architecture** → resolver imported in-process,
  called as DuckDB UDF on every row. We do **not** read the registry's
  internal `eval_results` table.
- **`harness_raw` form** → freeform string; concat of EEE's
  `eval_library.name` + `eval_library.version` (whatever's there).
  The resolver's normalize + fuzzy strategies eat noise.
- **`eval-entity-resolver` dependency** → uv workspace / local path dep
  for now (`../evalcard-registry/packages/eval-entity-resolver`); switch
  to git-URL pinned dep when the registry is properly published.
  Migration TODO is in `CLAUDE.md`.
- **`canonical_metrics` materialisation** → emit to
  `warehouse/<snapshot>/canonical_metrics.parquet` so the snapshot is
  self-contained for downstream readers.
- **Snapshot meta** → emit `warehouse/<snapshot>/snapshot_meta.json` with
  upstream HF dataset revisions (`eee_revision`, `registry_revision`,
  `cards_revision`). Reproducibility hook.
- **Model fields location in canonical_models** → top-level columns
  (`cm.released`, `cm.license`, `cm.modality`, `cm.access`,
  `cm.context_tokens`), not nested in `metadata` JSON. Producer's
  `models.parquet` SELECT currently writes `NULL::TYPE`; switch to
  `cm.<field>` once the registry adds the columns.
- **Card freshness behaviour** → proceed without card; LEFT JOIN gives
  `card_present = false` for benchmarks the cards source hasn't covered
  yet. No hard fail.
- **`fact_id` hash** → sha256(`evaluation_id:result_idx`)[:16] via
  `fact_id_udf` Python UDF. Matches registry's `eval_results.id` formula
  for cross-system referencing.
- **`variant_key` hash + setup normalisation** → sha256 of canonical-JSON
  of normalised setup dict. Per-field rules (`_norm_num`, `_norm_int`,
  `_norm_text`, `_norm_bool`) applied identically in `variant_key_udf`
  and in `_differing_setup_fields` divergence detection. No drift.
- **Stage B UNNEST pattern** → pinned to the `range()` + array-index
  approach. `result_idx` is 0-based to match registry.
- **Stage F cross-row signal computation** → Python UDFs
  (`compute_variant_divergence_udf`, `compute_cross_party_divergence_udf`)
  for the legacy-faithful `aggregated_setup` lower-median rule + NULL
  ordering + tie-breaking. Group-derived provenance fields stay in pure
  SQL window functions.
- **Operationalised completeness field set** → 28 fields ported from
  legacy. Stored as `registry/completeness_fields.json`; loaded once at
  module import. Detail in `01-`: "Operationalised completeness field set".
- **Helper UDF I/O contracts** → fully specified in this doc's
  "Helper UDF contracts" section. Behavioural parity with legacy
  verified by hand-built fixtures under `tests/fixtures/`.
- **Test fixtures** → seven hand-built EEE records covering edge cases
  (clean resolution, no_match, agentic-via-card, agentic-via-config,
  variant divergence, cross-party divergence, no-score). Three test
  layers: unit (signals/), UDF round-trip, end-to-end.
- **Smoke tests** → metrics logged per run; provisional FAIL set covers
  catastrophic empty cases only. Tighten after first calibration run.
- **DuckDB JSON access** → STRUCT dot notation for EEE records (loaded
  via `read_json_auto` with `union_by_name=true`) and for cards (loaded
  via temp JSONL). Variable-shape blobs (`agentic_eval_config`,
  `eval_plan`, `eval_limits`, `sandbox`) wrapped via `to_json` for
  divergence detection.
- **Score-NULL drop** → Stage E filters with `WHERE score IS NOT NULL`;
  counter increments. Identity-failure rows are kept (NULL canonical,
  raw preserved); score-failure rows are dropped.
- **Score scale anomaly** → flagged on row when `metric_unit='proportion'`
  but score ∉ [0, 1]. Surfaces probable metric-unit mislabelling without
  rejecting the row.
- **DuckDB JSON access pattern** → STRUCT dot notation throughout
  (trust `read_json_auto` inference). Cards loaded via temp JSONL so
  they get STRUCT typing too. Variable-shape blobs (`agentic_eval_config`,
  `eval_plan`, `eval_limits`, `sandbox`) wrapped via `to_json` for
  `canonical_json_udf` consumption.
- **Resolver logging philosophy** → no_match is *expected* (registry
  misses lots of community fine-tunes etc.); silent per-row, counter
  only. Resolver exceptions log first occurrence per
  `(entity_type, exception_class)` then count quietly. End-of-run
  summary lists miss counts and top unresolved raw strings.
- **Snapshot directory naming** → sanitise colons (`:` → `-`) for
  filesystem safety. Column value stays canonical ISO with colons.
- **Cold-start / unreachable upstream** → preflight checks fail fast on
  missing EEE or empty registry alias store; warn-and-proceed on
  missing AutoBenchmarkCards.
- **Benchmark slug regex (`is_agentic`)** → normalise canonical_id by
  collapsing `[^a-z0-9]+ → '_'` (matches legacy `helpers/benchmark_identity.py:6`)
  before matching the legacy underscore-pattern regex. Robust to any
  punctuation/separator convention.
