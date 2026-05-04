"""Composite / family / slice taxonomy loader.

Reads three curated YAML files from the registry seed directory and
materialises three small in-memory tables on the DuckDB connection:

- `composite_config_map(source_config, composite_slug, composite_display_name)`
- `family_membership(family_id, family_display_name, benchmark_id)`
- `slice_promotions(benchmark_id)`

The YAMLs live in `evalcard-registry/seed/`:
- `composites.yaml`  — leaderboard slug → list of EEE source_configs
- `families.yaml`    — multi-benchmark family slug → list of benchmark ids
- `slice_overrides.yaml` — `promote_to_benchmark: [...]`

Phase 1 (this module) reads from a configurable filesystem path. Phase 2
will move these into the registry's parquet dim tables (see
`notes/09-…` §9). The path defaults to `<registry_local_dir>/../seed/`,
which is correct for the workspace layout where eval-card-backend and
evalcard-registry are sibling clones. Missing files are tolerated: an
absent YAML produces an empty table (no curated entries → all defaults).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


_KEBAB_RE = re.compile(r"[^a-z0-9-]+")


def kebab_case(s: str) -> str:
    """Lowercase, replace non-alphanumeric runs with `-`, strip leading/
    trailing `-`. Used as the default composite_slug derivation when a
    source_config isn't curated.
    """
    s = s.lower().replace("_", "-")
    s = _KEBAB_RE.sub("-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _load_yaml(path: Path) -> dict | None:
    """Parse a YAML file, returning the top-level mapping or None when
    the file is absent. Returns None on parse failure with a warning;
    callers treat None and {} the same (no curated entries).
    """
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        log.warning(
            "PyYAML not installed; cannot read taxonomy seed at %s. "
            "Install with `uv add pyyaml` or skip curated taxonomy.", path,
        )
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        log.warning("taxonomy: failed to parse %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        log.warning(
            "taxonomy: expected top-level mapping in %s, got %s",
            path, type(data).__name__,
        )
        return None
    return data


def resolve_seed_dir(registry_root: Path | None, override: Path | None = None) -> Path | None:
    """Return the directory holding the taxonomy YAMLs.

    Resolution order:
      1. `override` if supplied (test injection / explicit path).
      2. `EVALCARD_REGISTRY_SEED_DIR` env var if set.
      3. Sibling of `registry_root`: `<registry_root>/../seed/` —
         correct when eval-card-backend and eval-card-registry are
         checked out side-by-side via the uv workspace dep.

    Returns None when no candidate exists.
    """
    import os

    if override is not None:
        return Path(override) if Path(override).exists() else None
    env = os.environ.get("EVALCARD_REGISTRY_SEED_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
    if registry_root is not None:
        # registry_root is typically `<repo>/.cache/registry/` — the seed
        # YAMLs live next to the registry source repo, not in the data
        # cache. Walk up to find a sibling `eval-card-registry/seed/`.
        for ancestor in [registry_root, *registry_root.parents]:
            candidate = ancestor / "eval-card-registry" / "seed"
            if candidate.exists():
                return candidate
            candidate = ancestor / "evalcard-registry" / "seed"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Loaders — return plain Python data; the SQL materialiser is below.
# ---------------------------------------------------------------------------


def load_composites(seed_dir: Path) -> dict[str, dict]:
    """Return `{composite_slug: {display: str, configs: [str, ...]}}`.

    `configs` is normalised to a list. Default-display compositional
    entries (key = slug, value = {display: ...} only) get
    `configs = [slug]` so a 1:1 mapping still produces a usable map.
    """
    data = _load_yaml(seed_dir / "composites.yaml")
    if not data:
        return {}
    out: dict[str, dict] = {}
    for slug, entry in data.items():
        if not isinstance(slug, str):
            continue
        if not isinstance(entry, dict):
            log.warning("composites.yaml: entry %r is not a mapping; skipping", slug)
            continue
        display = entry.get("display") or slug
        configs = entry.get("configs")
        if configs is None:
            # Display-only override: implicit single config = the slug
            # itself (e.g. helm-classic → [helm_classic]).
            configs = [slug.replace("-", "_")]
        if not isinstance(configs, list):
            log.warning(
                "composites.yaml: %r.configs must be a list, got %s",
                slug, type(configs).__name__,
            )
            continue
        out[slug] = {"display": str(display), "configs": [str(c) for c in configs]}
    return out


def load_families(seed_dir: Path) -> dict[str, dict]:
    """Return `{family_id: {display: str, benchmarks: [str, ...]}}`.

    Validation: a benchmark must appear in at most one family. Raises
    `ValueError` on conflict — the operator must fix the YAML before the
    pipeline can run.
    """
    data = _load_yaml(seed_dir / "families.yaml")
    if not data:
        return {}
    out: dict[str, dict] = {}
    seen_benchmarks: dict[str, str] = {}
    for fid, entry in data.items():
        if not isinstance(fid, str):
            continue
        if not isinstance(entry, dict):
            log.warning("families.yaml: entry %r is not a mapping; skipping", fid)
            continue
        display = entry.get("display") or fid
        benchmarks = entry.get("benchmarks") or []
        if not isinstance(benchmarks, list):
            log.warning(
                "families.yaml: %r.benchmarks must be a list, got %s",
                fid, type(benchmarks).__name__,
            )
            continue
        bench_ids = [str(b) for b in benchmarks]
        for bid in bench_ids:
            prior = seen_benchmarks.get(bid)
            if prior is not None and prior != fid:
                raise ValueError(
                    f"families.yaml: benchmark {bid!r} appears in both "
                    f"{prior!r} and {fid!r} — a benchmark can belong to at "
                    f"most one curated family."
                )
            seen_benchmarks[bid] = fid
        out[fid] = {"display": str(display), "benchmarks": bench_ids}
    return out


def load_slice_promotions(seed_dir: Path) -> set[str]:
    """Return the set of benchmark ids in `promote_to_benchmark`."""
    data = _load_yaml(seed_dir / "slice_overrides.yaml")
    if not data:
        return set()
    promote = data.get("promote_to_benchmark") or []
    if not isinstance(promote, list):
        log.warning(
            "slice_overrides.yaml: promote_to_benchmark must be a list, "
            "got %s", type(promote).__name__,
        )
        return set()
    return {str(b) for b in promote}


# ---------------------------------------------------------------------------
# DuckDB materialisation
# ---------------------------------------------------------------------------


def materialise_taxonomy_tables(
    con,
    composites: dict[str, dict],
    families: dict[str, dict],
    slice_promotions: set[str],
) -> None:
    """Create three small tables on the connection.

    Validation: if any source_config appears in two composites,
    raise — Stage D would otherwise pick an arbitrary one.
    """
    # Validate composites: each source_config in at most one composite.
    seen_configs: dict[str, str] = {}
    for slug, entry in composites.items():
        for cfg in entry["configs"]:
            prior = seen_configs.get(cfg)
            if prior is not None and prior != slug:
                raise ValueError(
                    f"composites.yaml: source_config {cfg!r} appears in "
                    f"both {prior!r} and {slug!r}."
                )
            seen_configs[cfg] = slug

    con.execute(
        "CREATE OR REPLACE TABLE composite_config_map ("
        "source_config VARCHAR, composite_slug VARCHAR, "
        "composite_display_name VARCHAR)"
    )
    composite_rows: list[tuple[str, str, str]] = []
    for slug, entry in composites.items():
        for cfg in entry["configs"]:
            composite_rows.append((cfg, slug, entry["display"]))
    if composite_rows:
        con.executemany(
            "INSERT INTO composite_config_map VALUES (?, ?, ?)",
            composite_rows,
        )

    con.execute(
        "CREATE OR REPLACE TABLE family_membership ("
        "family_id VARCHAR, family_display_name VARCHAR, benchmark_id VARCHAR)"
    )
    family_rows: list[tuple[str, str, str]] = []
    for fid, entry in families.items():
        for bid in entry["benchmarks"]:
            family_rows.append((fid, entry["display"], bid))
    if family_rows:
        con.executemany(
            "INSERT INTO family_membership VALUES (?, ?, ?)",
            family_rows,
        )

    con.execute(
        "CREATE OR REPLACE TABLE slice_promotions (benchmark_id VARCHAR)"
    )
    if slice_promotions:
        con.executemany(
            "INSERT INTO slice_promotions VALUES (?)",
            [(b,) for b in sorted(slice_promotions)],
        )


def load_and_materialise(
    con,
    registry_root: Path | None,
    seed_dir_override: Path | None = None,
) -> tuple[dict[str, dict], dict[str, dict], set[str]]:
    """One-shot helper: resolve seed dir, load YAMLs, materialise tables.

    Returns the parsed data so Stage A's slice-grouping pass can see the
    promotion set without re-reading the file.
    """
    seed_dir = resolve_seed_dir(registry_root, seed_dir_override)
    if seed_dir is None:
        log.info(
            "taxonomy: no seed dir found (override=%s, registry_root=%s); "
            "all composites/families/slices use defaults.",
            seed_dir_override, registry_root,
        )
        composites: dict[str, dict] = {}
        families: dict[str, dict] = {}
        promotions: set[str] = set()
    else:
        log.info("taxonomy: loading seed YAMLs from %s", seed_dir)
        composites = load_composites(seed_dir)
        families = load_families(seed_dir)
        promotions = load_slice_promotions(seed_dir)
        log.info(
            "taxonomy: %d composite(s), %d curated family(ies), "
            "%d slice promotion(s)",
            len(composites), len(families), len(promotions),
        )
    materialise_taxonomy_tables(con, composites, families, promotions)
    return composites, families, promotions
