"""Snapshot and read access for `evaleval/entity-registry-data`.

Provides:
  - `ensure_snapshot(local_dir, hf_token, force_refresh)`: download the registry
    parquets to local cache.
  - `load_alias_store(root)`: build an `AliasStore` for the resolver from the
    local cache.
  - `open_dim_paths(root)`: map of canonical_* table -> path (for DuckDB
    `read_parquet`).

Layout on disk after snapshot (the registry's HF dataset shape):
    <local_dir>/aliases/part-0.parquet
    <local_dir>/canonical_orgs/part-0.parquet
    <local_dir>/canonical_models/part-0.parquet
    <local_dir>/canonical_benchmarks/part-0.parquet
    <local_dir>/canonical_metrics/part-0.parquet
    <local_dir>/eval_harnesses/part-0.parquet
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

ENTITY_REGISTRY_DATASET_REPO = "evaleval/entity-registry-data"

DIM_TABLES: tuple[str, ...] = (
    "canonical_orgs",
    "canonical_models",
    "canonical_benchmarks",
    "canonical_metrics",
    "eval_harnesses",
)

ALIASES_TABLE = "aliases"
ALL_TABLES: tuple[str, ...] = DIM_TABLES + (ALIASES_TABLE,)


def _has_registry_data(target: Path) -> bool:
    return any((target / table).exists() for table in ALL_TABLES)


def ensure_snapshot(
    local_dir: str, hf_token: str | None, force_refresh: bool
) -> Path:
    target = Path(local_dir).resolve()
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if _has_registry_data(target):
        return target

    try:
        snapshot_download(
            repo_id=ENTITY_REGISTRY_DATASET_REPO,
            repo_type="dataset",
            local_dir=str(target),
            token=hf_token,
        )
    except Exception as exc:
        log.warning(
            "registry.ensure_snapshot: HF download failed (%s: %s); "
            "falling back to local-only mode at %s",
            type(exc).__name__,
            exc,
            target,
        )
    return target


def _resolve_table_path(root: Path, table: str) -> Path | None:
    """Return the parquet path for `table`, supporting either a single-file
    layout (`<table>.parquet`) or HF's directory layout (`<table>/part-*.parquet`).
    """
    direct = root / f"{table}.parquet"
    if direct.exists():
        return direct
    table_dir = root / table
    if table_dir.is_dir():
        parts = sorted(table_dir.glob("*.parquet"))
        if parts:
            return parts[0] if len(parts) == 1 else table_dir
    return None


def open_dim_paths(root: Path) -> dict[str, Path]:
    """Return {table: path} for every dim table that's present.

    Path is either a single parquet file or a directory of `part-*.parquet`
    files. DuckDB's `read_parquet` accepts either form (use a glob if it's a
    directory).
    """
    out: dict[str, Path] = {}
    for table in DIM_TABLES:
        path = _resolve_table_path(root, table)
        if path is not None:
            out[table] = path
    return out


def aliases_path(root: Path) -> Path | None:
    return _resolve_table_path(root, ALIASES_TABLE)


def load_alias_store(root: Path):
    """Build a read-only `AliasStore` from the local cache.

    AliasStore.from_parquet reads `<root>/aliases.parquet` directly. The HF
    layout is `<root>/aliases/part-0.parquet`, so we adapt by passing the
    parent of the part file (renamed once to the layout the resolver expects).
    """
    from eval_entity_resolver import AliasStore

    direct = root / "aliases.parquet"
    if direct.exists():
        return AliasStore.from_parquet(root, read_only=True)

    table_dir = root / "aliases"
    if table_dir.is_dir():
        parts = sorted(table_dir.glob("*.parquet"))
        if parts:
            # AliasStore.from_parquet expects `<dir>/aliases.parquet`. Materialise
            # a small symlink so we don't copy the data.
            link = root / "aliases.parquet"
            if not link.exists():
                try:
                    link.symlink_to(parts[0])
                except OSError:
                    # Symlinks may be unsupported (e.g. some Windows filesystems);
                    # fall back to a hardlink-or-copy.
                    import os
                    try:
                        os.link(parts[0], link)
                    except OSError:
                        shutil.copy2(parts[0], link)
            return AliasStore.from_parquet(root, read_only=True)

    log.warning(
        "registry.load_alias_store: no aliases.parquet at %s — "
        "returning empty alias store", root,
    )
    return AliasStore.from_parquet(root, read_only=True)
