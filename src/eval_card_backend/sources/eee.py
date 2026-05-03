"""Snapshot and read access for `evaleval/EEE_datastore`.

Layout on disk after snapshot:
    <local_dir>/data/<config>/<dev>/<model>/<uuid>.json
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

from eval_card_backend.config import EEE_DATASET_REPO


def ensure_snapshot(local_dir: str, hf_token: str | None, force_refresh: bool) -> Path:
    target = Path(local_dir).resolve()
    if force_refresh and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    data_dir = target / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        return target

    snapshot_download(
        repo_id=EEE_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=["data/**"],
        token=hf_token,
    )
    return target


def discover_configs(local_dir: Path | None, hf_token: str | None) -> list[str]:
    if local_dir is not None:
        data_root = Path(local_dir) / "data"
        if not data_root.exists():
            return []
        return sorted(p.name for p in data_root.iterdir() if p.is_dir())

    fs = HfFileSystem(token=hf_token)
    entries = fs.ls(f"datasets/{EEE_DATASET_REPO}/data", detail=True)
    return sorted({entry["name"].split("/")[-1] for entry in entries if entry.get("name")})


def list_json_files(
    config: str, local_dir: Path | None, hf_token: str | None
) -> list[str]:
    """Return repo-relative JSON paths (e.g. `data/<config>/.../record.json`)."""
    if local_dir is not None:
        root = Path(local_dir) / "data" / config
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(local_dir)).replace(os.sep, "/")
            for p in root.rglob("*.json")
            if p.is_file() and not p.name.endswith(".jsonl")
        )

    fs = HfFileSystem(token=hf_token)
    pattern = f"datasets/{EEE_DATASET_REPO}/data/{config}/**/*.json"
    prefix = f"datasets/{EEE_DATASET_REPO}/"
    return sorted(
        p[len(prefix):] for p in fs.glob(pattern) if not p.endswith(".jsonl")
    )


def read_record(
    dataset_path: str, local_dir: Path | None, hf_token: str | None
) -> dict[str, Any]:
    if local_dir is not None:
        return json.loads((Path(local_dir) / dataset_path).read_text(encoding="utf-8"))

    cached = hf_hub_download(
        repo_id=EEE_DATASET_REPO,
        filename=dataset_path,
        repo_type="dataset",
        token=hf_token,
    )
    return json.loads(Path(cached).read_text(encoding="utf-8"))


def iter_records(
    config: str, local_dir: Path | None, hf_token: str | None
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in list_json_files(config, local_dir, hf_token):
        yield path, read_record(path, local_dir, hf_token)
