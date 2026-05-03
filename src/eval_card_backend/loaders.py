"""Config-by-config streaming over EEE records.

Public surface:

    iter_config_results(config, eee_root, hf_token=None) -> Iterator[EvalResult]
        Stream one EvalResult per (record × evaluation_result entry) for a single
        config. Resilient: read/parse failures on individual files are logged
        and skipped.

    for_each_config(configs, handler, eee_root, hf_token=None,
                    parallel=False, max_workers=None) -> Iterator[T]
        Run `handler(config, results)` once per config. Yields handler results in
        input order. With parallel=True, runs configs across processes —
        `handler` must be importable (top-level def, not a lambda or closure).

Each `EvalResult` is a flat TypedDict with arrow-friendly scalar types plus
`raw_evaluation_result`, the original nested payload preserved verbatim.

`raw_evaluation_result` is a stable column, not scaffolding — it's the schema
escape hatch for fields that haven't been promoted to top-level columns yet.
Downstream extractors may pull from it; consumers that don't need it can drop
it via `pipeline.run(..., drop_raw=True)`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypedDict, TypeVar

from eval_card_backend.sources import eee

log = logging.getLogger(__name__)

T = TypeVar("T")


class EvalResult(TypedDict):
    config: str
    record_path: str
    evaluation_id: str | None
    schema_version: str | None
    retrieved_timestamp: str | None
    developer: str | None
    model_name: str | None
    model_id: str | None
    inference_platform: str | None
    source_name: str | None
    source_type: str | None
    evaluator_relationship: str | None
    evaluation_name: str | None
    dataset_name: str | None
    benchmark_key: str
    metric_id: str | None
    metric_name: str | None
    metric_unit: str | None
    metric_kind: str | None
    lower_is_better: bool | None
    score: float | None
    hf_repo: str | None
    raw_evaluation_result: dict[str, Any]


def normalize_key(value: Any) -> str:
    """Lowercase + non-alphanumerics → `_`. Used as the join key for benchmark cards."""
    text = str(value or "").strip().lower()
    out: list[str] = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_record(
    config: str, record_path: str, record: Any
) -> Iterator[EvalResult]:
    if not isinstance(record, dict):
        return
    model = _as_dict(record.get("model_info"))
    src = _as_dict(record.get("source_metadata"))
    base = {
        "config": config,
        "record_path": record_path,
        "evaluation_id": record.get("evaluation_id"),
        "schema_version": record.get("schema_version"),
        "retrieved_timestamp": record.get("retrieved_timestamp"),
        "developer": model.get("developer"),
        "model_name": model.get("name"),
        "model_id": model.get("id"),
        "inference_platform": model.get("inference_platform"),
        "source_name": src.get("source_name"),
        "source_type": src.get("source_type"),
        "evaluator_relationship": src.get("evaluator_relationship"),
    }

    results = record.get("evaluation_results")
    if not isinstance(results, list):
        return

    for er in results:
        if not isinstance(er, dict):
            continue
        metric = _as_dict(er.get("metric_config"))
        source_data = _as_dict(er.get("source_data"))
        score_details = _as_dict(er.get("score_details"))

        eval_name = er.get("evaluation_name")
        dataset_name = source_data.get("dataset_name")
        lower = metric.get("lower_is_better")

        yield EvalResult(
            **base,
            evaluation_name=eval_name,
            dataset_name=dataset_name,
            benchmark_key=normalize_key(eval_name) or normalize_key(dataset_name),
            metric_id=metric.get("metric_id"),
            metric_name=metric.get("metric_name"),
            metric_unit=metric.get("metric_unit"),
            metric_kind=metric.get("metric_kind"),
            lower_is_better=lower if isinstance(lower, bool) else None,
            score=_coerce_score(score_details.get("score")),
            hf_repo=source_data.get("hf_repo"),
            raw_evaluation_result=er,
        )


def iter_config_results(
    config: str,
    eee_root: Path | None,
    hf_token: str | None = None,
) -> Iterator[EvalResult]:
    for path in eee.list_json_files(config, eee_root, hf_token):
        try:
            record = eee.read_record(path, eee_root, hf_token)
        except Exception as exc:
            log.warning("loaders: failed to read %s: %s", path, exc)
            continue
        try:
            yield from _parse_record(config, path, record)
        except Exception as exc:
            log.warning("loaders: failed to parse %s: %s", path, exc)


def _worker(args: tuple[str, Callable[..., T], Path | None, str | None]) -> T:
    config, handler, eee_root, hf_token = args
    return handler(config, iter_config_results(config, eee_root, hf_token))


def for_each_config(
    configs: Iterable[str],
    handler: Callable[[str, Iterator[EvalResult]], T],
    eee_root: Path | None,
    hf_token: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Iterator[T]:
    configs = list(configs)
    if not parallel:
        for cfg in configs:
            yield handler(cfg, iter_config_results(cfg, eee_root, hf_token))
        return

    args = [(cfg, handler, eee_root, hf_token) for cfg in configs]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        yield from pool.map(_worker, args)
