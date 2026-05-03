"""End-to-end load: EEE eval results joined to auto-benchmarkcards.

`run()` is a generator yielding one `(config, DataFrame)` tuple per EEE config.
Memory stays bounded to one config at a time; the caller decides whether to
concat for in-memory work or stream-write per config.

Per-config pipeline order:

    list_json_files → read_record → parse → join cards
        → [resolve entities]                 # if `resolver` is provided
        → [extract from raw_evaluation_result]   # future stage; not implemented
        → optional drop of raw_evaluation_result
        → yield (config, frame)

The `extract from raw` step is deliberately a comment, not a parameter — the
seam is between `resolver` and `drop_raw`. Promote to a real parameter when a
caller actually needs it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path

import pandas as pd

from eval_card_backend.entities import Resolver
from eval_card_backend.loaders import EvalResult, for_each_config
from eval_card_backend.sources import benchmark_cards, eee


def _cards_to_df(cards: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, card in cards.items():
        details = card.get("benchmark_details") or {}
        purpose = card.get("purpose_and_intended_users") or {}
        rows.append(
            {
                "benchmark_key": key,
                "card_benchmark_name": details.get("name"),
                "card_languages": details.get("languages"),
                "card_domains": details.get("domains"),
                "card_appears_in": details.get("appears_in"),
                "card_tasks": purpose.get("tasks"),
            }
        )
    return pd.DataFrame(rows)


def _join_config(
    config: str, results: Iterator[EvalResult], cards_df: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.merge(cards_df, on="benchmark_key", how="left")
    return config, df


def _apply_resolver(df: pd.DataFrame, resolver: Resolver) -> pd.DataFrame:
    # Row-wise via to_dict('records') — faster than df.apply(axis=1) and the
    # right shape for resolution rules that consider multiple columns together.
    additions = pd.DataFrame(
        [resolver.resolve_row(r) for r in df.to_dict("records")]
    )
    if additions.empty:
        return df
    return pd.concat(
        [df.reset_index(drop=True), additions.reset_index(drop=True)], axis=1
    )


def run(
    eee_root: Path,
    cards_root: Path,
    *,
    configs: Iterable[str] | None = None,
    hf_token: str | None = None,
    resolver: Resolver | None = None,
    drop_raw: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    cards_df = _cards_to_df(benchmark_cards.load_cards(cards_root))
    if configs is None:
        configs = eee.discover_configs(eee_root, hf_token)

    handler = partial(_join_config, cards_df=cards_df)
    for cfg, df in for_each_config(
        configs,
        handler,
        eee_root,
        hf_token=hf_token,
        parallel=parallel,
        max_workers=max_workers,
    ):
        if df.empty:
            yield cfg, df
            continue
        if resolver is not None:
            df = _apply_resolver(df, resolver)
        if drop_raw and "raw_evaluation_result" in df.columns:
            df = df.drop(columns=["raw_evaluation_result"])
        yield cfg, df
