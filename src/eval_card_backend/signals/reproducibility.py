"""Per-row reproducibility-gap signal.

Active production rule: temperature + max_tokens for non-agentic; agentic adds
eval_plan + eval_limits. The spec's full 4-field rule (temperature, top_p,
max_tokens, prompt_template) lives as a constant for swap-back purposes; the
fact_results schema stores raw values + has_* flags for all 4 base fields so
restoring it is a single CTAS over the existing parquet.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from eval_card_backend.signals.setup import _coerce_json

log = logging.getLogger(__name__)


SPEC_BASE_REPRODUCIBILITY_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "prompt_template",
)
BASE_REPRODUCIBILITY_FIELDS: tuple[str, ...] = ("temperature", "max_tokens")
AGENTIC_REPRODUCIBILITY_FIELDS: tuple[str, ...] = ("eval_plan", "eval_limits")

AGENTIC_TASK_TOKENS = frozenset({"agentic", "tool_use", "multi_step_agent"})


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalise_benchmark_slug(s: str | None) -> str:
    if not s:
        return ""
    return _NON_ALNUM_RE.sub("_", s.lower()).strip("_")


_AGENTIC_REGEX_PATH = (
    Path(__file__).resolve().parent.parent / "registry" / "agentic_name_regex.json"
)


def _load_agentic_regex() -> re.Pattern[str]:
    data = json.loads(_AGENTIC_REGEX_PATH.read_text(encoding="utf-8"))
    return re.compile(data["pattern"])


AGENTIC_NAME_REGEX = _load_agentic_regex()


def is_agentic_py(
    benchmark_id: str | None,
    benchmark_card: object,
    generation_args: object,
) -> bool:
    """Three-rule union (see notes/01-: 'is_agentic rule').

    Rule 1: card.purpose_and_intended_users.tasks ∩ AGENTIC_TASK_TOKENS.
    Rule 2: generation_args.agentic_eval_config is not None.
    Rule 3: AGENTIC_NAME_REGEX matches the normalised benchmark slug.
    """
    card = _coerce_json(benchmark_card, caller="is_agentic_py.card")
    ga = _coerce_json(generation_args, caller="is_agentic_py.gen_args")

    if isinstance(card, dict):
        purpose = card.get("purpose_and_intended_users") or {}
        tasks = purpose.get("tasks") if isinstance(purpose, dict) else None
        if isinstance(tasks, list):
            for task in tasks:
                if (
                    isinstance(task, str)
                    and task.strip().lower() in AGENTIC_TASK_TOKENS
                ):
                    return True

    if isinstance(ga, dict) and ga.get("agentic_eval_config") is not None:
        return True

    slug = _normalise_benchmark_slug(benchmark_id)
    if slug and AGENTIC_NAME_REGEX.search(slug):
        return True

    return False


def required_repro_fields(is_agentic: bool) -> tuple[str, ...]:
    if is_agentic:
        return BASE_REPRODUCIBILITY_FIELDS + AGENTIC_REPRODUCIBILITY_FIELDS
    return BASE_REPRODUCIBILITY_FIELDS


def compute_repro_missing_py(
    is_agentic: bool,
    has_temperature: bool | None,
    has_max_tokens: bool | None,
    has_eval_plan: bool | None,
    has_eval_limits: bool | None,
) -> list[str]:
    """Return names of active-required reproducibility fields that are absent."""
    presence = {
        "temperature": bool(has_temperature),
        "max_tokens": bool(has_max_tokens),
        "eval_plan": bool(has_eval_plan),
        "eval_limits": bool(has_eval_limits),
    }
    return [f for f in required_repro_fields(bool(is_agentic)) if not presence.get(f)]
