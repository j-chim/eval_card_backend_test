"""Setup-field normalisation, canonical-JSON, variant_key, fact_id.

Same `normalize_setup` is used by `variant_key_py` and by the
divergence-detector's `_differing_setup_fields` so the two stay in sync —
cosmetic differences (whitespace, float-repr noise) collapse identically.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

log = logging.getLogger(__name__)


GENERATION_ARGS_COMPARISON_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "prompt_template",
    "reasoning",
    "agentic_eval_config",
)


_LINE_ENDING_RE = re.compile(r"\r\n|\r")


_malformed_json_counter: Counter[str] = Counter()


def _coerce_json(arg: Any, caller: str = "") -> Any:
    """Parse a JSON-typed UDF param. DuckDB hands JSON params over as VARCHAR strings.

    Pass-through if already a dict/list. Returns None on malformed input and
    increments a per-call-site counter so end-of-run summary surfaces corrupt
    payloads rather than silently degrading.
    """
    if arg is None:
        return None
    if isinstance(arg, str):
        try:
            return json.loads(arg)
        except (ValueError, TypeError):
            _malformed_json_counter[caller] += 1
            return None
    return arg


def reset_json_coerce_counter() -> None:
    _malformed_json_counter.clear()


def log_json_coerce_summary() -> None:
    if _malformed_json_counter:
        log.warning("--- malformed JSON coercion ---")
        for caller, count in _malformed_json_counter.most_common():
            log.warning(
                "  %s: %d rows had malformed JSON (treated as missing)", caller, count
            )


def canonical_json(obj: Any) -> str | None:
    """Stable canonical-JSON serialisation. None passes through as None."""
    if obj is None:
        return None
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _norm_num(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    try:
        return float(f"{float(v):.8g}")
    except (ValueError, TypeError):
        return v


def _norm_int(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    try:
        return int(v)
    except (ValueError, TypeError):
        return v


def _norm_text(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    return _LINE_ENDING_RE.sub("\n", v).strip()


def _norm_bool(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "t"):
            return True
        if s in ("false", "0", "no", "f", ""):
            return False
    return v


def normalize_setup(generation_args: Any) -> dict:
    """Normalised dict over the seven comparison fields. Always returns a dict
    with all seven keys; values may be None.

    Robust to dict OR JSON string input (matches UDF call-site convention).
    """
    ga = _coerce_json(generation_args, caller="normalize_setup")
    ga = ga if isinstance(ga, dict) else {}
    return {
        "temperature": _norm_num(ga.get("temperature")),
        "top_p": _norm_num(ga.get("top_p")),
        "top_k": _norm_num(ga.get("top_k")),
        "max_tokens": _norm_int(ga.get("max_tokens")),
        "prompt_template": _norm_text(ga.get("prompt_template")),
        "reasoning": _norm_bool(ga.get("reasoning")),
        # agentic_eval_config goes through canonical_json downstream when needed
        "agentic_eval_config": ga.get("agentic_eval_config"),
    }


def setup_canonical_json(generation_args: Any) -> str:
    return canonical_json(normalize_setup(generation_args))


def variant_key_py(generation_args: Any) -> str:
    """First 16 hex chars of sha256(setup_canonical_json)."""
    s = setup_canonical_json(generation_args)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def fact_id_py(evaluation_id: str | None, result_idx: int | None) -> str | None:
    """First 16 hex chars of sha256(f'{evaluation_id}:{result_idx}').

    Matches the registry's `eval_results.id` formula. Returns None on missing
    evaluation_id (we can't synthesise a deterministic id without one).
    """
    if not evaluation_id:
        return None
    if result_idx is None:
        result_idx = 0
    payload = f"{evaluation_id}:{result_idx}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def differing_setup_fields(setups: list[Any]) -> list[dict]:
    """For each comparison field, if canonical-JSON values differ across `setups`,
    record the field with the unique original values (first-seen order, deduped
    by canonical form).

    Used in both variant- and cross-party-divergence detection.
    """
    differing: list[dict] = []
    for field in GENERATION_ARGS_COMPARISON_FIELDS:
        seen_canon: set[str] = set()
        original_values: list[Any] = []
        for setup in setups:
            value = setup.get(field) if isinstance(setup, dict) else None
            canon = canonical_json(value) if value is not None else "null"
            if canon not in seen_canon:
                seen_canon.add(canon)
                original_values.append(value)
        if len(seen_canon) > 1:
            # Mirror DDL: STRUCT(field VARCHAR, "values" JSON)[]
            # JSON-encode the original values for the "values" field.
            differing.append(
                {
                    "field": field,
                    "values": json.dumps(
                        original_values, ensure_ascii=False, default=str
                    ),
                }
            )
    return differing
