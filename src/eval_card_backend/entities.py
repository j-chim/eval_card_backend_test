"""Entity resolution: canonical IDs for models, developers, benchmarks, metrics.

Composes an upstream resolver (`eval-entity-resolver` from the evalcard-registry
repo) with locally-defined rules. Upstream resolves first; local rules apply on
top, so a local rule wins where it matches.

Local rules live in `_apply_local` below — that's the extension point. Add a
rule when you find a real upstream gap that shouldn't be fixed upstream;
otherwise prefer contributing back to the registry package.

The upstream is typed as a `Protocol` so this module imports nothing from
`eval-entity-resolver` directly — keeps the dependency optional until something
actually instantiates a `Resolver` with a real upstream.
"""

from __future__ import annotations

from typing import Any, Protocol


class UpstreamResolver(Protocol):
    """Subset of `eval-entity-resolver`'s surface this module relies on."""

    def resolve(self, row: dict[str, Any]) -> dict[str, Any]:
        ...


class Resolver:
    def __init__(
        self,
        upstream: UpstreamResolver,
        local_rules: dict[str, Any] | None = None,
    ) -> None:
        self._upstream = upstream
        self._local_rules = local_rules or {}

    def resolve_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return only the canonical columns to merge into the joined frame."""
        canonical = dict(self._upstream.resolve(row))
        canonical.update(self._apply_local(row, canonical))
        return canonical

    def _apply_local(
        self, row: dict[str, Any], canonical: dict[str, Any]
    ) -> dict[str, Any]:
        """Repo-local resolution rules. Override or extend as the spec firms up."""
        return {}
