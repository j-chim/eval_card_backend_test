"""Pure interpretive-signal logic. I/O-free; safe to unit test in isolation.

Modules:
- reproducibility: per-row reproducibility-gap rule + active field set.
- provenance: per-row source-type collapse + group-derived multi-source / first-party-only.
- comparability: variant-divergence + cross-party-divergence over a group.
- completeness: per-benchmark reporting completeness over the operationalised field set.
- setup: setup-field normalisation + canonical-JSON helpers (variant_key + differing-field source).
"""
