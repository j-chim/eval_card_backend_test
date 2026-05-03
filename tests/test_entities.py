from eval_card_backend.entities import Resolver


class _StubUpstream:
    def resolve(self, row):
        return {"canonical_model_id": "upstream/model", "canonical_developer": "up"}


class _LocalOverrideResolver(Resolver):
    def _apply_local(self, row, canonical):
        # Override developer locally; leave model id to upstream.
        return {"canonical_developer": "local"}


def test_resolver_local_wins_over_upstream():
    r = _LocalOverrideResolver(_StubUpstream())
    out = r.resolve_row({"developer": "anthropic"})
    assert out["canonical_model_id"] == "upstream/model"  # untouched by local
    assert out["canonical_developer"] == "local"          # local override applied
