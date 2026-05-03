from eval_card_backend.loaders import EvalResult, _parse_record


def test_parse_record_happy_path():
    record = {
        "evaluation_id": "ace/foo_bar/123",
        "schema_version": "0.2.2",
        "retrieved_timestamp": "123",
        "model_info": {
            "developer": "foo",
            "name": "Bar",
            "id": "foo/Bar",
            "inference_platform": "unknown",
        },
        "source_metadata": {
            "source_name": "Mercor ACE",
            "source_type": "evaluation_run",
            "evaluator_relationship": "first_party",
        },
        "evaluation_results": [
            {
                "evaluation_name": "ACE",
                "source_data": {"dataset_name": "ace", "hf_repo": "Mercor/ACE"},
                "metric_config": {
                    "metric_id": "ace.score",
                    "metric_name": "Score",
                    "metric_unit": "proportion",
                    "metric_kind": "score",
                    "lower_is_better": False,
                },
                "score_details": {"score": 0.4},
            },
            {
                "evaluation_name": "Gaming",
                "source_data": {"dataset_name": "ace"},
                "metric_config": {"metric_id": "ace.gaming"},
                "score_details": {"score": "0.61"},
            },
        ],
    }

    results = list(_parse_record("ace", "data/ace/foo/bar/x.json", record))

    assert len(results) == 2
    assert set(results[0].keys()) == set(EvalResult.__annotations__.keys())
    assert results[0]["benchmark_key"] == "ace"
    assert results[0]["score"] == 0.4
    assert results[0]["lower_is_better"] is False
    assert results[0]["developer"] == "foo"
    assert results[0]["raw_evaluation_result"] is record["evaluation_results"][0]

    assert results[1]["benchmark_key"] == "gaming"
    assert results[1]["score"] == 0.61  # string coerced to float


def test_parse_record_resilience():
    # results=None: yields nothing, doesn't crash
    assert list(_parse_record("c", "p", {"evaluation_results": None})) == []

    # entry is a string instead of a dict: skipped silently
    record = {
        "model_info": {"developer": "foo"},
        "evaluation_results": ["junk", {"evaluation_name": "ok"}],
    }
    results = list(_parse_record("c", "p", record))
    assert len(results) == 1
    assert results[0]["evaluation_name"] == "ok"

    # whole record is not a dict: yields nothing
    assert list(_parse_record("c", "p", "not a dict")) == []
