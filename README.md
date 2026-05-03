# eval-card-backend

Loads upstream data from `evaleval/EEE_datastore` and `evaleval/auto-benchmarkcards`. Downstream pipeline stages (normalization, artifact emission, upload) are not implemented yet.

## Install

```bash
uv sync
```

## Use

Fetch snapshots and report what was loaded:

```bash
uv run eval-card-backend
```

Limit to specific configs:

```bash
uv run eval-card-backend --configs cnn_dailymail,xsum
```

## Environment variables

- `HF_TOKEN` — required for private datasets; optional for public ones.
- `EEE_LOCAL_DATASET_DIR` — local cache dir for EEE snapshot. Default: `.cache/eee_datastore`.
- `BENCHMARK_METADATA_LOCAL_DIR` — local cache dir for benchmark cards. Default: `.cache/auto_benchmarkcards`.
- `EEE_REFRESH_SNAPSHOT=1` — force re-download of EEE snapshot.
- `BENCHMARK_METADATA_REFRESH=1` — force re-download of benchmark cards.
