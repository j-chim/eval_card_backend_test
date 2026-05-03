import os
from dataclasses import dataclass

EEE_DATASET_REPO = "evaleval/EEE_datastore"
BENCHMARK_METADATA_DATASET_REPO = "evaleval/auto-benchmarkcards"

DEFAULT_EEE_LOCAL_DIR = ".cache/eee_datastore"
DEFAULT_BENCHMARK_METADATA_LOCAL_DIR = ".cache/auto_benchmarkcards"


@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    eee_local_dir: str
    benchmark_metadata_local_dir: str
    refresh_eee: bool
    refresh_benchmark_metadata: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            hf_token=os.environ.get("HF_TOKEN"),
            eee_local_dir=os.environ.get("EEE_LOCAL_DATASET_DIR") or DEFAULT_EEE_LOCAL_DIR,
            benchmark_metadata_local_dir=(
                os.environ.get("BENCHMARK_METADATA_LOCAL_DIR")
                or DEFAULT_BENCHMARK_METADATA_LOCAL_DIR
            ),
            refresh_eee=os.environ.get("EEE_REFRESH_SNAPSHOT") == "1",
            refresh_benchmark_metadata=os.environ.get("BENCHMARK_METADATA_REFRESH") == "1",
        )
