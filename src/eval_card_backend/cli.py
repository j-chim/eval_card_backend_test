import argparse
import json
import sys

from eval_card_backend.config import Settings
from eval_card_backend.sources import benchmark_cards, eee


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-card-backend")
    parser.add_argument(
        "--configs",
        help="Comma-separated EEE config names to load. Default: all discovered.",
    )
    parser.add_argument(
        "--config-limit",
        type=int,
        help="Stop after N configs (smoke-test convenience).",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()

    eee_root = eee.ensure_snapshot(
        settings.eee_local_dir, settings.hf_token, settings.refresh_eee
    )
    cards_root = benchmark_cards.ensure_snapshot(
        settings.benchmark_metadata_local_dir,
        settings.hf_token,
        settings.refresh_benchmark_metadata,
    )

    cards = benchmark_cards.load_cards(cards_root) if cards_root else {}

    configs = eee.discover_configs(eee_root, settings.hf_token)
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",") if c.strip()}
        configs = [c for c in configs if c in wanted]
    if args.config_limit:
        configs = configs[: args.config_limit]

    summary = {
        "eee_root": str(eee_root),
        "benchmark_cards_root": str(cards_root) if cards_root else None,
        "benchmark_card_count": len(cards),
        "config_count": len(configs),
        "configs": [
            {"name": c, "json_files": len(eee.list_json_files(c, eee_root, settings.hf_token))}
            for c in configs
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
