import argparse
import json
import logging
import sys

from eval_card_backend.config import Settings
from eval_card_backend.sources import benchmark_cards, eee


def _cmd_summary(args: argparse.Namespace) -> int:
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


def _cmd_canonicalise(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet down some noisy loggers from upstream deps unless caller asks otherwise
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    from eval_card_backend.canonicalise import pipeline

    settings = Settings.from_env()

    configs = None
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    out_dir = pipeline.run(
        settings,
        configs=configs,
        config_limit=args.config_limit,
        snapshot_id=args.snapshot_id,
        warehouse_dir=args.warehouse,
        registry_local_dir=args.registry_local_dir,
        skip_preflight=args.skip_preflight,
    )
    print(f"\nWrote snapshot to: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-card-backend")
    subs = parser.add_subparsers(dest="command")

    # Default behaviour: summary (preserves existing CLI shape).
    parser.add_argument(
        "--configs",
        help="Comma-separated EEE config names to load. Default: all discovered.",
    )
    parser.add_argument(
        "--config-limit",
        type=int,
        help="Stop after N configs (smoke-test convenience).",
    )

    canon = subs.add_parser(
        "canonicalise", help="Run the canonicalisation pipeline end-to-end."
    )
    canon.add_argument(
        "--configs",
        help="Comma-separated EEE config names to load.",
    )
    canon.add_argument(
        "--config-limit", type=int,
        help="Stop after N configs (smoke-test convenience).",
    )
    canon.add_argument(
        "--snapshot-id",
        help="ISO timestamp; default: now (UTC).",
    )
    canon.add_argument(
        "--warehouse", default="warehouse",
        help="Output warehouse root directory.",
    )
    canon.add_argument(
        "--registry-local-dir", default=".cache/entity_registry",
        help="Local cache dir for the evaleval/entity-registry-data dataset.",
    )
    canon.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip preflight checks (use only for diagnostic runs).",
    )

    args = parser.parse_args(argv)

    if args.command == "canonicalise":
        return _cmd_canonicalise(args)
    return _cmd_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
