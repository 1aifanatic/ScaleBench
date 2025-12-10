"""CLI entrypoint: loads config, runs experiments."""
import argparse
import sys
import logging
from pathlib import Path

from core.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmarking experiments for ScaleDown context compression."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments.yaml"),
        help="Path to experiments.yaml config file (default: config/experiments.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and validate without running actual experiments.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Run only the specified experiment ID (optional).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading config from {args.config}")

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(config.experiments)} experiment(s)")

    # Filter experiments if --experiment-id is specified
    experiments_to_run = config.experiments
    if args.experiment_id:
        experiments_to_run = [
            exp for exp in config.experiments if exp.id == args.experiment_id
        ]
        if not experiments_to_run:
            logger.error(
                f"Experiment '{args.experiment_id}' not found in config. "
                f"Available: {[e.id for e in config.experiments]}"
            )
            sys.exit(1)
        logger.info(f"Filtered to experiment: {args.experiment_id}")

    # Print experiment summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    for i, exp in enumerate(experiments_to_run, 1):
        logger.info(f"\n[{i}] {exp.id}")
        logger.info(f"    Model:    {exp.model.name} (mode={exp.model.mode})")
        logger.info(f"    Dataset:  {exp.dataset.name} (split={exp.dataset.split}, num_examples={exp.dataset.num_examples})")
        logger.info(f"    Task:     {exp.task.name}")
        logger.info(f"    Metrics:  {', '.join(exp.metrics) if exp.metrics else 'none specified'}")
    logger.info("\n" + "=" * 70)

    if args.dry_run:
        logger.info("DRY RUN: Config validated successfully. No experiments executed.")
        sys.exit(0)

    # TODO: Actual experiment execution will be implemented later
    logger.info("Experiment execution not yet implemented.")
    sys.exit(0)


if __name__ == "__main__":
    main()
