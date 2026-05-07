"""Run distributed negotiation simulation and print aggregated metrics."""
from __future__ import annotations

import argparse
import logging
import random
import yaml

from environment.distributed_sim import DistributedSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def make_scenario(ep_id: int) -> dict:
    random.seed(ep_id)
    reserve = random.uniform(800, 1000)
    target  = reserve * random.uniform(1.1, 1.4)
    return {
        "seller_ask":    target,
        "buyer_bid":     reserve,
        "reserve_price": reserve,
        "target_price":  target,
    }


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"].copy()
    if args.workers:
        sim_cfg["n_workers"] = args.workers
    sim_cfg["negotiation"] = cfg["negotiation"]

    sim = DistributedSimulator(sim_cfg)
    sim.init_workers()

    logger.info("Running %d episodes across %d workers", args.episodes, sim.n_workers)
    results = sim.run_episodes_sync(args.episodes, make_scenario)

    metrics = sim.aggregate_metrics(results)
    logger.info("=== Simulation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %-35s %.4f", k, v)
        else:
            logger.info("  %-35s %s", k, v)

    sim.shutdown()


if __name__ == "__main__":
    main()