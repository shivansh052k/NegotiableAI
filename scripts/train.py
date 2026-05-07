"""Entry point for training: PPO self-play followed by DQN bootstrap."""
from __future__ import annotations

import argparse
import logging
import yaml
import torch

from training.ppo_selfplay import SelfPlayPPOTrainer
from agents.dqn_strategy import DQNStrategyAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = cfg["training"]["ppo"]
    dqn_cfg = cfg["negotiation"]["dqn"]
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    save_every = cfg["training"].get("save_every_n_episodes", 500)
    total_episodes = args.episodes or cfg["simulation"]["total_episodes"]

    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting PPO self-play for %d episodes", total_episodes)
    ppo = SelfPlayPPOTrainer(ppo_cfg)

    for ep in range(total_episodes):
        # Placeholder rollout — replace with real env interaction
        batch_size = ppo_cfg.get("batch_size", 64)
        obs_dim = ppo_cfg["obs_dim"]
        n_actions = ppo_cfg["n_actions"]

        import torch
        from training.ppo_selfplay import PPOBatch
        dummy_batch = PPOBatch(
            states=torch.zeros(batch_size, obs_dim),
            actions=torch.zeros(batch_size, dtype=torch.long),
            log_probs=torch.zeros(batch_size),
            rewards=torch.zeros(batch_size),
            values=torch.zeros(batch_size),
            dones=torch.zeros(batch_size),
        )
        stats = ppo.update(dummy_batch)

        if (ep + 1) % save_every == 0:
            path = f"{checkpoint_dir}/ppo_ep{ep+1}.pt"
            ppo.save(path)
            logger.info("ep=%d  policy_loss=%.4f  saved=%s", ep + 1, stats["policy_loss"], path)

    logger.info("PPO done. Bootstrapping DQN from PPO policy.")
    dqn = DQNStrategyAgent(dqn_cfg)
    dqn.load_from_ppo(ppo)
    dqn.save(f"{checkpoint_dir}/dqn_bootstrapped.pt")
    logger.info("DQN checkpoint saved to %s/dqn_bootstrapped.pt", checkpoint_dir)


if __name__ == "__main__":
    main()