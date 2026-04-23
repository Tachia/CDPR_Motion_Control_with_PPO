from __future__ import annotations

import argparse

from .data import CDPRDataset
from .env import CDPREnvironment
from .model import ImprovedPPO
from .trainer import CDPRTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO for CDPR trajectory tracking")
    parser.add_argument("--data", required=True, help="Path to source CSV/XLSX dataset")
    parser.add_argument("--config", default="4-cable", choices=["4-cable", "3-cable", "2-cable"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--plot", action="store_true", help="Plot metrics after training")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset = CDPRDataset(args.data)
    env = CDPREnvironment(dataset, config=args.config)
    agent = ImprovedPPO(state_dim=10, action_dim=4)
    trainer = CDPRTrainer(env, agent)
    trainer.train(num_episodes=args.episodes)

    if args.plot:
        trainer.plot_training_metrics()


if __name__ == "__main__":
    main()
