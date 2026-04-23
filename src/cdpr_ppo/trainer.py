from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class CDPRTrainer:
    def __init__(self, env, ppo_agent):
        self.env = env
        self.agent = ppo_agent
        self.training_metrics = {
            "episode_rewards": [],
            "pose_errors": [],
            "tension_violations": [],
        }
        self.clear_buffer()

    def clear_buffer(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def train(self, num_episodes: int = 100) -> None:
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            pose_errors = []
            tension_violations = []

            self.agent.actor_critic.eval()

            while True:
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.log_probs.append(log_prob)
                self.values.append(value)

                episode_reward += reward
                pose_errors.append(np.linalg.norm(next_state[:6] - state[:6]))
                tension_violations.append(np.sum(next_state[6:] < 0))

                if done:
                    break

                state = next_state

            self.training_metrics["episode_rewards"].append(episode_reward)
            self.training_metrics["pose_errors"].append(float(np.mean(pose_errors)))
            self.training_metrics["tension_violations"].append(float(np.mean(tension_violations)))

            if episode % 10 == 0:
                print(
                    f"Episode {episode} | reward={episode_reward:.4f} "
                    f"pose_error={np.mean(pose_errors):.4f} "
                    f"tension_violations={np.mean(tension_violations):.4f}"
                )

            self.clear_buffer()

    def plot_training_metrics(self) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        ax1.plot(self.training_metrics["episode_rewards"])
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        ax2.plot(self.training_metrics["pose_errors"])
        ax2.set_title("Average Pose Errors")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Error (m)")

        ax3.plot(self.training_metrics["tension_violations"])
        ax3.set_title("Cable Tension Violations")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Violations")

        plt.tight_layout()
        plt.show()
