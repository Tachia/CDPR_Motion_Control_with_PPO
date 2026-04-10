from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import CDPRDataset


class CDPREnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: CDPRDataset, config: str = "4-cable"):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.data = dataset.get_configuration(config)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(4,),
            dtype=np.float32,
        )

        self._compute_normalization_params()
        self.current_step = 0
        self.max_steps = len(self.data.ee_pose) - 1
        self.reset()

    def _compute_normalization_params(self) -> None:
        self.ee_pose_mean = np.mean(self.data.ee_pose, axis=0)
        self.ee_pose_std = np.std(self.data.ee_pose, axis=0) + 1e-8

    @staticmethod
    def calculate_raw_tensions(cable_lengths: np.ndarray) -> np.ndarray:
        k = 1000.0
        l0 = 1.0
        return k * (cable_lengths - l0)

    def calculate_cable_tensions(self, cable_lengths: np.ndarray) -> np.ndarray:
        return np.clip(self.calculate_raw_tensions(cable_lengths), 0.0, None)

    def _normalize_state(self, ee_pose: np.ndarray, tensions: np.ndarray) -> np.ndarray:
        ee_pose_norm = (ee_pose - self.ee_pose_mean) / self.ee_pose_std
        tensions_norm = tensions / 1000.0
        return np.concatenate([ee_pose_norm, tensions_norm])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_pose = self.data.ee_pose[0].copy()
        self.current_lengths = self.data.cable_lengths[0].copy()
        self.current_tensions = self.calculate_cable_tensions(self.current_lengths)
        state = self._normalize_state(self.current_pose, self.current_tensions)
        return state.astype(np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        new_lengths = self.current_lengths + action

        target_pose = self.data.ee_pose[self.current_step + 1]

        raw_tensions = self.calculate_raw_tensions(new_lengths)
        new_tensions = np.clip(raw_tensions, 0.0, None)

        pose_error = np.linalg.norm(target_pose - self.current_pose)
        tension_penalty = np.sum(np.maximum(0.0, -raw_tensions))
        stability_reward = -np.std(new_tensions)
        reward = -pose_error - 0.1 * tension_penalty + 0.05 * stability_reward

        self.current_pose = target_pose
        self.current_lengths = new_lengths
        self.current_tensions = new_tensions
        self.current_step += 1

        state = self._normalize_state(self.current_pose, self.current_tensions)
        done = self.current_step >= self.max_steps

        return state.astype(np.float32), float(reward), done, False, {}
