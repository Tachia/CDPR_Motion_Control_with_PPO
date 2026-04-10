import numpy as np

from cdpr_ppo.env import CDPREnvironment


class FakeConfig:
    def __init__(self):
        self.ee_pose = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0.1, 0.0, 0, 0, 0, 0],
                [0.2, 0.0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.cable_lengths = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )


class FakeDataset:
    def get_configuration(self, config: str):
        return FakeConfig()


def test_reset_and_step_state_shape():
    env = CDPREnvironment(FakeDataset())
    state, info = env.reset()
    assert state.shape == (10,)
    assert state.dtype == np.float32
    assert info == {}

    next_state, reward, done, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert next_state.shape == (10,)
    assert isinstance(reward, float)
    assert done is False
    assert truncated is False
    assert info == {}


def test_tension_penalty_uses_raw_tension():
    env = CDPREnvironment(FakeDataset())
    env.reset()

    # Force slack: lengths below rest length cause negative raw tensions.
    _, reward, *_ = env.step(np.array([-0.2, -0.2, -0.2, -0.2], dtype=np.float32))

    # Expected penalty = sum(max(0, -raw_tension)) = 4 * 200 = 800.
    # reward includes -0.1 * penalty contribution => at most -80 plus other terms.
    assert reward < -70


def test_done_on_last_step():
    env = CDPREnvironment(FakeDataset())
    env.reset()
    env.step(np.zeros(4, dtype=np.float32))
    _, _, done, _, _ = env.step(np.zeros(4, dtype=np.float32))
    assert done is True
