# CDPR Motion Control with PPO

A cleaned and structured implementation of **Proximal Policy Optimization (PPO)** for trajectory tracking and motion control of **Cable-Driven Parallel Robots (CDPRs)**.

## Project Goals

- Track desired end-effector trajectories in a CDPR environment.
- Encourage physically plausible cable behavior (positive tension).
- Provide a reproducible Python package layout suitable for extension and testing.

## Repository Layout

```text
.
├── src/cdpr_ppo/
│   ├── __init__.py
│   ├── cli.py            # command-line entrypoint
│   ├── data.py           # dataset loading and preprocessing
│   ├── env.py            # Gymnasium environment
│   ├── model.py          # actor-critic model and PPO wrapper
│   └── trainer.py        # training loop + plots
├── tests/
│   └── test_environment.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tooling:

```bash
pip install -e .[dev]
```

## Usage

Run training from the command line:

```bash
cdpr-train --data ./ppo_data.xlsx --config 4-cable --episodes 100 --plot
```

### Arguments

- `--data`: path to input dataset (`.csv`, `.xls`, `.xlsx`)
- `--config`: `4-cable`, `3-cable`, or `2-cable`
- `--episodes`: number of training episodes
- `--plot`: display training metrics plots

## Testing

```bash
pytest
```

## Important Notes

- This codebase is structured for research prototyping and reproducibility.
- The reward function now computes tension penalty from **raw (unclipped)** cable tensions, so slack cables are penalized correctly.
- Original notebook-style script is preserved as `ppo_model_19.py` for historical reference.

## Roadmap

- Add true PPO optimization updates (policy/value loss over minibatches).
- Add environment validation against experimental CDPR logs.
- Add CI workflow (lint + tests) and model checkpointing.

## License

MIT (recommended for open research code). Update if your institution requires a different license.
