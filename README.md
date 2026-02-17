# jaxborg

Baselines for CybORG CAGE Challenge 4.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run pytest tests/ -v
uv run python baselines/eval_sleep.py --max-eps 100 --seed 42
uv run python baselines/eval_random.py --max-eps 100 --seed 42
uv run python baselines/train_ppo.py total_timesteps=250000
```

Training output (Hydra logs, tensorboard events, saved models) goes to `../jaxborg-exp/`.

## Guidelines

- **Pin external dependencies to git commits** for reproducibility. Never depend on local filesystem paths.
- The CybORG commit hash is pinned in `pyproject.toml`.
