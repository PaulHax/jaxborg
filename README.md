# jaxborg

JAX port of CybORG CAGE Challenge 4 using [JaxMARL](https://github.com/FLAIROx/JaxMARL) for GPU-accelerated parallel RL training.

CybORG CC4 is a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 3 mission phases). This project re-implements CC4's environment logic as JIT-compilable JAX arrays for massively parallel simulation on GPU, validated step-by-step against the original CybORG.

## Approach

- Implements JaxMARL's `MultiAgentEnv` interface
- Dynamic topology handled via pad-to-max with active-host masking (one JIT trace for all topologies)
- Differential testing against CybORG: run both environments in lockstep, compare state after every step
- Behavior catalog drives parameterized test generation across all action/target/state combinations
- Final validation: train PPO on both environments, compare learning curves

## Setup

```bash
uv sync
```

## Usage

```bash
uv run pytest tests/ -v
uv run python scripts/baselines/eval_sleep.py --max-eps 100 --seed 42
uv run python scripts/baselines/eval_random.py --max-eps 100 --seed 42
uv run python scripts/baselines/train_ppo.py total_timesteps=250000
```

Training output (Hydra logs, tensorboard events, saved models) goes to `../jaxborg-exp/`.

## Guidelines

- **Pin external dependencies to git commits** for reproducibility. Never depend on local filesystem paths.
- The CybORG and JaxMARL commit hashes are pinned in `pyproject.toml`.
