"""Compare JAX IPPO and SB3 PPO policies: action distributions, entropy, rewards."""

import json
import pickle
import sys
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env

EXP_DIR = Path(__file__).resolve().parents[2].parent / "jaxborg-exp"

ACTION_TYPE_NAMES = ["Sleep", "Monitor", "Analyse", "Remove", "Restore", "Decoy", "BlockTraffic", "AllowTraffic"]

ACTION_TYPE_RANGES = [
    (BLUE_SLEEP, BLUE_SLEEP + 1),
    (BLUE_MONITOR, BLUE_MONITOR + 1),
    (BLUE_ANALYSE_START, BLUE_ANALYSE_START + GLOBAL_MAX_HOSTS),
    (BLUE_REMOVE_START, BLUE_REMOVE_START + GLOBAL_MAX_HOSTS),
    (BLUE_RESTORE_START, BLUE_RESTORE_START + GLOBAL_MAX_HOSTS),
    (BLUE_DECOY_START, BLUE_BLOCK_TRAFFIC_START),
    (BLUE_BLOCK_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_START),
    (BLUE_ALLOW_TRAFFIC_START, BLUE_ALLOW_TRAFFIC_END),
]


def classify_action(action_idx: int) -> int:
    for i, (start, end) in enumerate(ACTION_TYPE_RANGES):
        if start <= action_idx < end:
            return i
    return 0


def rollout_jax_policy(net_params, network, num_episodes=3):
    env = FsmRedCC4Env(num_steps=500)

    all_actions = []
    all_rewards = []

    for ep in range(num_episodes):
        t0 = time.time()
        key = jax.random.PRNGKey(ep * 100)
        obs, env_state = env.reset(key)

        episode_reward = np.zeros(NUM_BLUE_AGENTS)
        episode_actions = []

        for step in range(500):
            key, step_key = jax.random.split(key)
            act_keys = jax.random.split(key, NUM_BLUE_AGENTS)

            actions = {}
            for agent_idx in range(NUM_BLUE_AGENTS):
                agent = f"blue_{agent_idx}"
                avail = compute_blue_action_mask(env_state.const, agent_idx)
                pi, _ = network.apply(net_params, obs[agent], avail)
                action = pi.sample(seed=act_keys[agent_idx])
                actions[agent] = action
                episode_actions.append(int(action))

            obs, env_state, rewards, dones, _ = env.step(step_key, env_state, actions)
            for agent_idx in range(NUM_BLUE_AGENTS):
                episode_reward[agent_idx] += float(rewards[f"blue_{agent_idx}"])

            if dones["__all__"]:
                break

        all_actions.extend(episode_actions)
        all_rewards.append(episode_reward)
        elapsed = time.time() - t0
        print(
            f"  Episode {ep + 1}/{num_episodes}: {step + 1} steps, "
            f"mean_reward={episode_reward.mean():.1f}, time={elapsed:.1f}s"
        )

    return np.array(all_actions), np.array(all_rewards)


def load_jax_metrics(path: Path):
    steps, rewards, entropies = [], [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            steps.append(record["steps"])
            rewards.append(record["episode_reward_mean"])
            entropies.append(record["entropy"])
    return np.array(steps), np.array(rewards), np.array(entropies)


def plot_action_distribution(actions, title, output_path):
    counts = np.zeros(len(ACTION_TYPE_NAMES))
    for a in actions:
        counts[classify_action(a)] += 1
    counts = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ACTION_TYPE_NAMES, counts)
    ax.set_ylabel("Fraction of Actions")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(counts):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_reward_comparison(jax_metrics_path, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if jax_metrics_path.exists():
        steps, rewards, entropies = load_jax_metrics(jax_metrics_path)
        axes[0].plot(steps, rewards, label="JAX IPPO (masked)", linewidth=2)
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Mean Per-Agent Episode Return")
        axes[0].set_title("Reward Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, entropies, label="JAX IPPO", linewidth=2)
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Policy Entropy")
        axes[1].set_title("Entropy Over Training")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def print_mask_summary():
    print("\n--- Action Mask Summary ---")
    env = FsmRedCC4Env(num_steps=100)
    key = jax.random.PRNGKey(42)
    _, env_state = env.reset(key)

    for agent_idx in range(NUM_BLUE_AGENTS):
        mask = np.array(compute_blue_action_mask(env_state.const, agent_idx))
        total_valid = mask.sum()
        by_type = []
        for name, (start, end) in zip(ACTION_TYPE_NAMES, ACTION_TYPE_RANGES):
            count = mask[start:end].sum()
            if count > 0:
                by_type.append(f"{name}={count}")
        print(f"  blue_{agent_idx}: {total_valid} valid actions: {', '.join(by_type)}")


def main():
    jax_dir = EXP_DIR / "ippo_cc4"
    output_dir = EXP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print_mask_summary()

    jax_metrics = jax_dir / "metrics.jsonl"
    if jax_metrics.exists():
        plot_reward_comparison(jax_metrics, output_dir / "comparison_reward_entropy.png")

    checkpoint_path = jax_dir / "checkpoint_final.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        config_path = jax_dir / "config.json"
        hidden_dim = 256
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                hidden_dim = config.get("HIDDEN_DIM", 256)

        sys.path.insert(0, str(Path(__file__).parent))
        from train_ippo_cc4 import ActorCritic

        network = ActorCritic(
            action_dim=BLUE_ALLOW_TRAFFIC_END,
            hidden_dim=hidden_dim,
            activation="tanh",
        )
        net_params = checkpoint["params"]

        print("\nRolling out JAX IPPO policy (3 episodes)...")
        jax_actions, jax_rewards = rollout_jax_policy(net_params, network, num_episodes=3)
        plot_action_distribution(
            jax_actions,
            "JAX IPPO Action Distribution (Masked)",
            output_dir / "comparison_jax_actions.png",
        )

        print(f"\nJAX IPPO rollout rewards (per agent, {len(jax_rewards)} episodes):")
        print(f"  Mean: {jax_rewards.mean():.2f}")
        print(f"  Std:  {jax_rewards.std():.2f}")

        print("\nAction type breakdown:")
        counts = np.zeros(len(ACTION_TYPE_NAMES))
        for a in jax_actions:
            counts[classify_action(a)] += 1
        total = counts.sum()
        for name, count in zip(ACTION_TYPE_NAMES, counts):
            print(f"  {name}: {count:.0f} ({count / total * 100:.1f}%)")
    else:
        print(f"No JAX checkpoint found at {checkpoint_path}")
        print("Run training first: uv run python scripts/baselines/train_ippo_cc4.py")

    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
