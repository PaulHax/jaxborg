import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
import pettingzoo
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium.spaces import Space
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import hydra
from omegaconf import DictConfig


class CybORGPzShim(ParallelEnv):
    metadata = {
        "render_modes": [],
        "name": "CybORG v4",
        "is_parallelizable": True,
        "has_manual_policy": False,
    }

    def __init__(self, env: BlueFlatWrapper):
        super().__init__()
        self.env = env

    def reset(self, seed=None, *args, **kwargs):
        return self.env.reset(seed=seed)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def action_spaces(self) -> dict[str, Space]:
        return self.env.action_spaces()

    def action_space(self, agent) -> Space:
        return self.env.action_space(agent)

    @property
    def observation_spaces(self) -> dict[str, Space]:
        return self.env.observation_spaces()

    def observation_space(self, agent) -> Space:
        return self.env.observation_space(agent)

    @property
    def action_masks(self) -> dict[str, torch.Tensor]:
        return {
            a: torch.tensor(self.env.action_masks[a], dtype=torch.bool)
            for a in self.env.agents
        }


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.observation_space = super().observation_space(self.agent_selection)
        self.action_space = super().action_space(self.agent_selection)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        super().step(action)
        return super().last()

    def action_mask(self):
        return self.infos[self.agent_selection]["action_mask"]


def mask_fn(env):
    return env.action_mask()


def make_env(pad_spaces=True):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    wrapped = BlueFlatWrapper(cyborg, pad_spaces=pad_spaces)
    pz = CybORGPzShim(wrapped)
    return parallel_to_aec(pz)


def train(total_timesteps, learning_rate, n_steps, batch_size, seed, model_save_path):
    env = make_env()
    env = SB3ActionMaskWrapper(env)
    env.reset(seed=seed)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log="logs/ppo",
    )
    model.set_random_seed(seed)
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    env.close()


EXP_DIR = Path(__file__).resolve().parents[2].parent / "jaxborg-exp"


@hydra.main(config_path="../configs", config_name="ppo_baseline", version_base=None)
def main(cfg: DictConfig):
    train(
        total_timesteps=cfg.total_timesteps,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        model_save_path=cfg.model_save_path,
    )


if __name__ == "__main__":
    import sys

    if not any("hydra.run.dir" in a for a in sys.argv):
        sys.argv.append(f"hydra.run.dir={EXP_DIR}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
    if not any("hydra.job.chdir" in a for a in sys.argv):
        sys.argv.append("hydra.job.chdir=True")
    main()
