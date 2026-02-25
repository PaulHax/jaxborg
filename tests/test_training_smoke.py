import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.constants import BLUE_OBS_SIZE

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def training_output():
    from scripts.baselines.train_ippo_cc4 import make_train

    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2,
        "NUM_STEPS": 8,
        "TOTAL_TIMESTEPS": 640,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "HIDDEN_DIM": 64,
        "ANNEAL_LR": False,
        "SEED": 0,
    }

    env, init_obsv, init_env_state, train_fn = make_train(config)
    rng = jax.random.PRNGKey(1)
    return jax.jit(train_fn)(rng, init_obsv, init_env_state)


class TestTrainingSmoke:
    def test_single_update_produces_finite_losses(self, training_output):
        metrics = training_output["metrics"]
        assert jnp.isfinite(metrics["total_loss"]).all()
        assert jnp.isfinite(metrics["actor_loss"]).all()
        assert jnp.isfinite(metrics["critic_loss"]).all()
        assert jnp.isfinite(metrics["entropy"]).all()

    def test_masked_training_lower_entropy(self, training_output):
        """Masking should produce lower entropy than unmasked (fewer valid actions)."""
        metrics = training_output["metrics"]
        assert jnp.isfinite(metrics["entropy"]).all()
        initial_entropy = float(metrics["entropy"][0])
        max_uniform_entropy = float(jnp.log(jnp.array(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.float32)))
        assert initial_entropy < max_uniform_entropy, (
            f"Initial entropy {initial_entropy:.2f} should be less than "
            f"uniform over full action space {max_uniform_entropy:.2f}"
        )

    def test_network_output_shape(self):
        from scripts.baselines.train_ippo_cc4 import ActorCritic

        network = ActorCritic(
            action_dim=BLUE_ALLOW_TRAFFIC_END,
            hidden_dim=64,
            activation="tanh",
        )
        rng = jax.random.PRNGKey(0)
        init_x = jnp.zeros((BLUE_OBS_SIZE,))
        params = network.init(rng, init_x)

        pi, value = network.apply(params, init_x)
        assert pi.logits.shape == (BLUE_ALLOW_TRAFFIC_END,)
        assert value.shape == ()
