import jax
import jax.numpy as jp
import mediapy as media
from pathlib import Path
import orbax.checkpoint as orbax
import json
import functools
import os

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.networks import make_inference_fn
from brax.training.acme import running_statistics
from mujoco_playground import registry
from ml_collections import config_dict


def load_trained_policy(checkpoint_dir: str, env):
    """
    Load a pre-trained Booster T1 locomotion policy from checkpoint directory.
    
    Parameters
    ----------
    checkpoint_dir : str
        Path to checkpoint directory
    env : brax environment
        Pre-loaded environment to get observation/action sizes from
    """
    
    # Load config
    config_path = Path(checkpoint_dir) / "env_config.json"
    with open(config_path, "r") as f:
        env_cfg = json.load(f)
    env_cfg = config_dict.ConfigDict(env_cfg)

    # Get sizes from provided environment
    obs_size = env.observation_size["state"]
    act_size = env.action_size

    # Hidden layer sizes
    policy_hidden_layer_sizes = tuple(env_cfg["network_factory"]["policy_hidden_layer_sizes"]) if "network_factory" in env_cfg else (512, 256, 128)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir) / "final_model"
    # Convert to absolute path for Orbax
    checkpoint_path = checkpoint_path.resolve()
    orbax_checkpointer = orbax.PyTreeCheckpointer()
    checkpoint = orbax_checkpointer.restore(str(checkpoint_path))
    normalizer_params, policy_params = checkpoint[:2]
    
    # Convert normalizer if needed
    if isinstance(normalizer_params, dict):
        from brax.training.acme.running_statistics import RunningStatisticsState
        normalizer_params = RunningStatisticsState(**normalizer_params)
    
    # Build network
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        preprocess_observations_fn=running_statistics.normalize,
    )
    networks = network_factory(obs_size, act_size)
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    policy_fn = make_inference_fn((normalizer_params, policy_params), deterministic=True)
    
    return policy_fn
