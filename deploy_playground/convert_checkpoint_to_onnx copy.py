#!/usr/bin/env python3
"""
Convert Brax checkpoint to ONNX format.
Based on the notebook: mujoco_playground/mujoco_playground/experimental/brax_network_to_onnx.ipynb
"""

import os
import sys

# CRITICAL: Unset LD_LIBRARY_PATH for JAX to find CUDA libraries properly
if "LD_LIBRARY_PATH" in os.environ and not os.environ.get("_CONVERT_SCRIPT_REEXEC"):
    os.environ["_CONVERT_SCRIPT_REEXEC"] = "1"
    env = os.environ.copy()
    del env["LD_LIBRARY_PATH"]
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import argparse
import numpy as np
import json
from pathlib import Path

# Set environment variables
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import onnxruntime as rt
import functools

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
import orbax.checkpoint as orbax
from ml_collections import config_dict


class MLP(tf.keras.Model):
    """TensorFlow MLP model matching the JAX/Flax structure."""
    
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        mean_std=None,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_init = kernel_init
        self.activate_final = activate_final
        self.bias = bias
        self.layer_norm = layer_norm

        if mean_std is not None:
            self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
            self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
        else:
            self.mean = None
            self.std = None

        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(self.layer_sizes):
            dense_layer = layers.Dense(
                size,
                activation=self.activation,
                kernel_initializer=self.kernel_init,
                name=f"hidden_{i}",
                use_bias=self.bias,
            )
            self.mlp_block.add(dense_layer)
            if self.layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
        if not self.activate_final and self.mlp_block.layers:
            if hasattr(self.mlp_block.layers[-1], 'activation') and self.mlp_block.layers[-1].activation is not None:
                self.mlp_block.layers[-1].activation = None

        self.submodules = [self.mlp_block]

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.mean is not None and self.std is not None:
            inputs = (inputs - self.mean) / self.std
        logits = self.mlp_block(inputs)
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)


def make_policy_network(
    param_size,
    mean_std,
    hidden_layer_sizes=[256, 256],
    activation=tf.nn.relu,
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    """Create a TensorFlow policy network."""
    policy_network = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        mean_std=mean_std,
    )
    return policy_network


def transfer_weights(jax_params, tf_model):
    """
    Transfer weights from a JAX parameter dictionary to the TensorFlow model.
    
    Parameters:
    - jax_params: dict
      Nested dictionary with structure {block_name: {layer_name: {params}}}.
    - tf_model: tf.keras.Model
      An instance of the MLP model containing named submodules and layers.
    """
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
        except ValueError:
            print(f"⚠️  Layer {layer_name} not found in TensorFlow model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            kernel = np.array(layer_params['kernel'])
            bias = np.array(layer_params['bias'])
            print(f"✅ Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}")
            tf_layer.set_weights([kernel, bias])
        else:
            print(f"⚠️  Unhandled layer type in {layer_name}: {type(tf_layer)}")

    print("✅ Weights transferred successfully.")


def load_checkpoint(checkpoint_dir: str):
    """Load checkpoint using orbax."""
    print(f"📦 Loading checkpoint from: {checkpoint_dir}")
    
    # Load config
    config_path = Path(checkpoint_dir) / "env_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        env_cfg = json.load(f)
    env_cfg = config_dict.ConfigDict(env_cfg)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir) / "final_model"
    checkpoint_path = checkpoint_path.resolve()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    print(f"📂 Loading from: {checkpoint_path}")
    orbax_checkpointer = orbax.PyTreeCheckpointer()
    checkpoint = orbax_checkpointer.restore(str(checkpoint_path))
    
    normalizer_params, policy_params = checkpoint[:2]
    
    # Convert normalizer if needed
    if isinstance(normalizer_params, dict):
        from brax.training.acme.running_statistics import RunningStatisticsState
        normalizer_params = RunningStatisticsState(**normalizer_params)
    
    return normalizer_params, policy_params, env_cfg


def convert_to_onnx(
    checkpoint_dir: str,
    output_path: str = None,
    env_name: str = None,
    opset: int = 11,
):
    """
    Convert Brax checkpoint to ONNX format.
    
    Parameters:
    - checkpoint_dir: Path to checkpoint directory
    - output_path: Path for output ONNX file (default: checkpoint_dir/policy.onnx)
    - env_name: Environment name (default: from training_results.json)
    - opset: ONNX opset version (default: 11 for Isaac Lab compatibility)
    """
    
    checkpoint_dir = Path(checkpoint_dir).resolve()
    
    # Determine environment name
    if env_name is None:
        training_results_path = checkpoint_dir / "training_results.json"
        if training_results_path.exists():
            try:
                with open(training_results_path, "r") as f:
                    training_results = json.load(f)
                env_name = training_results.get("environment", "T1JoystickFlatTerrain")
            except json.JSONDecodeError:
                # File exists but is incomplete/corrupted - extract environment name with regex
                print(f"⚠️  training_results.json is incomplete, extracting environment name...")
                with open(training_results_path, "r") as f:
                    content = f.read()
                import re
                match = re.search(r'"environment"\s*:\s*"([^"]+)"', content)
                if match:
                    env_name = match.group(1)
                    print(f"✅ Found environment name: {env_name}")
                else:
                    env_name = "T1JoystickFlatTerrain"
                    print(f"⚠️  Could not extract environment name, using default: {env_name}")
        else:
            env_name = "T1JoystickFlatTerrain"
            print(f"⚠️  training_results.json not found, using default env: {env_name}")
    
    print(f"🌍 Environment: {env_name}")
    
    # Load checkpoint
    normalizer_params, policy_params, env_cfg = load_checkpoint(str(checkpoint_dir))
    
    # Extract observation/action sizes from checkpoint (avoid loading environment to prevent CUDA issues)
    print(f"📊 Extracting sizes from checkpoint...")
    
    # Observation size from normalizer mean/std
    mean = normalizer_params.mean["state"]
    obs_size = int(mean.shape[0])
    
    # Action size from policy output layer (policy outputs mean and std, so divide by 2)
    # Get the last layer size from policy params
    policy_layers = list(policy_params['params'].keys())
    if policy_layers:
        last_layer = policy_layers[-1]
        last_layer_output = policy_params['params'][last_layer]['kernel'].shape[1]
        act_size = last_layer_output // 2  # Policy outputs [mean, std] concatenated
    else:
        # Fallback: try to infer from env_config if available
        if "network_factory" in env_cfg and "action_size" in env_cfg.get("network_factory", {}):
            act_size = env_cfg["network_factory"]["action_size"]
        else:
            # Default for T1 robot
            act_size = 23
            print(f"⚠️  Could not infer action size, using default: {act_size}")
    
    print(f"📊 Extracted from checkpoint:")
    print(f"   Observation size: {obs_size}")
    print(f"   Action size: {act_size}")
    
    # Get network configuration - infer from checkpoint
    def infer_hidden_layer_sizes(policy_params, obs_size, act_size):
        """Infer hidden layer sizes from policy parameters."""
        if 'params' not in policy_params:
            return (512, 256, 128)
        layer_keys = sorted([k for k in policy_params['params'].keys() if 'kernel' in policy_params['params'][k]])
        if not layer_keys:
            return (512, 256, 128)
        hidden_sizes = []
        for i, key in enumerate(layer_keys[:-1]):  # All layers except the last (output layer)
            kernel = policy_params['params'][key]['kernel']
            if len(kernel.shape) == 2:
                output_size = kernel.shape[1]
                if i == 0 and kernel.shape[0] == obs_size:
                    hidden_sizes.append(output_size)
                elif i > 0:
                    hidden_sizes.append(output_size)
        return tuple(hidden_sizes) if hidden_sizes else (512, 256, 128)
    
    if "network_factory" in env_cfg and "policy_hidden_layer_sizes" in env_cfg["network_factory"]:
        policy_hidden_layer_sizes = tuple(env_cfg["network_factory"]["policy_hidden_layer_sizes"])
        print(f"✅ Using network_factory from config: {policy_hidden_layer_sizes}")
    else:
        policy_hidden_layer_sizes = infer_hidden_layer_sizes(policy_params, obs_size, act_size)
        print(f"📊 Inferred hidden layer sizes from checkpoint: {policy_hidden_layer_sizes}")
    
    print(f"🔧 Network config:")
    print(f"   Hidden layer sizes: {policy_hidden_layer_sizes}")
    
    # Extract mean/std from normalizer
    mean = normalizer_params.mean["state"]
    std = normalizer_params.std["state"]
    
    print(f"📊 Normalizer stats:")
    print(f"   Mean shape: {mean.shape}, range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"   Std shape: {std.shape}, range: [{std.min():.3f}, {std.max():.3f}]")
    
    # Convert mean/std to TensorFlow tensors
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
    
    # Create TensorFlow policy network
    print("🔨 Creating TensorFlow model...")
    tf_policy_network = make_policy_network(
        param_size=act_size * 2,  # Policy outputs mean and std
        mean_std=mean_std,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=tf.nn.swish,  # Common activation for brax policies
    )
    
    # Test the model with dummy input
    example_input = tf.zeros((1, obs_size))
    example_output = tf_policy_network(example_input)
    print(f"✅ TensorFlow model created, output shape: {example_output.shape}")
    
    # Transfer weights from JAX to TensorFlow
    print("🔄 Transferring weights from JAX to TensorFlow...")
    transfer_weights(policy_params['params'], tf_policy_network)
    
    # Verify the model works
    print("🧪 Testing TensorFlow model...")
    test_input = [np.ones((1, obs_size), dtype=np.float32)]
    tensorflow_pred = tf_policy_network(test_input)
    print(f"✅ TensorFlow prediction shape: {tensorflow_pred.shape}")
    print(f"   Prediction range: [{tensorflow_pred.numpy().min():.3f}, {tensorflow_pred.numpy().max():.3f}]")
    
    # Set output name
    tf_policy_network.output_names = ['continuous_actions']
    
    # Define input signature for ONNX conversion
    spec = [tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")]
    
    # Convert to ONNX
    if output_path is None:
        output_path = checkpoint_dir / "policy.onnx"
    else:
        output_path = Path(output_path)
    
    print(f"🔄 Converting to ONNX (opset={opset})...")
    print(f"📤 Output path: {output_path}")
    
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network,
        input_signature=spec,
        opset=opset,
        output_path=str(output_path)
    )
    
    print(f"✅ ONNX model saved to: {output_path}")
    
    # Test ONNX model
    print("🧪 Testing ONNX model...")
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(str(output_path), providers=providers)
    
    onnx_input = {
        'obs': np.ones((1, obs_size), dtype=np.float32)
    }
    output_names = ['continuous_actions']
    onnx_pred = m.run(output_names, onnx_input)[0][0]
    
    print(f"✅ ONNX prediction shape: {onnx_pred.shape}")
    print(f"   Prediction range: [{onnx_pred.min():.3f}, {onnx_pred.max():.3f}]")
    
    # Compare TensorFlow and ONNX outputs
    print("\n📊 Comparing TensorFlow and ONNX outputs:")
    tf_pred_np = tensorflow_pred.numpy()[0]
    diff = np.abs(tf_pred_np - onnx_pred)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ TensorFlow and ONNX outputs match closely!")
    else:
        print("⚠️  Warning: TensorFlow and ONNX outputs differ significantly")
    
    # Also test with JAX inference for comparison
    print("\n🧪 Testing JAX inference for comparison...")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        preprocess_observations_fn=running_statistics.normalize,
    )
    networks = network_factory(obs_size, act_size)
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    inference_fn = make_inference_fn((normalizer_params, policy_params), deterministic=True)
    
    test_input_jax = {
        'state': jp.ones(obs_size),
    }
    jax_pred, _ = inference_fn(test_input_jax, jax.random.PRNGKey(0))
    jax_pred_np = np.array(jax_pred)
    
    print(f"✅ JAX prediction shape: {jax_pred_np.shape}")
    print(f"   Prediction range: [{jax_pred_np.min():.3f}, {jax_pred_np.max():.3f}]")
    
    # Compare JAX with ONNX
    print("\n📊 Comparing JAX and ONNX outputs:")
    diff_jax = np.abs(jax_pred_np - onnx_pred)
    max_diff_jax = diff_jax.max()
    mean_diff_jax = diff_jax.mean()
    print(f"   Max difference: {max_diff_jax:.6f}")
    print(f"   Mean difference: {mean_diff_jax:.6f}")
    
    if max_diff_jax < 1e-4:
        print("✅ JAX and ONNX outputs match closely!")
    else:
        print("⚠️  Warning: JAX and ONNX outputs differ (this may be expected due to numerical differences)")
    
    print(f"\n🎉 Conversion complete! ONNX model saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Brax checkpoint to ONNX format"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoint",
        help="Path to checkpoint directory (default: ./checkpoint)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path for output ONNX file (default: checkpoint_dir/policy.onnx)"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Environment name (default: from training_results.json)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11 for Isaac Lab compatibility)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_to_onnx(
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output_path,
            env_name=args.env_name,
            opset=args.opset,
        )
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

