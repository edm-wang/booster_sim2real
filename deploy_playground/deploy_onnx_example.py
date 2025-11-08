#!/usr/bin/env python3
"""
Example script showing how to deploy the ONNX model on hardware.
This demonstrates loading and running inference with the ONNX model.
"""

import numpy as np
import onnxruntime as rt
from pathlib import Path


def load_onnx_model(model_path: str, use_cuda: bool = True):
    """
    Load ONNX model for inference.
    
    Parameters:
    - model_path: Path to ONNX model file
    - use_cuda: Whether to use CUDA execution provider (default: True)
    
    Returns:
    - InferenceSession object
    """
    sess_options = rt.SessionOptions()
    sess_options.log_severity_level = 3  # Only show errors
    
    if use_cuda:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    try:
        session = rt.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        print(f"✅ Loaded ONNX model: {model_path}")
        print(f"   Using provider: {session.get_providers()[0]}")
        return session
    except Exception as e:
        print(f"⚠️  Failed to load with CUDA, falling back to CPU: {e}")
        session = rt.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
        return session


def get_model_info(session: rt.InferenceSession):
    """Get input/output information from the ONNX model."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print("\n📊 Model Information:")
    print("📥 Inputs:")
    for inp in inputs:
        print(f"   Name: {inp.name}")
        print(f"   Shape: {inp.shape}")
        print(f"   Type: {inp.type}")
    
    print("\n📤 Outputs:")
    for out in outputs:
        print(f"   Name: {out.name}")
        print(f"   Shape: {out.shape}")
        print(f"   Type: {out.type}")
    
    return inputs, outputs


def run_inference(session: rt.InferenceSession, observation: np.ndarray):
    """
    Run inference on the ONNX model.
    
    Parameters:
    - session: ONNX Runtime InferenceSession
    - observation: Input observation array (shape: [batch_size, obs_size] or [obs_size])
    
    Returns:
    - actions: Output actions array
    """
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Ensure observation is 2D (batch dimension)
    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
    
    # Ensure correct dtype
    observation = observation.astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: observation})
    
    # Return first output (actions)
    actions = outputs[0]
    
    # Remove batch dimension if single observation
    if actions.shape[0] == 1 and len(actions.shape) > 1:
        actions = actions[0]
    
    return actions


def example_usage():
    """Example of how to use the ONNX model."""
    model_path = "./checkpoint/policy.onnx"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print("🔧 Loading ONNX model...")
    session = load_onnx_model(model_path, use_cuda=True)
    
    # Get model info
    inputs, outputs = get_model_info(session)
    
    # Example: Run inference with dummy observation
    print("\n🧪 Running inference example...")
    obs_size = inputs[0].shape[1] if len(inputs[0].shape) > 1 else inputs[0].shape[0]
    dummy_obs = np.random.randn(obs_size).astype(np.float32)
    
    print(f"   Input observation shape: {dummy_obs.shape}")
    actions = run_inference(session, dummy_obs)
    print(f"   Output actions shape: {actions.shape}")
    print(f"   Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   Sample actions: {actions[:5]}")


if __name__ == "__main__":
    example_usage()

