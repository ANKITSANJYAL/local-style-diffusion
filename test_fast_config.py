#!/usr/bin/env python3
"""
Simple test script to verify the fast config works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.lpa_model import LPAModel
from src.models.baselines import BaselineModels

def test_fast_config():
    """Test the fast config with SD v1.5 model."""
    print("Testing fast config with SD v1.5...")
    
    try:
        # Test LPA model
        print("Loading LPA model...")
        lpa_model = LPAModel(
            model_name="runwayml/stable-diffusion-v1-5",
            device="cpu",
            dtype="float32",
            use_attention_slicing=True,
            use_memory_efficient_attention=False
        )
        print("✅ LPA model loaded successfully!")
        
        # Test baseline models
        print("Loading baseline models...")
        baseline_models = BaselineModels(
            model_name="runwayml/stable-diffusion-v1-5",
            device="cpu",
            dtype="float32"
        )
        print("✅ Baseline models loaded successfully!")
        
        # Test generation
        print("Testing generation...")
        prompt = "A tiger and a spaceship in cyberpunk style"
        
        # Test LPA generation
        lpa_result = lpa_model.generate(
            prompt=prompt,
            seed=42
        )
        print("✅ LPA generation successful!")
        
        # Test baseline generation
        baseline_result = baseline_models.generate_vanilla_sdxl(
            prompt=prompt,
            seed=42
        )
        print("✅ Baseline generation successful!")
        
        print("\n🎉 All tests passed! The fast config is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fast_config()
    sys.exit(0 if success else 1) 