#!/usr/bin/env python3
"""
Test script for Local Prompt Adaptation (LPA) project setup.

This script tests the basic project structure and functionality
without requiring all dependencies to be installed.
"""

import os
import sys
import json
import yaml
from pathlib import Path

def test_project_structure():
    """Test that all required directories and files exist."""
    print("Testing project structure...")
    
    required_dirs = [
        "src",
        "src/models",
        "src/utils", 
        "src/experiments",
        "configs",
        "data",
        "data/prompts",
        "data/results",
        "data/results/images",
        "data/results/attention_maps",
        "data/results/metrics",
        "data/results/tables",
        "experiments",
        "paper",
        "paper/figures",
        "paper/tables",
        "paper/results",
        "notebooks",
        "scripts"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

def test_config_files():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")
    
    # Test experiment config
    try:
        with open("configs/experiment_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ["experiment", "model", "lpa", "generation", "evaluation", "dataset"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing config keys: {missing_keys}")
            return False
        else:
            print("✅ Experiment config is valid")
    except Exception as e:
        print(f"❌ Error reading experiment config: {e}")
        return False
    
    return True

def test_prompt_dataset():
    """Test that the prompt dataset is valid."""
    print("\nTesting prompt dataset...")
    
    try:
        with open("data/prompts/test_prompts.json", 'r') as f:
            data = json.load(f)
        
        # Check metadata
        if "metadata" not in data:
            print("❌ Missing metadata in prompt dataset")
            return False
        
        # Check prompts
        if "prompts" not in data:
            print("❌ Missing prompts in dataset")
            return False
        
        prompts = data["prompts"]
        if len(prompts) != 50:
            print(f"❌ Expected 50 prompts, found {len(prompts)}")
            return False
        
        # Check prompt structure
        required_fields = ["id", "category", "prompt", "objects", "style", "complexity"]
        for i, prompt in enumerate(prompts):
            missing_fields = [field for field in required_fields if field not in prompt]
            if missing_fields:
                print(f"❌ Prompt {i} missing fields: {missing_fields}")
                return False
        
        print("✅ Prompt dataset is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error reading prompt dataset: {e}")
        return False

def test_source_files():
    """Test that source files can be imported (basic syntax check)."""
    print("\nTesting source files...")
    
    source_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/experiments/__init__.py",
        "src/utils/prompt_parser.py",
        "src/utils/attention_utils.py",
        "src/utils/evaluation.py",
        "src/utils/visualization.py",
        "src/models/lpa_model.py",
        "src/models/baselines.py",
        "src/experiments/run_experiments.py"
    ]
    
    missing_files = []
    for file_path in source_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing source files: {missing_files}")
        return False
    else:
        print("✅ All source files exist")
    
    # Test basic import (without dependencies)
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), "src"))
        
        # Test basic imports (these should work even without dependencies)
        import src.utils.prompt_parser
        print("✅ Prompt parser module can be imported")
        
        import src.utils.attention_utils
        print("✅ Attention utils module can be imported")
        
        import src.models.lpa_model
        print("✅ LPA model module can be imported")
        
        import src.models.baselines
        print("✅ Baselines module can be imported")
        
        import src.experiments.run_experiments
        print("✅ Experiment runner module can be imported")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False
    
    return True

def test_requirements():
    """Test that requirements.txt exists and is readable."""
    print("\nTesting requirements...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
        
        # Check for essential packages
        essential_packages = ["torch", "diffusers", "transformers", "spacy"]
        missing_packages = []
        
        for package in essential_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing essential packages in requirements.txt: {missing_packages}")
            return False
        else:
            print("✅ requirements.txt contains essential packages")
            return True
            
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Local Prompt Adaptation (LPA) Project Setup Test")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_config_files,
        test_prompt_dataset,
        test_source_files,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The project is ready for development.")
        print("\nNext steps:")
        print("1. Run: ./scripts/setup_environment.sh")
        print("2. Activate environment: conda activate lpa")
        print("3. Test with a few prompts: python -m src.experiments.run_experiments --config configs/experiment_config.yaml --max-prompts 2")
        return 0
    else:
        print("❌ Some tests failed. Please check the project structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 