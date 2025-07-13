#!/bin/bash

# Setup script for Local Prompt Adaptation (LPA) project
# This script sets up the environment and installs dependencies

set -e  # Exit on any error

echo "Setting up Local Prompt Adaptation (LPA) environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating environment..."
    
    # Create conda environment
    conda create -n lpa python=3.9 -y
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate lpa
    
    echo "Conda environment 'lpa' created and activated."
else
    echo "Conda not found. Using pip directly..."
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch and torchvision together (with CUDA support if available)
echo "Installing PyTorch and torchvision..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies (excluding torch and torchvision)
echo "Installing other dependencies..."
grep -v "torch\|torchvision" requirements.txt | pip install -r /dev/stdin

# Install spaCy model for NLP
echo "Installing spaCy English model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/results/{images,attention_maps,metrics,tables}
mkdir -p experiments
mkdir -p paper/{figures,tables,results}
mkdir -p logs

# Set up git hooks (optional)
if [ -d ".git" ]; then
    echo "Setting up git hooks..."
    cp scripts/pre-commit .git/hooks/ 2>/dev/null || echo "No pre-commit hook found"
fi

# Test installation
echo "Testing installation..."
python -c "
import torch
import torchvision
import diffusers
import transformers
import spacy
import matplotlib
import numpy as np
print('✓ All core dependencies installed successfully!')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ TorchVision version: {torchvision.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
"

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate lpa"
echo "2. Test the project: python -m src.experiments.run_experiments --config configs/experiment_config.yaml --max-prompts 2"
echo "3. Run full experiment: python -m src.experiments.run_experiments --config configs/experiment_config.yaml"
echo ""
echo "For development:"
echo "- Install development dependencies: pip install -r requirements-dev.txt (if available)"
echo "- Run tests: python -m pytest tests/ (if available)"
echo "- Format code: black src/ (if black is installed)"
echo ""
echo "Happy researching! 🚀" 