# Local Prompt Adaptation (LPA) for Style-Consistent Multi-Object Generation
[![arXiv](https://img.shields.io/badge/arXiv-2506.18208-b31b1b.svg)](https://arxiv.org/abs/2507.20094)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Project Overview
This repository contains the implementation of Local Prompt Adaptation (LPA), a training-free method for improving style consistency and multi-object spatial control in text-to-image diffusion models.

## Project Structure
```
local-style-diffusion/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── configs/                           # Configuration files
│   ├── experiment_config.yaml         # Main experiment configuration
│   └── model_config.yaml              # Model-specific settings
├── src/                               # Source code
│   ├── __init__.py
│   ├── models/                        # Model implementations
│   │   ├── __init__.py
│   │   ├── lpa_model.py              # LPA implementation
│   │   └── baselines.py              # Baseline models
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── prompt_parser.py          # Prompt parsing utilities
│   │   ├── attention_utils.py        # Attention map utilities
│   │   ├── evaluation.py             # Evaluation metrics
│   │   └── visualization.py          # Visualization tools
│   └── experiments/                   # Experiment scripts
│       ├── __init__.py
│       ├── run_experiments.py        # Main experiment runner
│       └── ablation_studies.py       # Ablation studies
├── data/                              # Data and prompts
│   ├── prompts/                       # Prompt datasets
│   │   ├── test_prompts.json         # 50 test prompts
│   │   └── prompt_categories.json    # Prompt categorization
│   └── results/                       # Experiment results
│       ├── images/                    # Generated images
│       ├── attention_maps/            # Attention visualizations
│       ├── metrics/                   # Evaluation metrics
│       └── tables/                    # Results tables
├── experiments/                       # Experiment outputs
│   ├── experiment_001/                # Experiment runs
│   ├── experiment_002/
│   └── ...
├── paper/                             # Paper materials
│   ├── figures/                       # Paper figures
│   ├── tables/                        # Paper tables
│   └── results/                       # Final results
├── notebooks/                         # Jupyter notebooks
│   ├── demo.ipynb                     # Interactive demo
│   └── analysis.ipynb                 # Results analysis
└── scripts/                           # Utility scripts
    ├── setup_environment.sh           # Environment setup
    └── run_all_experiments.sh         # Batch experiment runner
```

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n lpa python=3.9
conda activate lpa

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
# Run main experiments
python src/experiments/run_experiments.py --config configs/experiment_config.yaml

# Run ablation studies
python src/experiments/ablation_studies.py --config configs/experiment_config.yaml
```

### 3. Interactive Demo
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/demo.ipynb
```

## Key Features

### Research-Grade Implementation
- **Reproducible experiments** with configurable parameters
- **Comprehensive evaluation** with multiple metrics
- **Proper result storage** with timestamps and versioning
- **Visualization tools** for attention maps and results

### Evaluation Metrics
- **Style Consistency Score** (CLIP/DINO embeddings)
- **LPIPS** (Perceptual Distance)
- **CLIP-Text Alignment** (CLIPScore)
- **User Study Framework** (Optional)

### Baseline Comparisons
- Raw SDXL with full prompt
- SDXL with CFG tuning (high guidance = 12-18)
- Our Method: Local Prompt Adaptation

## Citation
```bibtex
@article{lpa2024,
  title={Local Prompt Adaptation for Style-Consistent Multi-Object Generation in Diffusion Models},
  author={Ankit Sanjyal},
  journal={	arXiv:2507.20094},
  year={2024}
}
```

## License
MIT License - see LICENSE file for details.

## Contributing
This is a research project. For questions or collaborations, please open an issue or contact the authors.
