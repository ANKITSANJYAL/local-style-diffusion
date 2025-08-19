"""
Local Prompt Adaptation (LPA) for Style-Consistent Multi-Object Generation

A training-free method for improving style consistency and multi-object spatial control 
in text-to-image diffusion models.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "contact@example.com"

from .models.lpa_model import LPAModel
from .models.baselines import BaselineModels
from .utils.prompt_parser import PromptParser
from .utils.evaluation import Evaluator
from .utils.visualization import Visualizer

__all__ = [
    "LPAModel",
    "BaselineModels", 
    "PromptParser",
    "Evaluator",
    "Visualizer"
] 