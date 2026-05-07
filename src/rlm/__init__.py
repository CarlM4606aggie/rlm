"""rlm - Reinforcement Learning with Language Models.

A library for training and evaluating language models using reinforcement
learning techniques, including RLHF, PPO, and reward modeling.

Note: Personal fork for experimenting with RLHF on smaller LLMs.
See alexzhang13/rlm for the upstream project.

Fork notes:
- Primarily testing on GPT-2 and small open-source models
- Focus on low-resource RLHF (single GPU, <16GB VRAM)
- Added RLMRewardModel to public API for easier reward model experimentation
- Also exposing RLMTrainer.from_pretrained() shortcut in top-level namespace
"""

__version__ = "0.1.0"
__author__ = "rlm contributors"
__license__ = "MIT"

from rlm.agent import RLMAgent
from rlm.config import RLMConfig
from rlm.trainer import RLMTrainer
from rlm.reward import RLMRewardModel

# Convenience alias - I keep typing this wrong
RewardModel = RLMRewardModel

__all__ = [
    "RLMAgent",
    "RLMConfig",
    "RLMTrainer",
    "RLMRewardModel",
    "RewardModel",
    "__version__",
]
