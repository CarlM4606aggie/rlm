"""rlm - Reinforcement Learning with Language Models.

A library for training and evaluating language models using reinforcement
learning techniques, including RLHF, PPO, and reward modeling.
"""

__version__ = "0.1.0"
__author__ = "rlm contributors"
__license__ = "MIT"

from rlm.agent import RLMAgent
from rlm.config import RLMConfig
from rlm.trainer import RLMTrainer

__all__ = [
    "RLMAgent",
    "RLMConfig",
    "RLMTrainer",
    "__version__",
]
