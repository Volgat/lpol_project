"""
LPOL Neural Architecture Package
Architecture révolutionnaire remplaçant les Transformers

Copyright © 2025 Amega Mike - Proprietary License
"""

from .lpol_neural_core import (
    LPOLModel,
    LPOLConfig,
    LPOLLayer,
    ExperienceMemory,
    LPOLAttention,
    get_default_config
)

__version__ = "1.0.0"
__author__ = "Amega Mike"
__license__ = "Proprietary"

__all__ = [
    'LPOLModel',
    'LPOLConfig', 
    'LPOLLayer',
    'ExperienceMemory',
    'LPOLAttention',
    'get_default_config'
]