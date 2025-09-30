"""
Chen & Dai 2025 AGN Jet Model Package

A modular implementation of the Chen & Dai (2025) model for AGN jet propagation
through accretion disk environments.

Based on: Chen & Dai (2025), ApJ, 987, 214
"""

__version__ = "1.0.0"
__author__ = "Julian Sommer"

# Import main classes and functions for easy access
from .constants import ModelParameters
from .evolution import ChenDaiModel, create_time_array
from .disk_model import AGNDiskModel
from .solver import BetaHSolver
from .plotting import EvolutionPlotter

__all__ = [
    "ModelParameters",
    "ChenDaiModel",
    "create_time_array",
    "AGNDiskModel",
    "BetaHSolver",
    "EvolutionPlotter",
]
