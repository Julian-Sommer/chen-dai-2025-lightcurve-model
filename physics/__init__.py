"""
Physical functions and equations for the Chen & Dai 2025 jet model.

This module contains all the core physics calculations, organized into
logical groups for clarity and maintainability.
"""

# Import all physics classes for backward compatibility
from .jet_head_shock_breakout_emission import JetHeadShockBreakoutEmission
from .jet_physics import JetPhysics
from .cocoon_physics import CocoonPhysics
from .disk_cocoon_emission import DiskCocoonEmission
from .jet_cocoon_emission import JetCocoonEmission
from .disk_physics import DiskPhysics
from .physics_calculator import PhysicsCalculator

# Maintain backward compatibility with the original physics.py interface
__all__ = [
    "JetHeadShockBreakoutEmission",
    "JetPhysics",
    "CocoonPhysics",
    "DiskCocoonEmission",
    "JetCocoonEmission",
    "DiskPhysics",
    "PhysicsCalculator",
]
