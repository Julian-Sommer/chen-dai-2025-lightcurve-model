"""
Physical functions and equations for the Chen & Dai 2025 jet model.

This module has been modularized for better maintainability and context management.
All classes are now in separate files within the physics/ directory.

For backward compatibility, all classes are imported here.
"""

# Import all physics classes from their individual modules
from physics.jet_head_shock_breakout_emission import JetHeadShockBreakoutEmission
from physics.jet_physics import JetPhysics
from physics.cocoon_physics import CocoonPhysics
from physics.disk_cocoon_emission import DiskCocoonEmission
from physics.disk_physics import DiskPhysics
from physics.dimensionless_parameters import DimensionlessParameters
from physics.physics_calculator import PhysicsCalculator

# Maintain backward compatibility by exposing all classes at the module level
__all__ = [
    "JetHeadShockBreakoutEmission",
    "JetPhysics",
    "CocoonPhysics",
    "DiskCocoonEmission",
    "DiskPhysics",
    "DimensionlessParameters",
    "PhysicsCalculator",
]
