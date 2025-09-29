"""
Disk physics calculations for the Chen & Dai 2025 model.

This module contains functions for calculating disk properties
such as density profiles and mean densities along the jet path.
"""

import numpy as np
from typing import Literal


class DiskPhysics:
    """Collection of functions for disk physics calculations."""

    @staticmethod
    def density_profile(
        rho_0: float,
        h: float,
        z: float,
        profile_type: Literal["uniform", "isothermal", "polytropic"] = "isothermal",
    ) -> float:
        """
        Calculate density profile as function of height.

        Parameters
        ----------
        rho_0 : float
            Central density [g/cm³]
        h : float
            Scale height [cm]
        z : float
            Height above disk [cm]
        profile_type : str
            Type of density profile

        Returns
        -------
        float
            Density at height z [g/cm³]
        """
        if profile_type == "uniform":
            # Uniform density profile: ρ(z) = ρ₀ everywhere
            return rho_0

        elif profile_type == "isothermal":
            # Isothermal profile: ρ(z) = ρ₀ exp(-z²/(2h²))
            return rho_0 * np.exp(-(z**2) / (2 * h**2))

        elif profile_type == "polytropic":
            # Polytropic profile: ρ(z) = ρ₀ (1 - z²/(6h²))³
            return rho_0 * (1 - z**2 / (6 * h**2)) ** 3

        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    @staticmethod
    def mean_z_density(
        rho_0: float,
        h: float,
        z_h: float,
        profile_type: Literal["uniform", "isothermal", "polytropic"] = "isothermal",
        n_points: int = 1000,
    ) -> float:
        """
        Calculate the mean density along the z-direction from 0 to z_h.

        This computes the spatial average: <ρ> = (1/z_h) ∫[0 to z_h] ρ(z') dz'

        Parameters
        ----------
        rho_0 : float
            Central density [g/cm³]
        h : float
            Scale height [cm]
        z_h : float
            Jet head height [cm]
        profile_type : str
            Type of density profile
        n_points : int
            Number of points for numerical integration

        Returns
        -------
        float
            Mean density from z=0 to z=z_h [g/cm³]
        """
        if z_h <= 0:
            return rho_0

        # Create integration points from 0 to z_h
        z_points = np.linspace(0, z_h, n_points)

        # Calculate density at each point using the density profile
        rho_values = np.array(
            [DiskPhysics.density_profile(rho_0, h, z, profile_type) for z in z_points]
        )

        # Compute spatial average using trapezoidal integration
        # <ρ> = (1/z_h) ∫[0 to z_h] ρ(z') dz'
        integrated_density = np.trapz(rho_values, z_points)
        mean_density = integrated_density / z_h

        return mean_density
