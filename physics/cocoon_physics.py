"""
Cocoon dynamics calculations for the Chen & Dai 2025 model.

This module contains functions for calculating cocoon physical properties
such as energy, velocity parameters, radius, volume, and cross-sections.
"""

import numpy as np

try:
    from ..constants import C_CGS
except ImportError:
    from constants import C_CGS


class CocoonPhysics:
    """Collection of functions for cocoon dynamics calculations."""

    @staticmethod
    def internal_energy(eta_h: float, l_j: float, beta_h: float, t: float) -> float:
        """
        Calculate cocoon energy E_c.

        Parameters
        ----------
        eta_h : float
            Efficiency parameter
        l_j : float
            Jet luminosity [erg/s]
        beta_h : float
            Jet head velocity parameter
        t : float
            Time [s]

        Returns
        -------
        float
            Cocoon energy [erg]
        """
        return eta_h * l_j * (1 - beta_h) * t

    @staticmethod
    def energy(eta_h: float, l_j: float, beta_h: float, t: float) -> float:
        """
        Calculate cocoon energy E_c.

        Backward compatibility alias for internal_energy.

        Parameters
        ----------
        eta_h : float
            Efficiency parameter
        l_j : float
            Jet luminosity [erg/s]
        beta_h : float
            Jet head velocity parameter
        t : float
            Time [s]

        Returns
        -------
        float
            Cocoon energy [erg]
        """
        return CocoonPhysics.internal_energy(eta_h, l_j, beta_h, t)

    @staticmethod
    def beta_c_paper_formula(
        energy: float, mean_rho_z: float, z_h: float, t: float
    ) -> tuple:
        """
        Calculate cocoon velocity parameter using the paper's exact formula.

        From the paper: β_c = [E_c / (3π ρ(z_h) c² z_h r_c²)]^(1/2)
        where r_c = β_c * c * t, creating a circular dependency.

        Solving: β_c = [E_c / (3π ρ(z_h) c⁴ z_h t²)]^(1/4) - this was the working version
        But the self-consistent solution gives: β_c = [E_c / (3π ρ(z_h) c⁴ z_h t²)]^(1/6)

        Parameters
        ----------
        energy : float
            Cocoon energy [erg]
        mean_rho_z : float
            Mean density at z_h [g/cm³]
        z_h : float
            Jet head height [cm]
        t : float
            Time [s]

        Returns
        -------
        tuple
            (beta_c, r_c) - both calculated self-consistently
        """
        # Protect against extremely small densities
        min_density = 1e-30  # Minimum physical density
        safe_mean_rho_z = max(mean_rho_z, min_density)

        # Use the correct paper formula with circular dependency resolved
        # β_c = [E_c / (3π ρ c⁴ z_h t²)]^(1/4) - this was the working version
        denominator = 3 * np.pi * safe_mean_rho_z * (C_CGS**4) * z_h * (t**2)

        # Additional safety check for numerical stability
        if denominator <= 0 or energy <= 0:
            beta_c = 1e-6  # Return very small value
            r_c = beta_c * C_CGS * t
            return beta_c, r_c

        try:
            beta_c = (energy / denominator) ** (1 / 4)

            # Check for NaN or inf
            if not np.isfinite(beta_c):
                beta_c = 1e-6

        except (OverflowError, ZeroDivisionError, RuntimeWarning):
            beta_c = 1e-6

        # Ensure β_c is physical (< 1)
        beta_c = min(beta_c, 0.99)
        beta_c = max(beta_c, 1e-6)  # Also ensure it's not too small

        # Calculate r_c using the solved β_c
        r_c = beta_c * C_CGS * t

        return beta_c, r_c

    @staticmethod
    def radius(beta_c: float, t: float) -> float:
        """
        Calculate cocoon radius.

        Parameters
        ----------
        beta_c : float
            Cocoon velocity parameter
        t : float
            Time [s]

        Returns
        -------
        float
            Cocoon radius [cm]
        """
        return beta_c * C_CGS * t

    @staticmethod
    def volume(r_c: float, z_h: float) -> float:
        """
        Calculate cocoon volume, approximated as a cylinder.

        Parameters
        ----------
        r_c : float
            Cocoon radius [cm]
        z_h : float
            Jet head height [cm]

        Returns
        -------
        float
            Cocoon volume [cm³]
        """
        return np.pi * r_c**2 * z_h

    @staticmethod
    def cross_section(
        l_j: float, z_h: float, r_c: float, e_c: float, theta_0: float
    ) -> float:
        """
        Calculate jet cross-sectional area Σ_h.

        Parameters
        ----------
        l_j : float
            Jet luminosity [erg/s]
        z_h : float
            Jet head height [cm]
        r_c : float
            Cocoon radius [cm]
        e_c : float
            Cocoon energy [erg]
        theta_0 : float
            Initial jet opening angle [rad]

        Returns
        -------
        float
            Cross-sectional area [cm²]
        """
        term1 = z_h**2
        term2 = 3 * l_j * z_h * r_c**2 / (4 * C_CGS * e_c)
        return np.pi * theta_0**2 * np.minimum(term1, term2)

    @staticmethod
    def theta_c(r_c: float, z_h: float) -> float:
        """
        Calculate cocoon opening angle.

        Parameters
        ----------
        r_c : float
            Cocoon radius [cm]
        z_h : float
            Jet head height [cm]

        Returns
        -------
        float
            Cocoon opening angle
        """
        return r_c / z_h
