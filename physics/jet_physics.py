"""
Jet dynamics calculations for the Chen & Dai 2025 model.

This module contains functions for calculating jet physical properties
such as jet head height, Lorentz factor, efficiency parameters, and
opening angles.
"""

import numpy as np

try:
    from ..constants import C_CGS
except ImportError:
    from constants import C_CGS


class JetPhysics:
    """Collection of functions for jet dynamics calculations."""

    @staticmethod
    def z_h(beta_h: float, t: float) -> float:
        """
        Calculate jet head height.

        Parameters
        ----------
        beta_h : float
            Jet head velocity parameter
        t : float
            Time [s]

        Returns
        -------
        float
            Jet head height [cm]
        """
        return beta_h * C_CGS * t

    @staticmethod
    def gamma_h(beta_h: float) -> float:
        """
        Calculate jet head Lorentz factor.

        Parameters
        ----------
        beta_h : float
            Jet head velocity parameter

        Returns
        -------
        float
            Jet head Lorentz factor
        """
        return 1 / np.sqrt(1 - beta_h**2)

    @staticmethod
    def eta_h(gamma_h: float, theta_h: float) -> float:
        """
        Calculate efficiency parameter (Equation 9).

        Parameters
        ----------
        gamma_h : float
            Jet head Lorentz factor
        theta_h : float
            Jet head opening angle [rad]

        Returns
        -------
        float
            Efficiency parameter
        """
        return np.minimum(2 / (gamma_h * theta_h), 1.0)

    @staticmethod
    def theta_h(sigma_h: float, z_h: float, theta_0: float) -> float:
        """
        Calculate jet head opening angle.

        Parameters
        ----------
        sigma_h : float
            Cross-sectional area [cm²]
        z_h : float
            Jet head height [cm]
        theta_0 : float
            Initial jet opening angle [rad]

        Returns
        -------
        float
            Jet head opening angle [rad]
        """
        # Natural evolution without artificial constraints
        # Let θ_h evolve according to the geometric relationship
        return np.sqrt(sigma_h / (np.pi * z_h**2))

    @staticmethod
    def small_gamma_h(gamma_h: float) -> float:
        """
        Calculate the initial Lorentz factor of the shocked gas.

        This is a convenience function that returns the Lorentz factor
        for the jet head.

        Parameters
        ----------
        gamma_h : float
            Jet head Lorentz factor

        Returns
        -------
        float
            Small gamma_h parameter
        """
        return gamma_h / np.sqrt(2)

    @staticmethod
    def l_tilde(l_j: float, sigma_h: float, rho_z: float) -> float:
        """
        Calculate dimensionless jet collimation parameter L̃ (Equation 8).

        Parameters
        ----------
        l_j : float
            Jet luminosity [erg/s]
        sigma_h : float
            Cross-sectional area [cm²]
        rho_z : float
            Local density [g/cm³]

        Returns
        -------
        float
            Dimensionless jet collimation parameter L̃
        """
        return l_j / (sigma_h * rho_z * C_CGS**3)
