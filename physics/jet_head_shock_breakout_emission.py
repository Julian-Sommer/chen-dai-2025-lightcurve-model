"""
Jet head shock breakout emission calculations for the Chen & Dai 2025 model.

This module contains functions for calculating jet head shock breakout
parameters including shell width, breakout energy, emission duration,
luminosity, temperature calculations, and relativistic evolution.
"""

import numpy as np

try:
    from ..constants import C_CGS, a, ev_in_k, m_pro
except ImportError:
    from constants import C_CGS, a, ev_in_k, m_pro


class JetHeadShockBreakoutEmission:
    """Collection of functions for jet head shock breakout emission calculations."""

    @staticmethod
    def shock_breakout_shell_width(
        kappa: float, rho_z_bre: float, beta_h_bre: float
    ) -> float:
        """
        Calculate the shell width at shock breakout.

        Formula: d_bre = 1 / (κ * ρ(z_bre) * β_h)

        Parameters
        ----------
        kappa : float
            Opacity [cm²/g]
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        beta_h_bre : float
            Jet head velocity parameter at shock breakout

        Returns
        -------
        float
            Shell width at shock breakout [cm]
        """
        # Protect against division by zero or very small values
        denominator = kappa * rho_z_bre * beta_h_bre

        if denominator <= 0:
            # Return a large shell width if denominator is invalid
            return 1e20  # cm (very large, indicating no meaningful breakout)

        return 1.0 / denominator

    @staticmethod
    def breakout_energy(
        rho_z_bre: float, beta_h_bre: float, sigma_h_bre: float, d_bre: float
    ) -> float:
        """
        Calculate the energy released during shock breakout.

        Parameters
        ----------
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        beta_h_bre : float
            Jet head velocity parameter at shock breakout
        sigma_h_bre : float
            Cross-sectional area at shock breakout [cm²]
        d_bre : float
            Shell width at shock breakout [cm]

        Returns
        -------
        float
            Breakout energy [erg]
        """
        return rho_z_bre * sigma_h_bre * d_bre * beta_h_bre**2 * C_CGS**2

    @staticmethod
    def duration_of_emission(
        kappa: float, rho_z_bre: float, beta_h_bre: float
    ) -> float:
        """
        Calculate the duration of shock breakout emission.

        Parameters
        ----------
        kappa : float
            Opacity [cm²/g]
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        beta_h_bre : float
            Jet head velocity parameter at shock breakout

        Returns
        -------
        float
            Duration of emission [s]
        """
        return 1 / (kappa * rho_z_bre * beta_h_bre**2 * C_CGS)

    @staticmethod
    def jet_head_shock_breakout_lum(
        sigma_h_bre: float, rho_z_bre: float, beta_h_bre: float
    ) -> float:
        """
        Calculate the jet head shock breakout luminosity.

        Formula: L_bre = σ_h * ρ(z_bre) * β_h³ * c³

        Parameters
        ----------
        sigma_h_bre : float
            Cross-sectional area at shock breakout [cm²]
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        beta_h_bre : float
            Jet head velocity parameter at shock breakout

        Returns
        -------
        float
            Shock breakout luminosity [erg/s]
        """
        return sigma_h_bre * rho_z_bre * beta_h_bre**3 * C_CGS**3

    @staticmethod
    def typical_radiation_temperature(rho_z_bre: float, beta_h_bre: float) -> float:
        """
        Calculate the typical radiation temperature.

        Parameters
        ----------
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        beta_h_bre : float
            Jet head velocity parameter at shock breakout

        Returns
        -------
        float
            Typical radiation temperature [K]
        """
        return (18 / (7 * a) * rho_z_bre * beta_h_bre**2 * C_CGS**2) ** 0.25

    @staticmethod
    def comptonization_radiation_temperature(
        rho_z_bre: float, beta_h_bre: float
    ) -> float:
        """
        Calculate the comptonization radiation temperature.

        Parameters
        ----------
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        beta_h_bre : float
            Jet head velocity parameter at shock breakout

        Returns
        -------
        float
            Comptonization radiation temperature [K]
        """
        number_density = rho_z_bre / m_pro
        exponent = (
            0.975
            + 1.735 * np.sqrt(beta_h_bre / 0.1)
            + (0.26 - 0.08 * np.sqrt(beta_h_bre / 0.1))
            * np.log10(number_density / 1e15)
        )
        return 10**exponent * ev_in_k

    @staticmethod
    def relativistic_jet_evolution(
        s_gamma_h: float, m_shell: float, kappa: float, rho_z: float, sigma_h: float
    ) -> tuple[float, float]:
        """
        Calculate the relativistic radiation temperature and luminosity.

        Uses the relativistic shell evolution to determine thermal parameters
        and calculates the observed temperature and luminosity for relativistic
        jet head shock breakout.

        Parameters
        ----------
        s_gamma_h : float
            Small gamma_h parameter (initial Lorentz factor)
        m_shell : float
            Shell mass [g]
        kappa : float
            Opacity [cm²/g]
        rho_z : float
            Density [g/cm³]
        sigma_h : float
            Cross-sectional area [cm²]

        Returns
        -------
        tuple[float, float]
            (temp_hbre_obs, l_hbre_rel) - Observed temperature [K] and luminosity [erg/s]
        """
        e_0 = s_gamma_h**2 * m_shell * C_CGS**2
        t_hth, gamma_hth, temp_frac = (
            JetHeadShockBreakoutEmission.relativistic_shell_evolution(
                kappa, rho_z, s_gamma_h, sigma_h
            )
        )
        e_hbre_rel = e_0 * gamma_hth / s_gamma_h * (1 / temp_frac)
        t_hbre_obs = t_hth / (2 * gamma_hth**2)
        l_hbre_rel = e_hbre_rel / t_hbre_obs
        temp_th = 50 * 1e3 * ev_in_k  # Thermal temperature in K
        temp_hbre_obs = temp_th * gamma_hth
        return temp_hbre_obs, l_hbre_rel

    @staticmethod
    def relativistic_shell_evolution(
        kappa: float, rho_z: float, gamma_h: float, sigma_h: float
    ) -> tuple[float, float, float]:
        """
        Calculate relativistic shell evolution following Chen & Dai 2025 Appendix A.

        Based on the adiabatic evolution of a relativistic breakout shell,
        this function determines the thermal time t_hth and corresponding
        Lorentz factor gamma_hth based on various physical conditions.

        Parameters
        ----------
        kappa : float
            Opacity [cm²/g]
        rho_z : float
            Density [g/cm³]
        gamma_h : float
            Initial jet head Lorentz factor
        sigma_h : float
            Cross-sectional area [cm²]

        Returns
        -------
        tuple[float, float, float]
            (t_hth, gamma_hth, temp_frac) - Thermal time, corresponding Lorentz factor, and temperature fraction
        """
        # Initialize output variables that will be determined by case distinctions
        t_hth = None
        gamma_hth = None

        # Calculate fundamental timescales and parameters (following paper notation)
        t_0 = 1 / (kappa * rho_z * C_CGS)  # Characteristic time
        t_f = t_0 * gamma_h ** (3 + np.sqrt(3))  # Final time for planar phase
        t_hs = (sigma_h / np.pi) ** (
            1 / 2
        ) / C_CGS  # Time when shell becomes transparent

        # Temperature parameters (from paper: T_h ~ 200 keV, T_th ~ 50 keV)
        temp_h = 200  # Hot temperature in keV
        temp_th = 50  # Thermal temperature in keV
        temp_frac = temp_h / temp_th  # Temperature ratio (= 4)

        # Critical Lorentz factor (Equation A5 from paper)
        gamma_hs = ((sigma_h / np.pi) ** (1 / 2) / (t_0 * C_CGS)) ** (
            1 / (3 + np.sqrt(3))
        )

        # Case distinction following the paper's logic structure
        # The paper shows different evolution modes based on γ_h vs γ_hs comparison

        if gamma_h <= gamma_hs:
            # Case 1: γ_h ≤ γ_hs (acceleration terminates during planar phase)

            # Calculate initial thermal time estimate
            initial_t_hth = t_0 * (temp_frac) ** (3 + np.sqrt(3))

            if initial_t_hth < t_f:
                # Subcase 1a: Thermal time occurs during planar acceleration
                t_hth = initial_t_hth
                gamma_hth = gamma_h * (t_hth / t_0) ** ((np.sqrt(3) - 1) / 2)

            elif initial_t_hth > t_f and initial_t_hth <= t_hs:
                # Subcase 1b: Thermal time occurs during spherical phase
                t_hth = t_0 * (temp_frac) ** 3 * gamma_h ** np.sqrt(3)
                # In spherical phase, use the dynamical evolution to find gamma at t_hth
                # Note: gamma_hth will be determined later by thermal conditions

            elif initial_t_hth > t_hs:
                # Subcase 1c: Thermal time occurs after shell becomes transparent
                t_hth = (
                    t_0
                    * temp_frac
                    * gamma_h ** (np.sqrt(3) / 3)
                    * (t_0 / t_hs) ** (1 / 3)
                )
                # gamma_hth will be determined by thermal conditions below

            # Additional conditions for gamma_hth determination based on temperature comparison
            if temp_frac < gamma_h:
                gamma_hth = gamma_h * temp_frac ** np.sqrt(3)
            elif temp_frac >= gamma_h:
                gamma_hth = gamma_h ** (np.sqrt(3) + 1)  # Use final Lorentz factor

        elif gamma_h > gamma_hs:
            # Case 2: γ_h > γ_hs (acceleration ends in spherical phase)

            initial_t_hth = t_0 * (temp_frac) ** (3 + np.sqrt(3))

            if initial_t_hth <= t_hs:
                # Subcase 2a: Thermal time before shell transparency
                t_hth = initial_t_hth
                gamma_hth = gamma_h * (temp_frac) ** np.sqrt(3)

            else:
                # Subcase 2b: Thermal time after shell transparency
                t_hth = t_hs
                gamma_hth = gamma_h * (t_0 / t_hs) ** ((1 - np.sqrt(3)) / 2)

        # Safety checks - ensure we have valid return values
        if t_hth is None:
            # Fallback to characteristic time
            t_hth = t_0

        if gamma_hth is None:
            # Fallback to initial Lorentz factor
            gamma_hth = gamma_h

        # Ensure physical values
        t_hth = max(t_hth, 0.0)  # Time must be positive
        gamma_hth = max(gamma_hth, 1.0)  # Lorentz factor must be ≥ 1

        return t_hth, gamma_hth, temp_frac

    @staticmethod
    def determine_temperature(
        beta_h_bre: float,
        rho_z_bre: float,
        s_gamma_h: float,
        m_shell: float,
        kappa: float,
        rho_z: float,
        sigma_h: float,
    ) -> float:
        """
        Determine the appropriate radiation temperature based on jet velocity.

        Parameters
        ----------
        beta_h_bre : float
            Jet head velocity parameter at shock breakout
        rho_z_bre : float
            Density at shock breakout height [g/cm³]
        s_gamma_h : float
            Small gamma_h parameter (initial Lorentz factor)
        m_shell : float
            Shell mass [g]
        kappa : float
            Opacity [cm²/g]
        rho_z : float
            Density [g/cm³]
        sigma_h : float
            Cross-sectional area [cm²]

        Returns
        -------
        float
            Radiation temperature [K]
        """
        if beta_h_bre < 0.03:
            # For very slow jets, use the typical radiation temperature
            return JetHeadShockBreakoutEmission.typical_radiation_temperature(
                rho_z_bre, beta_h_bre
            )
        elif beta_h_bre >= 0.03 and beta_h_bre < 0.4:
            # For moderate speeds, use the comptonization temperature
            return JetHeadShockBreakoutEmission.comptonization_radiation_temperature(
                rho_z_bre, beta_h_bre
            )
        else:
            # For relativistic speeds, use the full relativistic calculation
            temp_hbre_obs, _ = JetHeadShockBreakoutEmission.relativistic_jet_evolution(
                s_gamma_h, m_shell, kappa, rho_z, sigma_h
            )
            return temp_hbre_obs

    @staticmethod
    def shell_mass(rho_z: float, sigma_h: float, d: float) -> float:
        """
        Calculate the mass of the shell at shock breakout.

        Formula: M_shell = ρ(z) * σ_h * d

        Parameters
        ----------
        rho_z : float
            Density at shock breakout height [g/cm³]
        sigma_h : float
            Cross-sectional area [cm²]
        d : float
            Shell width at shock breakout [cm]

        Returns
        -------
        float
            Shell mass [g]
        """
        return rho_z * sigma_h * d
