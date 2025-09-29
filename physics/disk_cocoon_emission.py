"""
Disk-cocoon emission calculations for the Chen & Dai 2025 model.

This module handles the specific emission processes that occur when the
cocoon interacts with the disk material, subsequent to jet head shock breakout.
These calculations are distinct from jet-cocoon emission processes.
"""

import numpy as np

try:
    from ..constants import C_CGS, a, ev_in_k, m_pro, k_B_CGS, b, n
except ImportError:
    from constants import C_CGS, a, ev_in_k, m_pro, k_B_CGS, b, n


class DiskCocoonEmission:
    """
    Collection of functions for disk-cocoon emission calculations.

    This class handles the specific emission processes that occur when the
    cocoon interacts with the disk material, subsequent to jet head shock breakout.
    These calculations are distinct from jet-cocoon emission processes.
    """

    @staticmethod
    def cocoon_breakout_luminosity(r_c: float, rho_z: float, beta_c: float) -> float:
        """
        Calculate cocoon breakout luminosity. This function takes physical parameters
        at the time of shock breakout.

        Formula: L_c = π r_c² ρ(z) β_c³ c³

        Parameters
        ----------
        r_c : float
            Cocoon radius [cm]
        rho_z : float
            Local density at cocoon height [g/cm³]
        beta_c : float
            Cocoon velocity parameter

        Returns
        -------
        float
            Cocoon breakout luminosity [erg/s]
        """
        return np.pi * r_c**2 * rho_z * beta_c**3 * C_CGS**3

    @staticmethod
    def cocoon_breakout_emission_timescale(
        kappa: float, rho_z: float, beta_c: float
    ) -> float:
        """
        Calculate the cocoon breakout emission timescale.

        Formula: t_c = 1 / (κ * ρ(z) * β_c² c)

        Parameters
        ----------
        kappa : float
            Opacity [cm²/g]
        rho_z : float
            Local density at cocoon height [g/cm³]
        beta_c : float
            Cocoon velocity parameter

        Returns
        -------
        float
            Cocoon breakout emission timescale [s]
        """
        return 1 / (kappa * rho_z * beta_c**2 * C_CGS)

    @staticmethod
    def cocoon_initial_internal_energy(e_c: float, r_c: float, h: float) -> float:
        """
        Calculate the initial internal energy of the cocoon.

        Parameters
        ----------
        e_c : float
            Cocoon energy [erg]
        r_c : float
            Cocoon radius [cm]
        h : float
            Disk scale height [cm]

        Returns
        -------
        float
            Initial internal energy [erg]
        """
        return e_c * r_c / (2 * h)  # Initial internal energy

    @staticmethod
    def cocoon_mass(rho_z: float, v_c: float) -> float:
        """
        Calculate cocoon mass.

        Parameters
        ----------
        rho_z : float
            Local density at cocoon height [g/cm³]
        v_c : float
            Cocoon volume [cm³]

        Returns
        -------
        float
            Cocoon mass [g]
        """
        return rho_z * v_c

    @staticmethod
    def cocoon_initial_diffusion_timescale(
        kappa: float, r_c: float, h: float, m_c: float
    ) -> float:
        """
        Calculate the initial diffusion timescale for the cocoon.

        Parameters
        ----------
        kappa : float
            Opacity [cm²/g]
        r_c : float
            Cocoon radius [cm]
        h : float
            Disk scale height [cm]
        m_c : float
            Cocoon mass [g]

        Returns
        -------
        float
            Initial diffusion timescale [s]
        """
        return kappa * (r_c / h) * m_c / (b * C_CGS * r_c)

    @staticmethod
    def expansion_time_scale(r_c: float, beta_c: float) -> float:
        """
        Calculate the initial shell expansion timescale.

        This is a simplified version that assumes the shell is thin and
        uses the cocoon radius and velocity.

        Parameters
        ----------
        r_c : float
            Cocoon radius [cm]
        beta_c : float
            Cocoon velocity parameter

        Returns
        -------
        float
            Initial shell diffusion timescale [s]
        """
        return r_c / (beta_c * C_CGS)

    @staticmethod
    def effective_diffusion_timescale(t_c_init: float, t_s_init: float) -> float:
        """
        Calculate the effective diffusion timescale.

        This is a simple average of the initial cocoon and shell diffusion timescales.

        Parameters
        ----------
        t_c_init : float
            Initial cocoon diffusion timescale [s]
        t_s_init : float
            Initial shell diffusion timescale [s]

        Returns
        -------
        float
            Effective diffusion timescale [s]
        """
        return np.sqrt(t_c_init * t_s_init)

    @staticmethod
    def initial_spherical_cocoon_luminosity(e_c_init: float, t_c_init: float) -> float:
        """
        Calculate the initial cocoon luminosity assuming spherical symmetry.

        Formula: L_c_init = E_c_init / t_c_init

        Parameters
        ----------
        e_c_init : float
            Initial cocoon energy [erg]
        t_c_init : float
            Initial cocoon diffusion timescale [s]

        Returns
        -------
        float
            Initial cocoon luminosity [erg/s]
        """
        return e_c_init / t_c_init if t_c_init > 0 else 0.0

    @staticmethod
    def planar_timescale(t_c_bre: float, l_c_bre: float, l_c_sph: float) -> float:
        """
        Calculate the planar timescale for cocoon evolution.

        This is a simplified version that assumes the cocoon evolves in a planar manner
        and uses the breakout luminosity and spherical luminosity.

        Parameters
        ----------
        t_c_bre : float
            Cocoon breakout emission timescale [s]
        l_c_bre : float
            Cocoon breakout luminosity [erg/s]
        l_c_sph : float
            Spherical cocoon luminosity [erg/s]

        Returns
        -------
        float
            Planar timescale [s]
        """
        return t_c_bre * (l_c_bre / l_c_sph) ** (3 / 4)

    @staticmethod
    def blackbody_temp(rho_z_bre: float, beta_c_bre: float) -> float:
        return (18 * rho_z_bre * beta_c_bre**2 * C_CGS**2 / (7 * a)) ** 0.25

    @staticmethod
    def thermal_eq_temp(
        t: float,
        t_c_bre: float,
        t_c_pla: float,
        t_c_diff: float,
        rho_z_bre: float,
        beta_c_bre: float,
    ) -> float:
        temp_bb_cbre = DiskCocoonEmission.blackbody_temp(rho_z_bre, beta_c_bre)
        if t <= t_c_bre:
            return temp_bb_cbre * np.exp(1 - t_c_bre / t) ** (1 / 4)
        elif t_c_bre < t <= t_c_pla:
            return temp_bb_cbre * (t / t_c_bre) ** (-1 / 3)
        elif t > t_c_pla:
            return (
                temp_bb_cbre
                * np.exp(-((t - t_c_pla) * (t + t_c_pla)) / (2 * t_c_diff**2))
                ** (1 / 4)
                * (t / t_c_pla) ** (-1 / 2)
            )

    @staticmethod
    def cocoon_comptonization_radiation_temp(
        rho_z_bre: float, beta_c_bre: float
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
            + 1.735 * np.sqrt(beta_c_bre / 0.1)
            + (0.26 - 0.08 * np.sqrt(beta_c_bre / 0.1))
            * np.log10(number_density / 1e15)
        )
        return 10**exponent * ev_in_k

    @staticmethod
    def thermalization_degree(temp_bb: float, t_cp: float, rho_z_bre: float) -> float:
        n_bb = a * temp_bb**4 / (3 * k_B_CGS * temp_bb)
        ndot_ph = 3.5e36 * rho_z_bre**2 * temp_bb ** (-1 / 2)
        return n_bb / (t_cp * ndot_ph)

    @staticmethod
    def find_t_c_th(rho_z_bre: float, beta_c_bre: float, times: np.ndarray) -> float:
        """
        Find the thermalization time t_c_th where thermalization degree = 1.

        Optimized version using vectorized operations for better performance.

        Parameters
        ----------
        rho_z_bre : float
            Density at breakout height [g/cm³]
        beta_c_bre : float
            Cocoon velocity at breakout
        times : np.ndarray
            Time array to search through

        Returns
        -------
        float
            Thermalization time [s]
        """
        temp_bb = DiskCocoonEmission.blackbody_temp(rho_z_bre, beta_c_bre)

        # Vectorized calculation for much better performance
        n_bb = a * temp_bb**4 / (3 * k_B_CGS * temp_bb)
        ndot_ph = 3.5e36 * rho_z_bre**2 * temp_bb ** (-1 / 2)

        # Vectorized thermalization degree calculation
        therm_degree_array = n_bb / (times * ndot_ph)

        # Find closest to 1
        t_c_th_idx = np.argmin(np.abs(therm_degree_array - 1))
        return times[t_c_th_idx]

    @staticmethod
    def comptonization_temp(
        t: float,
        t_c_bre: float,
        t_c_th: float,
        t_c_pla: float,
        t_c_diff: float,
        rho_z_bre: float,
        beta_c_bre: float,
    ) -> float:
        """
        Calculate comptonization temperature with smooth transitions between regimes.

        This implementation ensures continuity at regime boundaries by enforcing
        that each regime matches the previous one at the transition point.
        The physical temperature scale is set by the first regime only.
        """
        temp_comp_cbre = DiskCocoonEmission.cocoon_comptonization_radiation_temp(
            rho_z_bre, beta_c_bre
        )

        # Regime 1: t ≤ t_c_bre (sets the physical temperature scale)
        if t <= t_c_bre:
            return temp_comp_cbre * np.exp(1 - t_c_bre / t) ** 2 * (t / t_c_bre) ** (-2)

        # Calculate temperature at end of regime 1 (t = t_c_bre) for continuity
        temp_1_end = temp_comp_cbre  # At t = t_c_bre, exp and power terms = 1

        # Regime 2: t_c_bre < t ≤ t_c_pla (pure scaling relation)
        if t <= t_c_pla:
            # Raw scaling relation (without temp_comp_cbre)
            scaling_2 = (t / t_c_bre) ** (-2 / 3)
            # Apply to regime 1 end temperature to ensure continuity
            return temp_1_end * scaling_2

        # Calculate temperature at end of regime 2 for continuity
        temp_2_end = temp_1_end * (t_c_pla / t_c_bre) ** (-2 / 3)

        # Regime 3: t_c_pla < t ≤ t_c_th (pure scaling relation)
        if t <= t_c_th:
            # Raw scaling relation (without temp_comp_cbre)
            scaling_3 = (t / t_c_pla) ** (-1)
            # Apply to regime 2 end temperature to ensure continuity
            return temp_2_end * scaling_3

        # Calculate temperature at end of regime 3 for continuity
        temp_3_end = temp_2_end * (t_c_th / t_c_pla) ** (-1)

        # Regime 4: t > t_c_th (pure scaling relation)
        # Raw scaling relation (without temp_comp_cbre)
        exp_factor = np.exp(-((t - t_c_pla) * (t + t_c_pla)) / (2 * t_c_diff**2)) ** (
            1 / 4
        )
        power_factor = (t / t_c_pla) ** (-1 / 2)

        # Scaling at boundary to match regime 3
        exp_factor_boundary = np.exp(
            -((t_c_th - t_c_pla) * (t_c_th + t_c_pla)) / (2 * t_c_diff**2)
        ) ** (1 / 4)
        power_factor_boundary = (t_c_th / t_c_pla) ** (-1 / 2)
        scaling_4_boundary = exp_factor_boundary * power_factor_boundary

        # Current scaling
        scaling_4 = exp_factor * power_factor

        # Continuity factor and final result
        if scaling_4_boundary != 0:
            return temp_3_end * (scaling_4 / scaling_4_boundary)
        else:
            return temp_3_end
