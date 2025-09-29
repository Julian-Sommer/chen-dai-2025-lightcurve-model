import numpy as np
from scipy.optimize import brentq

try:
    from ..constants import (
        C_CGS,
        a,
        ev_in_k,
        m_pro,
        k_B_CGS,
        b,
        f_beta_gamma,
        s_exponent,
        theta_cj,
        f_d,
    )
except ImportError:
    from constants import (
        C_CGS,
        a,
        ev_in_k,
        m_pro,
        k_B_CGS,
        b,
        f_beta_gamma,
        s_exponent,
        theta_cj,
        f_d,
    )


class JetCocoonEmission:
    """
    Collection of functions for jet-cocoon emission calculations.

    This class handles the specific emission processes that occur when the
    cocoon interacts with the jet material, subsequent to jet head shock breakout.
    These calculations are distinct from disk-cocoon emission processes.
    """

    # Pair-regulated temperature limits (observed-frame)
    # 200 keV upper cap while pairs regulate; 50 keV transparency threshold
    T_PAIR_MAX_K = 200e3 * ev_in_k
    T_PAIR_MIN_K = 50e3 * ev_in_k
    # Verbosity switch for diagnostic prints
    VERBOSE = False

    @staticmethod
    def critical_velocity(kappa: float, e_cj: float, r_c: float) -> float:
        """
        Calculate the critical velocity at which r_cj = r_c.
        """
        return kappa * f_beta_gamma * e_cj / (2 * np.pi * r_c**2 * C_CGS**2)

    @staticmethod
    def velocity_of_first_shell(beta_cr: float) -> float:
        """
        Calculate the velocity of the first shell that
        produces diffusing thermal emission.
        """
        return np.min([0.7, beta_cr])

    @staticmethod
    def total_jet_cocoon_energy(e_c: float) -> float:
        return e_c / 2

    @staticmethod
    def newtonian_jet_cocoon_mass(e_cj: float, beta_cj: float) -> float:
        return 2 * f_beta_gamma * e_cj / (beta_cj**2 * C_CGS**2)

    @staticmethod
    def initial_jet_cocoon_volume(r_c: float, z_bre: float) -> float:
        return f_beta_gamma * np.pi * r_c**2 * z_bre

    @staticmethod
    def diffusion_radius(kappa: float, e_cj: float, beta_cj: float) -> float:
        return np.sqrt(kappa * f_beta_gamma * e_cj / (2 * np.pi * beta_cj * C_CGS**2))

    @staticmethod
    def diffustion_time(kappa: float, e_cj: float, beta_cj: float) -> float:
        return np.sqrt(
            kappa * f_beta_gamma * e_cj / (2 * np.pi * beta_cj**3 * C_CGS**4)
        )

    @staticmethod
    def diffusion_time(kappa: float, e_cj: float, beta_cj: float) -> float:
        """
        Correctly spelled alias of diffustion_time (kept for backward compatibility).
        """
        return np.sqrt(
            kappa * f_beta_gamma * e_cj / (2 * np.pi * beta_cj**3 * C_CGS**4)
        )

    @staticmethod
    def emission_luminosity(
        e_cj: float, v_cj_beta: float, t_cj: float, r_cj: float
    ) -> float:
        """
        Calculate the emission luminosity from the jet cocoon.
        """
        return (f_beta_gamma * e_cj / t_cj) * (v_cj_beta ** (1 / 3) / r_cj)

    @staticmethod
    def radiative_time(t_cjs: float, beta_cjs: float, beta_c: float) -> float:
        """
        Calculate the radiative time for the jet cocoon emission.
        This is the time when the mixed jet cocoon becomes radiative.
        """
        return t_cjs * (beta_cjs / beta_c) ** ((s_exponent + 2) / s_exponent)

    @staticmethod
    def shell_distance(z_bre: float) -> float:
        """
        Calculate the distance of the shell from the breakout point.
        This is the distance at which the first shell produces diffusing thermal emission.
        """
        return f_d * z_bre

    @staticmethod
    def shell_escaping_time(d_cj: float, beta_cjs: float) -> float:
        """
        Calculate the time it takes for the shell to escape.
        This is the time when the first shell produces diffusing thermal emission.
        """
        return d_cj / (beta_cjs * C_CGS)

    @staticmethod
    def initial_bb_temp(e_cj: float, v_cj_beta: float) -> float:
        """
        Calculate the initial blackbody temperature of the jet cocoon.
        """
        # Paper normalization: T_BB,cj0 = (f_beta_gamma * E_cj / (a V))^(1/4)
        return (f_beta_gamma * e_cj / (a * v_cj_beta)) ** 0.25

    @staticmethod
    def m_cj_s(e_cj: float) -> float:
        """
        Deprecated helper. Use newtonian_jet_cocoon_mass(e_cj, beta) instead.
        """
        raise NotImplementedError(
            "m_cj_s is deprecated. Use newtonian_jet_cocoon_mass(e_cj, beta) instead."
        )

    @staticmethod
    def calculate_jet_cocoon_emission_evolution(
        t: float, l_cjs: float, t_cjs: float, t_sph_end: float
    ) -> float:
        if t <= t_cjs:
            return 0.0  # No emission before jet-cocoon diffusion time
        elif t_cjs < t <= t_sph_end:
            return l_cjs * (t / t_cjs) ** (-4 / (s_exponent + 2))
        elif t > t_sph_end:
            return (
                l_cjs
                * (t_sph_end / t_cjs) ** (-4 / (s_exponent + 2))
                * np.exp(-1 / 2 * (t**2 / (t_sph_end**2) - 1))
            )
        else:
            return 0.0  # Fallback case

    @staticmethod
    def initial_thermal_coupling_coefficient(
        e_cj: float, v_cj: float, beta_cjs: float, m_cjs: float, z_bre: float
    ) -> float:
        A = 1.05e37 * k_B_CGS / (a ** (1 / 8))
        return (
            e_cj ** (7 / 8)
            * v_cj ** (9 / 8)
            * beta_cjs
            * C_CGS
            / (A * m_cjs**2 * f_d * z_bre)
        )

    @staticmethod
    def thermal_coupling_coefficient(
        eta_escs: float, t_cjs: float, t_escs: float
    ) -> float:
        """
        Calculate the thermal coupling coefficient for the cocoon-jet material.
        """
        return eta_escs * (t_cjs / t_escs) ** (3 / 2)

    @staticmethod
    def thermal_eq_temperature(
        t: float, t_cjs: float, t_thph: float, t_sphend: float, temp_norm: float = 1.0
    ) -> float:
        """
        Calculate the thermal equilibrium temperature of the cocoon-jet material with smooth transitions.

        This implementation ensures continuity at regime boundaries by enforcing
        that each regime matches the previous one at the transition point.
        The physical temperature scale is set by the normalization temperature.

        Parameters
        ----------
        t : float
            Time [s]
        t_cjs : float
            Jet-cocoon diffusion time [s]
        t_thph : float
            Thermal to photospheric transition time [s]
        t_sphend : float
            End of spherical emission time [s]
        temp_norm : float, optional
            Normalization temperature [K]. Default is 1.0 for backward compatibility.

        Returns
        -------
        float
            Temperature [K]
        """
        if t <= t_cjs:
            return 0.0  # No temperature before jet-cocoon diffusion time

        # Regime 1: t_cjs < t ≤ t_thph (sets the physical temperature scale)
        if t <= t_thph:
            scaling_1 = (t / t_cjs) ** (
                -2
                * (5 * s_exponent**2 + 27 * s_exponent + 50)
                / ((s_exponent + 2) * (17 * s_exponent + 45))
            )
            return temp_norm * scaling_1

        # Calculate temperature at end of regime 1 for continuity
        temp_1_end = temp_norm * (t_thph / t_cjs) ** (
            -2
            * (5 * s_exponent**2 + 27 * s_exponent + 50)
            / ((s_exponent + 2) * (17 * s_exponent + 45))
        )

        # Regime 2: t_thph < t ≤ t_sphend (pure scaling relation)
        if t <= t_sphend:
            # Raw scaling relation (without temp_norm)
            scaling_2 = (t / t_thph) ** (
                -(s_exponent**2 + 5 * s_exponent + 8)
                / (2 * (s_exponent + 2) * (s_exponent + 3))
            )
            # Apply to regime 1 end temperature to ensure continuity
            return temp_1_end * scaling_2

        # Calculate temperature at end of regime 2 for continuity
        temp_2_end = temp_1_end * (t_sphend / t_thph) ** (
            -(s_exponent**2 + 5 * s_exponent + 8)
            / (2 * (s_exponent + 2) * (s_exponent + 3))
        )

        # Regime 3: t > t_sphend (exponential decay + power law)
        if t > t_sphend:
            # Raw scaling relation components
            exp_factor = np.exp(-1 / 2 * (t**2 / (t_sphend**2) - 1)) ** (1 / 4)
            power_factor = (t / t_sphend) ** (
                -(s_exponent + 1) / (2 * (s_exponent + 3))
            )

            # Scaling at boundary to match regime 2
            exp_factor_boundary = np.exp(
                -1 / 2 * (t_sphend**2 / (t_sphend**2) - 1)
            ) ** (
                1 / 4
            )  # = 1
            power_factor_boundary = (t_sphend / t_sphend) ** (
                -(s_exponent + 1) / (2 * (s_exponent + 3))
            )  # = 1
            scaling_3_boundary = exp_factor_boundary * power_factor_boundary  # = 1

            # Current scaling
            scaling_3 = exp_factor * power_factor

            # Apply to regime 2 end temperature to ensure continuity
            return temp_2_end * scaling_3

        else:
            return 0.0  # Fallback case

    @staticmethod
    def normalization_temp_thermal_eq(
        l_cjs: float, beta_cjs: float, r_cjs: float
    ) -> float:
        """
        Calculate the normalization temperature for the thermal equilibrium temperature.
        This is used to normalize the temperature in the thermal equilibrium calculations.
        """
        return (l_cjs / (a * C_CGS * beta_cjs * r_cjs**2)) ** (1 / 4)

    @staticmethod
    def transition_time_th_ph(t_cjs: float, beta_cjs: float, eta_cjs: float) -> float:
        """
        Calculate the transition time from thermal to photospheric emission.
        This is the time when the cocoon-jet material transitions from thermal to photospheric emission.
        """
        return (
            t_cjs
            * beta_cjs
            ** (
                -((s_exponent + 2) * (17 * s_exponent + 45))
                / (6 * s_exponent**2 + 46 * s_exponent + 96)
            )
            * eta_cjs
            ** (
                (4 * (s_exponent + 2) * (s_exponent + 3))
                / (3 * s_exponent**2 + 23 * s_exponent + 48)
            )
        )

    @staticmethod
    def rho_cj_s(m_cjs: float, v_cjs: float) -> float:
        """
        Calculate the density of the cocoon-jet material in the shell.
        This is the density at which the cocoon-jet material is mixed with the jet material.
        """
        return m_cjs / (v_cjs)

    @staticmethod
    def temp_esc(temp_bb_cj0: float, eta_escs: float, rho_cjs: float) -> float:
        """
        Solve for the Compton-modified escape temperature T_esc
        using the full condition including ξ(T).

        This solves the equation: T_esc * ξ(T_esc)² = T_BB,cj0 * η_esc²
        where ξ is the Compton parameter that depends on temperature and density.

        Parameters
        ----------
        temp_bb_cj0 : float
            Initial blackbody temperature [K]
        eta_escs : float
            Escape thermal coupling coefficient
        rho_cjs : float
            Cocoon-jet density [g/cm³]

        Returns
        -------
        float
            Escape temperature [K]
        """
        rhs = temp_bb_cj0 * eta_escs**2

        def equation(temp_eV):
            """
            Equation to solve: T_esc * ξ(T_esc)² - T_BB,cj0 * η_esc² = 0

            Parameters
            ----------
            temp_eV : float
                Temperature in eV

            Returns
            -------
            float
                Residual of the equation
            """
            # Calculate y_max parameter for Compton coupling
            y_max = 3 * (rho_cjs / 1e-9) ** (-0.5) * (temp_eV / 1e2) ** (9 / 4)

            # Calculate Compton parameter ξ
            xi = np.maximum(1, 0.5 * np.log(y_max) * (1.6 + np.log(y_max)))

            # Left-hand side: T_esc * ξ² (convert eV to Kelvin)
            lhs = temp_eV * ev_in_k * xi**2

            # Return residual
            return lhs - rhs

        try:
            # Adaptive bracketing without artificial caps
            # Start from very low temperature where residual < 0 is guaranteed
            T_lo = 1e-6  # eV (~1e-2 K)
            f_lo = equation(T_lo)
            if f_lo > 0:
                # Extend lower bound if needed
                for _ in range(20):
                    T_lo *= 0.1
                    f_lo = equation(T_lo)
                    if f_lo <= 0:
                        break

            # Increase upper bound until residual > 0
            T_hi = 1e2  # eV (~1e6 K)
            f_hi = equation(T_hi)
            grow_steps = 0
            while f_hi <= 0 and grow_steps < 60:
                T_hi *= 2.0
                f_hi = equation(T_hi)
                grow_steps += 1

            if f_lo * f_hi > 0:
                # As a last resort, scan logspace to find a bracket
                T_grid = np.logspace(np.log10(max(T_lo, 1e-8)), np.log10(T_hi), 200)
                bracket = None
                for i in range(len(T_grid) - 1):
                    f1, f2 = equation(T_grid[i]), equation(T_grid[i + 1])
                    if f1 == 0:
                        temp_solution_eV = T_grid[i]
                        break
                    if f1 * f2 < 0:
                        bracket = (T_grid[i], T_grid[i + 1])
                        break
                if "temp_solution_eV" not in locals():
                    if bracket is None:
                        raise RuntimeError("Failed to bracket T_esc root")
                    temp_solution_eV = brentq(equation, bracket[0], bracket[1])
            else:
                temp_solution_eV = brentq(equation, T_lo, T_hi)

            result_temp_K = temp_solution_eV * ev_in_k

            # Apply physical pair-production ceiling (200 keV) to the observed temperature
            if result_temp_K > JetCocoonEmission.T_PAIR_MAX_K:
                if JetCocoonEmission.VERBOSE:
                    print(
                        "[Pair cap] T_esc exceeds 200 keV; capping to pair-regulated maximum."
                    )
                result_temp_K = JetCocoonEmission.T_PAIR_MAX_K

            if JetCocoonEmission.VERBOSE:
                print("=" * 60)
                print("Escape temperature calculation:")
                print(f"  Blackbody temperature at breakout: {temp_bb_cj0:.2e} K")
                print(f"  Escape thermal coupling: η_esc = {eta_escs:.3f}")
                print(f"  Density: ρ_cjs = {rho_cjs:.2e} g/cm³")
                print(f"  Bracket used: [{T_lo:.2e}, {T_hi:.2e}] eV")
                print(
                    f"  Solution (capped@200 keV if needed): T_esc = {result_temp_K:.2e} K"
                )
                print("=" * 60)

            return result_temp_K

        except Exception as e:
            print(f"Error in temp_esc root finding: {e}")
            # If solver fails, raise to surface for debugging
            raise

    @staticmethod
    def normalization_temp_observed_compton(
        temp_escs: float, v_cjs: float, r_cjs: float
    ) -> float:
        """
        Observed Compton-dominated temperature normalization at t_cj,s.
        Per Appendix C, scale as T_esc,s * V^{1/3} / r_cj,s.
        """
        return float(temp_escs) * (v_cjs ** (1 / 3)) / float(r_cjs)

    # ===== Relativistic observables wrappers (paper Eqs. 45–49) =====
    @staticmethod
    def relativistic_observables_for_gamma(
        kappa: float, e_cj: float, r_c: float, z_bre: float, gamma: float
    ) -> dict:
        """
        Compute characteristic observables for a given Lorentz factor γ.

        Returns dict with keys: 't_cj', 'L_iso', 'T_obs_th', 'r_ph', 'v_ph_prime', 'v_c', 'T_bb0'.
        """
        v_c = JetCocoonEmission.initial_jet_cocoon_volume(r_c, z_bre)
        T_bb0 = JetCocoonEmission.initial_bb_temp(e_cj, v_c)

        r_ph = JetCocoonEmission.photon_gas_dec_radius(kappa, e_cj, gamma)
        t_cj = JetCocoonEmission.observed_gas_dec_duration(r_ph, gamma)
        v_ph_prime = JetCocoonEmission.comoving_lum_shell_volume(r_ph, gamma)

        L_iso = JetCocoonEmission.isotropic_equiv_lum(
            e_cj, t_cj, v_c, v_ph_prime, gamma
        )
        T_obs_th = JetCocoonEmission.observed_jet_cocoon_temp(
            T_bb0, v_c, gamma, v_ph_prime
        )

        return {
            "t_cj": t_cj,
            "L_iso": L_iso,
            "T_obs_th": T_obs_th,
            "r_ph": r_ph,
            "v_ph_prime": v_ph_prime,
            "v_c": v_c,
            "T_bb0": T_bb0,
        }

    @staticmethod
    def relativistic_nonthermal_observables_for_gamma(
        kappa: float, e_cj: float, r_c: float, z_bre: float, gamma: float
    ) -> dict:
        obs = JetCocoonEmission.relativistic_observables_for_gamma(
            kappa, e_cj, r_c, z_bre, gamma
        )
        obs["T_obs_nt"] = JetCocoonEmission.T_PAIR_MAX_K
        return obs

    @staticmethod
    def compton_temperature_at_time(
        t: float,
        t_cjs: float,
        temp_bb_cj0: float,
        eta_cjs: float,
        rho_cjs: float,
    ) -> float:
        """
        Solve early-regime Compton relation T ξ(T)^2 = T_BB,cj0 · η(t)^2 for T(t), capped at 1e9 K.

        Here η(t) = η_cjs · (t/t_cjs)^(-3/2). We intentionally do not apply time
        evolution to T_BB,cj0 in this Compton-dominated regime, per author's guidance.
        """
        if t <= t_cjs:
            return 0.0

        time_ratio = t / t_cjs
        eta_t = eta_cjs * time_ratio ** (-1.5)
        # Early Compton regime: T_BB,cj0 is constant; only η(t) evolves
        rhs = temp_bb_cj0 * (eta_t**2)

        def xi_of_T(T):
            T_eV = max(T / ev_in_k, 1e-12)
            y_max = 3 * (rho_cjs / 1e-9) ** (-0.5) * (T_eV / 1e2) ** (9 / 4)
            if y_max <= 0:
                return 1.0
            return float(np.maximum(1, 0.5 * np.log(y_max) * (1.6 + np.log(y_max))))

        def eqn(T):
            return T * (xi_of_T(T) ** 2) - rhs

        # Adaptive bracket without artificial cap
        # Start from an extremely low temperature to ensure a valid bracket even when rhs is tiny
        T_lo, f_lo = 1e-6, eqn(1e-6)  # K
        if f_lo > 0:
            # decrease lower bound
            for _ in range(30):
                T_lo *= 0.5
                f_lo = eqn(T_lo)
                if f_lo <= 0:
                    break

        T_hi = 1e7
        f_hi = eqn(T_hi)
        grow_steps = 0
        while f_hi <= 0 and grow_steps < 80:
            T_hi *= 1.8
            f_hi = eqn(T_hi)
            grow_steps += 1

        if f_lo * f_hi > 0:
            # Fallback: scan logspace
            T_grid = np.logspace(np.log10(max(T_lo, 1e-6)), np.log10(T_hi), 160)
            bracket = None
            for i in range(len(T_grid) - 1):
                f1, f2 = eqn(T_grid[i]), eqn(T_grid[i + 1])
                if f1 == 0:
                    T = float(T_grid[i])
                    break
                if f1 * f2 < 0:
                    bracket = (T_grid[i], T_grid[i + 1])
                    break
            if "T" not in locals():
                if bracket is None:
                    raise RuntimeError("Failed to bracket T_com(t) root")
                T = float(brentq(eqn, bracket[0], bracket[1]))
        else:
            T = float(brentq(eqn, T_lo, T_hi))

        # Apply physical pair-production ceiling (200 keV)
        if T > JetCocoonEmission.T_PAIR_MAX_K:
            T = JetCocoonEmission.T_PAIR_MAX_K
        return T

    @staticmethod
    def critical_time_one(t_cjs: float, eta_escs: float) -> float:
        """
        Calculate the first critical time using simplified relation.
        This is kept for backward compatibility.
        """
        return t_cjs * eta_escs ** ((s_exponent + 2) / (2 * (2 * s_exponent + 3)))

    @staticmethod
    def critical_time_one_full(
        t_cjs: float,
        eta_cjs: float,
        temp_bb_cj0: float,
        rho_cjs: float,
        tolerance: float = 1e-3,
        max_iterations: int = 50,
    ) -> float:
        """
            Compute t_cj,th1 by solving η(t) ≈ ξ(T_com(t)).

        We define η(t) = η_cjs (t/t_cjs)^(-3/2). For each t, T_com(t) satisfies
        T_com ξ(T_com)^2 = T_BB,cj0 · η(t)^2 (no time factor on T_BB,cj0 in this regime).
        Find the earliest t > t_cjs where
            η(t) - ξ(T_com(t)) changes sign.
        """

        def xi_of_T(temp_K: float) -> float:
            temp_eV = max(temp_K / ev_in_k, 1e-12)
            y_max = 3 * (rho_cjs / 1e-9) ** (-0.5) * (temp_eV / 1e2) ** (9 / 4)
            if y_max <= 0:
                return 1.0
            return float(np.maximum(1, 0.5 * np.log(y_max) * (1.6 + np.log(y_max))))

        def Tcom_at(t: float) -> float:
            if t <= t_cjs:
                return 0.0
            time_ratio = t / t_cjs
            eta_t = eta_cjs * time_ratio ** (-1.5)
            rhs = temp_bb_cj0 * (eta_t**2)

            def eqn(T):
                return T * (xi_of_T(T) ** 2) - rhs

            # Respect physical pair cap: if even at 200 keV LHS < RHS, temperature saturates
            T_cap = JetCocoonEmission.T_PAIR_MAX_K
            if eqn(T_cap) < 0:
                return T_cap

            # Adaptive bracketing within [T_lo, T_cap]
            T_lo, f_lo = 1e-6, eqn(1e-6)
            if f_lo > 0:
                for _ in range(30):
                    T_lo *= 0.5
                    f_lo = eqn(T_lo)
                    if f_lo <= 0:
                        break
            T_hi = min(1e7, T_cap)
            f_hi = eqn(T_hi)
            grow_steps = 0
            while f_hi <= 0 and grow_steps < 80:
                T_hi *= 1.8
                if T_hi > T_cap:
                    T_hi = T_cap
                f_hi = eqn(T_hi)
                grow_steps += 1
            if f_lo * f_hi > 0:
                # Try scanning for a sign change first
                T_grid = np.logspace(np.log10(max(T_lo, 1e-6)), np.log10(T_hi), 200)
                for i in range(len(T_grid) - 1):
                    f1, f2 = eqn(T_grid[i]), eqn(T_grid[i + 1])
                    if f1 == 0:
                        return float(T_grid[i])
                    if f1 * f2 < 0:
                        return float(brentq(eqn, T_grid[i], T_grid[i + 1]))
                # No sign change: choose T minimizing |eqn(T)| within [T_lo, T_hi]
                vals = [abs(eqn(Tg)) for Tg in T_grid]
                j = int(np.argmin(vals))
                T_min = float(T_grid[j])
                if not (1e-6 <= T_min <= T_cap):
                    T_min = min(max(T_min, 1e-6), T_cap)
                # Minimal diagnostic to aid case-3 behavior
                # print(f"[Tcom_at] Selected minimizer T={T_min:.3e} K, |res|={vals[j]:.3e}")
                return T_min
            return float(brentq(eqn, T_lo, T_hi))

        def eta_of_t(t: float) -> float:
            return float(eta_cjs * (t / t_cjs) ** (-1.5))

        try:
            t_simple = JetCocoonEmission.critical_time_one(t_cjs, eta_cjs)
            # Sample the condition over a broad time range with good resolution
            ts = np.logspace(np.log10(t_cjs * 1.01), np.log10(t_cjs * 1e6), 200)
            vals = []
            for tt in ts:
                Tc = Tcom_at(tt)
                vals.append(eta_of_t(tt) - xi_of_T(Tc))
            bracket = None
            for i in range(len(vals) - 1):
                if vals[i] == 0:
                    return ts[i]
                if vals[i] * vals[i + 1] < 0:
                    bracket = (ts[i], ts[i + 1])
                    break
            # If no sign change found, try extending the search window further in time
            if bracket is None:
                ts_ext = np.logspace(
                    np.log10(ts[-1] * 1.05), np.log10(t_cjs * 1e9), 120
                )
                last_val = vals[-1]
                for tt in ts_ext:
                    Tc = Tcom_at(tt)
                    v = eta_of_t(tt) - xi_of_T(Tc)
                    if last_val * v < 0:
                        bracket = (ts[-1], tt)
                        break
                    last_val = v
                # Merge arrays for potential minimization fallback
                ts = np.concatenate([ts, ts_ext])
                vals = list(vals) + [
                    eta_of_t(tt) - xi_of_T(Tcom_at(tt)) for tt in ts_ext
                ]
            if bracket is None:
                # Monotonic case: pick time that minimizes |η(t) - ξ(T(t))| as robust fallback
                idx_min = int(np.argmin(np.abs(vals)))
                t_min = float(ts[idx_min])
                # Additionally, if η never falls below ~1 within range, use η(t)=1 approximation
                # since ξ ≥ 1 asymptotically.
                t_eta1 = float(t_cjs * (eta_cjs ** (2.0 / 3.0)))
                # Choose the smaller of t_min and t_eta1 that is > t_cjs to avoid overshooting
                candidates = [t for t in [t_min, t_eta1] if t > t_cjs]
                if len(candidates) == 0:
                    return max(t_simple, 1.05 * t_cjs)
                t_choice = min(candidates)
                # Final guard against pathological values
                return max(t_choice, 1.05 * t_cjs)

            def root_fn(x):
                return eta_of_t(x) - xi_of_T(Tcom_at(x))

            sol = brentq(root_fn, bracket[0], bracket[1])
            if sol <= t_cjs:
                return max(t_simple, 1.05 * t_cjs)
            return sol
        except Exception as e:
            print(f"Error in critical_time_one_full: {e}")
            return JetCocoonEmission.critical_time_one(t_cjs, eta_cjs)

    @staticmethod
    def critical_time_two(t_cjs: float, eta_cjs: float) -> float:
        return t_cjs * eta_cjs ** ((4 * (s_exponent + 2)) / (3 * s_exponent + 31))

    @staticmethod
    def compton_temperature_evolution(
        t: float,
        t_cjs: float,
        t_cjth1: float,
        t_cjth2: float,
        t_thph: float,
        t_sphend: float,
        temp_norm: float = 1.0,
        temp_at_t_cjs: float = None,
        temp_at_t_cjth1: float = None,
    ) -> float:
        """
            Calculate the Compton temperature evolution of the cocoon-jet material with smooth transitions.

            This implementation properly accounts for the evolution of ξ by normalizing the power-law
            between computed anchor points, as suggested by the paper author to avoid extremely low temperatures.

        Note: No artificial temperature cap is applied here to aid debugging.

            Parameters
            ----------
            t : float
                Time [s]
            t_cjs : float
                Jet-cocoon diffusion time [s]
            t_cjth1 : float
                First critical time [s]
            t_cjth2 : float
                Second critical time [s]
            t_thph : float
                Thermal to photospheric transition time [s]
            t_sphend : float
                End of spherical emission time [s]
            temp_norm : float, optional
                Normalization temperature [K]. Default is 1.0 for backward compatibility.
            temp_at_t_cjs : float, optional
                Temperature at t_cjs from escape temperature calculation [K]
            temp_at_t_cjth1 : float, optional
                Temperature at t_cjth1 from escape temperature calculation [K]

            Returns
            -------
            float
                Temperature [K]
        """
        if t <= t_cjs:
            return 0.0  # No temperature before jet-cocoon diffusion time

        # Regime 1: t_cjs < t ≤ t_cjth1
        # Use proper normalization between anchor points to avoid extremely low temperatures
        if t <= t_cjth1:
            # Power-law exponent from the paper
            alpha = (9 * s_exponent + 12) / (s_exponent + 2)

            # If anchor temperatures are provided, use them for proper normalization
            if temp_at_t_cjs is not None and temp_at_t_cjth1 is not None:
                # Normalize to connect the two anchor points smoothly
                raw_scaling_t = (t / t_cjs) ** (-alpha)
                raw_scaling_th1 = (t_cjth1 / t_cjs) ** (-alpha)

                # Linear interpolation in log space between anchor points
                log_temp_cjs = np.log(temp_at_t_cjs)
                log_temp_th1 = np.log(temp_at_t_cjth1)
                log_t_ratio = np.log(t / t_cjs) / np.log(t_cjth1 / t_cjs)

                interpolated_log_temp = log_temp_cjs + log_t_ratio * (
                    log_temp_th1 - log_temp_cjs
                )
                result = np.exp(interpolated_log_temp)
                return result
            else:
                # Fallback to standard scaling with normalization
                scaling_1 = (t / t_cjs) ** (-alpha)
                result = temp_norm * scaling_1
                return result

        # Calculate temperature at end of regime 1 for continuity
        alpha = (9 * s_exponent + 12) / (s_exponent + 2)

        if temp_at_t_cjth1 is not None:
            temp_1_end = temp_at_t_cjth1
        else:
            temp_1_end = temp_norm * (t_cjth1 / t_cjs) ** (-alpha)

        # Regime 2: t_cjth1 < t ≤ t_cjth2 (pure scaling relation)
        if t <= t_cjth2:
            # Raw scaling relation (without temp_norm)
            scaling_2 = (t / t_cjth1) ** (-(s_exponent + 1) / (2 * (s_exponent + 2)))
            # Apply to regime 1 end temperature to ensure continuity
            result = temp_1_end * scaling_2
            return result

        # Calculate temperature at end of regime 2 for continuity
        temp_2_end = temp_1_end * (t_cjth2 / t_cjth1) ** (
            -(s_exponent + 1) / (2 * (s_exponent + 2))
        )

        # Regime 3: t_cjth2 < t ≤ t_thph (pure scaling relation)
        if t <= t_thph:
            # Raw scaling relation (without temp_norm)
            scaling_3 = (t / t_cjth2) ** (
                -(2 * (5 * s_exponent**2 + 27 * s_exponent + 50))
                / ((s_exponent + 2) * (17 * s_exponent + 45))
            )
            # Apply to regime 2 end temperature to ensure continuity
            result = temp_2_end * scaling_3
            return result

        # Calculate temperature at end of regime 3 for continuity
        temp_3_end = temp_2_end * (t_thph / t_cjth2) ** (
            -(2 * (5 * s_exponent**2 + 27 * s_exponent + 50))
            / ((s_exponent + 2) * (17 * s_exponent + 45))
        )

        # Regime 4: t_thph < t ≤ t_sphend (pure scaling relation)
        if t <= t_sphend:
            # Raw scaling relation (without temp_norm)
            scaling_4 = (t / t_thph) ** (
                -(s_exponent**2 + 5 * s_exponent + 8)
                / (2 * (s_exponent + 2) * (s_exponent + 3))
            )
            # Apply to regime 3 end temperature to ensure continuity
            result = temp_3_end * scaling_4
            return result

        # Calculate temperature at end of regime 4 for continuity
        temp_4_end = temp_3_end * (t_sphend / t_thph) ** (
            -(s_exponent**2 + 5 * s_exponent + 8)
            / (2 * (s_exponent + 2) * (s_exponent + 3))
        )

        # Regime 5: t > t_sphend (exponential decay + power law)
        if t > t_sphend:
            # Raw scaling relation components
            exp_factor = np.exp(-1 / 2 * (t**2 / (t_sphend**2) - 1)) ** (1 / 4)
            power_factor = (t / t_sphend) ** (
                -(s_exponent + 1) / (2 * (s_exponent + 3))
            )

            # Scaling at boundary to match regime 4
            exp_factor_boundary = np.exp(
                -1 / 2 * (t_sphend**2 / (t_sphend**2) - 1)
            ) ** (
                1 / 4
            )  # = 1
            power_factor_boundary = (t_sphend / t_sphend) ** (
                -(s_exponent + 1) / (2 * (s_exponent + 3))
            )  # = 1
            scaling_5_boundary = exp_factor_boundary * power_factor_boundary  # = 1

            # Current scaling
            scaling_5 = exp_factor * power_factor

            # Apply to regime 4 end temperature to ensure continuity
            result = temp_4_end * scaling_5
            return result

        else:
            return 0.0  # Fallback case

    @staticmethod
    def critical_baryon_lorentz_factor(
        e_cj: float, z_bre: float, theta_c: float
    ) -> float:
        """
        Calculate the critical baryon loading Lorentz factor.

        Parameters
        ----------
        e_cj : float
            Jet-cocoon energy [erg]
        z_bre : float
            Jet head breakout height [cm]
        theta_c : float
            Cocoon opening angle at time of breakout [rad]

        Returns
        -------
        float
            Critical baryon loading Lorentz factor
        """
        e_cj_50 = e_cj / 1e50
        z_bre_13 = z_bre / 1e13
        return (
            5.5
            * e_cj_50 ** (1 / 4)
            * z_bre_13 ** (-1 / 2)
            * theta_cj ** (-1 / 2)
            * theta_c ** (-1 / 4)
        )

    @staticmethod
    def critical_spread_lorentz_factor(e_cj: float, z_bre: float) -> float:
        """
        Calculate the critical spreading Lorentz factor.

        Parameters
        ----------
        e_cj : float
            Jet-cocoon energy [erg]
        z_bre : float
            Jet head breakout height [cm]

        Returns
        -------
        float
            Critical spreading Lorentz factor
        """
        e_cj_50 = e_cj / 1e50
        z_bre_13 = z_bre / 1e13
        f_beta_gamma_m1 = f_beta_gamma / 1e-1
        return (
            4.9
            * e_cj_50 ** (1 / 5)
            * z_bre_13 ** (-2 / 5)
            * f_beta_gamma_m1 ** (-1 / 5)
            * theta_cj ** (-2 / 5)
        )

    @staticmethod
    def critical_thermal_lorentz_factor(
        e_cj: float, z_bre: float, theta_c: float
    ) -> float:
        """
        Calculate the critical thermal Lorentz factor.

        Parameters
        ----------
        e_cj : float
            Jet-cocoon energy [erg]
        z_bre : float
            Jet head breakout height [cm]
        theta_c : float
            Cocoon opening angle at time of breakout [rad]

        Returns
        -------
        float
            Critical thermal Lorentz factor
        """
        e_cj_50 = e_cj / 1e50
        z_bre_13 = z_bre / 1e13
        f_beta_gamma_m1 = f_beta_gamma / 1e-1
        return (
            0.007
            * e_cj_50 ** (9 / 16)
            * z_bre_13 ** (-19 / 16)
            * f_beta_gamma_m1 ** (1 / 2)
            * theta_c ** (-9 / 8)
        )

    @staticmethod
    def gamma_bb_emission_duration(z_bre: float) -> float:
        return f_beta_gamma * z_bre / C_CGS

    @staticmethod
    def gamma_bb_luminosity(e_cj: float, z_bre: float) -> float:
        return e_cj * C_CGS / (z_bre * theta_cj**2)

    @staticmethod
    def gamma_boundaries(gamma_s: float) -> tuple:
        gamma_min = 3.0
        gamma_max = np.min([gamma_s, 10.0])
        # Ensure gamma_max >= gamma_min
        if gamma_max < gamma_min:
            gamma_max = gamma_min
        return gamma_min, gamma_max

    @staticmethod
    def photon_gas_dec_radius(kappa: float, e_cj: float, gamma: float) -> float:
        return np.sqrt(
            kappa * f_beta_gamma * e_cj / (2 * np.pi * theta_cj**2 * C_CGS**2 * gamma)
        )

    @staticmethod
    def observed_gas_dec_duration(r_gamma_ph: float, gamma: float) -> float:
        return r_gamma_ph / (2 * gamma**2 * C_CGS)

    @staticmethod
    def comoving_lum_shell_volume(r_gamma_ph: float, gamma: float) -> float:
        return 2 * np.pi * theta_cj**2 * r_gamma_ph**3 / gamma

    @staticmethod
    def isotropic_equiv_lum(
        e_cj: float, t_cj: float, v_c: float, v_ph_prime: float, gamma: float
    ) -> float:
        # v_c is the cocoon volume which is defined in cocoon_physics.py and v_ph_prime is the comoving_lum_shell_volume
        return (
            (f_beta_gamma * e_cj / (t_cj * theta_cj**2))
            * (f_beta_gamma * v_c / v_ph_prime) ** (1 / 3)
            * gamma
        )

    @staticmethod
    def observed_jet_cocoon_temp(
        temp_bb_cjo: float, v_c: float, gamma: float, v_ph_prime: float
    ) -> float:
        return temp_bb_cjo * (f_beta_gamma * v_c / v_ph_prime) ** (1 / 3) * gamma
