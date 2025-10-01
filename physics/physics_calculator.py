"""
Physics calculator that combines all physics calculations for the Chen & Dai 2025 model.

This module provides a unified interface to all physics functions and handles
parameter passing between them. It serves as the main entry point for physics
calculations in the package.
"""

import numpy as np
from typing import Union, Callable, Optional

try:
    from ..constants import ModelParameters, ev_in_k
    from .jet_physics import JetPhysics
    from .cocoon_physics import CocoonPhysics
    from .disk_physics import DiskPhysics
    from .disk_cocoon_emission import DiskCocoonEmission
    from .jet_cocoon_emission import JetCocoonEmission
    from .jet_head_shock_breakout_emission import JetHeadShockBreakoutEmission
except ImportError:
    from constants import ModelParameters, ev_in_k
    from physics.jet_physics import JetPhysics
    from physics.cocoon_physics import CocoonPhysics
    from physics.disk_physics import DiskPhysics
    from physics.disk_cocoon_emission import DiskCocoonEmission
    from physics.jet_cocoon_emission import JetCocoonEmission
    from physics.jet_head_shock_breakout_emission import JetHeadShockBreakoutEmission


class PhysicsCalculator:
    """
    Convenience wrapper that combines all physics calculations.

    This class provides a unified interface to all physics functions
    and handles parameter passing between them.
    """

    def __init__(self, params: ModelParameters, l_j: float):
        """Initialize with static jet luminosity l_j (erg/s)."""
        self.params = params
        if l_j is None:
            raise ValueError("Static jet luminosity l_j must be provided.")
        self.l_j = float(l_j)

        # Instantiate physics component helpers
        self.jet = JetPhysics()
        self.cocoon = CocoonPhysics()
        self.disk = DiskPhysics()
        self.disk_cocoon_emission = DiskCocoonEmission()
        self.jet_cocoon_emission = JetCocoonEmission()
        self.jet_head_shock_breakout_emission = JetHeadShockBreakoutEmission()

    def get_current_jet_luminosity(self) -> float:
        """Return static jet luminosity (compatibility method)."""
        return self.l_j

    def calculate_all_quantities(
        self,
        beta_h: float,
        t: float,
        rho_0: float,
        h: float,
        density_profile: str = "isothermal",
    ) -> dict:
        """
        Calculate all physical quantities for given beta_h and time.

        Parameters
        ----------
        beta_h : float
            Jet head velocity parameter
        t : float
            Time [s]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        density_profile : str
            Density profile type for mean density calculation

        Returns
        -------
        dict
            Dictionary containing all calculated quantities
        """
        # Use static jet luminosity
        self.l_j = self.get_current_jet_luminosity()

        # Jet quantities
        z_h = self.jet.z_h(beta_h, t)
        gamma_h = self.jet.gamma_h(beta_h)

        # Disk density at jet head
        rho_z = self.disk.density_profile(rho_0, h, z_h, density_profile)

        # Mean density from z=0 to z=z_h for cocoon calculations
        mean_rho_z = self.disk.mean_z_density(rho_0, h, z_h, density_profile)

        # Efficiency and energy
        eta_h = self.jet.eta_h(gamma_h, self.params.theta_0)
        e_c = self.cocoon.internal_energy(eta_h, self.l_j, beta_h, t)

        # Cocoon properties (using paper's formula)
        beta_c, r_c = self.cocoon.beta_c_paper_formula(e_c, mean_rho_z, z_h, t)

        # Cross-section and angles
        sigma_h = self.cocoon.cross_section(
            self.l_j, z_h, r_c, e_c, self.params.theta_0
        )
        theta_h = self.jet.theta_h(sigma_h, z_h, self.params.theta_0)
        theta_c = self.cocoon.theta_c(r_c, z_h)

        # Dimensionless parameter
        l_tilde = self.jet.l_tilde(self.l_j, sigma_h, rho_z)

        return {
            "z_h": z_h,
            "gamma_h": gamma_h,
            "rho_z": rho_z,
            "mean_rho_z": mean_rho_z,
            "eta_h": eta_h,
            "e_c": e_c,
            "beta_c": beta_c,
            "r_c": r_c,
            "sigma_h": sigma_h,
            "theta_h": theta_h,
            "theta_c": theta_c,
            "l_tilde": l_tilde,
        }

    def calculate_shock_breakout_parameters(
        self,
        beta_h_bre: float,
        z_bre: float,
        rho_0: float,
        h: float,
        density_profile: str = "isothermal",
    ) -> dict:
        """
        Calculate all key physical parameters at shock breakout.

        This function serves as a centralized calculation point for breakout parameters
        that are needed by multiple other functions (temperatures, luminosities, etc.).
        This eliminates repetitive calculations and ensures consistency.

        Parameters
        ----------
        beta_h_bre : float
            Jet head velocity parameter at shock breakout
        z_bre : float
            Height at shock breakout [cm]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        density_profile : str
            Density profile type

        Returns
        -------
        dict
            Dictionary containing all key breakout parameters:
            - beta_h_bre, z_bre: Input parameters
            - rho_z_bre: Density at breakout height [g/cm³]
            - gamma_h_bre: Lorentz factor at breakout
            - s_gamma_h: Small gamma_h parameter
            - d_bre: Shell width at breakout [cm]
            - sigma_h_bre: Cross-sectional area at breakout [cm²]
            - m_shell: Shell mass [g]
            - theta_h_bre: Jet opening angle at breakout [rad]
        """
        # Basic parameters
        rho_z_bre = self.disk.density_profile(rho_0, h, z_bre, density_profile)
        gamma_h_bre = self.jet.gamma_h(beta_h_bre)
        s_gamma_h = self.jet.small_gamma_h(gamma_h_bre)

        # Shell width at breakout
        d_bre = self.jet_head_shock_breakout_emission.shock_breakout_shell_width(
            self.params.kappa, rho_z_bre, beta_h_bre
        )

        # Cross-sectional area at breakout (approximate)
        sigma_h_bre = np.pi * (self.params.theta_0 * z_bre) ** 2

        # Shell mass
        m_shell = self.jet_head_shock_breakout_emission.shell_mass(
            rho_z_bre, sigma_h_bre, d_bre
        )

        # Jet opening angle at breakout
        theta_h_bre = self.jet.theta_h(sigma_h_bre, z_bre, self.params.theta_0)

        return {
            "beta_h_bre": beta_h_bre,
            "z_bre": z_bre,
            "rho_z_bre": rho_z_bre,
            "gamma_h_bre": gamma_h_bre,
            "s_gamma_h": s_gamma_h,
            "d_bre": d_bre,
            "sigma_h_bre": sigma_h_bre,
            "m_shell": m_shell,
            "theta_h_bre": theta_h_bre,
        }

    def calculate_shock_breakout_temperatures(
        self,
        beta_h_bre: float,
        z_bre: float,
        rho_0: float,
        h: float,
        density_profile: str = "isothermal",
    ) -> dict:
        """
        Calculate all temperature variants at shock breakout.

        Note: For efficiency, consider using calculate_shock_breakout_parameters()
        first if you need multiple breakout quantities.

        Parameters
        ----------
        beta_h_bre : float
            Jet head velocity parameter at shock breakout
        z_bre : float
            Height at shock breakout [cm]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        density_profile : str
            Density profile type

        Returns
        -------
        dict
            Dictionary containing all temperature calculations
        """
        # Get all breakout parameters at once
        breakout_params = self.calculate_shock_breakout_parameters(
            beta_h_bre, z_bre, rho_0, h, density_profile
        )

        # Extract what we need
        rho_z_bre = breakout_params["rho_z_bre"]
        s_gamma_h = breakout_params["s_gamma_h"]
        m_shell = breakout_params["m_shell"]
        sigma_h_bre = breakout_params["sigma_h_bre"]

        # Calculate all temperature variants
        T_typical = self.jet_head_shock_breakout_emission.typical_radiation_temperature(
            rho_z_bre, beta_h_bre
        )
        T_compton = (
            self.jet_head_shock_breakout_emission.comptonization_radiation_temperature(
                rho_z_bre, beta_h_bre
            )
        )

        # Use the determine_temperature method which handles all regimes
        T_determined = self.jet_head_shock_breakout_emission.determine_temperature(
            beta_h_bre,
            rho_z_bre,
            s_gamma_h,
            m_shell,
            self.params.kappa,
            rho_z_bre,
            sigma_h_bre,
        )

        # Determine regime
        if beta_h_bre < 0.03:
            regime = "typical"
        elif beta_h_bre >= 0.03 and beta_h_bre < 0.4:
            regime = "comptonization"
        else:
            regime = "relativistic"

        return {
            "T_typical": T_typical,
            "T_comptonization": T_compton,
            "T_determined": T_determined,
            "regime": regime,
            "rho_z_bre": rho_z_bre,
            "beta_h_bre": beta_h_bre,
            "temperature_ratio": T_compton / T_typical,
        }

    def calculate_timescales(
        self,
        beta_c_bre: float,
        r_c_bre: float,
        rho_z_bre: float,
        kappa: float,
        e_c_bre: float,
        h: float,
        z_h_bre: float,
    ) -> dict:
        """
        Calculate all timescales needed for cocoon emission calculations.

        This function only calculates timescales, not luminosities or energies.

        Parameters
        ----------
        beta_c_bre : float
            Cocoon velocity at breakout
        r_c_bre : float
            Cocoon radius at breakout [cm]
        rho_z_bre : float
            Density at breakout height [g/cm³]
        kappa : float
            Opacity [cm²/g]
        e_c_bre : float
            Cocoon energy at breakout [erg]
        h : float
            Disk scale height [cm]
        z_h_bre : float
            Jet head height at breakout [cm]

        Returns
        -------
        dict
            Dictionary containing only the calculated timescales
        """
        # Calculate breakout emission timescale
        t_c_bre = self.disk_cocoon_emission.cocoon_breakout_emission_timescale(
            kappa, rho_z_bre, beta_c_bre
        )

        # Calculate initial diffusion timescale
        v_c_bre = self.cocoon.volume(r_c_bre, z_h_bre)
        m_c_bre = self.disk_cocoon_emission.cocoon_mass(rho_z_bre, v_c_bre)
        t_c_init = self.disk_cocoon_emission.cocoon_initial_diffusion_timescale(
            kappa, r_c_bre, h, m_c_bre
        )

        # Calculate expansion timescale
        t_s_init = self.disk_cocoon_emission.expansion_time_scale(r_c_bre, beta_c_bre)

        # Calculate effective diffusion timescale using paper's exact formula
        # OLD IMPLEMENTATION (commented out - was too fast by factor ~36):
        # t_c_diff = self.disk_cocoon_emission.effective_diffusion_timescale_old(
        #     t_c_init, t_s_init
        # )

        # NEW IMPLEMENTATION using paper's exact formula: t_c,diff = sqrt(2/πb) * κM_c / (cH_d)
        t_c_diff = self.disk_cocoon_emission.effective_diffusion_timescale(
            kappa, m_c_bre, h
        )  # For planar timescale, we need to calculate luminosities temporarily
        # Calculate initial internal energy and spherical luminosity
        e_c_init = self.disk_cocoon_emission.cocoon_initial_internal_energy(
            e_c_bre, r_c_bre, h
        )
        l_c_sph = self.disk_cocoon_emission.initial_spherical_cocoon_luminosity(
            e_c_init, t_c_init
        )

        # Calculate breakout luminosity
        l_c_bre = self.disk_cocoon_emission.cocoon_breakout_luminosity(
            r_c_bre, rho_z_bre, beta_c_bre
        )

        # Calculate planar timescale
        t_c_pla = self.disk_cocoon_emission.planar_timescale(t_c_bre, l_c_bre, l_c_sph)

        # Return only timescales
        return {
            "t_c_bre": t_c_bre,
            "t_c_init": t_c_init,
            "t_s_init": t_s_init,
            "t_c_diff": t_c_diff,
            "t_c_pla": t_c_pla,
        }

    def calculate_disk_cocoon_emission(
        self,
        t: float,
        beta_c_bre: float,
        r_c_bre: float,
        rho_z_bre: float,
        kappa: float,
        e_c_bre: float,
        h: float,
        z_h_bre: float,
        t_break: float = None,
        warn_on_mask: bool = True,
    ):
        """
        Calculate disk cocoon emission luminosity at time t.

        This function uses the calculate_timescales method for timescales and
        calculates luminosities and energies directly as needed.

        Parameters
        ----------
        t : float
            Time at which to calculate emission [s]
        beta_c_bre : float
            Cocoon velocity at breakout
        r_c_bre : float
            Cocoon radius at breakout [cm]
        rho_z_bre : float
            Density at breakout height [g/cm³]
        kappa : float
            Opacity [cm²/g]
        e_c_bre : float
            Cocoon energy at breakout [erg]
        h : float
            Disk scale height [cm]
        z_h_bre : float
            Jet head height at breakout [cm]
        t_break : float, optional
            Shock breakout time [s]. If provided, emission before this time is set to zero.
        warn_on_mask : bool, optional
            Whether to print a warning when masking emission before breakout

        Returns
        -------
        float
            Cocoon luminosity [erg/s]
        """
        # PHYSICAL CONSTRAINT: No emission before shock breakout
        if t_break is not None and t < t_break:
            if warn_on_mask and hasattr(self, "_warned_emission_mask") == False:
                print(
                    f"Warning: Masking cocoon emission for t < t_break ({t_break:.1e} s)"
                )
                print(
                    f"         Cocoon emission should only occur after shock breakout."
                )
                self._warned_emission_mask = True
            return 0.0
        # Get timescales only
        timescales = self.calculate_timescales(
            beta_c_bre, r_c_bre, rho_z_bre, kappa, e_c_bre, h, z_h_bre
        )

        # Extract timescale values
        t_c_bre = timescales["t_c_bre"]
        t_c_diff = timescales["t_c_diff"]
        t_c_pla = timescales["t_c_pla"]
        t_c_init = timescales["t_c_init"]

        # Calculate luminosities and energies needed for emission
        e_c_init = self.disk_cocoon_emission.cocoon_initial_internal_energy(
            e_c_bre, r_c_bre, h
        )
        l_c_sph = self.disk_cocoon_emission.initial_spherical_cocoon_luminosity(
            e_c_init, t_c_init
        )
        l_c_bre = self.disk_cocoon_emission.cocoon_breakout_luminosity(
            r_c_bre, rho_z_bre, beta_c_bre
        )

        # Apply the luminosity evolution
        # t_c_diff = 5 * t_c_diff
        if t <= t_c_bre:
            return l_c_bre * np.exp(1 - t_c_bre / t)
        elif t > t_c_bre and t <= t_c_pla:
            return l_c_bre * (t / t_c_bre) ** (-4 / 3)
        elif t > t_c_pla:
            return l_c_sph * np.exp(-(t - t_c_pla) * (t + t_c_pla) / (2 * t_c_diff**2))

    def calculate_disk_cocoon_temperature(
        self,
        t: float,
        beta_c_bre: float,
        r_c_bre: float,
        rho_z_bre: float,
        kappa: float,
        e_c_bre: float,
        h: float,
        z_h_bre: float,
        times: np.ndarray = None,
        t_c_th: float = None,
    ) -> float:
        """
        Calculate disk cocoon temperature at time t with automatic regime detection.

        This function automatically determines the appropriate temperature evolution:
        - If beta_c < 0.03: Use thermal_eq_temp (slow cocoon regime)
        - If beta_c >= 0.03: Use comptonization_temp (fast cocoon regime)

        For fast cocoons, if no times array is provided, one is automatically generated
        with appropriate range based on the timescales.

        Parameters
        ----------
        t : float
            Time at which to calculate temperature [s]
        beta_c_bre : float
            Cocoon velocity at breakout
        r_c_bre : float
            Cocoon radius at breakout [cm]
        rho_z_bre : float
            Density at breakout height [g/cm³]
        kappa : float
            Opacity [cm²/g]
        e_c_bre : float
            Cocoon energy at breakout [erg]
        h : float
            Disk scale height [cm]
        z_h_bre : float
            Jet head height at breakout [cm]
        times : np.ndarray, optional
            Time array for thermalization time calculation. If None and needed,
            automatically generated based on relevant timescales.
        t_c_th : float, optional
            Pre-calculated thermalization time. If provided, avoids recalculation.

        Returns
        -------
        float
            Temperature evolution [K]
        """
        # Get all timescales and related quantities
        timescales = self.calculate_timescales(
            beta_c_bre, r_c_bre, rho_z_bre, kappa, e_c_bre, h, z_h_bre
        )

        # Extract the values we need
        t_c_bre = timescales["t_c_bre"]
        t_c_diff = timescales["t_c_diff"]
        t_c_pla = timescales["t_c_pla"]

        # Case distinction based on cocoon velocity (determined by physics)
        if beta_c_bre < 0.03:
            # Slow cocoon regime - use thermal equilibrium temperature
            return self.disk_cocoon_emission.thermal_eq_temp(
                t, t_c_bre, t_c_pla, t_c_diff, rho_z_bre, beta_c_bre
            )
        else:
            # Fast cocoon regime - use comptonization temperature
            if t_c_th is None:
                t_c_th = self.disk_cocoon_emission.find_t_c_th(
                    rho_z_bre, beta_c_bre, times
                )

            return self.disk_cocoon_emission.comptonization_temp(
                t, t_c_bre, t_c_th, t_c_pla, t_c_diff, rho_z_bre, beta_c_bre
            )

    def calculate_shock_breakout_shell_width(
        self,
        beta_h_bre: float,
        z_bre: float,
        rho_0: float,
        h: float,
        density_profile: str = "isothermal",
    ) -> float:
        """
        Calculate the shell width at shock breakout using model parameters.

        Note: For efficiency, consider using calculate_shock_breakout_parameters()
        if you need multiple breakout quantities.

        Parameters
        ----------
        beta_h_bre : float
            Jet head velocity parameter at shock breakout
        z_bre : float
            Height at shock breakout [cm]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        density_profile : str
            Density profile type

        Returns
        -------
        float
            Shell width at shock breakout [cm]
        """
        # Calculate density at breakout height
        rho_z_bre = self.disk.density_profile(rho_0, h, z_bre, density_profile)

        # Use the opacity from model parameters (kappa)
        return self.jet_head_shock_breakout_emission.shock_breakout_shell_width(
            self.params.kappa, rho_z_bre, beta_h_bre
        )

    def find_jet_cocoon_critical_time(
        self,
        results,
        t_break: float,
        idx_break: int,
        search_range_factor: float = 0.5,
        tolerance: float = 1e-3,
    ) -> dict:
        """
        Find the time when r_cj = r_c condition is satisfied.

        This method searches for the time when the diffusion radius r_cj equals
        the cocoon radius r_c, which defines the critical velocity β_cr.

        Parameters
        ----------
        results : EvolutionResults
            Evolution results containing time-dependent parameters
        t_break : float
            Shock breakout time [s]
        idx_break : int
            Index of shock breakout in results arrays
        search_range_factor : float, optional
            Start search at t_break * search_range_factor. Default 0.5.
        tolerance : float, optional
            Relative tolerance for r_cj = r_c condition. Default 1e-3.

        Returns
        -------
        dict
            Dictionary containing critical time and parameters:
            - 'critical_time': Time when r_cj = r_c [s]
            - 'critical_time_idx': Index in results arrays
            - 'parameters': Parameters at critical time
            - 'condition_satisfied': Whether condition was found within tolerance
        """
        print(f"Finding critical time when r_cj = r_c...")
        print(f"  Search starting from t = {t_break * search_range_factor:.2e} s")

        # Get time-dependent parameters from results
        times = results.times
        e_c_evolution = results.e_c
        r_c_evolution = results.r_c
        z_h_evolution = results.z_h
        beta_c_evolution = results.beta_c

        # Calculate r_cj for each time step in search range
        best_error = float("inf")
        best_idx = idx_break
        best_time = t_break

        search_start_idx = max(
            0, idx_break - int(idx_break * (1 - search_range_factor))
        )

        # Extended search: check more points after breakout and some before
        for i in range(search_start_idx, min(len(times), idx_break + 250)):
            t = times[i]
            if t < t_break * search_range_factor:
                continue

            e_c_t = e_c_evolution[i]
            r_c_t = r_c_evolution[i]
            z_h_t = z_h_evolution[i]
            beta_c_t = beta_c_evolution[i]

            # Calculate jet-cocoon energy (half of cocoon energy)
            e_cj = self.jet_cocoon_emission.total_jet_cocoon_energy(e_c_t)

            # Calculate critical velocity for this r_c
            beta_cr = self.jet_cocoon_emission.critical_velocity(
                self.params.kappa, e_cj, r_c_t
            )

            # Calculate velocity of first shell
            beta_cjs = self.jet_cocoon_emission.velocity_of_first_shell(beta_cr)

            # Calculate diffusion radius r_cj
            r_cj = self.jet_cocoon_emission.diffusion_radius(
                self.params.kappa, e_cj, beta_cjs
            )

            # Check how close r_cj is to r_c
            relative_error = abs(r_cj - r_c_t) / r_c_t

            if relative_error < best_error:
                best_error = relative_error
                best_idx = i
                best_time = t

        # If still no good match, try searching further back in time
        # Sometimes the optimal condition occurs before the search window
        if (
            best_error > 0.1 and search_start_idx > 50
        ):  # Only if we have room to search back
            extended_search_start = max(0, search_start_idx - 200)
            for i in range(extended_search_start, search_start_idx):
                t = times[i]
                e_c_t = e_c_evolution[i]
                r_c_t = r_c_evolution[i]

                # Calculate jet-cocoon properties
                e_cj = self.jet_cocoon_emission.total_jet_cocoon_energy(e_c_t)
                beta_cr = self.jet_cocoon_emission.critical_velocity(
                    self.params.kappa, e_cj, r_c_t
                )
                beta_cjs = self.jet_cocoon_emission.velocity_of_first_shell(beta_cr)
                r_cj = self.jet_cocoon_emission.diffusion_radius(
                    self.params.kappa, e_cj, beta_cjs
                )

                relative_error = abs(r_cj - r_c_t) / r_c_t

                if relative_error < best_error:
                    best_error = relative_error
                    best_idx = i
                    best_time = t

        # Extract parameters at best time found
        critical_time_idx = best_idx
        critical_time = best_time

        e_c_crit = e_c_evolution[critical_time_idx]
        r_c_crit = r_c_evolution[critical_time_idx]
        z_h_crit = z_h_evolution[critical_time_idx]
        beta_c_crit = beta_c_evolution[critical_time_idx]

        # Calculate final jet-cocoon properties at critical time
        e_cj_crit = self.jet_cocoon_emission.total_jet_cocoon_energy(e_c_crit)
        beta_cr_crit = self.jet_cocoon_emission.critical_velocity(
            self.params.kappa, e_cj_crit, r_c_crit
        )
        beta_cjs_crit = self.jet_cocoon_emission.velocity_of_first_shell(beta_cr_crit)
        r_cj_crit = self.jet_cocoon_emission.diffusion_radius(
            self.params.kappa, e_cj_crit, beta_cjs_crit
        )

        condition_satisfied = best_error <= tolerance

        print(
            f"  Critical time found: t = {critical_time:.2e} s (index {critical_time_idx})"
        )
        print(
            f"  Condition r_cj = r_c satisfied with relative error = {best_error:.2e}"
        )
        print(f"  Tolerance check: {condition_satisfied} (tolerance = {tolerance:.2e})")

        if not condition_satisfied:
            print(f"  WARNING: Could not satisfy r_cj = r_c within tolerance!")
            print(f"  Using best match found with error = {best_error:.2e}")

        return {
            "critical_time": critical_time,
            "critical_time_idx": critical_time_idx,
            "breakout_time": t_break,
            "breakout_time_idx": idx_break,
            "condition_satisfied": condition_satisfied,
            "relative_error": best_error,
            "parameters": {
                "e_c": e_c_crit,
                "r_c": r_c_crit,
                "z_h": z_h_crit,
                "beta_c": beta_c_crit,
                "e_cj": e_cj_crit,
                "beta_cr": beta_cr_crit,
                "beta_cjs": beta_cjs_crit,
                "r_cj": r_cj_crit,
            },
        }

    def calculate_jet_cocoon_breakout_properties(
        self,
        e_c_bre: float = None,
        r_c_bre: float = None,
        z_bre: float = None,
        beta_c_bre: float = None,
        critical_time_data: dict = None,
    ) -> dict:
        """
        Calculate time-independent jet-cocoon properties.

        This method can work in two modes:
        1. Legacy mode: Use breakout parameters directly (may not satisfy r_cj = r_c)
        2. Critical time mode: Use parameters from when r_cj = r_c is satisfied

        Parameters
        ----------
        e_c_bre : float, optional
            Cocoon energy [erg] (legacy mode)
        r_c_bre : float, optional
            Cocoon radius [cm] (legacy mode)
        z_bre : float, optional
            Jet head height [cm] (legacy mode)
        beta_c_bre : float, optional
            Cocoon velocity (legacy mode)
        critical_time_data : dict, optional
            Data from find_jet_cocoon_critical_time() (critical time mode)

        Returns
        -------
        dict
            Dictionary containing all time-independent jet-cocoon properties
        """

        if critical_time_data is not None:
            # Critical time mode: use parameters when r_cj = r_c
            print("Using CRITICAL TIME parameters (r_cj = r_c condition satisfied)")
            params = critical_time_data["parameters"]
            e_c = params["e_c"]
            r_c = params["r_c"]
            z_h = params["z_h"]
            beta_c = params["beta_c"]

            # These are already calculated and satisfy r_cj = r_c
            e_cj = params["e_cj"]
            beta_cr = params["beta_cr"]
            beta_cjs = params["beta_cjs"]
            r_cjs = params["r_cj"]

            print(f"  Critical time: {critical_time_data['critical_time']:.2e} s")
            print(f"  Condition error: {critical_time_data['relative_error']:.2e}")

        else:
            # Legacy mode: use breakout parameters directly
            print("Using BREAKOUT parameters (may not satisfy r_cj = r_c)")
            if any(param is None for param in [e_c_bre, r_c_bre, z_bre, beta_c_bre]):
                raise ValueError(
                    "Either critical_time_data or all breakout parameters must be provided"
                )

            e_c = e_c_bre
            r_c = r_c_bre
            z_h = z_bre
            beta_c = beta_c_bre

            # Calculate jet-cocoon properties (may not satisfy r_cj = r_c exactly)
            e_cj = self.jet_cocoon_emission.total_jet_cocoon_energy(e_c)
            beta_cr = self.jet_cocoon_emission.critical_velocity(
                self.params.kappa, e_cj, r_c
            )
            beta_cjs = self.jet_cocoon_emission.velocity_of_first_shell(beta_cr)
            r_cjs = self.jet_cocoon_emission.diffusion_radius(
                self.params.kappa, e_cj, beta_cjs
            )

        # Common calculations using the determined parameters
        m_cjs = self.jet_cocoon_emission.newtonian_jet_cocoon_mass(e_cj, beta_cjs)
        v_cjs = self.jet_cocoon_emission.initial_jet_cocoon_volume(r_c, z_h)

        print("&" * 50)
        print(f"r_cjs: {r_cjs:.2e} cm, r_c: {r_c:.2e} cm")
        print(f"|r_cjs - r_c|/r_c = {abs(r_cjs - r_c)/r_c:.2e}")
        print("&" * 50)
        t_cjs = self.jet_cocoon_emission.diffustion_time(
            self.params.kappa, e_cj, beta_cjs
        )
        l_cjs = self.jet_cocoon_emission.emission_luminosity(e_cj, v_cjs, t_cjs, r_cjs)
        t_sph_end = self.jet_cocoon_emission.radiative_time(t_cjs, beta_cjs, beta_c)

        # Shell and thermal properties (time-independent)
        d_cjs = self.jet_cocoon_emission.shell_distance(z_h)
        t_escs = self.jet_cocoon_emission.shell_escaping_time(d_cjs, beta_cjs)
        temp_bb_cj0 = self.jet_cocoon_emission.initial_bb_temp(e_cj, v_cjs)
        rho_cjs = self.jet_cocoon_emission.rho_cj_s(m_cjs, v_cjs)
        eta_escs = self.jet_cocoon_emission.initial_thermal_coupling_coefficient(
            e_cj, v_cjs, beta_cjs, m_cjs, z_h
        )
        eta_cjs = self.jet_cocoon_emission.thermal_coupling_coefficient(
            eta_escs, t_cjs, t_escs
        )
        t_thph = self.jet_cocoon_emission.transition_time_th_ph(
            t_cjs, beta_cjs, eta_cjs
        )

        # Temperature normalization factors (time-independent)
        # Following coauthor's suggestion: Use anchor temperature approach even for η ~ 1
        # because ξ evolution becomes important in this regime

        # rho_cjs already computed above

        if eta_escs <= 1:
            # η ≤ 1: still use Compton-aware anchoring without arbitrary fallbacks
            print(f"Using ξ-aware anchor temperatures for η = {eta_escs:.6f} (≤ 1)")

            # Thermal normalization for later regimes
            temp_thermal_norm = self.jet_cocoon_emission.normalization_temp_thermal_eq(
                l_cjs, beta_cjs, r_cjs
            )

            # Compute anchors via full relation T ξ^2 = T_BB η^2 (with time scaling)
            # Avoid exactly t=t_cjs to prevent 0 from helper
            t_anchor = t_cjs * 1.001
            temp_at_t_cjs = self.jet_cocoon_emission.compton_temperature_at_time(
                t_anchor, t_cjs, temp_bb_cj0, eta_cjs, rho_cjs
            )

            # Critical time using ξ-aware condition
            try:
                t_cj_th1 = self.jet_cocoon_emission.critical_time_one_full(
                    t_cjs, eta_cjs, temp_bb_cj0, rho_cjs
                )
            except Exception:
                t_cj_th1 = self.jet_cocoon_emission.critical_time_one(t_cjs, eta_cjs)

            # Safety clamp: ensure progression and avoid extreme span
            if t_cj_th1 <= t_cjs:
                t_cj_th1 = t_cjs * 5
            if t_cj_th1 > t_cjs * 1e3:
                t_cj_th1 = t_cjs * 1e2

            temp_at_t_cjth1 = self.jet_cocoon_emission.compton_temperature_at_time(
                t_cj_th1, t_cjs, temp_bb_cj0, eta_cjs, rho_cjs
            )

            t_cj_th2 = self.jet_cocoon_emission.critical_time_two(t_cjs, eta_cjs)
            temp_norm = temp_thermal_norm
            thermal_regime = "thermal_equilibrium_with_anchoring"

        else:
            # Standard Compton-dominated regime (η >> 1)
            temp_escs = self.jet_cocoon_emission.temp_esc(
                temp_bb_cj0, eta_escs, rho_cjs
            )
            # Observed Compton normalization uses t_escs, not temperature
            temp_norm = self.jet_cocoon_emission.normalization_temp_observed_compton(
                temp_escs, v_cjs, r_cjs
            )

            # Use the full method to calculate t_cj_th1 using Equation C14
            try:
                t_cj_th1 = self.jet_cocoon_emission.critical_time_one_full(
                    t_cjs, eta_cjs, temp_bb_cj0, rho_cjs
                )
            except:
                # Fallback to simplified method if full method fails
                print(
                    "Warning: Full critical time calculation failed, using simplified method."
                )
                t_cj_th1 = self.jet_cocoon_emission.critical_time_one(t_cjs, eta_cjs)

            t_cj_th2 = self.jet_cocoon_emission.critical_time_two(t_cjs, eta_cjs)

            # Calculate anchor temperatures for proper normalization
            # Temperature anchors using full ξ-aware relation
            temp_at_t_cjs = temp_escs
            temp_at_t_cjth1 = self.jet_cocoon_emission.compton_temperature_at_time(
                t_cj_th1, t_cjs, temp_bb_cj0, eta_cjs, rho_cjs
            )

            thermal_regime = "compton_dominated"

        properties = {
            # Basic properties
            "e_cj": e_cj,
            "beta_cr": beta_cr,
            "beta_cjs": beta_cjs,
            "m_cjs": m_cjs,
            "v_cjs": v_cjs,
            "r_cjs": r_cjs,
            "t_cjs": t_cjs,
            "l_cjs": l_cjs,
            "t_sph_end": t_sph_end,
            # Thermal properties
            "eta_escs": eta_escs,
            "d_cjs": d_cjs,
            "t_escs": t_escs,
            "eta_cjs": eta_cjs,
            "t_thph": t_thph,
            "temp_bb_cj0": temp_bb_cj0,
            "temp_norm": temp_norm,
            "thermal_regime": thermal_regime,
            # Anchor temperatures (may be None for thermal equilibrium)
            "temp_at_t_cjs": temp_at_t_cjs,
            "temp_at_t_cjth1": temp_at_t_cjth1,
        }

        # Enforce regime ordering consistency and add informational logs
        if "t_cj_th1" in locals() and "t_cj_th2" in locals():
            if t_cj_th2 <= t_cj_th1:
                print(
                    "Note: Adjusting critical time order to enforce t_cj_th1 < t_cj_th2."
                )
                t_cj_th1, t_cj_th2 = t_cj_th2, t_cj_th1
                # Recompute anchor for the (new) t_cj_th1
                temp_at_t_cjth1 = self.jet_cocoon_emission.compton_temperature_at_time(
                    t_cj_th1, t_cjs, temp_bb_cj0, eta_cjs, rho_cjs
                )
                properties["temp_at_t_cjth1"] = temp_at_t_cjth1
                properties["t_cj_th1"] = t_cj_th1
                properties["t_cj_th2"] = t_cj_th2

            # Keep t_cj_th2 within the t_thph boundary
            if t_cj_th2 > t_thph:
                print(
                    f"Info: Clamping t_cj_th2 ({t_cj_th2:.2e} s) to t_thph ({t_thph:.2e} s)."
                )
                t_cj_th2 = t_thph
                properties["t_cj_th2"] = t_cj_th2

        # Add regime-specific properties
        if thermal_regime == "compton_dominated":
            properties.update(
                {
                    "rho_cjs": rho_cjs,
                    "temp_escs": temp_escs,
                    "t_cj_th1": t_cj_th1,
                    "t_cj_th2": t_cj_th2,
                }
            )
        elif thermal_regime == "thermal_equilibrium_with_anchoring":
            properties.update(
                {
                    "rho_cjs": rho_cjs,
                    "t_cj_th1": t_cj_th1,
                    "t_cj_th2": t_cj_th2,
                }
            )

        return properties

    def calculate_jet_cocoon_emission_with_critical_time(
        self,
        results,
        t_break: float,
        idx_break: int,
        use_critical_time: bool = True,
        search_range_factor: float = 0.5,
        tolerance: float = 1e-3,
    ) -> dict:
        """
        Calculate jet-cocoon properties using either critical time or breakout time approach.

        This is the recommended method for jet-cocoon calculations as it automatically
        finds the time when r_cj = r_c condition is satisfied (critical time approach)
        or falls back to breakout parameters (legacy approach).

        Parameters
        ----------
        results : EvolutionResults
            Evolution results containing time-dependent parameters
        t_break : float
            Shock breakout time [s]
        idx_break : int
            Index of shock breakout in results arrays
        use_critical_time : bool, optional
            Whether to use critical time approach (True) or legacy breakout approach (False)
        search_range_factor : float, optional
            Start search at t_break * search_range_factor. Default 0.5.
        tolerance : float, optional
            Relative tolerance for r_cj = r_c condition. Default 1e-3.

        Returns
        -------
        dict
            Dictionary containing jet-cocoon properties and metadata:
            - 'properties': Time-independent jet-cocoon properties
            - 'approach': 'critical_time' or 'breakout'
            - 'critical_time_data': If critical time approach, contains timing info
            - 'timing_comparison': Comparison between critical and breakout times
        """
        print(
            f"\nCalculating jet-cocoon properties using {'CRITICAL TIME' if use_critical_time else 'BREAKOUT'} approach"
        )

        # Extract breakout parameters for comparison and potential fallback
        e_c_bre = results.e_c[idx_break]
        r_c_bre = results.r_c[idx_break]
        z_bre = results.z_h[idx_break]
        beta_c_bre = results.beta_c[idx_break]

        if use_critical_time:
            try:
                # Find critical time when r_cj = r_c
                critical_time_data = self.find_jet_cocoon_critical_time(
                    results, t_break, idx_break, search_range_factor, tolerance
                )

                # Calculate properties using critical time parameters
                properties = self.calculate_jet_cocoon_breakout_properties(
                    critical_time_data=critical_time_data
                )

                approach = "critical_time"
                print(f"SUCCESS: Using critical time approach")
                print(f"  Critical time: {critical_time_data['critical_time']:.2e} s")
                print(f"  Breakout time: {t_break:.2e} s")
                print(
                    f"  Time ratio: {critical_time_data['critical_time']/t_break:.3f}"
                )

                # Calculate timing comparison
                timing_comparison = {
                    "critical_time": critical_time_data["critical_time"],
                    "breakout_time": t_break,
                    "time_ratio": critical_time_data["critical_time"] / t_break,
                    "critical_earlier_by_factor": t_break
                    / critical_time_data["critical_time"],
                }

                return {
                    "properties": properties,
                    "approach": approach,
                    "critical_time_data": critical_time_data,
                    "timing_comparison": timing_comparison,
                }

            except Exception as e:
                print(f"WARNING: Critical time approach failed: {e}")
                print(f"         Falling back to breakout approach")
                use_critical_time = False

        if not use_critical_time:
            # Legacy breakout approach
            properties = self.calculate_jet_cocoon_breakout_properties(
                e_c_bre, r_c_bre, z_bre, beta_c_bre
            )

            approach = "breakout"
            print(f"Using breakout time approach")
            print(f"  Breakout time: {t_break:.2e} s")

            return {
                "properties": properties,
                "approach": approach,
                "critical_time_data": None,
                "timing_comparison": None,
            }

    def calculate_jet_cocoon_emission(
        self,
        t: float,
        jet_cocoon_properties: dict = None,
        e_c_bre: float = None,
        r_c_bre: float = None,
        z_bre: float = None,
        beta_c_bre: float = None,
        results=None,
        t_break: float = None,
        idx_break: int = None,
        use_critical_time: bool = True,
    ) -> float:
        """
        Calculate jet-cocoon emission luminosity at time t.

        This method can work in multiple modes:
        1. Efficient mode: Pass pre-calculated jet_cocoon_properties
        2. Critical time mode: Pass evolution results and use critical time approach (recommended)
        3. Legacy mode: Pass breakout parameters (uses breakout timing)

        Parameters
        ----------
        t : float
            Time [s]
        jet_cocoon_properties : dict, optional
            Pre-calculated jet-cocoon properties from calculate_jet_cocoon_breakout_properties()
        e_c_bre : float, optional
            Cocoon energy at breakout [erg] (for legacy mode)
        r_c_bre : float, optional
            Cocoon radius at breakout [cm] (for legacy mode)
        z_bre : float, optional
            Jet head height at breakout [cm] (for legacy mode)
        beta_c_bre : float, optional
            Cocoon velocity at breakout (for legacy mode)
        results : EvolutionResults, optional
            Evolution results for critical time calculation (critical time mode)
        t_break : float, optional
            Shock breakout time [s] (critical time mode)
        idx_break : int, optional
            Index of shock breakout (critical time mode)
        use_critical_time : bool, optional
            Whether to use critical time approach when results are provided. Default True.

        Returns
        -------
        float
            Jet-cocoon luminosity at time t [erg/s]
        """
        if jet_cocoon_properties is not None:
            # Efficient mode: use pre-calculated properties
            props = jet_cocoon_properties
        elif results is not None and t_break is not None and idx_break is not None:
            # Critical time mode: calculate properties using critical time approach
            jet_cocoon_data = self.calculate_jet_cocoon_emission_with_critical_time(
                results, t_break, idx_break, use_critical_time=use_critical_time
            )
            props = jet_cocoon_data["properties"]
        else:
            # Legacy mode: calculate properties on-the-fly using breakout parameters
            if any(param is None for param in [e_c_bre, r_c_bre, z_bre, beta_c_bre]):
                raise ValueError(
                    "Must provide either: "
                    "1) jet_cocoon_properties, or "
                    "2) results + t_break + idx_break (critical time mode), or "
                    "3) all breakout parameters e_c_bre, r_c_bre, z_bre, beta_c_bre (legacy mode)"
                )
            props = self.calculate_jet_cocoon_breakout_properties(
                e_c_bre, r_c_bre, z_bre, beta_c_bre
            )

        # Calculate time-dependent luminosity using the determined properties
        l_cj_of_t = self.jet_cocoon_emission.calculate_jet_cocoon_emission_evolution(
            t, props["l_cjs"], props["t_cjs"], props["t_sph_end"]
        )
        return l_cj_of_t if l_cj_of_t is not None else 0.0

    def calculate_jet_cocoon_temperature(
        self,
        t: float,
        jet_cocoon_properties: dict = None,
        e_c_bre: float = None,
        r_c_bre: float = None,
        z_bre: float = None,
        beta_c_bre: float = None,
        results=None,
        t_break: float = None,
        idx_break: int = None,
        use_critical_time: bool = True,
    ) -> float:
        """
        Calculate jet-cocoon emission temperature at time t.

        This method can work in multiple modes:
        1. Efficient mode: Pass pre-calculated jet_cocoon_properties
        2. Critical time mode: Pass evolution results and use critical time approach (recommended)
        3. Legacy mode: Pass breakout parameters (uses breakout timing)

        Parameters
        ----------
        t : float
            Time [s]
        jet_cocoon_properties : dict, optional
            Pre-calculated jet-cocoon properties from calculate_jet_cocoon_breakout_properties()
        e_c_bre : float, optional
            Cocoon energy at breakout [erg] (for legacy mode)
        r_c_bre : float, optional
            Cocoon radius at breakout [cm] (for legacy mode)
        z_bre : float, optional
            Jet head height at breakout [cm] (for legacy mode)
        beta_c_bre : float, optional
            Cocoon velocity at breakout (for legacy mode)
        results : EvolutionResults, optional
            Evolution results for critical time calculation (critical time mode)
        t_break : float, optional
            Shock breakout time [s] (critical time mode)
        idx_break : int, optional
            Index of shock breakout (critical time mode)
        use_critical_time : bool, optional
            Whether to use critical time approach when results are provided. Default True.

        Returns
        -------
        float
            Jet-cocoon temperature at time t [K]
        """
        if jet_cocoon_properties is not None:
            # Efficient mode: use pre-calculated properties
            props = jet_cocoon_properties
        elif results is not None and t_break is not None and idx_break is not None:
            # Critical time mode: calculate properties using critical time approach
            jet_cocoon_data = self.calculate_jet_cocoon_emission_with_critical_time(
                results, t_break, idx_break, use_critical_time=use_critical_time
            )
            props = jet_cocoon_data["properties"]
        else:
            # Legacy mode: calculate properties on-the-fly using breakout parameters
            if any(param is None for param in [e_c_bre, r_c_bre, z_bre, beta_c_bre]):
                raise ValueError(
                    "Must provide either: "
                    "1) jet_cocoon_properties, or "
                    "2) results + t_break + idx_break (critical time mode), or "
                    "3) all breakout parameters e_c_bre, r_c_bre, z_bre, beta_c_bre (legacy mode)"
                )
            props = self.calculate_jet_cocoon_breakout_properties(
                e_c_bre, r_c_bre, z_bre, beta_c_bre
            )

        # Calculate temperature based on thermal coupling regime
        if props["thermal_regime"] == "thermal_equilibrium":
            # Traditional thermal equilibrium regime (if we ever use it)
            temp_scaling = self.jet_cocoon_emission.thermal_eq_temperature(
                t,
                props["t_cjs"],
                props["t_thph"],
                props["t_sph_end"],
                props["temp_norm"],
            )
            return temp_scaling

        elif props["thermal_regime"] == "thermal_equilibrium_with_anchoring":
            # Thermal equilibrium with anchor temperature approach (coauthor's suggestion)
            # Use Compton evolution but with anchor temperatures appropriate for η ~ 1
            temp_at_t_cjs = props.get("temp_at_t_cjs", None)
            temp_at_t_cjth1 = props.get("temp_at_t_cjth1", None)

            temp_scaling = self.jet_cocoon_emission.compton_temperature_evolution(
                t,
                props["t_cjs"],
                props["t_cj_th1"],
                props["t_cj_th2"],
                props["t_thph"],
                props["t_sph_end"],
                props["temp_norm"],
                temp_at_t_cjs,
                temp_at_t_cjth1,
            )
            return temp_scaling

        else:
            # Compton-dominated regime (η >> 1) - use new method with anchor temperatures
            temp_at_t_cjs = props.get("temp_at_t_cjs", None)
            temp_at_t_cjth1 = props.get("temp_at_t_cjth1", None)

            temp_scaling = self.jet_cocoon_emission.compton_temperature_evolution(
                t,
                props["t_cjs"],
                props["t_cj_th1"],
                props["t_cj_th2"],
                props["t_thph"],
                props["t_sph_end"],
                props["temp_norm"],
                temp_at_t_cjs,
                temp_at_t_cjth1,
            )
            return temp_scaling

    def calculate_rel_jet_cocoon_lum_and_temp(
        self, e_cj: float, r_c_bre: float, z_bre: float
    ):
        theta_c = self.cocoon.theta_c(r_c_bre, z_bre)
        gamma_b = self.jet_cocoon_emission.critical_baryon_lorentz_factor(
            e_cj, z_bre, theta_c
        )
        gamma_s = self.jet_cocoon_emission.critical_spread_lorentz_factor(e_cj, z_bre)
        gamma_bb = self.jet_cocoon_emission.critical_thermal_lorentz_factor(
            e_cj, z_bre, theta_c
        )
        gamma_min, gamma_max = self.jet_cocoon_emission.gamma_boundaries(gamma_s)
        if gamma_max < gamma_min:
            # No valid relativistic shell range; suppress plateau
            return None, None, None

        # If radiation cannot accelerate to relativistic (gamma_b <= 1.4)
        if gamma_b <= 1.4:
            # Duration and luminosity per Nakar17 Eq. (rel-Newtonian branch)
            t_cj = self.jet_cocoon_emission.gamma_bb_emission_duration(z_bre)
            l_cj = self.jet_cocoon_emission.gamma_bb_luminosity(e_cj, z_bre)
            # Temperature: pair regulated ~200 keV when out of equilibrium; else return 0 to signal BB evaluation downstream
            if gamma_bb < 1:
                temp_cj = self.jet_cocoon_emission.T_PAIR_MAX_K
            else:
                temp_cj = 0.0
            return t_cj, l_cj, temp_cj

        # gamma_b > 1.4: relativistic shell observables; choose a representative gamma in [gamma_min, gamma_max]
        # Use gamma representative = min(gamma_bb, gamma_max) for thermal, else gamma_min for non-thermal
        if gamma_bb > gamma_max:
            # Fully thermalized relativistic regime; use gamma_max
            obs = self.jet_cocoon_emission.relativistic_observables_for_gamma(
                self.params.kappa, e_cj, r_c_bre, z_bre, gamma_max
            )
            # Eqs. 48–49 give temporal evolution; here return instantaneous values at representative gamma
            return obs["t_cj"], obs["L_iso"], obs["T_obs_th"]

        if gamma_bb < 1.4:
            # Entire relativistic segment non-thermal/pair-regulated; use gamma in [gamma_min, gamma_max]
            obs = (
                self.jet_cocoon_emission.relativistic_nonthermal_observables_for_gamma(
                    self.params.kappa, e_cj, r_c_bre, z_bre, gamma_min
                )
            )
            return obs["t_cj"], obs["L_iso"], obs["T_obs_nt"]

        # Mixed case: gamma_min < gamma_bb < gamma_max
        # Report thermalized part at gamma_bb as characteristic; non-thermal handled elsewhere by time-evolution
        gamma_char = max(gamma_min, min(gamma_bb, gamma_max))
        obs = self.jet_cocoon_emission.relativistic_observables_for_gamma(
            self.params.kappa, e_cj, r_c_bre, z_bre, gamma_char
        )
        return obs["t_cj"], obs["L_iso"], obs["T_obs_th"]
