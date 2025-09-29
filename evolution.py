"""
Time evolution of the jet system.

This module implements the main evolution class that coordinates all components
to simulate the time evolution of the Chen & Dai 2025 jet model.
"""

import numpy as np
from typing import Dict, Optional, Union

try:
    from .constants import ModelParameters, C_CGS
    from .disk_model import AGNDiskModel, create_disk_model
    from .solver import BetaHSolver
    from .physics import PhysicsCalculator
except ImportError:
    # Fallback for when not imported as a package
    from constants import ModelParameters, C_CGS
    from disk_model import AGNDiskModel, create_disk_model
    from solver import BetaHSolver
    from physics import PhysicsCalculator
from scipy.integrate import quad


def density_profiles_with_edges(rho0: float, h: float, z: float, profile: str) -> float:
    """
    Calculate density profiles with proper disk edges based on optical depth.

    Following Chen & Dai 2025:
    - Uniform: disk edge at z = H_d
    - Polytropic: disk edge at z = sqrt(6) * H_d ≈ 2.45 * H_d
    - Isothermal: no disk edge (z_edge = infinity)

    Parameters
    ----------
    rho0 : float
        Central density [g/cm³]
    h : float
        Disk scale height [cm]
    z : float
        Height above disk [cm]
    profile : str
        Density profile type

    Returns
    -------
    float
        Density at height z [g/cm³]
    """
    if profile == "uniform":
        # Sharp cutoff at scale height H_d
        if z <= h:
            return rho0
        else:
            return 1e-30 * rho0  # Very small but non-zero density beyond disk edge

    elif profile == "polytropic":
        # Disk edge at sqrt(6) * H_d where profile naturally goes to zero
        z_edge = np.sqrt(6) * h  # ≈ 2.45 * h
        if z <= z_edge:
            return rho0 * (1 - z**2 / (6 * h**2)) ** 3
        else:
            return 1e-30 * rho0  # Very small but non-zero density beyond disk edge

    elif profile == "isothermal":
        # No disk edge - continues indefinitely
        return rho0 * np.exp(-(z**2) / (2 * h**2))

    else:
        raise ValueError(f"Unknown density profile type: {profile}")


def optical_depth_to_surface(
    rho0: float,
    h: float,
    z_start: float,
    density_profile: str,
    kappa_opacity: float = 0.34,
    breakout_threshold: Optional[float] = None,
) -> tuple:
    """
    Calculate optical depth from z_start to disk surface.

    τ_s(z) = ∫[z to z_edge] κρ(z')dz'

    Parameters
    ----------
    rho0 : float
        Central density [g/cm³]
    h : float
        Disk scale height [cm]
    z_start : float
        Starting height [cm]
    density_profile : str
        Density profile type
    kappa_opacity : float
        Opacity [cm²/g]
    breakout_threshold : float, optional
        Optional threshold for breakout flag. If None, breakout flag is always False.
        Note: The physical breakout condition is τ(z_h) = 1/β_h, not a fixed threshold.

    Returns
    -------
    tuple
        (optical_depth, breakout_flag) where breakout_flag indicates τ_s < threshold
    """
    # Define integration limits based on profile
    if density_profile == "uniform":
        z_edge = h  # H_d
    elif density_profile == "polytropic":
        z_edge = np.sqrt(6) * h  # sqrt(6) * H_d
    elif density_profile == "isothermal":
        z_edge = 10 * h  # Approximate infinity (integrate to large distance)
    else:
        raise ValueError(f"Unknown density profile: {density_profile}")

    # If already beyond disk edge, optical depth is zero
    if z_start >= z_edge:
        return 0.0, True  # Broken out

    # Numerical integration from z_start to z_edge
    z_points = np.linspace(z_start, z_edge, 1000)

    # Calculate density at each point
    rho_values = np.array(
        [density_profiles_with_edges(rho0, h, z, density_profile) for z in z_points]
    )

    # Integrate τ = ∫ κρ dz
    tau_s = kappa_opacity * np.trapz(rho_values, z_points)

    # Apply breakout threshold if provided
    if breakout_threshold is not None:
        breakout = tau_s < breakout_threshold
    else:
        breakout = False  # No threshold provided

    return tau_s, breakout


def find_shock_breakout_time(
    result, model, density_profile: str = "isothermal", kappa_opacity: float = 0.34
):
    """
    Find the time when shock breakout occurs (τ(z_h) = 1/β_h).

    Parameters
    ----------
    result : EvolutionResults
        Evolution results
    model : ChenDaiModel
        Model instance for accessing parameters
    density_profile : str
        Density profile type
    kappa_opacity : float
        Opacity for optical depth calculation [cm²/g]

    Returns
    -------
    tuple
        (breakout_time, breakout_index) or (None, None) if no breakout
    """
    import numpy as np

    times = result.times if hasattr(result, "times") else np.array(result.data["times"])
    beta_h = (
        result.beta_h if hasattr(result, "beta_h") else np.array(result.data["beta_h"])
    )
    z_h = result.z_h if hasattr(result, "z_h") else np.array(result.data["z_h"])

    for i, (t, bh, zh) in enumerate(zip(times, beta_h, z_h)):
        # Calculate optical depth at current z_h
        tau_zh, _ = optical_depth_to_surface(
            model.rho_0, model.h, zh, density_profile, kappa_opacity
        )

        # Check breakout condition: τ(z_h) = 1/β_h
        if tau_zh <= 1.0 / bh:
            print(f"  Shock breakout at t = {t:.2e} s (index {i})")
            print(f"    τ(z_h) = {tau_zh:.3f}, 1/β_h = {1.0/bh:.3f}")
            print(f"    z_h/h = {zh/model.h:.3f}")
            return t, i

    print("  No shock breakout detected in time range")
    return None, None


class EvolutionResults:
    """Container for evolution results with convenient access methods."""

    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initialize results container.

        Parameters
        ----------
        data : dict
            Dictionary containing evolution data
        """
        self.data = data

        # Make data accessible as attributes
        for key, value in data.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> np.ndarray:
        """Allow dictionary-style access."""
        return self.data[key]

    def keys(self):
        """Return data keys."""
        return self.data.keys()

    def summary(self) -> None:
        """Print summary of results."""
        print("=== Evolution Results Summary ===")
        for key, values in self.data.items():
            if key == "times":
                print(f"Time range: {values[0]:.1e} - {values[-1]:.1e} s")
            else:
                print(f"{key}: {np.min(values):.6g} - {np.max(values):.6g}")

        print(f"\nFinal values:")
        for key, values in self.data.items():
            if key != "times":
                print(f"  {key} = {values[-1]:.6g}")


class ChenDaiModel:
    """
    Main class for Chen & Dai 2025 jet evolution model.

    This class coordinates all components to perform time evolution
    simulations of AGN jets propagating through disk environments.
    """

    def __init__(
        self,
        l_j: float,
        params: Optional[ModelParameters] = None,
        disk_filename: str = "chen_and_dai_25",
        disk_radius_rg: float = 1e3,
        data_dir: str = "./agn_disks",
        auto_generate_disk: bool = False,
    ):
        """
        Initialize Chen & Dai model.

        Parameters
        ----------
        l_j : float
            Jet luminosity [erg/s]
        params : ModelParameters, optional
            Model parameters (uses defaults if None)
        disk_filename : str
            Name of disk data file
        disk_radius_rg : float
            Disk radius in gravitational radii where jet is launched
        data_dir : str
            Directory containing disk data
        auto_generate_disk : bool
            Whether to auto-generate disk model if it doesn't exist
        """
        self.l_j = l_j
        self.params = params or ModelParameters()
        self.disk_filename = disk_filename
        self.disk_radius_rg = disk_radius_rg
        self.data_dir = data_dir
        self.auto_generate_disk = auto_generate_disk

        # Initialize components
        self.disk_model = None
        self.solver = None
        self.physics = None
        self.rho_0 = None
        self.h = None

        # Setup
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all model components."""
        # Load disk model
        self.disk_model = create_disk_model(
            self.params,
            self.disk_filename,
            self.data_dir,
            auto_generate=self.auto_generate_disk,
        )

        # Get local disk properties
        self.rho_0, self.h = self.disk_model.get_local_properties(self.disk_radius_rg)

        # Initialize solver and physics
        self.solver = BetaHSolver(self.params, self.l_j)
        self.physics = PhysicsCalculator(self.params, self.l_j)

        print(f"\n=== Chen & Dai Model Initialized ===")
        print(f"Jet luminosity: L_j = {self.l_j:.2e} erg/s")
        print(f"Launch radius: {self.disk_radius_rg:.1f} r_g")
        print(f"Model parameters: {self.params}")

    def evolve(
        self,
        times: np.ndarray,
        density_profile: str = "uniform",
        initial_guess: float = 0.1,
        progress_interval: int = 100,
        store_diagnostics: bool = False,
        respect_profile_limits: bool = True,
        use_optical_depth: bool = True,
        kappa_opacity: float = 0.34,
        use_numerical_integration: bool = False,
        integration_method: str = "trapz",
    ) -> EvolutionResults:
        """
        Evolve jet system over specified times.

        Parameters
        ----------
        times : np.ndarray
            Time array [s]
        density_profile : str
            Density profile type: "uniform", "isothermal", or "polytropic"
        initial_guess : float
            Initial guess for beta_h
        progress_interval : int
            Print progress every N steps
        store_diagnostics : bool
            Whether to store additional diagnostic quantities
        respect_profile_limits : bool
            Whether to apply profile-specific stopping criteria
        use_optical_depth : bool
            Whether to include optical depth calculations and breakout
        kappa_opacity : float
            Opacity for optical depth calculation [cm²/g]
        use_numerical_integration : bool
            Whether to use proper numerical integration of differential equations
        integration_method : str
            Integration method: "trapz", "simpson", or "quad"

        Returns
        -------
        EvolutionResults
            Evolution results
        """
        if use_numerical_integration:
            return self.evolve_with_numerical_integration(
                times,
                density_profile,
                initial_guess,
                progress_interval,
                store_diagnostics,
                use_optical_depth,
                kappa_opacity,
                integration_method,
            )
        elif respect_profile_limits:
            return self.evolve_with_profile_logic(
                times,
                density_profile,
                initial_guess,
                progress_interval,
                store_diagnostics,
                use_optical_depth,
                kappa_opacity,
            )

        # Original evolution logic (for backward compatibility)
        return self._evolve_original(
            times, initial_guess, progress_interval, store_diagnostics
        )

    def _evolve_original(
        self,
        times: np.ndarray,
        initial_guess: float = 0.1,
        progress_interval: int = 100,
        store_diagnostics: bool = False,
    ) -> EvolutionResults:
        """
        Evolve the jet system over specified times (original method).

        Parameters
        ----------
        times : np.ndarray
            Time array [s]
        initial_guess : float
            Initial guess for beta_h
        progress_interval : int
            Interval for progress reporting
        store_diagnostics : bool
            Whether to store diagnostic information

        Returns
        -------
        EvolutionResults
            Evolution results
        """
        print(f"\n=== Evolving Jet System ===")
        print(f"Time steps: {len(times)}")
        print(f"Time range: {times[0]:.1e} - {times[-1]:.1e} s")

        # Initialize storage arrays
        results = {
            "times": times,
            "beta_h": np.zeros_like(times),
            "z_h": np.zeros_like(times),
            "zh_over_h": np.zeros_like(times),
            "e_c": np.zeros_like(times),
            "theta_h": np.zeros_like(times),
            "theta_c": np.zeros_like(times),
            "gamma_h": np.zeros_like(times),
        }

        if store_diagnostics:
            results.update(
                {
                    "l_tilde": np.zeros_like(times),
                    "rho_z": np.zeros_like(times),
                    "beta_c": np.zeros_like(times),
                    "r_c": np.zeros_like(times),
                }
            )

        # Evolution loop
        beta_h_prev = initial_guess

        for i, t in enumerate(times):
            if i % progress_interval == 0:
                print(f"Processing step {i+1}/{len(times)} (t = {t:.2e} s)")

            # Solve for beta_h
            beta_h = self.solver.solve_single(t, self.rho_0, self.h, beta_h_prev)

            # Validate and store
            if not 0 < beta_h < 1:
                print(f"Warning: beta_h = {beta_h:.6f} out of bounds at t = {t:.2e} s")
                beta_h = max(0.001, min(0.999, beta_h))

            # Calculate all derived quantities
            quantities = self.physics.calculate_all_quantities(
                beta_h, t, self.rho_0, self.h, "isothermal"
            )

            # Store main results
            results["beta_h"][i] = beta_h
            results["z_h"][i] = quantities["z_h"]
            results["zh_over_h"][i] = quantities["z_h"] / self.h
            results["e_c"][i] = quantities["e_c"]
            results["theta_h"][i] = quantities["theta_h"]
            results["theta_c"][i] = quantities["theta_c"]
            results["gamma_h"][i] = quantities["gamma_h"]

            # Store diagnostics if requested
            if store_diagnostics:
                results["l_tilde"][i] = quantities["l_tilde"]
                results["rho_z"][i] = quantities["rho_z"]
                results["beta_c"][i] = quantities["beta_c"]
                results["r_c"][i] = quantities["r_c"]

            # Diagnostic output for some steps
            if i % progress_interval == 0:
                print(f"  β_h = {beta_h:.6f}, z_h/h = {quantities['z_h']/self.h:.3f}")
                print(
                    f"  L̃ = {quantities['l_tilde']:.2e}, θ_h = {quantities['theta_h']:.4f} rad"
                )

            # Update guess for next iteration
            beta_h_prev = beta_h

        evolution_results = EvolutionResults(results)
        evolution_results.summary()

        return evolution_results

    def evolve_to_breakout(
        self, max_time: float = 1e6, breakout_criterion: float = 2.0, **kwargs
    ) -> EvolutionResults:
        """
        Evolve until jet breaks out of disk.

        Parameters
        ----------
        max_time : float
            Maximum evolution time [s]
        breakout_criterion : float
            z_h/h ratio for breakout
        **kwargs
            Additional arguments passed to evolve()

        Returns
        -------
        EvolutionResults
            Evolution results up to breakout
        """
        # Start with coarse time sampling
        times_coarse = np.logspace(1, np.log10(max_time), 100)

        # Evolve with coarse sampling
        results_coarse = self.evolve(times_coarse, **kwargs)

        # Find approximate breakout time
        breakout_indices = np.where(results_coarse.zh_over_h >= breakout_criterion)[0]

        if len(breakout_indices) == 0:
            print(f"Warning: Breakout not reached within {max_time:.1e} s")
            return results_coarse

        breakout_index = breakout_indices[0]
        t_breakout = times_coarse[breakout_index]

        # Refine around breakout
        t_start = times_coarse[max(0, breakout_index - 1)]
        times_refined = np.logspace(1, np.log10(t_breakout * 1.2), 1000)

        print(f"\nBreakout detected around t = {t_breakout:.2e} s")
        print(f"Refining evolution up to {t_breakout*1.2:.2e} s...")

        return self.evolve(times_refined, **kwargs)

    def evolve_with_profile_logic(
        self,
        times: np.ndarray,
        density_profile: str = "isothermal",
        initial_guess: float = 0.1,
        progress_interval: int = 100,
        store_diagnostics: bool = False,
        use_optical_depth: bool = True,
        kappa_opacity: float = 0.34,
    ) -> EvolutionResults:
        """
        Evolve jet system with density profile-aware stopping criteria and optical depth.

        Parameters
        ----------
        times : np.ndarray
            Time array [s]
        density_profile : str
            Density profile type: "uniform", "isothermal", or "polytropic"
        initial_guess : float
            Initial guess for beta_h
        progress_interval : int
            Print progress every N steps
        store_diagnostics : bool
            Whether to store additional diagnostic quantities
        use_optical_depth : bool
            Whether to include optical depth calculations and breakout
        kappa_opacity : float
            Opacity for optical depth calculation [cm²/g]

        Returns
        -------
        EvolutionResults
            Evolution results with profile-appropriate stopping
        """
        print(f"\n=== Evolving Jet System (Profile: {density_profile}) ===")
        print(f"Time steps: {len(times)}")
        print(f"Time range: {times[0]:.1e} - {times[-1]:.1e} s")

        # Print disk edge information
        if density_profile == "uniform":
            print(f"Disk edge: z_edge = H_d = 1.0 h")
        elif density_profile == "polytropic":
            print(f"Disk edge: z_edge = √6 H_d = {np.sqrt(6):.2f} h")
        elif density_profile == "isothermal":
            print(f"Disk edge: z_edge = ∞")

        if use_optical_depth:
            print(f"Optical depth breakout: enabled (κ = {kappa_opacity} cm²/g)")

        # Define stopping criteria based on density profile
        if density_profile == "uniform":
            stop_criterion = (
                1.0  # Stop at z_h/h = 1 (no physics beyond uniform profile)
            )
            criterion_name = "uniform profile boundary"
        elif density_profile in ["isothermal", "polytropic"]:
            stop_criterion = None  # No spatial limit - evolve for full time range
            criterion_name = "time-based evolution"
        else:
            raise ValueError(f"Unknown density profile: {density_profile}")

        if stop_criterion is not None:
            print(
                f"Spatial stopping criterion: z_h/h >= {stop_criterion} ({criterion_name})"
            )
        else:
            print(f"Evolution mode: {criterion_name} (no spatial limits)")

        # Initialize storage arrays
        results = {
            "times": [],
            "beta_h": [],
            "z_h": [],
            "zh_over_h": [],
            "e_c": [],
            "theta_h": [],
            "theta_c": [],
            "gamma_h": [],
        }

        # Add optical depth tracking if enabled
        if use_optical_depth:
            results.update(
                {
                    "tau_s": [],
                }
            )

        if store_diagnostics:
            results.update(
                {
                    "l_tilde": [],
                    "rho_z": [],
                    "beta_c": [],
                    "r_c": [],
                }
            )

        # Evolution loop with profile-aware stopping
        beta_h_prev = initial_guess
        stopped_early = False

        for i, t in enumerate(times):
            if i % progress_interval == 0:
                print(f"Processing step {i+1}/{len(times)} (t = {t:.2e} s)")

            # Solve for beta_h using specified density profile
            beta_h = self.solver.solve_single(t, self.rho_0, self.h, beta_h_prev)

            # Validate and store
            if not 0 < beta_h < 1:
                print(f"Warning: beta_h = {beta_h:.6f} out of bounds at t = {t:.2e} s")
                beta_h = max(0.001, min(0.999, beta_h))

            # Calculate all derived quantities with specified profile
            quantities = self._calculate_quantities_with_profile(
                beta_h, t, density_profile
            )

            # Calculate optical depth if enabled
            tau_s = None
            if use_optical_depth:
                tau_s, _ = optical_depth_to_surface(
                    self.rho_0,
                    self.h,
                    quantities["z_h"],
                    density_profile,
                    kappa_opacity,
                )
                # Note: Proper breakout detection (τ(z_h) = 1/β_h) is done separately
                # using find_shock_breakout_time() function

            # Check stopping criterion (only for uniform profile)
            zh_over_h = quantities["z_h"] / self.h
            if stop_criterion is not None and zh_over_h >= stop_criterion:
                print(
                    f"\nStopping criterion reached: z_h/h = {zh_over_h:.3f} >= {stop_criterion}"
                )
                print(f"Evolution stopped at t = {t:.2e} s (step {i+1}/{len(times)})")
                print(
                    f"Reason: {criterion_name} - no meaningful physics beyond this point"
                )
                stopped_early = True

            # Store results
            results["times"].append(t)
            results["beta_h"].append(beta_h)
            results["z_h"].append(quantities["z_h"])
            results["zh_over_h"].append(zh_over_h)
            results["e_c"].append(quantities["e_c"])
            results["theta_h"].append(quantities["theta_h"])
            results["theta_c"].append(quantities["theta_c"])
            results["gamma_h"].append(quantities["gamma_h"])

            # Store optical depth results if enabled
            if use_optical_depth:
                results["tau_s"].append(tau_s if tau_s is not None else 0.0)

            # Store diagnostics if requested
            if store_diagnostics:
                results["l_tilde"].append(quantities["l_tilde"])
                results["rho_z"].append(quantities["rho_z"])
                results["beta_c"].append(quantities["beta_c"])
                results["r_c"].append(quantities["r_c"])

            # Diagnostic output for some steps
            if i % progress_interval == 0:
                print(f"  β_h = {beta_h:.6f}, z_h/h = {zh_over_h:.3f}")
                if use_optical_depth:
                    print(f"  τ_s = {tau_s:.3f}")
                print(
                    f"  L̃ = {quantities['l_tilde']:.2e}, θ_h = {quantities['theta_h']:.4f} rad"
                )
                if density_profile == "uniform" and zh_over_h > 0.8:
                    print(f"  → Approaching uniform profile boundary (z_h/h = 1)")
                elif density_profile in ["isothermal", "polytropic"] and zh_over_h > 1:
                    print(f"  → Jet beyond disk scale height (z_h/h > 1)")

            # Update guess for next iteration
            beta_h_prev = beta_h

            # Stop if criterion reached (only for uniform profile)
            if stopped_early:
                break

        # Convert lists to arrays
        for key in results:
            results[key] = np.array(results[key])

        evolution_results = EvolutionResults(results)

        # Enhanced summary with profile info
        print(f"\n=== Evolution Results Summary ({density_profile} profile) ===")
        print(f"Density profile: {density_profile}")
        if stop_criterion is not None:
            print(f"Stopping criterion: z_h/h >= {stop_criterion}")
        else:
            print(f"Evolution mode: Time-based (no spatial limits)")
        print(f"Final z_h/h: {results['zh_over_h'][-1]:.3f}")

        # Optical depth summary
        if use_optical_depth:
            final_tau = results["tau_s"][-1]
            print(f"Final optical depth: τ_s = {final_tau:.3f}")
            print(
                f"Note: Use find_shock_breakout_time() for proper breakout detection (τ(z_h) = 1/β_h)"
            )

        if stopped_early:
            print(f"Evolution stopped early due to profile constraints")
        else:
            print(f"Evolution completed for full time range")
        evolution_results.summary()

        return evolution_results

    def evolve_with_numerical_integration(
        self,
        times: np.ndarray,
        density_profile: str = "isothermal",
        initial_guess: float = 0.1,
        progress_interval: int = 100,
        store_diagnostics: bool = False,
        use_optical_depth: bool = True,
        kappa_opacity: float = 0.34,
        integration_method: str = "trapz",
    ) -> EvolutionResults:
        """
        Evolve jet system using proper numerical integration of differential equations.

        This method implements the actual differential equations (4-6) from Chen & Dai 2025:
        - dz_h/dt = β_h * c (equation 4)
        - dE_c/dt = η_h * L_j * (1 - β_h) (equation 5)
        - dr_c/dt = β_c * c (equation 6)

        Integration is performed numerically between time steps, providing more
        accurate results than assuming zero integration constants.

        Parameters
        ----------
        times : np.ndarray
            Time array [s]
        density_profile : str
            Density profile type: "uniform", "isothermal", or "polytropic"
        initial_guess : float
            Initial guess for beta_h
        progress_interval : int
            Print progress every N steps
        store_diagnostics : bool
            Whether to store additional diagnostic quantities
        use_optical_depth : bool
            Whether to include optical depth calculations and breakout
        kappa_opacity : float
            Opacity for optical depth calculation [cm²/g]
        integration_method : str
            Numerical integration method: "trapz", "simpson", or "quad"

        Returns
        -------
        EvolutionResults
            Evolution results with numerical integration
        """
        print(f"\n=== Evolving Jet System (Numerical Integration) ===")
        print(f"Profile: {density_profile}")
        print(f"Integration method: {integration_method}")
        print(f"Time steps: {len(times)}")
        print(f"Time range: {times[0]:.1e} - {times[-1]:.1e} s")

        # Print disk edge information
        if density_profile == "uniform":
            print(f"Disk edge: z_edge = H_d = 1.0 h")
        elif density_profile == "polytropic":
            print(f"Disk edge: z_edge = √6 H_d = {np.sqrt(6):.2f} h")
        elif density_profile == "isothermal":
            print(f"Disk edge: z_edge = ∞")

        if use_optical_depth:
            print(f"Optical depth breakout: enabled (κ = {kappa_opacity} cm²/g)")

        # Initialize storage arrays
        n_times = len(times)
        results = {
            "times": times,
            "beta_h": np.zeros(n_times),
            "z_h": np.zeros(n_times),
            "e_c": np.zeros(n_times),
            "r_c": np.zeros(n_times),
            "zh_over_h": np.zeros(n_times),
            "theta_h": np.zeros(n_times),
            "theta_c": np.zeros(n_times),
            "gamma_h": np.zeros(n_times),
            "beta_c": np.zeros(n_times),
        }

        # Add optical depth tracking if enabled
        if use_optical_depth:
            results.update(
                {
                    "tau_s": np.zeros(n_times),
                }
            )

        if store_diagnostics:
            results.update(
                {
                    "l_tilde": np.zeros(n_times),
                    "rho_z": np.zeros(n_times),
                    "eta_h": np.zeros(n_times),
                    "sigma_h": np.zeros(n_times),
                }
            )

        # ========================================================================
        # FIRST TIME STEP: Use algebraic forms (integration constants = 0)
        # ========================================================================
        t0 = times[0]
        print(f"\nFirst time step (t = {t0:.2e} s): Using algebraic forms")

        # Solve for beta_h using the self-consistent equation
        beta_h_0 = self.solver.solve_single(t0, self.rho_0, self.h, initial_guess)
        results["beta_h"][0] = beta_h_0

        # Calculate initial state using algebraic forms (integration constants = 0)
        results["z_h"][0] = beta_h_0 * C_CGS * t0  # z_h = β_h * c * t
        results["gamma_h"][0] = 1 / np.sqrt(1 - beta_h_0**2)
        eta_h_0 = np.minimum(2 / (results["gamma_h"][0] * self.params.theta_0), 1.0)
        results["e_c"][0] = (
            eta_h_0 * self.l_j * (1 - beta_h_0) * t0
        )  # E_c = η_h * L_j * (1-β_h) * t

        # Calculate initial density and cocoon parameters
        rho_zh_0 = density_profiles_with_edges(
            self.rho_0, self.h, results["z_h"][0], density_profile
        )
        denominator = (
            3 * np.pi * rho_zh_0 * C_CGS**2 * results["z_h"][0] * (C_CGS * t0) ** 2
        )
        results["beta_c"][0] = (results["e_c"][0] / denominator) ** (1 / 4)
        results["r_c"][0] = results["beta_c"][0] * C_CGS * t0  # r_c = β_c * c * t

        results["zh_over_h"][0] = results["z_h"][0] / self.h

        # Calculate cross-sectional area and opening angle
        sigma_h_0 = (
            np.pi
            * self.params.theta_0**2
            * np.minimum(
                results["z_h"][0] ** 2,
                3
                * self.l_j
                * results["z_h"][0]
                * results["r_c"][0] ** 2
                / (4 * C_CGS * results["e_c"][0]),
            )
        )
        results["theta_h"][0] = np.sqrt(sigma_h_0 / (np.pi * results["z_h"][0] ** 2))

        # Cocoon opening angle
        results["theta_c"][0] = results["r_c"][0] / results["z_h"][0]

        # Store diagnostics for first step
        if store_diagnostics:
            results["eta_h"][0] = eta_h_0
            results["rho_z"][0] = rho_zh_0
            results["sigma_h"][0] = sigma_h_0
            results["l_tilde"][0] = self.l_j / (sigma_h_0 * rho_zh_0 * C_CGS**3)

        # Calculate optical depth for first step
        if use_optical_depth:
            tau_s_0, _ = optical_depth_to_surface(
                self.rho_0, self.h, results["z_h"][0], density_profile, kappa_opacity
            )
            results["tau_s"][0] = tau_s_0

        print(f"  Initial: β_h = {beta_h_0:.6f}, z_h/h = {results['zh_over_h'][0]:.3f}")

        # ========================================================================
        # NUMERICAL INTEGRATION FOR SUBSEQUENT TIME STEPS
        # ========================================================================
        print(
            f"\nSubsequent steps: Using numerical integration of differential equations"
        )

        stopped_early = False

        for i in range(1, n_times):
            if i % progress_interval == 0:
                print(f"Processing step {i+1}/{n_times} (t = {times[i]:.2e} s)")

            t_prev = times[i - 1]
            t_curr = times[i]
            dt = t_curr - t_prev

            # Previous state
            z_h_prev = results["z_h"][i - 1]
            e_c_prev = results["e_c"][i - 1]
            r_c_prev = results["r_c"][i - 1]

            # ====================================================================
            # STEP 1: Solve for β_h at current time using self-consistent equation
            # Use constrained solver to prevent large jumps
            # ====================================================================
            beta_h_curr = self.solver.solve_single_constrained(
                t_curr,
                self.rho_0,
                self.h,
                results["beta_h"][i - 1],
                max_relative_change=0.15,
            )
            results["beta_h"][i] = beta_h_curr
            results["gamma_h"][i] = 1 / np.sqrt(1 - beta_h_curr**2)

            # ====================================================================
            # STEP 2: Numerically integrate differential equations
            # ====================================================================

            # Create time points for integration
            if integration_method == "trapz":
                # Use trapezoidal rule with current and previous values
                # dz_h/dt = β_h * c
                dzh_dt_prev = results["beta_h"][i - 1] * C_CGS
                dzh_dt_curr = beta_h_curr * C_CGS
                results["z_h"][i] = z_h_prev + 0.5 * (dzh_dt_prev + dzh_dt_curr) * dt

            elif integration_method == "simpson":
                # Use Simpson's rule with midpoint
                t_mid = 0.5 * (t_prev + t_curr)
                beta_h_mid = 0.5 * (
                    results["beta_h"][i - 1] + beta_h_curr
                )  # Linear interpolation

                dzh_dt_prev = results["beta_h"][i - 1] * C_CGS
                dzh_dt_mid = beta_h_mid * C_CGS
                dzh_dt_curr = beta_h_curr * C_CGS

                results["z_h"][i] = z_h_prev + (dt / 6) * (
                    dzh_dt_prev + 4 * dzh_dt_mid + dzh_dt_curr
                )

            else:  # integration_method == "quad"
                # Use scipy.integrate.quad for higher accuracy
                def beta_h_interp(t):
                    # Linear interpolation between previous and current β_h
                    return (
                        results["beta_h"][i - 1]
                        + (beta_h_curr - results["beta_h"][i - 1]) * (t - t_prev) / dt
                    )

                def dzh_dt(t):
                    return beta_h_interp(t) * C_CGS

                dz_h_integrated, _ = quad(dzh_dt, t_prev, t_curr)
                results["z_h"][i] = z_h_prev + dz_h_integrated

            # ====================================================================
            # STEP 3: Calculate current density and efficiency
            # ====================================================================
            rho_zh_curr = density_profiles_with_edges(
                self.rho_0, self.h, results["z_h"][i], density_profile
            )
            eta_h_curr = np.minimum(
                2 / (results["gamma_h"][i] * self.params.theta_0), 1.0
            )

            # ====================================================================
            # STEP 4: Integrate cocoon energy equation dE_c/dt = η_h * L_j * (1 - β_h)
            # ====================================================================
            if integration_method == "trapz":
                eta_h_prev = np.minimum(
                    2 / (results["gamma_h"][i - 1] * self.params.theta_0), 1.0
                )
                dec_dt_prev = eta_h_prev * self.l_j * (1 - results["beta_h"][i - 1])
                dec_dt_curr = eta_h_curr * self.l_j * (1 - beta_h_curr)
                results["e_c"][i] = e_c_prev + 0.5 * (dec_dt_prev + dec_dt_curr) * dt

            elif integration_method == "simpson":
                eta_h_prev = np.minimum(
                    2 / (results["gamma_h"][i - 1] * self.params.theta_0), 1.0
                )
                eta_h_mid = 0.5 * (eta_h_prev + eta_h_curr)
                beta_h_mid = 0.5 * (results["beta_h"][i - 1] + beta_h_curr)

                dec_dt_prev = eta_h_prev * self.l_j * (1 - results["beta_h"][i - 1])
                dec_dt_mid = eta_h_mid * self.l_j * (1 - beta_h_mid)
                dec_dt_curr = eta_h_curr * self.l_j * (1 - beta_h_curr)

                results["e_c"][i] = e_c_prev + (dt / 6) * (
                    dec_dt_prev + 4 * dec_dt_mid + dec_dt_curr
                )

            else:  # quad

                def gamma_h_interp(t):
                    beta_h_t = beta_h_interp(t)
                    return 1 / np.sqrt(1 - beta_h_t**2)

                def eta_h_interp(t):
                    gamma_h_t = gamma_h_interp(t)
                    return np.minimum(2 / (gamma_h_t * self.params.theta_0), 1.0)

                def dec_dt(t):
                    return eta_h_interp(t) * self.l_j * (1 - beta_h_interp(t))

                de_c_integrated, _ = quad(dec_dt, t_prev, t_curr)
                results["e_c"][i] = e_c_prev + de_c_integrated

            # ====================================================================
            # STEP 5: Calculate β_c from current energy and position
            # ====================================================================
            # Use mean density along z-direction (from z=0 to z_h) as per Chen & Dai 2025
            mean_rho_zh_curr = self.physics.disk.mean_z_density(
                self.rho_0, self.h, results["z_h"][i], density_profile
            )

            denominator = (
                3
                * np.pi
                * mean_rho_zh_curr
                * C_CGS**2
                * results["z_h"][i]
                * (C_CGS * t_curr) ** 2
            )
            results["beta_c"][i] = (results["e_c"][i] / denominator) ** (1 / 4)

            # ====================================================================
            # STEP 6: Integrate cocoon radius equation dr_c/dt = β_c * c
            # ====================================================================
            if integration_method == "trapz":
                drc_dt_prev = results["beta_c"][i - 1] * C_CGS
                drc_dt_curr = results["beta_c"][i] * C_CGS
                results["r_c"][i] = r_c_prev + 0.5 * (drc_dt_prev + drc_dt_curr) * dt

            elif integration_method == "simpson":
                beta_c_mid = 0.5 * (results["beta_c"][i - 1] + results["beta_c"][i])
                drc_dt_prev = results["beta_c"][i - 1] * C_CGS
                drc_dt_mid = beta_c_mid * C_CGS
                drc_dt_curr = results["beta_c"][i] * C_CGS

                results["r_c"][i] = r_c_prev + (dt / 6) * (
                    drc_dt_prev + 4 * drc_dt_mid + drc_dt_curr
                )

            else:  # quad - β_c depends on current state, so we use simpler approach
                results["r_c"][i] = r_c_prev + results["beta_c"][i] * C_CGS * dt

            # ====================================================================
            # STEP 7: Calculate derived quantities
            # ====================================================================
            results["zh_over_h"][i] = results["z_h"][i] / self.h

            # Cross-sectional area
            sigma_h_curr = (
                np.pi
                * self.params.theta_0**2
                * np.minimum(
                    results["z_h"][i] ** 2,
                    3
                    * self.l_j
                    * results["z_h"][i]
                    * results["r_c"][i] ** 2
                    / (4 * C_CGS * results["e_c"][i]),
                )
            )
            results["theta_h"][i] = np.sqrt(
                sigma_h_curr / (np.pi * results["z_h"][i] ** 2)
            )

            # Cocoon opening angle
            results["theta_c"][i] = results["r_c"][i] / results["z_h"][i]

            # Store diagnostics
            if store_diagnostics:
                results["eta_h"][i] = eta_h_curr
                results["rho_z"][i] = rho_zh_curr
                results["sigma_h"][i] = sigma_h_curr
                results["l_tilde"][i] = self.l_j / (
                    sigma_h_curr * rho_zh_curr * C_CGS**3
                )

            # Calculate optical depth
            if use_optical_depth:
                tau_s_curr, _ = optical_depth_to_surface(
                    self.rho_0,
                    self.h,
                    results["z_h"][i],
                    density_profile,
                    kappa_opacity,
                )
                results["tau_s"][i] = tau_s_curr

            # Check stopping criterion (for uniform profile)
            if density_profile == "uniform" and results["zh_over_h"][i] >= 1.0:
                print(
                    f"\nStopping criterion reached: z_h/h = {results['zh_over_h'][i]:.3f} >= 1.0"
                )
                print(f"Evolution stopped at t = {t_curr:.2e} s (step {i+1}/{n_times})")
                stopped_early = True

                # Fill remaining arrays with final values
                for j in range(i + 1, n_times):
                    for key in results:
                        if key != "times":
                            results[key][j] = results[key][i]
                break

            # Diagnostic output
            if i % progress_interval == 0:
                print(
                    f"  β_h = {beta_h_curr:.6f}, z_h/h = {results['zh_over_h'][i]:.3f}"
                )
                if use_optical_depth:
                    print(f"  τ_s = {results['tau_s'][i]:.3f}")

        # Convert to EvolutionResults
        evolution_results = EvolutionResults(results)

        # Enhanced summary
        print(f"\n=== Numerical Integration Results Summary ===")
        print(f"Integration method: {integration_method}")
        print(f"Density profile: {density_profile}")
        print(f"Final z_h/h: {results['zh_over_h'][-1]:.3f}")

        if use_optical_depth:
            final_tau = results["tau_s"][-1]
            print(f"Final optical depth: τ_s = {final_tau:.3f}")
            print(
                f"Note: Use find_shock_breakout_time() for proper breakout detection (τ(z_h) = 1/β_h)"
            )

        if stopped_early:
            print(f"Evolution stopped early due to profile constraints")
        else:
            print(f"Evolution completed for full time range")

        evolution_results.summary()
        return evolution_results

    def _calculate_quantities_with_profile(
        self, beta_h: float, t: float, density_profile: str
    ) -> dict:
        """
        Calculate physical quantities using specified density profile.

        Parameters
        ----------
        beta_h : float
            Jet head velocity parameter
        t : float
            Time [s]
        density_profile : str
            Density profile type

        Returns
        -------
        dict
            Dictionary containing calculated quantities
        """
        # Jet quantities
        z_h = self.physics.jet.z_h(beta_h, t)
        gamma_h = self.physics.jet.gamma_h(beta_h)

        # Disk density at jet head using specified profile
        rho_z = self.physics.disk.density_profile(
            self.rho_0, self.h, z_h, profile_type=density_profile
        )

        # Mean density from z=0 to z=z_h for cocoon calculations
        mean_rho_z = self.physics.disk.mean_z_density(
            self.rho_0, self.h, z_h, density_profile
        )

        # Efficiency and energy
        eta_h = self.physics.jet.eta_h(gamma_h, self.params.theta_0)
        e_c = self.physics.cocoon.energy(eta_h, self.l_j, beta_h, t)

        # Cocoon properties (using paper's formula)
        beta_c, r_c = self.physics.cocoon.beta_c_paper_formula(e_c, mean_rho_z, z_h, t)

        # Cross-section and angles
        sigma_h = self.physics.cocoon.cross_section(
            self.l_j, z_h, r_c, e_c, self.params.theta_0
        )
        theta_h = self.physics.jet.theta_h(sigma_h, z_h, self.params.theta_0)
        theta_c = self.physics.cocoon.theta_c(r_c, z_h)

        # Dimensionless parameter
        l_tilde = self.physics.jet.l_tilde(self.l_j, sigma_h, rho_z)

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

    def get_model_info(self) -> Dict:
        """
        Get summary of model configuration.

        Returns
        -------
        dict
            Model configuration information
        """
        return {
            "l_j": self.l_j,
            "params": self.params,
            "disk_filename": self.disk_filename,
            "disk_radius_rg": self.disk_radius_rg,
            "rho_0": self.rho_0,
            "h": self.h,
            "r_g": self.params.r_g,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def run_chen_dai_evolution(
    l_j: float, times: np.ndarray, mass_smbh: float = 1e8, **kwargs
) -> EvolutionResults:
    """
    Run Chen & Dai evolution with minimal setup.

    Parameters
    ----------
    l_j : float
        Jet luminosity [erg/s]
    times : np.ndarray
        Time array [s]
    mass_smbh : float
        SMBH mass in solar masses
    **kwargs
        Additional parameters

    Returns
    -------
    EvolutionResults
        Evolution results
    """
    params = ModelParameters(mass_smbh=mass_smbh)
    model = ChenDaiModel(l_j, params, **kwargs)
    return model.evolve(times)


def create_time_array(
    t_min: float = 10.0, t_max: float = 1e5, n_points: int = 1000
) -> np.ndarray:
    """
    Create logarithmically spaced time array.

    Parameters
    ----------
    t_min : float
        Minimum time [s]
    t_max : float
        Maximum time [s]
    n_points : int
        Number of points

    Returns
    -------
    np.ndarray
        Time array
    """
    return np.logspace(np.log10(t_min), np.log10(t_max), n_points)


def find_optical_breakout_time(
    results: EvolutionResults, tau_threshold: float = 1.0
) -> Optional[float]:
    """
    Find the time when optical depth breakout occurs (τ_s < threshold).

    Parameters
    ----------
    results : EvolutionResults
        Evolution results containing optical depth data
    tau_threshold : float
        Optical depth threshold for breakout

    Returns
    -------
    float or None
        Breakout time [s], or None if no breakout found
    """
    if not hasattr(results, "tau_s"):
        print("Warning: No optical depth data in results")
        return None

    breakout_indices = np.where(results.tau_s < tau_threshold)[0]

    if len(breakout_indices) == 0:
        return None

    return results.times[breakout_indices[0]]


def analyze_optical_depth_evolution(results: EvolutionResults) -> dict:
    """
    Analyze optical depth evolution and breakout characteristics.

    Parameters
    ----------
    results : EvolutionResults
        Evolution results containing optical depth data

    Returns
    -------
    dict
        Analysis results including breakout time, final τ_s, etc.
    """
    analysis = {}

    if not hasattr(results, "tau_s"):
        print("Warning: No optical depth data in results")
        return analysis

    # Find breakout time
    breakout_time = find_optical_breakout_time(results)
    analysis["breakout_time"] = breakout_time

    if breakout_time is not None:
        # Find breakout conditions
        breakout_idx = np.where(results.times >= breakout_time)[0][0]
        analysis["breakout_zh_h"] = results.zh_over_h[breakout_idx]
        analysis["breakout_beta_h"] = results.beta_h[breakout_idx]
        analysis["breakout_tau"] = results.tau_s[breakout_idx]

    # Final conditions
    analysis["final_tau"] = results.tau_s[-1]
    analysis["final_zh_h"] = results.zh_over_h[-1]
    analysis["min_tau"] = np.min(results.tau_s)
    analysis["max_tau"] = np.max(results.tau_s)

    return analysis


def compare_profile_breakouts(results_dict: dict) -> None:
    """
    Compare optical depth breakout characteristics across density profiles.

    Parameters
    ----------
    results_dict : dict
        Dictionary of {profile_name: EvolutionResults} for comparison
    """
    print("\n" + "=" * 70)
    print("OPTICAL DEPTH BREAKOUT COMPARISON")
    print("=" * 70)

    for profile, results in results_dict.items():
        print(f"\n{profile.upper()} PROFILE:")
        analysis = analyze_optical_depth_evolution(results)

        if analysis.get("breakout_time") is not None:
            print(f"  Breakout time: {analysis['breakout_time']:.2e} s")
            print(f"  Breakout z_h/h: {analysis['breakout_zh_h']:.3f}")
            print(f"  Breakout β_h: {analysis['breakout_beta_h']:.6f}")
            print(f"  Breakout τ_s: {analysis['breakout_tau']:.3f}")
        else:
            print(f"  No breakout in time range")

        print(f"  Final τ_s: {analysis['final_tau']:.3f}")
        print(f"  Final z_h/h: {analysis['final_zh_h']:.3f}")
        print(f"  τ_s range: {analysis['min_tau']:.3f} - {analysis['max_tau']:.3f}")


def truncate_result_at_breakout(result, breakout_index):
    """
    Truncate evolution result at the breakout time.

    This function takes a single evolution result and truncates all data
    arrays just before the breakout index, excluding the first unphysical
    time step where the shock breakout condition is satisfied.

    Parameters
    ----------
    result : EvolutionResults
        Single evolution result to truncate
    breakout_index : int or None
        Index at which breakout occurs. Data will be truncated to exclude
        this index and all subsequent indices. If None, returns original result.

    Returns
    -------
    EvolutionResults
        Truncated evolution result (includes indices 0 to breakout_index-1)
    """
    if breakout_index is None:
        # No breakout, return original result
        return result

    # Create truncated data dictionary
    truncated_data = {}

    # Handle both attribute-style and dictionary-style access
    if hasattr(result, "data") and isinstance(result.data, dict):
        # Result has a data dictionary
        for key, data in result.data.items():
            if isinstance(data, (list, np.ndarray)) and len(data) > breakout_index:
                truncated_data[key] = data[:breakout_index]  # Exclude breakout index
            else:
                truncated_data[key] = data
    else:
        # Result stores data as attributes
        for attr_name in dir(result):
            if not attr_name.startswith("_"):  # Skip private attributes
                data = getattr(result, attr_name)
                if isinstance(data, (list, np.ndarray)) and len(data) > breakout_index:
                    truncated_data[attr_name] = data[
                        :breakout_index
                    ]  # Exclude breakout index
                elif not callable(data):  # Skip methods
                    truncated_data[attr_name] = data

    # Return new EvolutionResults object
    return EvolutionResults(truncated_data)


def truncate_results_at_breakout(results_dict, breakout_indices_dict):
    """
    Truncate multiple evolution results at their respective breakout times.

    This function is designed for comparing multiple evolution methods.
    For single evolution results, use truncate_result_at_breakout instead.

    Parameters
    ----------
    results_dict : dict
        Dictionary of evolution results with method names as keys
    breakout_indices_dict : dict
        Dictionary of breakout indices for each method

    Returns
    -------
    dict
        Dictionary of truncated evolution results
    """
    truncated_results = {}

    for method_key, result in results_dict.items():
        breakout_idx = breakout_indices_dict[method_key]
        truncated_results[method_key] = truncate_result_at_breakout(
            result, breakout_idx
        )

    return truncated_results
