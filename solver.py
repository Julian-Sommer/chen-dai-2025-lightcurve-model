"""
Numerical solver for beta_h determination.

This module implements robust numerical methods to solve for the jet head
velocity parameter beta_h following the Chen & Dai 2025 approach.
"""

import numpy as np
from typing import Optional, Callable
from scipy.optimize import brentq, fsolve

try:
    from .constants import ModelParameters
    from .physics import PhysicsCalculator
except ImportError:
    from constants import ModelParameters
    from physics import PhysicsCalculator


class BetaHSolver:
    """
    Numerical solver for beta_h using the Chen & Dai approach.

    This class implements the equation β_h = β_j / (1 + L̃^(-1/2))
    where all quantities are expressed in terms of beta_h for self-consistent solution.
    """

    def __init__(self, params: ModelParameters, l_j: float):
        """
        Initialize solver.

        Parameters
        ----------
        params : ModelParameters
            Model parameters
        l_j : float
            Jet luminosity [erg/s]
        """
        self.params = params
        self.l_j = l_j
        self.physics = PhysicsCalculator(params, l_j)

        # Solver settings
        self.tolerance = 1e-10
        self.max_iterations = 100
        self.bounds = (1e-6, 0.99)

    def beta_h_equation(
        self, beta_h_guess: float, t: float, rho_0: float, h: float
    ) -> float:
        """
        Equation to solve for beta_h.

        Following equation (7): β_h = β_j / (1 + L̃^(-1/2))
        where L̃ is given by equation (8): L̃ = L_j / (Σ_h ρ(z_h) c³)

        Parameters
        ----------
        beta_h_guess : float
            Current guess for beta_h
        t : float
            Time [s]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]

        Returns
        -------
        float
            Residual (should be zero when beta_h is correct)
        """
        # Calculate all quantities from beta_h_guess
        z_h = self.physics.jet.z_h(beta_h_guess, t)
        gamma_h = self.physics.jet.gamma_h(beta_h_guess)

        # Disk density at jet head
        rho_z = self.physics.disk.density_profile(rho_0, h, z_h)

        # Calculate efficiency and cocoon energy
        eta_h = self.physics.jet.eta_h(gamma_h, self.params.theta_0)
        e_c = self.physics.cocoon.energy(eta_h, self.l_j, beta_h_guess, t)

        # Cocoon properties (using local density as approximation)
        beta_c, r_c = self.physics.cocoon.beta_c_paper_formula(e_c, rho_z, z_h, t)

        # Cross-sectional area
        sigma_h = self.physics.cocoon.cross_section(
            self.l_j, z_h, r_c, e_c, self.params.theta_0
        )

        # Calculate L̃ (dimensionless parameter)
        l_tilde = self.physics.jet.l_tilde(self.l_j, sigma_h, rho_z)

        # Apply equation (7): β_h = β_j / (1 + L̃^(-1/2))
        computed_beta_h = self.params.beta_j / (1 + l_tilde ** (-0.5))

        # Return residual
        return beta_h_guess - computed_beta_h

    def solve_single(
        self, t: float, rho_0: float, h: float, initial_guess: Optional[float] = None
    ) -> float:
        """
        Solve for beta_h at a single time point.

        Parameters
        ----------
        t : float
            Time [s]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        initial_guess : float, optional
            Initial guess for beta_h

        Returns
        -------
        float
            Solved beta_h value
        """
        if initial_guess is None:
            initial_guess = 0.1

        # First try Brent's method (most robust for well-behaved functions)
        try:
            beta_h_solution = brentq(
                self.beta_h_equation,
                self.bounds[0],
                self.bounds[1],
                args=(t, rho_0, h),
                xtol=self.tolerance,
                maxiter=self.max_iterations,
            )

            if self._validate_solution(beta_h_solution):
                return beta_h_solution
            else:
                raise ValueError(f"Solution out of bounds: {beta_h_solution}")

        except (ValueError, RuntimeError):
            # Fall back to fsolve with multiple initial guesses
            return self._solve_with_fsolve(t, rho_0, h, initial_guess)

    def _solve_with_fsolve(
        self, t: float, rho_0: float, h: float, initial_guess: float
    ) -> float:
        """
        Solve using fsolve with multiple initial guesses.

        Parameters
        ----------
        t : float
            Time [s]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        initial_guess : float
            Initial guess for beta_h

        Returns
        -------
        float
            Solved beta_h value
        """
        # Try multiple initial guesses
        guesses = [initial_guess, 0.01, 0.1, 0.5, 0.8]

        for guess in guesses:
            try:
                result = fsolve(
                    self.beta_h_equation,
                    guess,
                    args=(t, rho_0, h),
                    xtol=self.tolerance,
                    full_output=True,
                )

                beta_h_solution = result[0][0]
                converged = result[2] == 1

                if converged and self._validate_solution(beta_h_solution):
                    return beta_h_solution

            except Exception:
                continue

        # If all methods fail, use fallback
        print(f"Warning: Numerical solver failed at t={t:.2e}s. Using fallback value.")
        return self._get_fallback_value(initial_guess)

    def _validate_solution(self, beta_h: float) -> bool:
        """
        Validate that solution is physically reasonable.

        Parameters
        ----------
        beta_h : float
            Solution to validate

        Returns
        -------
        bool
            True if solution is valid
        """
        return 0 < beta_h < 1

    def _get_fallback_value(self, initial_guess: float) -> float:
        """
        Get fallback value when solver fails.

        Parameters
        ----------
        initial_guess : float
            Initial guess

        Returns
        -------
        float
            Fallback value
        """
        return max(0.001, min(0.5, initial_guess * 0.5))

    def set_solver_options(
        self,
        tolerance: float = 1e-10,
        max_iterations: int = 100,
        bounds: tuple = (1e-6, 0.99),
    ) -> None:
        """
        Set solver options.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum number of iterations
        bounds : tuple
            (lower_bound, upper_bound) for beta_h
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.bounds = bounds

    def get_equation_residual(
        self, beta_h: float, t: float, rho_0: float, h: float
    ) -> float:
        """
        Get equation residual for diagnostics.

        Parameters
        ----------
        beta_h : float
            Beta_h value
        t : float
            Time [s]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]

        Returns
        -------
        float
            Equation residual
        """
        return self.beta_h_equation(beta_h, t, rho_0, h)

    def solve_single_constrained(
        self,
        t: float,
        rho_0: float,
        h: float,
        previous_beta_h: float,
        max_relative_change: float = 0.2,
    ) -> float:
        """
        Solve for beta_h with constraints to prevent large jumps.

        This method constrains the search to stay within a reasonable range of the
        previous solution, preventing the solver from jumping to distant solutions
        that might be mathematically valid but physically implausible for smooth evolution.

        Parameters
        ----------
        t : float
            Time [s]
        rho_0 : float
            Central disk density [g/cm³]
        h : float
            Disk scale height [cm]
        previous_beta_h : float
            Previous β_h value for constraint
        max_relative_change : float, optional
            Maximum allowed relative change from previous value (default: 0.2 = 20%)

        Returns
        -------
        float
            Solved beta_h value
        """
        # Define constrained bounds around previous solution
        lower_bound = max(self.bounds[0], previous_beta_h * (1 - max_relative_change))
        upper_bound = min(self.bounds[1], previous_beta_h * (1 + max_relative_change))

        # Ensure bounds are valid
        if lower_bound >= upper_bound:
            # If bounds are too tight, expand them slightly
            lower_bound = max(self.bounds[0], previous_beta_h * 0.8)
            upper_bound = min(self.bounds[1], previous_beta_h * 1.2)

        # Try constrained Brent's method first
        try:
            beta_h_solution = brentq(
                self.beta_h_equation,
                lower_bound,
                upper_bound,
                args=(t, rho_0, h),
                xtol=self.tolerance,
                maxiter=self.max_iterations,
            )

            if self._validate_solution(beta_h_solution):
                return beta_h_solution
            else:
                raise ValueError(
                    f"Constrained solution out of bounds: {beta_h_solution}"
                )

        except (ValueError, RuntimeError):
            # Fall back to fsolve with initial guess close to previous value
            try:
                result = fsolve(
                    self.beta_h_equation,
                    previous_beta_h,  # Start from previous solution
                    args=(t, rho_0, h),
                    xtol=self.tolerance,
                    full_output=True,
                )

                beta_h_solution = result[0][0]
                converged = result[2] == 1

                if converged and self._validate_solution(beta_h_solution):
                    # Check if solution is within reasonable bounds
                    relative_change = (
                        abs(beta_h_solution - previous_beta_h) / previous_beta_h
                    )
                    if (
                        relative_change <= max_relative_change * 2
                    ):  # Allow 2x the constraint for fsolve
                        return beta_h_solution

            except Exception:
                pass  # Final fallback: use unconstrained solver but warn about large change
            unconstrained_solution = self.solve_single(t, rho_0, h, previous_beta_h)
            relative_change = (
                abs(unconstrained_solution - previous_beta_h) / previous_beta_h
            )

            if relative_change > 0.5:  # 50% change
                print(
                    f"Warning: Large β_h change detected at t={t:.2e}s: "
                    f"{previous_beta_h:.6f} → {unconstrained_solution:.6f} ({relative_change:.1%})"
                )
                print(f"  This may indicate the jet head has reached the disk edge.")
                # For extreme changes, use a more conservative estimate
                if relative_change > 2.0:  # 200% change - likely numerical instability
                    conservative_solution = previous_beta_h * 1.2  # 20% increase max
                    print(f"  Using conservative estimate: {conservative_solution:.6f}")
                    return min(conservative_solution, 0.99)

            return unconstrained_solution


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_solver(params: ModelParameters, l_j: float) -> BetaHSolver:
    """
    Create BetaHSolver with given parameters.

    Parameters
    ----------
    params : ModelParameters
        Model parameters
    l_j : float
        Jet luminosity [erg/s]

    Returns
    -------
    BetaHSolver
        Configured solver
    """
    return BetaHSolver(params, l_j)


def solve_beta_h_simple(
    beta_j: float, l_j: float, t: float, rho_0: float, h: float, theta_0: float
) -> float:
    """
    Simple function to solve for beta_h with minimal setup.

    Parameters
    ----------
    beta_j : float
        Initial jet velocity parameter
    l_j : float
        Jet luminosity [erg/s]
    t : float
        Time [s]
    rho_0 : float
        Central disk density [g/cm³]
    h : float
        Disk scale height [cm]
    theta_0 : float
        Initial jet opening angle [rad]

    Returns
    -------
    float
        Solved beta_h value
    """
    params = ModelParameters(theta_0=theta_0)
    params.beta_j = beta_j  # Override with custom value

    solver = BetaHSolver(params, l_j)
    return solver.solve_single(t, rho_0, h)
