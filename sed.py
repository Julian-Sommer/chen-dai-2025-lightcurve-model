"""
Spectral Energy Distribution (SED) calculations for the Chen & Dai 2025 model.

The module provides functions to:
1. Calculate blackbody spectral energy density νB_ν(T)
2. Calculate total blackbody spectral energy density B(T)
3. Convert bolometric luminosity to spectral luminosity at specific frequencies/wavelengths
4. Calculate magnitudes in standard photometric bands
"""

import numpy as np
from typing import Union, Dict, List
import astropy.constants as const
import astropy.units as u

# Physical constants in CGS units
C_CGS = const.c.cgs.value  # Speed of light [cm/s]
H_CGS = const.h.cgs.value  # Planck constant [erg⋅s]
KB_CGS = const.k_B.cgs.value  # Boltzmann constant [erg/K]
SIGMA_SB_CGS = const.sigma_sb.cgs.value  # Stefan-Boltzmann constant [erg/cm²/s/K⁴]

# Unit conversions
ANGSTROM_TO_CM = 1e-8  # Convert Angstroms to cm
JY_TO_CGS = 1e-23  # Convert Jy to erg/s/cm²/Hz


class BlackbodySED:
    """
    Class for calculating spectral energy distributions using blackbody radiation.

    This implementation follows the approach described by Chen & Dai 2025 authors:
    assuming blackbody radiation with bolometric luminosity equal to the disk
    cocoon luminosities, then using the ratio νB_ν(T)/B(T) to determine
    luminosity at specific frequencies.
    """

    @staticmethod
    def planck_function_nu(
        nu: Union[float, np.ndarray], T: float
    ) -> Union[float, np.ndarray]:
        """
        Calculate Planck function B_ν(T) in frequency space.

        Parameters
        ----------
        nu : float or array
            Frequency [Hz]
        T : float
            Temperature [K]

        Returns
        -------
        float or array
            Planck function B_ν(T) [erg/s/cm²/Hz/sr]
        """
        if T <= 0:
            return np.zeros_like(nu) if hasattr(nu, "__len__") else 0.0

        x = H_CGS * nu / (KB_CGS * T)

        # Avoid overflow for large x values
        x = np.clip(x, 0, 700)  # exp(700) is close to machine limit

        # Calculate Planck function
        return (2 * H_CGS * nu**3 / C_CGS**2) / (np.exp(x) - 1)

    @staticmethod
    def planck_function_lambda(
        wavelength: Union[float, np.ndarray], T: float
    ) -> Union[float, np.ndarray]:
        """
        Calculate Planck function B_λ(T) in wavelength space.

        Parameters
        ----------
        wavelength : float or array
            Wavelength [cm]
        T : float
            Temperature [K]

        Returns
        -------
        float or array
            Planck function B_λ(T) [erg/s/cm²/cm/sr]
        """
        if T <= 0:
            return np.zeros_like(wavelength) if hasattr(wavelength, "__len__") else 0.0

        # Avoid division by zero for very small wavelengths or temperatures
        with np.errstate(divide="ignore", invalid="ignore"):
            x = H_CGS * C_CGS / (wavelength * KB_CGS * T)

        # Avoid overflow for large x values
        x = np.clip(x, 0, 700)

        # Calculate Planck function in wavelength space
        return (2 * H_CGS * C_CGS**2 / wavelength**5) / (np.exp(x) - 1)

    @staticmethod
    def spectral_energy_density_nu(
        nu: Union[float, np.ndarray], T: float
    ) -> Union[float, np.ndarray]:
        """
        Calculate spectral energy density νB_ν(T).

        Parameters
        ----------
        nu : float or array
            Frequency [Hz]
        T : float
            Temperature [K]

        Returns
        -------
        float or array
            Spectral energy density νB_ν(T) [erg/s/cm²/sr]
        """
        return nu * BlackbodySED.planck_function_nu(nu, T)

    @staticmethod
    def spectral_energy_density_lambda(
        wavelength: Union[float, np.ndarray], T: float
    ) -> Union[float, np.ndarray]:
        """
        Calculate spectral energy density λB_λ(T).

        Parameters
        ----------
        wavelength : float or array
            Wavelength [cm]
        T : float
            Temperature [K]

        Returns
        -------
        float or array
            Spectral energy density λB_λ(T) [erg/s/cm²/sr]
        """
        return wavelength * BlackbodySED.planck_function_lambda(wavelength, T)

    @staticmethod
    def total_spectral_energy_density(T: float) -> float:
        """
        Calculate total blackbody spectral energy density B(T) = σT⁴/π.

        This is the integral of B_ν over all frequencies (or B_λ over all wavelengths).

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        float
            Total spectral energy density B(T) [erg/s/cm²/sr]
        """
        if T <= 0:
            return 0.0
        return SIGMA_SB_CGS * T**4 / np.pi

    @classmethod
    def frequency_to_wavelength(
        cls, nu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert frequency to wavelength."""
        return C_CGS / nu

    @classmethod
    def wavelength_to_frequency(
        cls, wavelength: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert wavelength to frequency."""
        return C_CGS / wavelength

    @classmethod
    def spectral_luminosity_at_frequency(
        cls, L_bol: float, T: float, nu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate spectral luminosity at a specific frequency.

        Correct relation for a blackbody emitter of bolometric luminosity L_bol:
        L_ν = L_bol × [B_ν(T) / B(T)], where B(T) = σT⁴/π.

        Parameters
        ----------
        L_bol : float
            Bolometric luminosity [erg/s]
        T : float
            Temperature [K]
        nu : float or array
            Frequency [Hz]

        Returns
        -------
        float or array
            Spectral luminosity L_ν [erg/s/Hz]
        """
        if T <= 0 or L_bol <= 0:
            return np.zeros_like(nu) if hasattr(nu, "__len__") else 0.0

        # Planck function B_ν(T) [erg/s/cm²/Hz/sr]
        B_nu = cls.planck_function_nu(nu, T)

        # Total spectral energy density B(T) [erg/s/cm²/sr]
        B_total = cls.total_spectral_energy_density(T)

        if B_total == 0:
            return np.zeros_like(nu) if hasattr(nu, "__len__") else 0.0

        # L_ν scales with B_ν/B(T)
        return L_bol * B_nu / B_total

    @classmethod
    def spectral_luminosity_at_wavelength(
        cls, L_bol: float, T: float, wavelength: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate spectral luminosity at a specific wavelength.

        Correct relation for a blackbody emitter of bolometric luminosity L_bol:
        L_λ = L_bol × [B_λ(T) / B(T)], where B(T) = σT⁴/π.

        Parameters
        ----------
        L_bol : float
            Bolometric luminosity [erg/s]
        T : float
            Temperature [K]
        wavelength : float or array
            Wavelength [cm]

        Returns
        -------
        float or array
            Spectral luminosity L_λ [erg/s/cm]
        """
        if T <= 0 or L_bol <= 0:
            return np.zeros_like(wavelength) if hasattr(wavelength, "__len__") else 0.0

        # Planck function B_λ(T) [erg/s/cm²/cm/sr]
        B_lambda = cls.planck_function_lambda(wavelength, T)

        # Total spectral energy density B(T) [erg/s/cm²/sr]
        B_total = cls.total_spectral_energy_density(T)

        if B_total == 0:
            return np.zeros_like(wavelength) if hasattr(wavelength, "__len__") else 0.0

        # L_λ scales with B_λ/B(T)
        return L_bol * B_lambda / B_total

    @classmethod
    def calculate_band_luminosity(
        cls, L_bol: float, T: float, wavelength_central: float
    ) -> float:
        """
        Calculate luminosity in a specific photometric band.

        This uses the central wavelength as a representative wavelength
        for the band, which is a common approximation.

        Parameters
        ----------
        L_bol : float
            Bolometric luminosity [erg/s]
        T : float
            Temperature [K]
        wavelength_central : float
            Central wavelength of the band [cm]

        Returns
        -------
        float
            Band luminosity [erg/s/cm] (spectral luminosity at central wavelength)
        """
        return cls.spectral_luminosity_at_wavelength(L_bol, T, wavelength_central)

    @classmethod
    def luminosity_to_magnitude(
        cls, L_lambda: float, wavelength: float, distance: float
    ) -> float:
        """
        Convert spectral luminosity to apparent magnitude.

        Parameters
        ----------
        L_lambda : float
            Spectral luminosity [erg/s/cm]
        wavelength : float
            Wavelength [cm]
        distance : float
            Luminosity distance [cm]

        Returns
        -------
        float
            Apparent magnitude (AB system)
        """
        if L_lambda <= 0 or distance <= 0:
            return 999.0  # Return very faint magnitude for invalid inputs

        # Convert L_λ to flux density F_λ [erg/s/cm²/cm]
        F_lambda = L_lambda / (4 * np.pi * distance**2)

        # Convert to frequency units: F_ν = F_λ × λ²/c [erg/s/cm²/Hz]
        F_nu = F_lambda * wavelength**2 / C_CGS

        # Convert to magnitude (AB system: m = -2.5 log₁₀(F_ν/3631 Jy))
        F_nu_jy = F_nu / JY_TO_CGS  # Convert to Jy

        if F_nu_jy <= 0:
            return 999.0

        magnitude = -2.5 * np.log10(F_nu_jy / 3631.0)

        return magnitude

    @classmethod
    def spectral_luminosity_to_nu_f_nu(
        cls, L_lambda: float, wavelength: float, distance: float
    ) -> float:
        """
        Convert spectral luminosity to νf_ν flux density.

        Parameters
        ----------
        L_lambda : float
            Spectral luminosity [erg/s/cm]
        wavelength : float
            Wavelength [cm]
        distance : float
            Luminosity distance [cm]

        Returns
        -------
        float
            νf_ν flux density [erg/s/cm²]
        """
        if L_lambda <= 0 or distance <= 0:
            return 0.0

        # Convert L_λ to flux density F_λ [erg/s/cm²/cm]
        F_lambda = L_lambda / (4 * np.pi * distance**2)

        # Convert to frequency units: F_ν = F_λ × λ²/c [erg/s/cm²/Hz]
        F_nu = F_lambda * wavelength**2 / C_CGS

        # Convert wavelength to frequency: ν = c/λ [Hz]
        nu = C_CGS / wavelength

        # Calculate νf_ν [erg/s/cm²]
        nu_f_nu = nu * F_nu

        return nu_f_nu


def calculate_multiband_lightcurve(
    times: np.ndarray,
    L_bol_array: np.ndarray,
    T_array: np.ndarray,
    band_wavelengths: Dict[str, float],
    distance: float,
    t_break: float = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate multiband light curves for given bolometric luminosity and temperature evolution.

    Parameters
    ----------
    times : array
        Time array [s]
    L_bol_array : array
        Bolometric luminosity evolution [erg/s]
    T_array : array
        Temperature evolution [K]
    band_wavelengths : dict
        Dictionary mapping band names to central wavelengths [Angstroms]
    distance : float
        Luminosity distance [cm]
    t_break : float, optional
        Shock breakout time [s]. If provided, emission before this time
        will be masked to zero to prevent unphysical pre-breakout emission.

    Returns
    -------
    dict
        Dictionary mapping band names to magnitude arrays
    """
    sed_calc = BlackbodySED()
    magnitudes = {}

    # Apply physical constraint if t_break is provided
    if t_break is not None:
        # Count how many points are before breakout
        pre_breakout_mask = times < t_break
        n_masked = np.sum(pre_breakout_mask)

        if n_masked > 0:
            print(
                f"Warning: Masking {n_masked} time points before shock breakout (t < {t_break:.1f} s)"
            )
            print(f"         This prevents unphysical cocoon emission before breakout.")

            # Create masked arrays
            L_bol_masked = L_bol_array.copy()
            T_masked = T_array.copy()
            L_bol_masked[pre_breakout_mask] = 0.0
            T_masked[pre_breakout_mask] = 0.0
        else:
            L_bol_masked = L_bol_array
            T_masked = T_array
    else:
        L_bol_masked = L_bol_array
        T_masked = T_array

    for band, wavelength_ang in band_wavelengths.items():
        # Convert wavelength from Angstroms to cm
        wavelength_cm = wavelength_ang * ANGSTROM_TO_CM

        # Calculate magnitudes for this band
        band_magnitudes = []
        for i, (t, L_bol, T) in enumerate(zip(times, L_bol_masked, T_masked)):
            # Check if this time point was masked due to being before breakout
            if t_break is not None and t < t_break:
                # Set magnitude to infinity for masked points (no physical emission)
                magnitude = np.inf
            else:
                # Calculate spectral luminosity at band central wavelength
                L_lambda = sed_calc.spectral_luminosity_at_wavelength(
                    L_bol, T, wavelength_cm
                )

                # Convert to magnitude
                magnitude = sed_calc.luminosity_to_magnitude(
                    L_lambda, wavelength_cm, distance
                )
            band_magnitudes.append(magnitude)

        magnitudes[band] = np.array(band_magnitudes)

    return magnitudes


def calculate_multiband_nu_f_nu(
    times: np.ndarray,
    L_bol_array: np.ndarray,
    T_array: np.ndarray,
    band_wavelengths: Dict[str, float],
    distance: float,
    t_break: float = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate multiband νf_ν light curves for given bolometric luminosity and temperature evolution.

    Parameters
    ----------
    times : array
        Time array [s]
    L_bol_array : array
        Bolometric luminosity evolution [erg/s]
    T_array : array
        Temperature evolution [K]
    band_wavelengths : dict
        Dictionary mapping band names to central wavelengths [Angstroms]
    distance : float
        Luminosity distance [cm]
    t_break : float, optional
        Shock breakout time [s]. If provided, emission before this time
        will be masked to zero to prevent unphysical pre-breakout emission.

    Returns
    -------
    dict
        Dictionary mapping band names to νf_ν arrays [erg/s/cm²]
    """
    sed_calc = BlackbodySED()
    nu_f_nu_dict = {}

    # Apply physical constraint if t_break is provided
    if t_break is not None:
        # Count how many points are before breakout
        pre_breakout_mask = times < t_break
        n_masked = np.sum(pre_breakout_mask)

        if n_masked > 0:
            print(
                f"Warning: Masking {n_masked} time points before shock breakout (t < {t_break:.1f} s)"
            )
            print(f"         This prevents unphysical cocoon emission before breakout.")

            # Create masked arrays
            L_bol_masked = L_bol_array.copy()
            T_masked = T_array.copy()
            L_bol_masked[pre_breakout_mask] = 0.0
            T_masked[pre_breakout_mask] = 0.0
        else:
            L_bol_masked = L_bol_array
            T_masked = T_array
    else:
        L_bol_masked = L_bol_array
        T_masked = T_array

    for band, wavelength_ang in band_wavelengths.items():
        # Convert wavelength from Angstroms to cm
        wavelength_cm = wavelength_ang * ANGSTROM_TO_CM

        # Calculate νf_ν for this band
        band_nu_f_nu = []
        for i, (t, L_bol, T) in enumerate(zip(times, L_bol_masked, T_masked)):
            # Check if this time point was masked due to being before breakout
            if t_break is not None and t < t_break:
                # Set νf_ν to zero for masked points (no physical emission)
                nu_f_nu = 0.0
            else:
                # Calculate spectral luminosity at band central wavelength
                L_lambda = sed_calc.spectral_luminosity_at_wavelength(
                    L_bol, T, wavelength_cm
                )

                # Convert to νf_ν flux density
                nu_f_nu = sed_calc.spectral_luminosity_to_nu_f_nu(
                    L_lambda, wavelength_cm, distance
                )
            band_nu_f_nu.append(nu_f_nu)

        nu_f_nu_dict[band] = np.array(band_nu_f_nu)

    return nu_f_nu_dict
