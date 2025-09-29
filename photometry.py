"""Photometry utilities for multiband lightcurves.

Functions here convert bolometric or spectral luminosities into flux densities
and AB magnitudes for specified bands using central wavelengths defined in
constants.BAND_CENTRAL_WAVELENGTH_ANG.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Dict

try:
    from .constants import BAND_CENTRAL_WAVELENGTH_ANG, C_CGS, AB_ZEROPOINT_FLUX_CGS
except ImportError:  # pragma: no cover
    from constants import BAND_CENTRAL_WAVELENGTH_ANG, C_CGS, AB_ZEROPOINT_FLUX_CGS


def wavelength_from_band(band: str) -> float:
    if band not in BAND_CENTRAL_WAVELENGTH_ANG:
        raise ValueError(
            f"Unknown band '{band}'. Available: {list(BAND_CENTRAL_WAVELENGTH_ANG)}"
        )
    return BAND_CENTRAL_WAVELENGTH_ANG[band]


def frequency_from_wavelength_angstrom(wavelength_ang: float) -> float:
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    wavelength_cm = wavelength_ang * 1e-8
    return C_CGS / wavelength_cm


def luminosity_nu_to_flux_density(l_nu: float, distance_mpc: float) -> float:
    """Convert monochromatic luminosity L_nu [erg/s/Hz] to flux density F_nu [erg/s/cm^2/Hz]."""
    d_cm = distance_mpc * 3.086e24
    return l_nu / (4 * np.pi * d_cm**2)


def ab_mag_from_flux_density(f_nu: float) -> float:
    if f_nu <= 0:
        return np.inf
    return -2.5 * np.log10(f_nu) - 48.6


def ab_magnitude_from_lnu(l_nu: float, distance_mpc: float) -> float:
    return ab_mag_from_flux_density(luminosity_nu_to_flux_density(l_nu, distance_mpc))


def planck_lnu(temperature: float, radius: float, frequency: float) -> float:
    """Approximate spectral luminosity L_nu for a blackbody sphere.

    L_nu = 4 * pi * R^2 * pi * B_nu(T)  (surface area times hemispheric intensity)
    B_nu = 2 h nu^3 / c^2 / (exp(h nu / kT) - 1) -- we fold constants numerically.
    """
    from math import exp

    # Fundamental constants (CGS)
    h = 6.62607015e-27
    k = 1.380649e-16
    c = C_CGS
    if temperature <= 0:
        return 0.0
    pre = 8 * np.pi**2 * radius**2 * h * frequency**3 / c**2
    denom = exp(h * frequency / (k * temperature)) - 1.0
    if denom <= 0:
        return 0.0
    return pre / denom


def build_simple_blackbody_lnu_series(
    times: np.ndarray, temps: np.ndarray, radii: np.ndarray, band: str
) -> np.ndarray:
    """Generate L_nu time series for a band using blackbody approximation.

    Parameters
    ----------
    times : array
        Time array (unused except for shape).
    temps : array
        Temperature evolution [K].
    radii : array
        Effective emission radius evolution [cm].
    band : str
        Photometric band key (g,r,i,z,J).
    """
    nu = frequency_from_wavelength_angstrom(wavelength_from_band(band))
    lnu = np.array([planck_lnu(T, R, nu) for T, R in zip(temps, radii)])
    return lnu


def multi_band_ab_magnitudes(
    l_nu_map: Dict[str, np.ndarray], distance_mpc: float
) -> Dict[str, np.ndarray]:
    out = {}
    for band, lnu_series in l_nu_map.items():
        fnu = luminosity_nu_to_flux_density(lnu_series, distance_mpc)
        with np.errstate(divide="ignore"):
            mags = -2.5 * np.log10(np.where(fnu > 0, fnu, np.nan)) - 48.6
        out[band] = mags
    return out
