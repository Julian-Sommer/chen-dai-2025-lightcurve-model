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
