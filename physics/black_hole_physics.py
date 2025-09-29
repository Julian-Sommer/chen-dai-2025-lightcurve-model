"""Black hole physics utilities for an embedded secondary BH in an AGN disk.

Overview
--------
This module provides utilities to compute the accretion rate onto an embedded
lower-mass black hole moving within an AGN disk and the resulting jet power.

Uses the Bondi–Hoyle–Lyttleton accretion rate with the specified relative velocity
(quadrature of local sound speed and kick velocity). The jet luminosity is then

    L_j = eta_jet * Mdot_Bondi * c^2

No Eddington cap is applied - calculations are based purely on Bondi accretion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

try:
    from ..constants import C_CGS, G_CGS, M_SUN_CGS, ETA_JET_DEFAULT
except ImportError:  # pragma: no cover
    from constants import C_CGS, G_CGS, M_SUN_CGS, ETA_JET_DEFAULT


@dataclass
class EmbeddedBHParameters:
    """Parameters describing the embedded (secondary) black hole.

    Attributes
    ----------
    mass_bh : float
        Mass of the embedded BH [M_sun]
    radius_rg : float
        Radial position in gravitational radii of the *central SMBH*.
    v_kick : float
        Kick / relative velocity in km/s (will be converted to cm/s).
    eta_jet : float
        Jet production efficiency (fraction of rest-mass energy into jet power).
    """

    mass_bh: float
    radius_rg: float
    v_kick: float
    eta_jet: float = ETA_JET_DEFAULT

    def to_dict(self) -> Dict[str, float]:  # convenience
        return {
            "mass_bh": self.mass_bh,
            "radius_rg": self.radius_rg,
            "v_kick": self.v_kick,
            "eta_jet": self.eta_jet,
        }


def bondi_hoyle_accretion_rate(
    mass_bh_msun: float,
    density_g_cm3: float,
    sound_speed_cm_s: float,
    v_kick_km_s: float,
) -> float:
    """Compute the Bondi–Hoyle–Lyttleton accretion rate for an embedded BH.

    Parameters
    ----------
    mass_bh_msun : float
        Mass of the BH in solar masses.
    density_g_cm3 : float
        Local gas density [g/cm^3].
    sound_speed_cm_s : float
        Local sound speed [cm/s].
    v_kick_km_s : float
        Kick/relative velocity [km/s].

    Returns
    -------
    mdot_bondi : float
        Bondi accretion rate in g/s.
    """
    m_bh_g = mass_bh_msun * M_SUN_CGS
    v_kick_cm_s = v_kick_km_s * 1e5
    v_eff = np.sqrt(sound_speed_cm_s**2 + v_kick_cm_s**2)

    # Bondi radius
    r_bondi = 2 * G_CGS * m_bh_g / v_eff**2
    # Bondi accretion rate
    mdot_bondi = np.pi * r_bondi**2 * density_g_cm3 * v_eff

    return mdot_bondi


def compute_jet_luminosity(
    bh_params: EmbeddedBHParameters,
    density_g_cm3: float,
    sound_speed_cm_s: float,
) -> Dict[str, float]:
    """Compute jet luminosity for an embedded BH using Bondi accretion.

    This function:
    1. Computes the Bondi–Hoyle–Lyttleton accretion rate
    2. Converts to jet luminosity via L_j = eta_jet * Mdot * c^2

    Parameters
    ----------
    bh_params : EmbeddedBHParameters
        Parameters for the embedded BH.
    density_g_cm3 : float
        Local gas density [g/cm^3].
    sound_speed_cm_s : float
        Local sound speed [cm/s].

    Returns
    -------
    dict
        Dictionary containing:
        - 'mdot_bondi': Bondi accretion rate [g/s]
        - 'mdot_cap': Same as mdot_bondi (for compatibility)
        - 'L_jet': Jet luminosity [erg/s]
    """
    # Compute Bondi accretion rate
    mdot_bondi = bondi_hoyle_accretion_rate(
        bh_params.mass_bh, density_g_cm3, sound_speed_cm_s, bh_params.v_kick
    )

    # Compute jet luminosity
    L_jet = bh_params.eta_jet * mdot_bondi * C_CGS**2

    return {
        "mdot_bondi": mdot_bondi,
        "mdot_cap": mdot_bondi,  # For backward compatibility
        "L_jet": L_jet,
    }
