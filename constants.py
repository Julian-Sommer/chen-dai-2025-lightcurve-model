"""
Physical constants and parameters for the Chen & Dai 2025 model.

This module centralizes all physical constants and model parameters,
making them easy to modify and maintain consistency across the package.
"""

import numpy as np
import astropy.constants as c

# ============================================================================
# PHYSICAL CONSTANTS (CGS units)
# ============================================================================

# Basic constants
C_CGS = c.c.cgs.value  # Speed of light [cm/s]
G_CGS = c.G.cgs.value  # Gravitational constant [cm³/g/s²]
M_SUN_CGS = c.M_sun.cgs.value  # Solar mass [g]
m_pro = c.m_p.cgs.value  # Proton mass [g]
k_B_CGS = c.k_B.cgs.value  # Boltzmann constant [erg/K]
sigma_SB_CGS = c.sigma_sb.cgs.value  # Stefan-Boltzmann constant [erg/cm²/s/K⁴]
# Additional frequently used fundamental constants
SIGMA_T_CGS = c.sigma_T.cgs.value  # Thomson cross-section [cm^2]
M_PROTON_CGS = m_pro  # Alias for clarity in BH physics module
a = 7.5657e-15  # Radiation constant [erg/cm^3/K^4]
b = 13.8  # Constant for cocoon diffusion timescale from Arnett (1982)
f_beta_gamma = 0.1  # Fraction of the jet-cocoon energy deposited
s_exponent = 1  # Exponent for the jet-cocoon energy scaling
n = 3  # Cocoon ejecta power law index
theta_cj = 0.5
f_d = 0.5

# Unit conversions
PC_CGS = 3.086e18  # Parsec in cm
SI_to_gcm3 = 1e-3  # Convert SI density to g/cm³
SI_to_cms = 1e2  # Convert SI velocity to cm/s
ev_in_k = 11604.5250061657  # eV to Kelvin conversion factor
ev_in_angstrom = 1 / 12398.4193043616  # (Deprecated) reciprocal of standard factor
# Conversion constant: λ[Å] = EV_TO_ANGSTROM / E[eV]
EV_TO_ANGSTROM = 12398.4193043616  # hc/e in Å·eV
jy_to_cgs = 1e-23  # Convert Jy to erg/s/cm^2/Hz
ANGSTROM_TO_CM = 1e-8  # Convert Angstroms to cm

# Jet physics parameters
ETA_JET_DEFAULT = 0.1  # Default jet production efficiency (fraction of rest-mass energy into jet power)


def energy_ev_to_wavelength_angstrom(E_ev: float) -> float:
    """Convert photon energy in eV to wavelength in Angstrom.

    λ[Å] = 12398.4193043616 / E[eV]
    """
    if E_ev <= 0:
        raise ValueError("Energy must be positive (eV)")
    return EV_TO_ANGSTROM / E_ev


# =========================================================================
# PHOTOMETRY / BAND DEFINITIONS (optical + NIR for embedded BH lightcurves)
# =========================================================================

# Central wavelengths (Å). Extend as needed.
BAND_CENTRAL_WAVELENGTH_ANG = {
    "g": 4770.0,  # SDSS g-band
    "r": 6231.0,  # SDSS r-band
    "i": 7625.0,  # SDSS i-band
    "z": 9134.0,  # SDSS z-band
    "J": 12350.0,  # 2MASS J-band
    "H": 16620.0,  # 2MASS H-band
    "K": 21590.0,  # 2MASS K-band
}

# AB magnitude zero point constant: m_AB = -2.5 log10(F_nu [cgs]) - 48.6
AB_ZEROPOINT_FLUX_CGS = 10 ** (-(48.6) / 2.5)  # ~3.631e-20 erg/s/cm^2/Hz


# ============================================================================
# MODEL PARAMETERS
# ============================================================================


class DiskConfiguration:
    """Handles disk parameter sets and filename generation."""

    # Parameter sets
    CHEN_DAI_2025 = {
        "name": "chen_and_dai_25",
        "description": "Chen & Dai 2025 parameters",
        "eddington_ratio": 0.1,
        "alpha_viscosity": 0.1,
    }

    PAGN_DEFAULT = {
        "name": "sirko_and_goodman_03",
        "description": "PAGN default (Sirko & Goodman 2003) parameters",
        "eddington_ratio": 0.5,
        "alpha_viscosity": 0.01,
    }

    @staticmethod
    def get_config(use_pagn_default: bool = False):
        """Get disk configuration based on parameter choice.

        Parameters
        ----------
        use_pagn_default : bool
            If True, use PAGN default parameters. Otherwise use Chen & Dai.

        Returns
        -------
        dict
            Configuration dictionary with name, eddington_ratio, alpha_viscosity
        """
        return (
            DiskConfiguration.PAGN_DEFAULT
            if use_pagn_default
            else DiskConfiguration.CHEN_DAI_2025
        )

    @staticmethod
    def get_disk_filename(mass_smbh: float, use_pagn_default: bool = False) -> str:
        """Generate appropriate disk filename based on parameters.

        Parameters
        ----------
        mass_smbh : float
            SMBH mass in solar masses
        use_pagn_default : bool
            If True, use PAGN naming. Otherwise use Chen & Dai naming.

        Returns
        -------
        str
            Disk filename (without extension)
        """
        config = DiskConfiguration.get_config(use_pagn_default)
        mass_str = f"{mass_smbh:.0e}"
        return f"{config['name']}_{mass_str}"


class ModelParameters:
    """Container for Chen & Dai 2025 model parameters."""

    def __init__(
        self,
        mass_smbh: float = 1e8,
        gamma_j: float = 100.0,
        theta_0: float = 0.17,
        kappa: float = 0.34,
        eddington_ratio: float = 0.1,
        alpha_viscosity: float = 0.1,
        use_chatzopoulos_tdiff: bool = False,
    ):
        """
        Initialize model parameters.

        Parameters
        ----------
        mass_smbh : float
            SMBH mass in solar masses
        gamma_j : float
            Initial jet Lorentz factor
        theta_0 : float
            Initial jet opening angle [rad]
        kappa : float
            Opacity [cm²/g]
        eddington_ratio : float
            Eddington luminosity ratio
        alpha_viscosity : float
            Shakura-Sunyaev viscosity parameter
        """
        self.mass_smbh = mass_smbh
        self.gamma_j = gamma_j
        self.theta_0 = theta_0
        self.kappa = kappa
        self.eddington_ratio = eddington_ratio
        self.alpha_viscosity = alpha_viscosity
        self.use_chatzopoulos_tdiff = use_chatzopoulos_tdiff

        # Derived parameters
        self.beta_j = np.sqrt(1 - 1 / gamma_j**2)
        self.r_g = (
            G_CGS * mass_smbh * M_SUN_CGS / (C_CGS**2)
        )  # Gravitational radius [cm]

    def __repr__(self):
        return (
            f"ModelParameters(M_SMBH={self.mass_smbh:.1e} M_sun, "
            f"Γ_j={self.gamma_j}, θ_0={self.theta_0:.3f} rad)"
        )

    @classmethod
    def from_disk_config(
        cls,
        mass_smbh: float,
        use_pagn_default: bool = False,
        gamma_j: float = 100.0,
        theta_0: float = 0.17,
        kappa: float = 0.34,
        use_chatzopoulos_tdiff: bool = False,
    ):
        """Create ModelParameters using disk configuration.

        Parameters
        ----------
        mass_smbh : float
            SMBH mass in solar masses
        use_pagn_default : bool
            If True, use PAGN default disk parameters. Otherwise use Chen & Dai.
        gamma_j : float
            Initial jet Lorentz factor
        theta_0 : float
            Initial jet opening angle [rad]
        kappa : float
            Opacity [cm²/g]

        Returns
        -------
        ModelParameters
            Configured model parameters
        """
        config = DiskConfiguration.get_config(use_pagn_default)
        return cls(
            mass_smbh=mass_smbh,
            gamma_j=gamma_j,
            theta_0=theta_0,
            kappa=kappa,
            eddington_ratio=config["eddington_ratio"],
            alpha_viscosity=config["alpha_viscosity"],
            use_chatzopoulos_tdiff=use_chatzopoulos_tdiff,
        )

    def get_disk_filename(self) -> str:
        """Get appropriate disk filename for these parameters.

        Returns
        -------
        str
            Disk filename (without extension)
        """
        # Determine which parameter set is being used
        use_pagn = self.eddington_ratio == 0.5 and self.alpha_viscosity == 0.01
        return DiskConfiguration.get_disk_filename(self.mass_smbh, use_pagn)


# Default parameters for Chen & Dai 2025
DEFAULT_PARAMS = ModelParameters()
