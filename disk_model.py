"""
AGN disk model utilities.

This module handles the generation, loading, and manipulation of AGN disk models
based on the Sirko & Goodman approach.
"""

import numpy as np
import os
from typing import Tuple, Optional

try:
    from .constants import (
        PC_CGS,
        SI_to_gcm3,
        SI_to_cms,
        ModelParameters,
        C_CGS,
        jy_to_cgs,
        ANGSTROM_TO_CM,
    )
except ImportError:
    from constants import (
        PC_CGS,
        SI_to_gcm3,
        SI_to_cms,
        ModelParameters,
        C_CGS,
        jy_to_cgs,
        ANGSTROM_TO_CM,
    )


class AGNDiskModel:
    """Class for handling AGN disk models based on Sirko & Goodman profiles."""

    def __init__(self, params: ModelParameters, data_dir: str = "./agn_disks"):
        """Initialize the AGN disk model container.

        Parameters
        ----------
        params : ModelParameters
            Global model parameters
        data_dir : str
            Directory where disk .dat files are stored or generated
        """
        # Store parameters and placeholders for arrays
        self.params = params
        self.data_dir = data_dir
        self.r = None
        self.density = None
        self.height = None
        self.tau = None
        self.sound_speed = None  # Midplane sound speed [cm/s]
        self._loaded = False

    # ------------------------------------------------------------------
    # I/O HELPERS
    # ------------------------------------------------------------------
    def file_exists(self, filename: str) -> bool:
        file_path = os.path.join(self.data_dir, f"{filename}.dat")
        return os.path.exists(file_path)

    def load_disk_model(self, filename: str) -> None:
        """Load an existing disk model from a .dat file."""
        file_path = os.path.join(self.data_dir, f"{filename}.dat")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Disk file not found: {file_path}")

        data = np.loadtxt(file_path)
        # Columns: r(pc), omega, T, rho(SI), h(m), c_s(m/s), tau, Q, Teff
        self.r = data[:, 0] * PC_CGS
        self.density = data[:, 3] * SI_to_gcm3
        self.height = data[:, 4] * 100.0  # m -> cm
        self.sound_speed = data[:, 5] * SI_to_cms
        self.tau = data[:, 6]

        self._loaded = True
        print(f"Loaded AGN disk model from {filename}.dat")
        print(
            f"  Radial range: {self.r[0]/self.params.r_g:.1f} - {self.r[-1]/self.params.r_g:.1f} r_g"
        )
        print(
            f"  Density range: {np.min(self.density):.2e} - {np.max(self.density):.2e} g/cm³"
        )
        print(
            f"  Scale height range: {np.min(self.height):.2e} - {np.max(self.height):.2e} cm"
        )
        print(
            f"  Sound speed range: {np.min(self.sound_speed):.2e} - {np.max(self.sound_speed):.2e} cm/s"
        )

    def generate_disk_model(self, filename: str) -> None:
        """Generate and save a new disk model (requires pagn)."""
        if self.file_exists(filename):
            print(
                f"Disk file {filename}.dat already exists. Loading existing file instead of regenerating."
            )
            self.load_disk_model(filename)
            return

        try:
            from pagn import Sirko
            import pagn.constants as ct
        except ImportError as e:
            raise ImportError("pagn package required for disk model generation") from e

        os.makedirs(self.data_dir, exist_ok=True)

        Mbh = self.params.mass_smbh * ct.MSun
        le = self.params.eddington_ratio
        alpha = self.params.alpha_viscosity

        rho_arr = np.logspace(-15, -4, 10)
        temp_arr = np.logspace(1, np.log10(999999), 1001)
        kappa_arr = np.ones((len(rho_arr), len(temp_arr))) * (self.params.kappa / 10.0)

        print(
            f"Generating AGN disk model with M_SMBH = {self.params.mass_smbh:.1e} M_sun..."
        )
        sk = Sirko.SirkoAGN(
            Mbh=Mbh,
            le=le,
            Mdot=None,
            alpha=alpha,
            X=0.7,
            b=0,
            opacity=(kappa_arr, rho_arr, temp_arr),
        )
        sk.solve_disk(N=int(1e4))

        pgas = sk.rho * sk.T * ct.Kb / ct.massU
        prad = 4.0 * ct.sigmaSB * (sk.T**4) / (3.0 * ct.c)
        cs = np.sqrt((pgas + prad) / sk.rho)

        file_path = os.path.join(self.data_dir, f"{filename}.dat")
        output_data = np.vstack(
            [
                sk.R / ct.pc,
                sk.Omega,
                sk.T,
                sk.rho,
                sk.h,
                cs,
                sk.tauV,
                sk.Q,
                sk.Teff4,
            ]
        ).T
        np.savetxt(file_path, output_data)
        print(f"Disk model saved to {filename}.dat")
        self.load_disk_model(filename)

    # ------------------------------------------------------------------
    # QUERY METHODS
    # ------------------------------------------------------------------
    def get_local_properties(self, radius_rg: float) -> Tuple[float, float]:
        if not self._loaded:
            raise RuntimeError("Disk model not loaded. Call load_disk_model() first.")
        radius_cm = radius_rg * self.params.r_g
        r_index = int(np.argmin(np.abs(self.r - radius_cm)))
        density = self.density[r_index]
        height = self.height[r_index]
        print(f"Local disk properties at {radius_rg:.1f} r_g:")
        print(f"  Density: ρ₀ = {density:.2e} g/cm³")
        print(f"  Scale height: h = {height:.2e} cm")
        return density, height

    def get_local_properties_with_sound_speed(
        self, radius_rg: float
    ) -> Tuple[float, float, float]:
        if not self._loaded:
            raise RuntimeError("Disk model not loaded. Call load_disk_model() first.")
        radius_cm = radius_rg * self.params.r_g
        r_index = int(np.argmin(np.abs(self.r - radius_cm)))
        density = self.density[r_index]
        height = self.height[r_index]
        c_s = self.sound_speed[r_index]
        print(
            f"Local disk properties at {radius_rg:.1f} r_g: ρ₀={density:.2e} g/cm³, h={height:.2e} cm, c_s={c_s:.2e} cm/s"
        )
        return density, height, c_s

    def get_disk_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._loaded:
            raise RuntimeError("Disk model not loaded. Call load_disk_model() first.")
        return self.r, self.density, self.height, self.tau

    # ------------------------------------------------------------------
    # LOADING WRAPPER
    # ------------------------------------------------------------------
    def ensure_disk_loaded(self, filename: str, auto_generate: bool = False) -> None:
        if self._loaded:
            print("Disk model already loaded.")
            return
        full_path = os.path.join(self.data_dir, f"{filename}.dat")
        print(f"Checking for disk file: {full_path}")
        if self.file_exists(filename):
            print(f"Found existing disk file: {filename}.dat")
            self.load_disk_model(filename)
        elif auto_generate:
            print(f"Disk file {filename}.dat not found. Generating new disk model...")
            self.generate_disk_model(filename)
        else:
            raise FileNotFoundError(
                f"Disk file {filename}.dat not found at {full_path}. Set auto_generate=True to generate automatically."
            )

    def read_model_for_lum(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        - Units used to solve the disk are: pc, msun, yr
        - Constants are given in SI units
        """
        file_path = os.path.join(self.data_dir, f"{filename}.dat")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Disk file not found: {file_path}")
        r = np.loadtxt(file_path, usecols=(0)) * PC_CGS
        teff4 = np.loadtxt(file_path, usecols=(8))  # In SI units
        return r, teff4

    def deltar(self, rmin, rmax, n):
        Redge = np.logspace(np.log10(rmin), np.log10(rmax), n + 1)
        R = (Redge[:-1] + Redge[1:]) * 0.5
        deltaR = Redge[1:] - Redge[:-1]
        return R, deltaR

    def lum_per_wavelength(self, radius, lamb, teff):
        """
        Integrate the r integral for a given wavelength and temperature.

        Parameters
        ----------
        radius : array
            Radius values (in cm).
        lamb : float
            Wavelength (in cm).
        teff : array
            Effective temperature (in Kelvin).

        Returns
        -------
        float
            Integrated luminosity over the radius.
        """
        from scipy.integrate import simpson
        import astropy.constants as c

        r_arr, delta_r = self.deltar(np.min(radius), np.max(radius), len(radius))
        expo = c.h.cgs.value * c.c.cgs.value / (lamb * c.k_B.cgs.value * teff)
        planck = (2 * c.h.cgs.value * c.c.cgs.value**2) / (lamb**5) / (np.expm1(expo))
        r_integrand = 2 * np.pi * r_arr * planck
        result = simpson(r_integrand, r_arr)
        return result

    def bolometric_luminosity(self, radius, teff4):
        """
        Compute the bolometric luminosity by integrating over wavelength.

        Parameters
        ----------
        radius : array
            Radius values (in cm).
        teff : array
            Effective temperature (in Kelvin).

        Returns
        -------
        float
            Bolometric luminosity (in erg/s).
        """
        teff = teff4**0.25
        wavelengths_cm = np.logspace(
            np.log10(1e-6), np.log10(1e-2), 1000
        )  # 100 Å to 100,000 Å
        lum_per_lam = [
            self.lum_per_wavelength(radius, lam, teff) for lam in wavelengths_cm
        ]
        bol_lum = np.trapz(lum_per_lam, wavelengths_cm)  # erg/s
        return bol_lum

    def agn_magnitudes(
        self,
        radius,
        teff4,
        band_wavelengths,
        distance_cm,
        luminosity_distance,
    ):
        """
        Compute AB and absolute magnitudes for each band for the AGN disk.

        Parameters
        ----------
        radius : array
            Radius values (in cm).
        teff4 : array
            Effective temperature^4 (in K^4).
        band_wavelengths : dict
            Dictionary of band name to central wavelength (in Angstrom).
        distance_cm : float
            Luminosity distance in cm.
        luminosity_distance : float
            Luminosity distance in Mpc.
        C_CGS : float
            Speed of light in cm/s.
        jy_to_cgs : float
            Jansky to cgs conversion factor.
        ANGSTROM_TO_CM : float
            Angstrom to cm conversion factor.

        Returns
        -------
        dict
            Dictionary mapping band to (AB magnitude, absolute magnitude).
        """
        import numpy as np

        agn_luminosities = {}
        teff = teff4**0.25
        ab_zero_flux = 3631 * jy_to_cgs
        D_L_pc = luminosity_distance * 1e6  # Mpc to pc
        for band, wavelength_ang in band_wavelengths.items():
            print(f"\nBand: {band}")
            lam = wavelength_ang * ANGSTROM_TO_CM
            agn_lum_per_lam = self.lum_per_wavelength(radius, lam, teff)
            print(f"agn_lum_per_lam (erg/s/cm): {agn_lum_per_lam:.3e}")
            f_lambda = agn_lum_per_lam / (4 * np.pi * distance_cm**2)
            print(f"f_lambda (erg/s/cm^2/cm): {f_lambda:.3e}")
            f_nu = f_lambda * lam**2 / C_CGS
            print(f"f_nu (erg/s/cm^2/Hz): {f_nu:.3e}")
            if f_nu > 0:
                mags = -2.5 * np.log10(f_nu / ab_zero_flux)
            else:
                mags = np.inf
            if np.isfinite(mags):
                abs_mag = mags - 5 * np.log10(D_L_pc / 10)
            else:
                abs_mag = np.inf
            print(f"AB magnitude: {mags}")
            print(f"Absolute magnitude: {abs_mag}")
            agn_luminosities[band] = (mags, abs_mag)
        return agn_luminosities


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_disk_model(
    params: ModelParameters,
    filename: str,
    data_dir: str = "./agn_disks",
    auto_generate: bool = True,
) -> AGNDiskModel:
    """
    Create and load AGN disk model.

    Parameters
    ----------
    params : ModelParameters
        Model parameters
    filename : str
        Disk model filename
    data_dir : str
        Data directory
    auto_generate : bool
        Whether to auto-generate if needed

    Returns
    -------
    AGNDiskModel
        Loaded disk model
    """
    disk_model = AGNDiskModel(params, data_dir)
    disk_model.ensure_disk_loaded(filename, auto_generate)
    return disk_model
