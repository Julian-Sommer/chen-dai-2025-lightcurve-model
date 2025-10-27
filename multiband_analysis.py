#!/usr/bin/env python3
"""
Multiband emission analysis for Chen & Dai (2025) jet model.

This module provides the core functionality for computing comprehensive
multiband lightcurves including jet-head shock breakout, disk cocoon,
and jet-cocoon emission components.

Based on: Chen & Dai (2025), ApJ, 987, 214
"""

import numpy as np
import os
from pathlib import Path

from evolution import ChenDaiModel, find_shock_breakout_time
from constants import (
    ModelParameters,
    DiskConfiguration,
    ETA_JET_DEFAULT,
    C_CGS,
    jy_to_cgs,
)
from disk_model import create_disk_model
from physics import PhysicsCalculator, DiskPhysics
from physics.black_hole_physics import EmbeddedBHParameters, compute_jet_luminosity
from sed import BlackbodySED, ANGSTROM_TO_CM
from photometry import wavelength_from_band
from plotting import (
    plot_multiband_ab_magnitudes,
    plot_multiband_ab_magnitudes_linear_days,
    plot_multiband_ab_magnitudes_with_agn,
)
from utils import generate_parameter_subdir, create_plots_subdir


class MultibandAnalysisResults:
    """Container for multiband analysis results."""

    def __init__(self):
        self.times = None
        self.band_results = {}
        self.emission_components = {}
        self.breakout_data = {}
        self.model_info = {}
        self.output_paths = {}


def run_multiband_analysis(
    bh_mass,
    vkick,
    radial_distance,
    luminosity_distance,
    smbh_mass,
    bands=None,
    t_max=1e6,
    time_bins=5000,
    use_pagn_default=False,
    gamma_j=100.0,
    theta_0=0.17,
    kappa=0.34,
    output_dir="./plots",
    save_data=True,
    use_chatzopoulos_tdiff=False,
    verbose=True,
):
    """
    Run comprehensive multiband emission analysis.

    Parameters
    ----------
    bh_mass : float
        Embedded BH mass in solar masses
    vkick : float
        Kick velocity in km/s
    radial_distance : float
        BH radial distance in gravitational radii
    luminosity_distance : float
        Observer distance in Mpc
    smbh_mass : float
        Supermassive BH mass in solar masses
    bands : list, optional
        Photometric bands to analyze (default: ['g', 'r', 'i'])
    t_max : float, optional
        Maximum time in seconds (default: 1e6)
    time_bins : int, optional
        Number of time bins (default: 5000)
    use_pagn_default : bool, optional
        Use PAGN default parameters (default: False)
    gamma_j : float, optional
        Jet Lorentz factor (default: 100.0)
    theta_0 : float, optional
        Initial jet opening angle in radians (default: 0.17)
    kappa : float, optional
        Opacity in cm²/g (default: 0.34)
    output_dir : str, optional
        Output directory for plots and data (default: "./plots")
    save_data : bool, optional
        Whether to save data to CSV files (default: True)
    verbose : bool, optional
        Whether to print detailed output (default: True)

    Returns
    -------
    MultibandAnalysisResults
        Object containing all analysis results
    """
    if bands is None:
        bands = ["g", "r", "i"]

    results = MultibandAnalysisResults()

    if verbose:
        print("=" * 60)
        print("MULTIBAND EMISSION ANALYSIS")
        print("=" * 60)

    # Create model parameters using elegant disk configuration
    params = ModelParameters.from_disk_config(
        mass_smbh=smbh_mass,
        use_pagn_default=use_pagn_default,
        gamma_j=gamma_j,
        theta_0=theta_0,
        kappa=kappa,
        use_chatzopoulos_tdiff=use_chatzopoulos_tdiff,
    )

    # Get configuration info for user feedback
    config = DiskConfiguration.get_config(use_pagn_default)
    if verbose:
        print(
            f"Using {config['description']}: le={config['eddington_ratio']}, alpha={config['alpha_viscosity']}"
        )
        print(f"Disk filename will be: {params.get_disk_filename()}")

    disk = create_disk_model(params, params.get_disk_filename())

    if verbose:
        print(f"Embedded BH mass: {bh_mass} M_sun")
        print(f"SMBH mass (disk): {smbh_mass} M_sun")
        print(f"Model parameters: {params}")

    bh_params = EmbeddedBHParameters(
        mass_bh=bh_mass,
        radius_rg=radial_distance,
        v_kick=vkick,
        eta_jet=ETA_JET_DEFAULT,
    )
    rho_local, h_local, c_s_local = disk.get_local_properties_with_sound_speed(
        radial_distance
    )
    lj_data = compute_jet_luminosity(bh_params, rho_local, c_s_local)
    l_j = lj_data["L_jet"]

    if verbose:
        print(f"Jet Luminosity: {l_j:.3e} erg/s")

    # Create model using package's automatic disk filename generation
    model = ChenDaiModel(
        l_j=l_j,
        params=params,
        disk_filename=params.get_disk_filename(),
        disk_radius_rg=radial_distance,
        data_dir="./agn_disks",
        auto_generate_disk=True,
    )

    # Generate time grid using specified number of bins
    times = np.logspace(-1, np.log10(t_max), time_bins)
    results.times = times

    if verbose:
        print(f"\n=== Evolution Parameters ===")
        print(f"Time steps: {time_bins}")
        print(f"Time range: {times[0]:.1e} - {times[-1]:.1e} s")

    # Evolve the jet system
    profile_type = "uniform"
    evolution_results = model.evolve_with_numerical_integration(
        times,
        density_profile=profile_type,
        initial_guess=0.1,
        progress_interval=max(500, time_bins // 10),
        store_diagnostics=True,
        use_optical_depth=False,
        integration_method="trapz",
    )

    # Find shock breakout time
    t_break, idx_break = find_shock_breakout_time(
        evolution_results, model, density_profile=profile_type
    )

    if t_break is None or idx_break is None:
        if verbose:
            print("Warning: no shock breakout detected")
        return results

    if verbose:
        print(f"\n=== Breakout Results ===")
        print(f"Shock breakout: t_break = {t_break:.2e} s")
        print(f"Breakout index: {idx_break}")

    # Store breakout data
    results.breakout_data = {"t_break": t_break, "idx_break": idx_break}

    # Calculate emission properties at breakout
    calc = PhysicsCalculator(params, l_j)

    # Get parameters at breakout
    beta_h_break = evolution_results.beta_h[idx_break]
    z_h_break = evolution_results.z_h[idx_break]
    sigma_h_break = evolution_results.sigma_h[idx_break]

    # Calculate density at breakout height using the appropriate profile
    rho_z_break = DiskPhysics.density_profile(
        rho_0=model.rho_0, h=model.h, z=z_h_break, profile_type=profile_type
    )

    if verbose:
        print(f"Physics parameters at breakout:")
        print(f"  β_h = {beta_h_break:.6f}")
        print(f"  z_h = {z_h_break:.2e} cm")
        print(f"  ρ_z = {rho_z_break:.2e} g/cm³")
        print(f"  σ_h = {sigma_h_break:.2e} cm²")

    # ===== JET-HEAD SHOCK BREAKOUT EMISSION =====
    if verbose:
        print(f"\n=== Jet-Head Shock Breakout Emission ===")

    # Calculate jet-head shock breakout luminosity
    L_h = calc.jet_head_shock_breakout_emission.jet_head_shock_breakout_lum(
        sigma_h_break, rho_z_break, beta_h_break
    )

    # Calculate additional parameters needed for temperature determination
    gamma_h_break = calc.jet.gamma_h(beta_h_break)
    s_gamma_h = calc.jet.small_gamma_h(gamma_h_break)
    shell_width_break = (
        calc.jet_head_shock_breakout_emission.shock_breakout_shell_width(
            params.kappa, rho_z_break, beta_h_break
        )
    )
    m_shell = calc.jet_head_shock_breakout_emission.shell_mass(
        rho_z_break, sigma_h_break, shell_width_break
    )

    # Calculate temperature using comprehensive method
    T_h = calc.jet_head_shock_breakout_emission.determine_temperature(
        beta_h_break,
        rho_z_break,
        s_gamma_h,
        m_shell,
        params.kappa,
        rho_z_break,
        sigma_h_break,
    )

    # Calculate duration of emission
    t_h = calc.jet_head_shock_breakout_emission.duration_of_emission(
        params.kappa, rho_z_break, beta_h_break
    )

    if verbose:
        print(f"Jet-head emission properties:")
        print(f"  Luminosity: {L_h:.2e} erg/s")
        print(f"  Temperature: {T_h:.2e} K")
        print(f"  Duration: {t_h:.2e} s")
        print(f"  Shell width: {shell_width_break:.2e} cm")
        print(f"  Shell mass: {m_shell:.2e} g")

    # ===== DISK-COCOON EMISSION =====
    if verbose:
        print(f"\n=== Disk-Cocoon Emission ===")

    # Extract cocoon parameters at breakout
    e_c_break = evolution_results.e_c[idx_break]
    beta_c_break = evolution_results.beta_c[idx_break]
    r_c_break = evolution_results.r_c[idx_break]
    h_disk = model.h

    # Calculate thermalization time t_c_th
    t_c_th = None
    if beta_c_break >= 0.03:
        t_c_th = calc.disk_cocoon_emission.find_t_c_th(rho_z_break, beta_c_break, times)
        if verbose:
            print(f"Thermalization time: t_c_th = {t_c_th:.2e} s")
    else:
        if verbose:
            print(
                f"Thermalization time: Not applicable (β_c = {beta_c_break:.3f} < 0.03)"
            )

    # Calculate timescales for diagnostics
    try:
        all_timescales = calc.calculate_timescales(
            beta_c_break,
            r_c_break,
            rho_z_break,
            params.kappa,
            e_c_break,
            h_disk,
            z_h_break,
        )
        if verbose:
            print(f"Timescale analysis completed")
    except Exception as e:
        if verbose:
            print(f"Timescale calculation failed: {e}")

    if verbose:
        print(f"Disk-cocoon parameters:")
        print(f"  E_c = {e_c_break:.2e} erg")
        print(f"  β_c = {beta_c_break:.6f}")
        print(f"  r_c = {r_c_break:.2e} cm")
        print(f"  h = {h_disk:.2e} cm")

    # Calculate disk-cocoon emission over time
    if verbose:
        print("Calculating disk-cocoon emission evolution...")
    L_dc = []
    T_dc = []
    for t in times:
        L_dc.append(
            calc.calculate_disk_cocoon_emission(
                t,
                beta_c_break,
                r_c_break,
                rho_z_break,
                params.kappa,
                e_c_break,
                h_disk,
                z_h_break,
            )
        )
        T_dc.append(
            calc.calculate_disk_cocoon_temperature(
                t,
                beta_c_break,
                r_c_break,
                rho_z_break,
                params.kappa,
                e_c_break,
                h_disk,
                z_h_break,
                times=times,
                t_c_th=t_c_th,
            )
        )
    L_dc = np.array(L_dc)
    T_dc = np.array(T_dc)

    valid_dc_lum = L_dc[L_dc > 0]
    valid_dc_temp = T_dc[T_dc > 0]
    if verbose:
        print(f"Disk-cocoon emission results:")
        if len(valid_dc_lum) > 0:
            print(f"  Peak luminosity: {np.max(valid_dc_lum):.2e} erg/s")
        if len(valid_dc_temp) > 0:
            print(f"  Peak temperature: {np.max(valid_dc_temp):.2e} K")

    # ===== JET-COCOON EMISSION =====
    if verbose:
        print(f"\n=== Jet-Cocoon Emission ===")
        print("Calculating jet-cocoon emission with critical time approach...")

    try:
        # Use advanced critical time approach
        jc_data = calc.calculate_jet_cocoon_emission_with_critical_time(
            results=evolution_results,
            t_break=t_break,
            idx_break=idx_break,
            use_critical_time=True,
            search_range_factor=0.5,
            tolerance=1e-3,
        )

        jc_props = jc_data["properties"]
        approach_used = jc_data["approach"]

        # Report approach details
        if verbose:
            print(f"Jet-cocoon approach: {approach_used.upper()}")
            if approach_used == "critical_time":
                critical_data = jc_data["critical_time_data"]
                timing = jc_data["timing_comparison"]
                print(f"  Critical time: {timing['critical_time']:.2e} s")
                print(f"  Breakout time: {timing['breakout_time']:.2e} s")
                print(
                    f"  Critical occurs {timing['critical_earlier_by_factor']:.2f}x earlier"
                )
                print(f"  r_cj = r_c error: {critical_data['relative_error']:.2e}")

            # Report thermal regime
            thermal_regime = jc_props["thermal_regime"]
            eta_escs = jc_props["eta_escs"]
            print(f"  Thermal regime: {thermal_regime.upper()}")
            print(f"  η_esc = {eta_escs:.3f}")
            print(f"  t_sph_end = {jc_props['t_sph_end']:.2e} s")

        # Calculate jet-cocoon emission over time
        if verbose:
            print("Computing jet-cocoon emission evolution...")
        L_jc = []
        T_jc = []
        for t in times:
            L_jc.append(
                calc.calculate_jet_cocoon_emission(t, jet_cocoon_properties=jc_props)
            )
            T_jc.append(
                calc.calculate_jet_cocoon_temperature(t, jet_cocoon_properties=jc_props)
            )
        L_jc = np.array(L_jc)
        T_jc = np.array(T_jc)

        valid_jc_lum = L_jc[L_jc > 0]
        valid_jc_temp = T_jc[T_jc > 0]
        if verbose:
            print(f"Jet-cocoon emission results:")
            if len(valid_jc_lum) > 0:
                print(f"  Peak luminosity: {np.max(valid_jc_lum):.2e} erg/s")
            if len(valid_jc_temp) > 0:
                print(f"  Peak temperature: {np.max(valid_jc_temp):.2e} K")

    except Exception as e:
        if verbose:
            print(f"Jet-cocoon computation failed: {e}")
            print("Falling back to zero emission")
        L_jc = np.zeros_like(times)
        T_jc = np.zeros_like(times)

    # ===== MULTIBAND PHOTOMETRY CALCULATION =====
    if verbose:
        print(f"\n=== Multiband Photometry ===")
        print(f"Computing photometry for bands: {bands}")

    # Create jet-head emission time series
    L_jh_arr = np.zeros_like(times)
    T_jh_arr = np.zeros_like(times)
    mask_jh = (times >= t_break) & (times <= t_break + t_h)
    L_jh_arr[mask_jh] = L_h
    T_jh_arr[mask_jh] = T_h

    # Distance for flux calculations
    distance_cm = luminosity_distance * 3.086e24  # Mpc to cm

    # AB magnitude conversion constants
    AB_ZEROPOINT = 48.60

    def nu_f_nu_to_ab_mag(nu_f_nu_array, wavelength_cm):
        """Convert nu f_nu to AB magnitude"""
        c = 2.99792458e10  # cm/s
        nu = c / wavelength_cm
        f_nu = np.where(nu_f_nu_array > 0, nu_f_nu_array / nu, -1.0)
        mags = np.full_like(f_nu, np.inf, dtype=float)
        pos = f_nu > 0
        mags[pos] = -2.5 * np.log10(f_nu[pos]) - AB_ZEROPOINT
        return mags

    def compute_component_nu_f_nu(
        times, L_array, T_array, wavelength_cm, distance_cm, mask_before=None
    ):
        """Compute nu f_nu for a component"""
        sed = BlackbodySED()
        out = np.zeros_like(times)
        for i, (t, L, T) in enumerate(zip(times, L_array, T_array)):
            if mask_before is not None and t < mask_before:
                continue
            if L <= 0 or T <= 0:
                continue
            L_lambda = sed.spectral_luminosity_at_wavelength(L, T, wavelength_cm)
            out[i] = sed.spectral_luminosity_to_nu_f_nu(
                L_lambda, wavelength_cm, distance_cm
            )
        return out

    def compute_component_nu_f_nu_without_time(
        L_array, T_array, wavelength_cm, distance_cm
    ):
        """Compute nu f_nu for a component"""
        sed = BlackbodySED()
        out = np.zeros_like(L_array)
        for i, (L, T) in enumerate(zip(L_array, T_array)):
            if L <= 0 or T <= 0:
                continue
            L_lambda = sed.spectral_luminosity_at_wavelength(L, T, wavelength_cm)
            out[i] = sed.spectral_luminosity_to_nu_f_nu(
                L_lambda, wavelength_cm, distance_cm
            )
        return out

    # Calculate photometry for each band
    band_results = {}
    for band in bands:
        wavelength_ang = wavelength_from_band(band)
        wavelength_cm = wavelength_ang * ANGSTROM_TO_CM

        # Calculate nu f_nu for each component
        nu_f_nu_jh = compute_component_nu_f_nu(
            times, L_jh_arr, T_jh_arr, wavelength_cm, distance_cm
        )
        nu_f_nu_dc = compute_component_nu_f_nu(
            times, L_dc, T_dc, wavelength_cm, distance_cm, mask_before=t_break
        )
        nu_f_nu_jc = compute_component_nu_f_nu(
            times, L_jc, T_jc, wavelength_cm, distance_cm, mask_before=t_break
        )

        # Total flux and magnitude
        total = nu_f_nu_jh + nu_f_nu_dc + nu_f_nu_jc
        mags_total = nu_f_nu_to_ab_mag(total, wavelength_cm)

        band_results[band] = {
            "nu_f_nu_jh": nu_f_nu_jh,
            "nu_f_nu_dc": nu_f_nu_dc,
            "nu_f_nu_jc": nu_f_nu_jc,
            "nu_f_nu_total": total,
            "ab_mag_total": mags_total,
            "wavelength_cm": wavelength_cm,
        }

    # ===== AGN DISK LUMINOSITY CALCULATION =====
    band_wavelengths = {}
    for band in bands:
        wavelength_ang = wavelength_from_band(band)
        wavelength_cm = wavelength_ang * ANGSTROM_TO_CM
        band_wavelengths[band] = wavelength_ang

    (
        radial_dist,
        teff4,
    ) = disk.read_model_for_lum(params.get_disk_filename())
    bol_lum = disk.bolometric_luminosity(radial_dist, teff4)
    print(f"\nBolometric disk luminosity (integrated): {bol_lum:.3e} erg/s")

    (
        radial_dist,
        teff4,
    ) = disk.read_model_for_lum(params.get_disk_filename())
    lums_from_function = disk.agn_magnitudes(
        radial_dist,
        teff4,
        band_wavelengths,
        distance_cm,
        luminosity_distance,
    )
    print(f"")

    # Add AGN disk AB magnitudes to band_results (non-destructive, after agn_magnitudes call)
    for band in lums_from_function:
        ab_mag_agn = lums_from_function[band][0]  # AB magnitude
        if band in band_results:
            band_results[band]["ab_mag_agn"] = ab_mag_agn
            # Physically correct sum: add in flux space, then convert back to mag
            ab_zero_flux = 3631 * jy_to_cgs
            ab_mag_total = band_results[band]["ab_mag_total"]
            # ab_mag_total may be an array (time-dependent), ab_mag_agn is a float
            f_trans = ab_zero_flux * 10 ** (-0.4 * ab_mag_total)
            f_agn = ab_zero_flux * 10 ** (-0.4 * ab_mag_agn)
            f_sum = f_trans + f_agn
            ab_mag_total_agn = -2.5 * np.log10(f_sum / ab_zero_flux)
            band_results[band]["ab_mag_total_agn"] = ab_mag_total_agn
            print(
                f"Band {band}: AGN disk AB mag = {ab_mag_agn:.2f}, transient + AGN peak AB mag = {np.min(ab_mag_total_agn):.2f}"
            )
        else:
            band_results[band] = {"ab_mag_agn": ab_mag_agn}

    results.band_results = band_results

    # Store emission component data
    results.emission_components = {
        "jet_head": {"L": L_jh_arr, "T": T_jh_arr, "t_duration": t_h},
        "disk_cocoon": {"L": L_dc, "T": T_dc},
        "jet_cocoon": {"L": L_jc, "T": T_jc},
    }

    # Store model info
    results.model_info = {
        "bh_mass": bh_mass,
        "vkick": vkick,
        "radial_distance": radial_distance,
        "luminosity_distance": luminosity_distance,
        "smbh_mass": smbh_mass,
        "l_j": l_j,
        "params": params,
        "bands": bands,
        "t_max": t_max,
        "time_bins": time_bins,
    }

    # Print photometry summary
    if verbose:
        print(f"\nPhotometry results:")
        for band in bands:
            total = band_results[band]["nu_f_nu_total"]
            print(
                f"  {band}: peak (nu f_nu) = {np.max(total):.3e} erg s^-1 cm^-2 Hz^-1"
            )

    # ===== OUTPUT AND PLOTTING =====
    if save_data or output_dir:
        if verbose:
            print(f"\n=== Output Generation ===")

        # Create data subdirectory structure
        data_base_dir = os.path.join(".", "data", "multiband_lc")
        subdir_name = generate_parameter_subdir(
            bh_mass=bh_mass,
            vkick=vkick,
            radial_distance=radial_distance,
            luminosity_distance=luminosity_distance,
            smbh_mass=smbh_mass,
        )
        data_dir = create_plots_subdir(data_base_dir, subdir_name)

        if save_data:
            # Generate unique filename based on parameters
            bands_str = "_".join(bands)
            csv_filename = f"multiband_lc_{bands_str}.csv"
            out_file = os.path.join(data_dir, csv_filename)

            # Prepare data for CSV output
            header_cols = ["time_s"]
            data_cols = [times]

            for b in bands:
                # Always present columns
                header_cols.extend(
                    [
                        f"nu_f_nu_total_{b}",
                        f"ab_mag_total_{b}",
                    ]
                )
                data_cols.extend(
                    [
                        band_results[b]["nu_f_nu_total"],
                        band_results[b]["ab_mag_total"],
                    ]
                )

                # Add AGN-only column if present
                if "ab_mag_agn" in band_results[b]:
                    header_cols.append(f"ab_mag_agn_{b}")
                    # ab_mag_agn is a scalar, so broadcast to shape of times
                    ab_mag_agn_arr = np.full_like(
                        times, band_results[b]["ab_mag_agn"], dtype=float
                    )
                    data_cols.append(ab_mag_agn_arr)

                # Add AGN+transient column if present
                if "ab_mag_total_agn" in band_results[b]:
                    header_cols.append(f"ab_mag_total_agn_{b}")
                    # ab_mag_total_agn is an array (same shape as times)
                    data_cols.append(band_results[b]["ab_mag_total_agn"])

            # Stack all data
            all_data = np.vstack(data_cols).T

            # Filter out rows where all flux values are zero (before emission starts)
            flux_cols = [
                i for i, col in enumerate(header_cols) if "nu_f_nu_total" in col
            ]

            # Keep header and rows where at least one flux value is non-zero
            nonzero_mask = np.any(all_data[:, flux_cols] > 0, axis=1)
            filtered_data = all_data[nonzero_mask]

            if verbose:
                print(
                    f"Filtered data: {len(all_data)} -> {len(filtered_data)} rows (removed {len(all_data) - len(filtered_data)} zero-flux rows)"
                )

            # Save CSV data with simple citation header
            header = ",".join(header_cols)
            citation_header = "# Based on: Chen & Dai (2025), ApJ, 987, 214"
            np.savetxt(
                out_file,
                filtered_data,
                delimiter=",",
                header=header,
                comments="",  # This ensures the header line is not commented
            )
            # Optionally, prepend the citation line manually if you want
            with open(out_file, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(citation_header.rstrip("\r\n") + "\n" + content)
            if verbose:
                print(f"Saved data: {out_file}")

            results.output_paths["csv"] = out_file

        # Generate plots with parameter-specific subdirectory
        if output_dir:
            plots_base_dir = os.path.join(output_dir)
            plots_dir = create_plots_subdir(plots_base_dir, subdir_name)

            if verbose:
                print(f"Plots will be saved to: {plots_dir}")

            # Determine disk type suffix for filenames
            disk_suffix = "_pagn_disk" if use_pagn_default else "_cd_disk"
            
            # Add timescale method suffix
            tdiff_suffix = "_chat_tdiff" if use_chatzopoulos_tdiff else "_paper_tdiff"

            # Only generate AB magnitude plots (not nu f_nu plots)
            mag_path = os.path.join(
                plots_dir, f"multiband_ab_magnitudes{disk_suffix}{tdiff_suffix}.png"
            )
            mag_linear_path = os.path.join(
                plots_dir, f"multiband_ab_magnitudes_linear_days{disk_suffix}{tdiff_suffix}.png"
            )
            mag_with_agn_linear_path = os.path.join(
                plots_dir,
                f"multiband_ab_magnitudes_with_agn_linear_days{disk_suffix}{tdiff_suffix}.png",
            )
            mag_with_agn_log_path = os.path.join(
                plots_dir, f"multiband_ab_magnitudes_with_agn_log_days{disk_suffix}{tdiff_suffix}.png"
            )

            plot_multiband_ab_magnitudes(times, band_results, bands, t_break, mag_path)
            plot_multiband_ab_magnitudes_linear_days(
                times, band_results, bands, t_break, mag_linear_path
            )

            plot_multiband_ab_magnitudes_with_agn(
                times,
                band_results,
                bands,
                t_break,
                mag_with_agn_linear_path,
                title=r"Multiband AB Magnitudes (Transient + AGN)",
                mag_key="ab_mag_total_agn",
                use_log_scale=False,
            )

            plot_multiband_ab_magnitudes_with_agn(
                times,
                band_results,
                bands,
                t_break,
                mag_with_agn_log_path,
                title=r"Multiband AB Magnitudes (Transient + AGN)",
                mag_key="ab_mag_total_agn",
                use_log_scale=True,
            )

            if verbose:
                print(f"Generated plots:")
                print(f"  {mag_path}")
                print(f"  {mag_linear_path}")
                print(f"  {mag_with_agn_linear_path}")
                print(f"  {mag_with_agn_log_path}")

            results.output_paths["plots"] = [
                mag_path,
                mag_linear_path,
                mag_with_agn_linear_path,
                mag_with_agn_log_path,
            ]

    if verbose:
        print(f"\n=== Analysis Complete ===")
        print(
            f"Successfully computed comprehensive emission analysis with multiband photometry"
        )
        print(f"Time bins used: {time_bins}")
        print(f"Bands analyzed: {bands}")

    return results
