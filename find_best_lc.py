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
from tqdm import tqdm
import sys
import warnings

from evolution import ChenDaiModel, find_shock_breakout_time
from multiband_analysis import run_multiband_analysis
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
    """Container for best lightcurve results."""

    def __init__(self):
        self.times = None
        self.band_results = {}
        self.emission_components = {}
        self.breakout_data = {}
        self.model_info = {}
        self.output_paths = {}


import argparse


def run_find_best_lc(
    bh_mass,
    vkick=None,
    radial_distance=None,
    luminosity_distance=None,
    smbh_mass=None,
    bands=None,
    t_max=1e6,
    time_bins=5000,
    use_pagn_default=False,
    gamma_j=100.0,
    theta_0=0.17,
    kappa=0.34,
    output_dir="./plots",
    save_data=True,
    verbose=True,
    vkick_range=None,
    radial_distance_range=None,
    n_elements=5,
):
    """
    Run comprehensive multiband emission analysis for a grid or single values of vkick and radial_distance.

    SMBH mass (smbh_mass) and luminosity distance (luminosity_distance) must always be provided as single values (not arrays or ranges).

    If vkick or radial_distance are not provided as arrays, generate them from the specified range and n_elements.
    """
    # Check that smbh_mass and luminosity_distance are single values
    if isinstance(smbh_mass, (list, np.ndarray)) and len(np.atleast_1d(smbh_mass)) > 1:
        raise ValueError("smbh_mass must be a single value, not an array or range.")
    if (
        isinstance(luminosity_distance, (list, np.ndarray))
        and len(np.atleast_1d(luminosity_distance)) > 1
    ):
        raise ValueError(
            "luminosity_distance must be a single value, not an array or range."
        )
    # Handle vkick
    if vkick is not None:
        if isinstance(vkick, (list, np.ndarray)):
            vkick_arr = np.array(vkick)
        else:
            vkick_arr = np.array([vkick])
    elif vkick_range is not None:
        vmin, vmax = vkick_range
        vkick_arr = np.round(np.linspace(vmin, vmax, n_elements), 2)
    else:
        raise ValueError("You must provide either vkick or vkick_range.")

    # Handle radial_distance
    if radial_distance is not None:
        if isinstance(radial_distance, (list, np.ndarray)):
            radial_arr = np.array(radial_distance)
        else:
            radial_arr = np.array([radial_distance])
    elif radial_distance_range is not None:
        rmin, rmax = radial_distance_range
        radial_arr = np.round(np.linspace(rmin, rmax, n_elements), 2)
    else:
        raise ValueError(
            "You must provide either radial_distance or radial_distance_range."
        )

    if verbose:
        print(f"vkick array: {vkick_arr}")
        print(f"radial_distance array: {radial_arr}")

    # 1. Suppress RuntimeWarnings globally
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 2. Save log file in the main package directory (not output_dir)
    main_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(main_dir, "find_best_lc.log")

    total_iterations = len(vkick_arr) * len(radial_arr)
    results_grid = []  # Store results for later evaluation
    with open(log_path, "w") as logfile:
        original_stdout = sys.stdout
        sys.stdout = logfile
        try:
            # 3. Only create the progress bar once, outside redirected stdout
            sys.stdout = original_stdout
            with tqdm(total=total_iterations, desc="Parameter grid", ncols=80) as pbar:
                sys.stdout = logfile
                for vk in vkick_arr:
                    for rd in radial_arr:
                        result = run_multiband_analysis(
                            bh_mass=bh_mass,
                            vkick=vk,
                            radial_distance=rd,
                            luminosity_distance=luminosity_distance,
                            smbh_mass=smbh_mass,
                            bands=bands,
                            t_max=t_max,
                            time_bins=time_bins,
                            use_pagn_default=use_pagn_default,
                            gamma_j=gamma_j,
                            theta_0=theta_0,
                            kappa=kappa,
                            output_dir=output_dir,
                            save_data=save_data,
                            verbose=True,  # Now verbose output goes to log file
                        )
                        results_grid.append(
                            {"vkick": vk, "radial_distance": rd, "result": result}
                        )
                        sys.stdout = original_stdout
                        pbar.update(1)
                        sys.stdout = logfile
            sys.stdout = original_stdout
        finally:
            sys.stdout = original_stdout

    print(f"All verbose output during grid search was saved to {log_path}")
