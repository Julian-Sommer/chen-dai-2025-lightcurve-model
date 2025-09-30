#!/usr/bin/env python3
"""
BBH-Counterpart Command Line Interface

This is a unified command-line interface for the Chen & Dai 2025 jet model
providing various scientific analysis functionalities.

Based on: Chen & Dai (2025), ApJ, 987, 214

Usage:
    python bbh_counterpart_cli.py <command> [options]

Commands:
    generate-multiband-lc        Generate multiband lightcurves
    find-best-lc                 Find best lightcurve based on BH and SMBH mass
    profile-comparison           Compare different density profiles  

Examples:
    # Generate multiband lightcurves
    python bbh_counterpart_cli.py generate-multiband-lc --bh_mass 150 --vkick 100 --radial_distance 1000 \
                    --luminosity_distance 300 --bands g r i J --t_max 1e6 \
                    --smbh_mass 1e8 --time_bins 5000 --use-pagn-default

Based on: Chen & Dai (2025), ApJ, 987, 214
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules for consistent Bondi-based analysis
from constants import ModelParameters, ETA_JET_DEFAULT
from physics.black_hole_physics import EmbeddedBHParameters, compute_jet_luminosity
from disk_model import create_disk_model
from utils import generate_parameter_subdir, create_plots_subdir


def calculate_jet_luminosity_bondi(args):
    """
    Calculate jet luminosity using Bondi accretion.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing BH parameters

    Returns
    -------
    dict
        Dictionary containing:
        - 'L_jet': Jet luminosity [erg/s]
        - 'params': ModelParameters object
        - 'disk': Disk model object
        - 'bondi_data': Full Bondi accretion results
        - 'local_properties': Local disk properties
    """
    print("=" * 60)
    print("BONDI ACCRETION CALCULATION")
    print("=" * 60)

    # Create consistent parameters
    params = ModelParameters.from_disk_config(
        mass_smbh=args.smbh_mass,
        use_pagn_default=getattr(args, "use_pagn_default", False),
        gamma_j=getattr(args, "gamma_j", 100.0),
        theta_0=getattr(args, "theta_0", 0.17),
        kappa=0.34,
    )

    print(f"SMBH mass: {args.smbh_mass:.1e} M_sun")
    print(f"Embedded BH mass: {args.bh_mass:.1f} M_sun")
    print(f"Kick velocity: {args.vkick:.1f} km/s")
    print(f"Radial distance: {args.radial_distance:.1f} r_g")

    # Create disk and get local properties
    disk = create_disk_model(params, params.get_disk_filename())
    rho_local, h_local, c_s_local = disk.get_local_properties_with_sound_speed(
        args.radial_distance
    )

    print(f"Local density: {rho_local:.2e} g/cm³")
    print(f"Local scale height: {h_local:.2e} cm")
    print(f"Local sound speed: {c_s_local:.2e} cm/s")

    # Calculate jet luminosity using Bondi accretion
    bh_params = EmbeddedBHParameters(
        mass_bh=args.bh_mass,
        radius_rg=args.radial_distance,
        v_kick=args.vkick,
        eta_jet=ETA_JET_DEFAULT,
    )
    bondi_data = compute_jet_luminosity(bh_params, rho_local, c_s_local)
    l_j = bondi_data["L_jet"]

    # Check for research override
    if hasattr(args, "override_lj") and args.override_lj is not None:
        print(f"\n⚠️  RESEARCH OVERRIDE ACTIVE ⚠️")
        print(f"Bondi-calculated L_j: {l_j:.2e} erg/s")
        print(f"Override L_j:         {args.override_lj:.2e} erg/s")
        print(f"Override factor:      {args.override_lj/l_j:.2f}x")
        l_j = args.override_lj
        print(f"Using override luminosity for research purposes")
    else:
        print(f"\nBondi-derived jet luminosity: {l_j:.2e} erg/s")
        print(f"Bondi accretion rate: {bondi_data['mdot_bondi']:.2e} g/s")
        print(f"Jet efficiency: {bh_params.eta_jet:.3f}")

    return {
        "L_jet": l_j,
        "params": params,
        "disk": disk,
        "bondi_data": bondi_data,
        "bh_params": bh_params,
        "local_properties": {
            "rho_local": rho_local,
            "h_local": h_local,
            "c_s_local": c_s_local,
        },
    }


def create_main_parser():
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        prog="python bbh_counterpart_cli.py",
        description="Chen & Dai 2025 BBH Jet Model Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available analysis commands", metavar="<command>"
    )

    # === MULTIBAND LIGHTCURVES ===
    generate_multiband_parser = subparsers.add_parser(
        "generate-multiband-lc",
        help="Generate comprehensive multiband lightcurves with three emission components",
    )
    add_multiband_arguments(generate_multiband_parser)

    find_best_lc_parser = subparsers.add_parser(
        "find-best-lc",
        help="Find best lightcurve out of various radial distances and kick velocities based on BH and SMBH mass",
    )
    add_find_best_lc_arguments(find_best_lc_parser)

    # === PROFILE COMPARISON ===
    profile_parser = subparsers.add_parser(
        "profile-comparison",
        help="Compare jet evolution under different density profiles",
    )
    add_profile_comparison_arguments(profile_parser)

    return parser


def add_multiband_arguments(parser):
    """Add arguments for multiband lightcurve generation."""

    # Physical parameters
    parser.add_argument(
        "--bh_mass",
        type=float,
        default=100.0,
        help="Embedded BH mass in solar masses (default: 100)",
    )
    parser.add_argument(
        "--vkick",
        type=float,
        default=100.0,
        help="Kick velocity in km/s (default: 100)",
    )
    parser.add_argument(
        "--radial_distance",
        type=float,
        default=1000.0,
        help="Launch radius in gravitational radii (default: 1000)",
    )
    parser.add_argument(
        "--luminosity_distance",
        type=float,
        default=300.0,
        help="Luminosity distance in Mpc (default: 300)",
    )
    parser.add_argument(
        "--smbh_mass",
        type=float,
        default=1e8,
        help="SMBH mass in solar masses (default: 1e8)",
    )

    # Technical parameters
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["g", "r", "i", "J"],
        choices=["u", "g", "r", "i", "z", "J", "H", "K", "B", "V", "R", "I"],
        help="Photometric bands to analyze",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=1e6,
        help="Maximum evolution time in seconds (default: 1e6)",
    )
    parser.add_argument(
        "--time_bins",
        type=int,
        default=5000,
        help="Number of time steps (default: 5000)",
    )

    # Disk options
    parser.add_argument(
        "--use-pagn-default",
        action="store_true",
        help="Use PAGN default parameters (Sirko & Goodman 2003)",
    )
    parser.add_argument(
        "--le", type=float, default=None, help="Eddington ratio (if not using defaults)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha viscosity parameter (if not using defaults)",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Output directory for plots and data (default: ./plots)",
    )
    parser.add_argument(
        "--save_data", action="store_true", default=True, help="Save data to CSV files"
    )


def add_find_best_lc_arguments(parser):
    """Add arguments for finding the best multiband lightcurve."""

    # Physical parameters
    parser.add_argument(
        "--bh_mass",
        type=float,
        default=100.0,
        help="Embedded BH mass in solar masses (default: 100)",
    )
    parser.add_argument(
        "--vkick",
        type=float,
        default=None,
        help="Kick velocity in km/s",
    )
    parser.add_argument(
        "--radial_distance",
        type=float,
        default=None,
        help="Launch radius in gravitational radii",
    )
    parser.add_argument(
        "--luminosity_distance",
        type=float,
        default=300.0,
        help="Luminosity distance in Mpc (default: 300)",
    )
    parser.add_argument(
        "--smbh_mass",
        type=float,
        default=1e8,
        help="SMBH mass in solar masses (default: 1e8)",
    )

    # Technical parameters
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["g", "r", "i", "J"],
        choices=["u", "g", "r", "i", "z", "J", "H", "K", "B", "V", "R", "I"],
        help="Photometric bands to analyze",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=1e6,
        help="Maximum evolution time in seconds (default: 1e6)",
    )
    parser.add_argument(
        "--time_bins",
        type=int,
        default=5000,
        help="Number of time steps (default: 5000)",
    )

    # Disk options
    parser.add_argument(
        "--use-pagn-default",
        action="store_true",
        help="Use PAGN default parameters (Sirko & Goodman 2003)",
    )
    parser.add_argument(
        "--le", type=float, default=None, help="Eddington ratio (if not using defaults)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha viscosity parameter (if not using defaults)",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Output directory for plots and data (default: ./plots)",
    )
    parser.add_argument(
        "--save_data", action="store_true", default=True, help="Save data to CSV files"
    )

    parser.add_argument(
        "--vkick_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("VKICK_MIN", "VKICK_MAX"),
        help="Range of vkick values (min max) in km/s",
    )
    parser.add_argument(
        "--radial_distance_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("RMIN", "RMAX"),
        help="Range of radial distance values (min max) in r_g",
    )
    parser.add_argument(
        "--n_elements",
        type=int,
        default=5,
        help="Number of elements for vkick/radial_distance arrays",
    )


def add_profile_comparison_arguments(parser):
    """Add arguments for profile comparison."""

    # Physical parameters (match generate-multiband-lc)
    parser.add_argument(
        "--bh_mass",
        type=float,
        default=100.0,
        help="Embedded BH mass in solar masses (default: 100)",
    )
    parser.add_argument(
        "--vkick",
        type=float,
        default=100.0,
        help="Kick velocity in km/s (default: 100)",
    )
    parser.add_argument(
        "--radial_distance",
        type=float,
        default=1000.0,
        help="Launch radius in gravitational radii (default: 1000)",
    )
    parser.add_argument(
        "--smbh_mass",
        type=float,
        default=1e8,
        help="SMBH mass in solar masses (default: 1e8)",
    )
    parser.add_argument(
        "--gamma_j", type=float, default=100.0, help="Jet Lorentz factor (default: 100)"
    )
    parser.add_argument(
        "--theta_0",
        type=float,
        default=0.17,
        help="Initial jet opening angle in radians (default: 0.17)",
    )
    parser.add_argument(
        "--t_min",
        type=float,
        default=10,
        help="Minimum evolution time in seconds (default: 10)",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=1e4,
        help="Maximum evolution time in seconds (default: 1e4)",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=500,
        help="Number of time steps (default: 500)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="profile_comparison.png",
        help="Output plot filename",
    )


# ============================================================================
# COMMAND IMPLEMENTATIONS
# ============================================================================


def run_generate_multiband_lc(args):
    """Run multiband lightcurve analysis."""
    from multiband_analysis import run_multiband_analysis

    # Run the analysis with proper parameters
    results = run_multiband_analysis(
        bh_mass=args.bh_mass,
        vkick=args.vkick,
        radial_distance=args.radial_distance,
        luminosity_distance=args.luminosity_distance,
        smbh_mass=args.smbh_mass,
        bands=args.bands,
        t_max=args.t_max,
        time_bins=args.time_bins,
        use_pagn_default=args.use_pagn_default,
        gamma_j=100.0,
        theta_0=0.17,
        kappa=0.34,
        output_dir=args.output_dir,
        save_data=args.save_data,
        verbose=True,
    )

    print("\nMultiband lightcurve analysis completed!")
    return results


def run_find_best_lc(args):
    """Find the best lightcurve based on BH and SMBH mass."""
    from find_best_lc import run_find_best_lc

    # Run the analysis with proper parameters
    run_find_best_lc(
        bh_mass=args.bh_mass,
        vkick=args.vkick,
        radial_distance=args.radial_distance,
        luminosity_distance=args.luminosity_distance,
        smbh_mass=args.smbh_mass,
        bands=args.bands,
        t_max=args.t_max,
        time_bins=args.time_bins,
        use_pagn_default=args.use_pagn_default,
        gamma_j=100.0,
        theta_0=0.17,
        kappa=0.34,
        output_dir=args.output_dir,
        save_data=args.save_data,
        verbose=True,
        vkick_range=args.vkick_range,
        radial_distance_range=args.radial_distance_range,
        n_elements=args.n_elements,
    )

    return None


def run_profile_comparison(args):
    """Run profile comparison analysis using Bondi accretion."""
    print("=" * 60)
    print("DENSITY PROFILE COMPARISON (Bondi-Based)")
    print("=" * 60)

    # matplotlib imports removed - using plotting module functions
    from pathlib import Path

    # Calculate Bondi-based jet luminosity
    bondi_results = calculate_jet_luminosity_bondi(args)
    l_j = bondi_results["L_jet"]
    params = bondi_results["params"]

    print(f"Model parameters: {params}")
    print(f"Jet luminosity: {l_j:.2e} erg/s")

    # Time range for comparison
    times = np.logspace(1, 4, 500)  # 10 to 10,000 seconds

    # Create model
    from evolution import ChenDaiModel

    model = ChenDaiModel(
        l_j=l_j,
        params=params,
        disk_filename=params.get_disk_filename(),
        disk_radius_rg=args.radial_distance,
    )

    results = {}
    profiles = ["uniform", "isothermal", "polytropic"]

    for profile in profiles:
        print(f"\\n{'='*20} {profile.upper()} PROFILE {'='*20}")

        # Run evolution with profile-specific logic
        result = model.evolve(
            times,
            density_profile=profile,
            progress_interval=100,
            respect_profile_limits=True,
        )

        results[profile] = result

    # Create parameter-specific subdirectory for plots
    import os

    plots_base_dir = os.path.join(".", "plots")
    subdir_name = generate_parameter_subdir(
        bh_mass=args.bh_mass, vkick=args.vkick, radial_distance=args.radial_distance
    )
    plots_dir = create_plots_subdir(plots_base_dir, subdir_name)
    output_path = os.path.join(plots_dir, "profile_comparison.png")

    # Create comparison plot
    create_profile_comparison_plot(results, output_path)

    # Print summary comparison
    print(f"\\n{'='*60}")
    print("PROFILE COMPARISON SUMMARY")
    print(f"{'='*60}")

    for profile in profiles:
        result = results[profile]
        final_zh_h = result.zh_over_h[-1]
        final_time = result.times[-1]
        n_steps = len(result.times)

        print(
            f"{profile.upper():>12}: "
            f"z_h/h = {final_zh_h:.3f}, "
            f"t_final = {final_time:.2e} s, "
            f"steps = {n_steps}"
        )

    print(f"\\nProfile comparison completed!")


def create_profile_comparison_plot(results, filename):
    """Create comparison plot for different profiles using plotting module."""
    from plotting import plot_profile_comparison
    from pathlib import Path

    # Create output directory
    Path(filename).parent.mkdir(exist_ok=True)

    # Use plotting module
    plot_profile_comparison(results, filename)

    # Print detailed info about what's plotted
    print("\\nPlot details:")
    for profile, result in results.items():
        print(
            f"  {profile:>10}: {len(result.times):3d} points, "
            f"t = {result.times[0]:.0f} - {result.times[-1]:.0f} s, "
            f"final z_h/h = {result.zh_over_h[-1]:.3f}"
        )
    print("\\nNote: Lines may overlap when profiles behave similarly!")
    print("      Different line styles help distinguish overlapping curves.")


# ============================================================================
# MAIN CLI ENTRY POINT
# ============================================================================


def main():
    """Main CLI entry point."""

    parser = create_main_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Route to appropriate command function
    command_functions = {
        "generate-multiband-lc": run_generate_multiband_lc,
        "find-best-lc": run_find_best_lc,
        "profile-comparison": run_profile_comparison,
    }

    try:
        command_functions[args.command](args)
        return 0
    except KeyboardInterrupt:
        print("\\n\\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\\nError running {args.command}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
