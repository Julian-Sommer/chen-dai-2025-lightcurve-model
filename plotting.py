"""
Visualization and plotting utilities for the Chen & Dai model.

This module provides comprehensive plotting capabilities for analyzing
and presenting jet evolution results.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Union


# Configure matplotlib for publication-quality plots with LaTeX fallback
def _configure_matplotlib():
    """Configure matplotlib with LaTeX support and fallback to serif fonts."""
    try:
        # Check if LaTeX is available
        import subprocess

        result = subprocess.run(
            ["latex", "--version"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            mpl.rc("text", usetex=True)
            mpl.rc("font", family="serif", size=12)
            mpl.rc("text.latex", preamble=r"\usepackage{amsmath}")
            mpl.rc("figure", dpi=100)
            print("Using LaTeX for text rendering")
            return True

    except Exception as e:
        print(f"LaTeX not available ({e}), using serif fonts")

    # Fallback to serif fonts if LaTeX is not available
    mpl.rc("text", usetex=False)
    mpl.rc("font", family="serif", size=12)
    mpl.rc("figure", dpi=100)
    print("Using serif fonts")
    return False


# Configure matplotlib on import
_LATEX_AVAILABLE = _configure_matplotlib()


# ============================================================================
# MULTIBAND AB MAGNITUDE PLOTTING FUNCTIONS
# ============================================================================


def plot_multiband_ab_magnitudes(
    times,
    band_results,
    bands,
    t_break,
    save_path,
    title=r"Multiband AB Magnitudes",
    figsize=(8, 5),
    dpi=300,
):
    """
    Plot multiband AB magnitude lightcurves with consistent LaTeX styling.

    Parameters
    ----------
    times : array
        Time array in seconds
    band_results : dict
        Dictionary with band names as keys, containing 'ab_mag_total' arrays
    bands : list
        List of band names to plot
    t_break : float
        Shock breakout time [s]
    duration_emission : float
        Duration of jet-head emission [s]
    save_path : str
        Path to save the figure
    title : str
        Plot title (LaTeX-formatted)
    figsize : tuple
        Figure size
    dpi : int
        Figure DPI

    Returns
    -------
    str
        Path to saved figure
    """
    # Set up LaTeX styling consistent with other plots

    fig, ax = plt.subplots(figsize=figsize)

    # Get colors from matplotlib color cycle
    colors = (
        mpl.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    )

    # Plot each band
    for i, band in enumerate(bands):
        color = colors[i % len(colors)]
        mags = band_results[band]["ab_mag_total"]
        ax.plot(times, mags, label=f"{band}", color=color, linewidth=2.0)

    # Formatting consistent with other plots
    ax.set_xscale("log")
    ax.set_xlabel("Time [s]", fontsize=14)
    ax.set_ylabel("AB magnitude", fontsize=14)
    ax.invert_yaxis()

    # Add breakout time marker
    ax.axvline(
        t_break,
        color="k",
        ls="--",
        alpha=0.7,
        lw=1.5,
        label=r"$t_{\mathrm{break}}$",
    )

    # Calculate y-limits as specified
    idx_break_interp = np.argmin(np.abs(times - t_break))
    mag_at_breakout = np.min(
        [band_results[b]["ab_mag_total"][idx_break_interp] for b in bands]
    )
    peak_mag = np.min([np.min(band_results[b]["ab_mag_total"]) for b in bands])

    ax.set_ylim(
        mag_at_breakout + 3,  # minimum y-lim: magnitude at breakout + 3
        peak_mag - 1,  # maximum y-lim: peak (minimum magnitude value)
    )

    # Legend styling without grid
    ax.legend(frameon=True, framealpha=0.9, fontsize=12, loc="best")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Save with consistent settings
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Multiband AB magnitudes plot saved to {save_path}")
    return save_path


def plot_multiband_ab_magnitudes_linear_days(
    times,
    band_results,
    bands,
    t_break,
    save_path,
    title=r"Multiband AB Magnitudes",
    figsize=(8, 5),
    dpi=300,
):
    """
    Plot multiband AB magnitude lightcurves with linear time scale in days.

    Parameters
    ----------
    times : array
        Time array in seconds
    band_results : dict
        Dictionary with band names as keys, containing 'ab_mag_total' arrays
    bands : list
        List of band names to plot
    t_break : float
        Shock breakout time [s]
    duration_emission : float
        Duration of jet-head emission [s]
    save_path : str
        Path to save the figure
    title : str
        Plot title (LaTeX-formatted)
    figsize : tuple
        Figure size
    dpi : int
        Figure DPI
    mag_key : str
        Key in band_results to use for magnitudes (default: 'ab_mag_total')

    Returns
    -------
    str
        Path to saved figure
    """
    # Set up LaTeX styling consistent with other plots

    fig, ax = plt.subplots(figsize=figsize)

    # Convert time to days
    SECONDS_PER_DAY = 86400.0
    times_days = times / SECONDS_PER_DAY
    t_break_days = t_break / SECONDS_PER_DAY

    # Get colors from matplotlib color cycle
    colors = (
        mpl.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    )

    # Plot each band
    for i, band in enumerate(bands):
        color = colors[i % len(colors)]
        mags = band_results[band]["ab_mag_total"]
        ax.plot(times_days, mags, label=f"{band}", color=color, linewidth=2.0)

    # Formatting with linear scale and days
    ax.set_xlabel("Time [days]", fontsize=14)
    ax.set_ylabel("AB magnitude", fontsize=14)
    ax.invert_yaxis()

    # Add breakout time marker
    ax.axvline(
        t_break_days,
        color="k",
        ls="--",
        alpha=0.7,
        lw=1.5,
        label=r"$t_{\mathrm{break}}$",
    )

    # Calculate y-limits as specified
    idx_break_interp = np.argmin(np.abs(times - t_break))
    mag_at_breakout = np.min(
        [band_results[b]["ab_mag_total"][idx_break_interp] for b in bands]
    )
    peak_mag = np.min([np.min(band_results[b]["ab_mag_total"]) for b in bands])

    y_min = mag_at_breakout + 3  # minimum y-lim: magnitude at breakout + 3
    y_max = peak_mag - 1  # maximum y-lim: peak (minimum magnitude value)

    ax.set_ylim(y_min, y_max)

    # Find the time when the lower y-limit (y_min) is reached and set x-limit
    # Find the last time any band reaches the y_min magnitude
    max_time_at_y_min_days = 0
    for band in bands:
        mags = band_results[band]["ab_mag_total"]
        indices = np.where(mags <= y_min)[0]
        if len(indices) > 0:
            last_index = indices[-1]
            time_at_y_min_days = times_days[last_index]
            max_time_at_y_min_days = max(max_time_at_y_min_days, time_at_y_min_days)

    # Set x-limit to that time + 1 day
    if max_time_at_y_min_days > 0:
        ax.set_xlim(left=times_days[0], right=max_time_at_y_min_days + 1)

    # Legend styling without grid
    ax.legend(frameon=True, framealpha=0.9, fontsize=12, loc="best")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Save with consistent settings
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Multiband AB magnitudes (linear days) plot saved to {save_path}")
    return save_path


def plot_multiband_ab_magnitudes_with_agn(
    times,
    band_results,
    bands,
    t_break,
    save_path,
    title=r"Multiband AB Magnitudes (Transient + AGN)",
    figsize=(8, 5),
    dpi=300,
    mag_key="ab_mag_total_agn",
    agn_label="AGN+Transient",
    agn_color="black",
    agn_linestyle="-",
    agn_alpha=0.8,
    show_legend=True,
    xlim_days=None,
    ylim=None,
    use_log_scale=False,
):
    """
    Plot multiband AB magnitude lightcurves for the sum of transient and AGN disk.

    Parameters
    ----------
    times : array
        Time array in seconds
    band_results : dict
        Dictionary with band names as keys, containing magnitude arrays
    bands : list
        List of band names to plot
    save_path : str
        Path to save the figure
    title : str
        Plot title (LaTeX-formatted)
    figsize : tuple
        Figure size
    dpi : int
        Figure DPI
    mag_key : str
        Key in band_results to use for magnitudes (default: 'ab_mag_total_agn')
    agn_label : str
        Label for AGN+transient curve
    agn_color : str
        Color for AGN+transient curve
    agn_linestyle : str
        Linestyle for AGN+transient curve
    agn_alpha : float
        Alpha for AGN+transient curve
    show_legend : bool
        Whether to show legend
    xlim_days : tuple or None
        (xmin, xmax) in days for x-axis
    ylim : tuple or None
        (ymin, ymax) for y-axis
    use_log_scale : bool
        Whether to use logarithmic x-axis (default: False for linear scale)

    Returns
    -------
    str
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    SECONDS_PER_DAY = 86400.0
    times_days = times / SECONDS_PER_DAY
    t_break_days = t_break / SECONDS_PER_DAY

    # Get colors from matplotlib color cycle
    colors = (
        mpl.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])
    )

    # Add breakout time marker
    ax.axvline(
        t_break_days,
        color="k",
        ls="--",
        alpha=0.7,
        lw=1.5,
        label=r"$t_{\mathrm{break}}$",
    )

    # Plot each band
    for i, band in enumerate(bands):
        color = colors[i % len(colors)]
        mags = band_results[band][mag_key]
        ax.plot(times_days, mags, label=f"{band}", color=color, linewidth=2.0)

    # Set x-axis scale based on use_log_scale parameter
    if use_log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("Time [days]", fontsize=14)
    else:
        ax.set_xlabel("Time [days]", fontsize=14)

    ax.set_ylabel("AB magnitude", fontsize=14)
    ax.invert_yaxis()

    # Handle x-limits based on scale type
    if use_log_scale:
        # For log scale, use time range similar to other log plots
        if xlim_days is not None:
            ax.set_xlim(*xlim_days)
        # Don't auto-compute xlim for log scale as it can be tricky
    else:
        # Sensible x-lim for linear scale: find the earliest time after the minimum where n consecutive identical magnitudes occur in any band
        n_identical = 3
        min_x_limit = times_days[-1]
        for band in bands:
            mags = band_results[band][mag_key]
            # Find the index of the minimum magnitude
            min_index = np.argmin(mags)
            # Check for n consecutive identical values after the minimum
            for i in range(min_index + 1, len(mags) - n_identical + 1):
                if np.all(mags[i : i + n_identical] == mags[i]):
                    min_x_limit = min(min_x_limit, times_days[i] + 1)  # +1 day buffer
                    break

        # Set xlim: use user-provided xlim_days if given, else use the computed min_x_limit
        if xlim_days is not None:
            ax.set_xlim(*xlim_days)
        else:
            ax.set_xlim(left=times_days[0], right=min_x_limit)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_legend:
        ax.legend(frameon=True, framealpha=0.9, fontsize=12, loc="best")
    ax.set_title(title, fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    scale_type = "logarithmic" if use_log_scale else "linear"
    print(
        f"Multiband AB magnitudes with AGN component ({scale_type} scale) plot saved to {save_path}"
    )
    return save_path


def plot_profile_comparison(
    results_dict,
    save_path,
    title="Chen and Dai 2025: Density Profile Comparison",
    figsize=(12, 10),
    dpi=300,
):
    """
    Create density profile comparison plot.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping profile names to EvolutionResults
    save_path : str
        Path to save the figure
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (default: (12, 10))
    dpi : int, optional
        Figure DPI (default: 300)

    Returns
    -------
    str
        Path to saved figure
    """
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path

    # Create output directory
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Check LaTeX availability for proper labels

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = {"uniform": "red", "isothermal": "blue", "polytropic": "green"}
    linestyles = {"uniform": "-", "isothermal": "--", "polytropic": "-."}

    # Plot 1: β_h vs time
    ax = axes[0, 0]
    for profile, result in results_dict.items():
        ax.loglog(
            result.times,
            result.beta_h,
            color=colors[profile],
            linestyle=linestyles[profile],
            label=f"{profile} (n={len(result.times)})",
            linewidth=2,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\beta_h$")
    ax.set_title("Jet Head Velocity")
    ax.legend()

    # Plot 2: z_h/h vs time
    ax = axes[0, 1]
    for profile, result in results_dict.items():
        ax.loglog(
            result.times,
            result.zh_over_h,
            color=colors[profile],
            linestyle=linestyles[profile],
            label=f"{profile} (n={len(result.times)})",
            linewidth=2,
        )
    ax.axhline(y=1, color="black", linestyle=":", alpha=0.7, label="z_h/h = 1")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$z_h/h$")
    ax.set_title("Normalized Jet Height")
    ax.legend()

    # Plot 3: Cocoon energy vs time
    ax = axes[1, 0]
    for profile, result in results_dict.items():
        ax.loglog(
            result.times,
            result.e_c,
            color=colors[profile],
            linestyle=linestyles[profile],
            label=f"{profile} (n={len(result.times)})",
            linewidth=2,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$E_c$ [erg]")
    ax.set_title("Cocoon Energy")
    ax.legend()

    # Plot 4: Opening angle vs time
    ax = axes[1, 1]
    for profile, result in results_dict.items():
        ax.loglog(
            result.times,
            result.theta_h,
            color=colors[profile],
            linestyle=linestyles[profile],
            label=f"{profile} (n={len(result.times)})",
            linewidth=2,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\theta_h$ [rad]")
    ax.set_title("Jet Opening Angle")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return save_path
