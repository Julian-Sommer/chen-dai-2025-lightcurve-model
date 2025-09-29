"""
Utility functions for the Chen & Dai model package.

This module contains helper functions used by the CLI interface.
"""

from typing import Optional


def generate_parameter_subdir(
    bh_mass: float, 
    vkick: float, 
    radial_distance: float, 
    luminosity_distance: Optional[float] = None,
    smbh_mass: Optional[float] = None
) -> str:
    """
    Generate a parameter-specific subdirectory name for organized plot storage.
    
    Parameters
    ----------
    bh_mass : float
        Embedded BH mass [M_sun]
    vkick : float
        Kick velocity [km/s]
    radial_distance : float
        Radial distance [r_g]
    luminosity_distance : float, optional
        Luminosity distance [Mpc]
    smbh_mass : float, optional
        SMBH mass [M_sun]
        
    Returns
    -------
    str
        Directory name in format: bh{mass}_v{vkick}_r{radius}_d{lumdist}_smbh{smbhmass}
    """
    # Format numbers for directory names (avoid scientific notation issues)
    def format_for_dir(val, label):
        if val is None:
            return ""
        if val >= 1e6:
            return f"_{label}{val:.0e}".replace('+', '').replace('e0', 'e')
        elif val >= 1000:
            return f"_{label}{int(val)}"
        else:
            return f"_{label}{val:g}"
    
    # Build directory name
    dirname = f"bh{int(bh_mass)}_v{int(vkick)}_r{int(radial_distance)}"
    
    if luminosity_distance is not None:
        dirname += f"_d{int(luminosity_distance)}"
        
    if smbh_mass is not None:
        dirname += format_for_dir(smbh_mass, "smbh")
        
    return dirname


def create_plots_subdir(base_dir: str, subdir_name: str) -> str:
    """
    Create a subdirectory in the plots directory and return the full path.
    
    Parameters
    ----------
    base_dir : str
        Base directory (usually 'plots')
    subdir_name : str
        Subdirectory name
        
    Returns
    -------
    str
        Full path to the created subdirectory
    """
    import os
    full_path = os.path.join(base_dir, subdir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path