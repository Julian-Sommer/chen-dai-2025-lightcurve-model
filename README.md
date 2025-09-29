# Chen & Dai 2025 Jet Model

A Python package for modeling jet-head shock breakout emission from embedded black holes in supermassive black hole accretion disks, following the Chen & Dai (2025) model.

## Overview

This package simulates the emission signatures of massive stellar mass black holes (BHs) embedded in supermassive black hole (SMBH) accretion disks. The model computes:

- **Jet-head breakout emission** from relativistic jets breaking out of the disk
- **Disk cocoon emission** from shocked disk material
- **Jet-cocoon emission** from the relativistic jet itself
- **Multiband lightcurves** across optical and near-infrared bands

## Installation

```bash
# Clone the repository
git clone https://github.com/Juli-Sommer/chen-dai-2025-jet-model.git
cd chen-dai-2025-jet-model

# Install dependencies (requires Python 3.8+)
pip install numpy scipy matplotlib astropy
```

## Quick Start

### Command Line Interface

The package provides a unified CLI for all analysis modes:

```bash
# Generate multiband lightcurves
python bbh_counterpart_cli.py generate-multiband-lc \
    --bh_mass 150 --vkick 100 --radial_distance 1000 \
    --luminosity_distance 300 --smbh_mass 1e8 --bands g r i
```

## Core Commands

### 1. Multiband Lightcurves (`generate-multiband-lc`)

Generate complete multiband lightcurves with all emission components:

```bash
python bbh_counterpart_cli.py generate-multiband-lc \
    --bh_mass 150 \              # Embedded BH mass [M_sun]
    --vkick 100 \                # Kick velocity [km/s]
    --radial_distance 1000 \     # Distance from SMBH [r_g]
    --luminosity_distance 300 \  # Observer distance [Mpc]
    --smbh_mass 1e8 \           # SMBH mass [M_sun]
    --bands g r i J \           # Photometric bands
    --t_max 1e6 \               # Maximum evolution time [s]
    --time_bins 5000 \          # Time resolution
    --use-pagn-default          # Optional: use PAGN disk parameters
```

**Outputs:**
- `multiband_ab_magnitudes.png` - AB magnitude lightcurves
- `multiband_ab_magnitudes_linear_days.png` - AB magnitudes (linear time scale)
- `multiband_lc_t{time_bins}_b{bands}.csv` - Numerical results (filtered, non-zero only)

### 2. Optical Depth Analysis (`optical-depth`)

Study breakout conditions across different density profiles:

```bash
python bbh_counterpart_cli.py optical-depth \
    --bh_mass 100 \
    --vkick 100 \
    --radial_distance 1000 \
    --density_profile all \     # or 'gaussian', 'exponential', 'power_law'
    --t_max 1e4
```

### Fixing Jet Luminosity

For specific analysis needs, you can override the Bondi-calculated jet luminosity:

```bash
python bbh_counterpart_cli.py generate-multiband-lc \
    --bh_mass 150 --vkick 100 --radial_distance 1000 \
    --override_lj 1e46         # Override L_j [erg/s]
```

**Note:** This bypasses the default physics calculation and uses the specified luminosity value.

## Data Storage Structure

```
package_github/
├── data/                           # Output data files
│   └── multiband_lc/              # Multiband lightcurve data
│       └── bh150_v100_r1000_d300_smbh1e8/
│           └── multiband_lc_t5000_bg_r_i.csv
├── plots/                          # Output plots (organized by parameters)
│   └── bh150_v100_r1000_d300_smbh1e8/
│       ├── multiband_ab_magnitudes.png
│       └── multiband_ab_magnitudes_linear_days.png
├── agn_disks/                      # AGN disk models
└── physics/                        # Physics modules
```

### Output Files

- **CSV files:** Numerical lightcurve data in `data/multiband_lc/` with parameter-specific subdirectories (zero-flux rows filtered out)
- **PNG plots:** AB magnitude visualizations in parameter-specific subdirectories in `plots/`
- **Terminal output:** Key parameters and breakout times displayed

## Model Parameters

### Default Disk Configurations

**Chen & Dai defaults:**
- Eddington ratio: `l_e = 0.1`
- Viscosity parameter: `α = 0.1`

**PAGN defaults (use `--use-pagn-default`):**
- Eddington ratio: `l_e = 0.5`
- Viscosity parameter: `α = 0.01`

### Key Physical Parameters

- **Jet opening angle:** `θ_0 = 0.17` rad (≈10°)
- **Lorentz factor:** `γ_j = 100`
- **Jet efficiency:** `η_j = 0.1`
- **Opacity:** `κ = 0.34` cm²/g

### Available Photometric Bands

The model supports the following photometric bands:

**Optical bands:**
- `g` - SDSS g-band (4770 Å)
- `r` - SDSS r-band (6231 Å)  
- `i` - SDSS i-band (7625 Å)
- `z` - SDSS z-band (9134 Å)

**Near-infrared bands:**
- `J` - 2MASS J-band (12350 Å)
- `H` - 2MASS H-band (16620 Å)
- `K` - 2MASS K-band (21590 Å)

**Usage example:**
```bash
--bands g r i z J H K    # All available bands
--bands g r i            # Optical only
--bands J H K            # Near-infrared only
```

## Troubleshooting

### Common Issues

1. **ImportError:** Ensure all dependencies are installed
2. **Slow evolution:** Reduce `--time_bins` for faster computation
3. **No breakout found:** Increase `--t_max` or check jet parameters
4. **Memory issues:** Reduce time resolution for large parameter studies

### Performance Tips

- Use `--time_bins 1000` for quick tests
- Use `--time_bins 5000` for publication-quality results
- Parameter studies automatically optimize time resolution

### Physics Validation

The code automatically validates physics consistency:
- Reports Bondi accretion rates
- Compares override values to calculated values
- Warns about unphysical parameter combinations
