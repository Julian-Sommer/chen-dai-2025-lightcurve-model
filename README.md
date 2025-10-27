# Chen & Dai 2025 Lightcurve Model

A Python package for modeling jet-head shock breakout emission from embedded black holes in supermassive black hole accretion disks, following the Chen & Dai (2025) model ([ApJ, 987, 214](https://iopscience.iop.org/article/10.3847/1538-4357/addb48)).

## Overview

This package simulates the emission signatures of massive stellar mass black holes (BHs) embedded in supermassive black hole (SMBH) accretion disks. The model computes:

- **Jet-head breakout emission** from relativistic jets breaking out of the disk
- **Disk cocoon emission** from shocked disk material
- **Jet-cocoon emission** from the relativistic jet itself
- **Multiband lightcurves** across optical and near-infrared bands

## Installation

```bash
# Clone the repository
git clone https://github.com/Juli-Sommer/chen-dai-2025-lightcurve-model.git
cd chen-dai-2025-lightcurve-model

# Install dependencies (requires Python 3.8+)
pip install pagn numpy scipy matplotlib astropy tqdm
```

## Core Commands

### 1. Multiband Lightcurves (`generate-multiband-lc`)

Generate complete multiband lightcurves with all emission components:

```bash
python bbh_counterpart_cli.py generate-multiband-lc \
    --bh_mass 150 \
    --vkick 100 \
    --radial_distance 1000 \
    --luminosity_distance 300 \
    --smbh_mass 1e8 \
    --bands g r i J \
    --t_max 1e6 \
    --time_bins 5000 \
    --use-pagn-default
```

**Outputs:**
- `multiband_ab_magnitudes_{disk}_{tdiff}.png` - AB magnitude lightcurves (log scale)
- `multiband_ab_magnitudes_linear_days_{disk}_{tdiff}.png` - AB magnitudes (linear time scale)
- `multiband_ab_magnitudes_with_agn_linear_days_{disk}_{tdiff}.png` - Transient + AGN (linear scale)
- `multiband_ab_magnitudes_with_agn_log_days_{disk}_{tdiff}.png` - Transient + AGN (log scale)
- `multiband_lc_t{time_bins}_b{bands}.csv` - Numerical results (filtered, non-zero only)

Where `{disk}` is either `cd_disk` (Chen & Dai) or `pagn_disk` (PAGN defaults), and `{tdiff}` is either `paper_tdiff` (default) or `chat_tdiff` (Chatzopoulos formula).


### 2. Best Lightcurve Selection (`find-best-lc`)

Automate parameter grid searches and identify the best lightcurve (the parameter combination with the strongest transient–AGN contrast) for your chosen bands.

**How It Works:**
- Runs a grid of models over `vkick` and `radial_distance` (either as single values or ranges).
- For each parameter combination, generates multiband lightcurves and saves results to CSV and plots.
- After the grid search, analyzes only the CSVs generated in the current run.
- For the bluest band (shortest wavelength in your selection), computes the difference between the combined (transient+AGN) and AGN-only magnitude at each time step.
- Finds the minimum (brightest) difference for each combination and reports the top 3 parameter sets with the strongest transient–AGN contrast.
- Prints the CSV and plot file paths for the best results.

**Example Usage:**

*Grid Search with Ranges (limits):*
```bash
python bbh_counterpart_cli.py find-best-lc \
    --bh_mass 150 \
    --vkick_range 50 200 \
    --radial_distance_range 500 2000 \
    --luminosity_distance 300 \
    --smbh_mass 1e8 \
    --bands g r i z J H K \
    --n_elements 7 \
    --t_max 1e7 \
    --time_bins 5000
```
This will sweep `vkick` from 50 to 200 and `radial_distance` from 500 to 2000, using 7 steps for each parameter (default: 5 if not specified).

*Single Value Search (only one of `vkick` or `radial_distance` should be a single value; the other must be a range):*

```bash

# Example: vkick is a single value, radial_distance is a range
python bbh_counterpart_cli.py find-best-lc \
    --bh_mass 150 \
    --vkick 100 \
    --radial_distance_range 500 2000 \
    --luminosity_distance 300 \
    --smbh_mass 1e8 \
    --bands g r i \
    --n_elements 7 \
    --t_max 1e7 \
    --time_bins 5000

# Example: radial_distance is a single value, vkick is a range
python bbh_counterpart_cli.py find-best-lc \
    --bh_mass 150 \
    --vkick_range 50 200 \
    --radial_distance 1000 \
    --luminosity_distance 300 \
    --smbh_mass 1e8 \
    --bands g r i \
    --n_elements 7 \
    --t_max 1e7 \
    --time_bins 5000
```
> **Note:** A single value search only makes sense if *either* `vkick` *or* `radial_distance` is a single value, not both. If both are single values, only one model will be run and the grid search logic is not used.
This will run a single model for the specified parameters.


**Outputs:**
- Prints the top 3 parameter combinations with the strongest transient–AGN contrast in the bluest band.
- For each, prints the CSV file path and associated plot paths.
- All output files are organized in parameter-specific subdirectories under `data/multiband_lc/` and `plots/`.
- A log file `find_best_lc.log` is generated, containing verbose output from each simulation. If you want to check detailed progress or debug issues, refer to this log file.

## Physical Parameter Insights

- **vkick:** The lowest `vkick` value always returns the strongest signature in the lightcurve, because the jet luminosity $L_j$ directly scales with the Bondi accretion rate, which increases as `vkick` decreases.
- **SMBH mass:** The smaller the SMBH mass, the stronger the transient–AGN contrast. However, this also results in a lower overall magnitude in redder bands and causes shorter transient emission durations.

## Data Storage Structure

```
package_github/
├── data/                           # Output data files
│   └── multiband_lc/              # Multiband lightcurve data
│       └── bh150_v100_r1000_d300_smbh1e8/
│           └── multiband_lc_t5000_bg_r_i.csv
├── plots/                          # Output plots (organized by parameters)
│   └── bh150_v100_r1000_d300_smbh1e8/
│       ├── multiband_ab_magnitudes_cd_disk_paper.png
│       ├── multiband_ab_magnitudes_linear_days_cd_disk_paper.png
│       ├── multiband_ab_magnitudes_with_agn_linear_days_cd_disk_paper.png
│       └── multiband_ab_magnitudes_with_agn_log_days_cd_disk_paper.png
├── agn_disks/                      # AGN disk models
└── physics/                        # Physics modules
```

### Output Files

- **CSV files:** Numerical lightcurve data in `data/multiband_lc/` with parameter-specific subdirectories (zero-flux rows filtered out)
- **PNG plots:** AB magnitude visualizations in parameter-specific subdirectories in `plots/`
- **Terminal output:** Key parameters and breakout times displayed
- **Logfile:** When using find-best-lc an output file will be generated

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

### Diffusion Timescale Options

The disk cocoon diffusion timescale can be calculated using two different formulas:

- **Default (Chen & Dai 2025):** Uses the paper's approximation from Equation 26
- **Chatzopoulos et al. (2012):** Uses `√(t_c,0 × t_s,0)` formula with `--use-chatzopoulos-tdiff`

Note: The paper's approximation has been found to differ from the Chatzopoulos formula. The default preserves the paper's original implementation.

## Troubleshooting

### Common Issues

1. **ImportError:** Ensure all dependencies are installed
2. **Slow evolution:** Reduce `--time_bins` for faster computation
3. **No breakout found:** Increase `--t_max` or check jet parameters
4. **Memory issues:** Reduce time resolution for large parameter studies

### Performance Tips

- Use `--time_bins 1000` for quick tests
- Use `--time_bins 5000` for higher quality results

## Acknowledgements & Development Notes

This package provides a numerical implementation of the light curve model presented in **Chen & Dai (2025)**.  
The goal of this work is to make the equations and methods from that paper easily accessible for numerical studies and model comparison.  
The implementation follows the original formulation as closely as possible, with minor extensions introduced by the author to improve applicability and flexibility.

During development, **GitHub Copilot** was used as an auxiliary tool to assist with code development and documentation.  
All suggestions were reviewed by the author, and the final implementation reflects the author's scientific and technical judgment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite both the software and the underlying paper (Chen & Dai 2025). See [CITATION.cff](CITATION.cff) for citation details, or use the "Cite this repository" button on GitHub.
