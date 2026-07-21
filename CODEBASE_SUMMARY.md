# TXPipe Codebase Summary

## What is TXPipe?

TXPipe is the **Dark Energy Science Collaboration (DESC) 3×2pt Pipeline** — a modular, MPI-parallel Python framework for measuring weak gravitational lensing and galaxy clustering statistics from imaging survey data. It transforms raw galaxy catalogs into cosmological two-point measurements ready for dark energy analysis.

The name "3×2pt" refers to three combinations of two-point correlation functions:
- **Cosmic shear** (shape × shape): ξ+(θ), ξ-(θ) / C_ell^EE
- **Galaxy–galaxy lensing** (shape × position): γ_t(θ) / C_ell^Eκ
- **Galaxy clustering** (position × position): ω(θ) / C_ell^κκ

TXPipe is built on the **Ceci** workflow framework, which manages stage dependencies as a DAG and supports local (mini), Parsl, and CWL launchers for distributed execution.

---

## Repository Layout

```
TXPipe/
├── txpipe/                  # Main Python package (~50 modules)
│   ├── __init__.py          # Registers all pipeline stages
│   ├── base_stage.py        # PipelineStage base class
│   ├── data_types.py        # HDF5/FITS file type definitions
│   ├── ingest/              # Data import (GCR, FITS, mocks, LSST, Rubin DP0/DP1)
│   ├── source_selection/    # WL source selection (metacal, metadetect, lensfit, HSC)
│   ├── lens_selector.py     # Lens sample selection
│   ├── calibrate.py         # Apply m/c shear bias corrections
│   ├── shear_calibration/   # Compute shear bias per tomographic bin
│   ├── photoz_stack.py      # Photo-z n(z) stacking
│   ├── rail/                # RAIL photo-z framework integration
│   ├── binning/             # Tomographic bin assignment
│   ├── maps.py              # Shear (g1,g2) and density HEALPix maps
│   ├── auxiliary_maps.py    # PSF, flag, depth systematic maps
│   ├── masks.py             # Survey mask generation
│   ├── noise_maps.py        # Noise estimation (rotation / half-splits)
│   ├── mapping/             # Dask-parallel map generation utilities
│   ├── twopoint.py          # Real-space 2pt via TreeCorr (~1200 lines)
│   ├── twopoint_fourier.py  # Fourier 2pt via NaMaster (~1000 lines)
│   ├── twopoint_null_tests.py # Systematic null tests (γ_t around stars, randoms)
│   ├── blinding.py          # Muir et al. seed-based result blinding
│   ├── covariance.py        # Gaussian covariance (TJPCov)
│   ├── covariance_nmt.py    # NaMaster mode-coupling covariance
│   ├── theory.py            # CCL theory predictions
│   ├── convergence.py       # Kaiser–Squires κ maps
│   ├── psf_diagnostics.py   # PSF quality control (Rowe statistics)
│   ├── diagnostics.py       # Data QA plots
│   ├── jackknife.py         # Jackknife patch creation
│   ├── random_cats.py       # Random catalog generation
│   ├── lssweights.py        # LSS systematic weight calculation
│   ├── extensions/          # Optional: cluster counts, CMB lensing, HOS
│   └── utils/               # Shared utilities (patches, NaMaster, pixelization)
├── examples/                # 25+ pipeline configuration examples
│   ├── metadetect/          # Primary: MetaDetect shear (Rubin-era)
│   ├── metacal/             # Metacalibrated shear (DES/HSC style)
│   ├── lensfit/             # KiDS/Lensfit shear
│   ├── desy1/, desy3/       # DES Year 1 & 3
│   ├── hscy1/, hscy3/       # Hyper Suprime-Cam Y1 & Y3
│   ├── cosmodc2/, dp0.2/, dp1/ # Rubin simulations and data products
│   ├── cmb/                 # CMB lensing cross-correlations
│   └── cluster_counts/      # Cluster lensing extension
├── docs/src/                # Sphinx documentation source
├── data/                    # Test data and fiducial cosmology
├── notebooks/               # Jupyter analysis notebooks
├── submodules/              # External: WLMassMap, pyfsb
└── bin/                     # Utility and installation scripts
```

---

## Pipeline Data Flow

### Inputs
- **Shear catalog** (HDF5): `ra`, `dec`, `g1`, `g2`, `flags`, `psf_g1/g2`, `T`, `s2n`, `mag_*`
  - Metacal variant: also `_1p`, `_1m`, `_2p`, `_2m` response columns
- **Photometry catalog** (HDF5): `ra`, `dec`, `mag_*`, `mag_err_*`
- **Exposure metadata**: PSF properties, depth, flags per exposure
- **Star catalog**: Bright stars for systematic tests

### Processing Phases

**Phase 1 — Data Preparation**
1. Ingest raw catalogs (GCR, FITS, mocks, Rubin butler)
2. Estimate photometric redshift PDFs via RAIL (BPZ-lite, DIR method)
3. Source selection: S/N and size cuts → `ShearCatalog`
4. Lens selection: magnitude and photo-z cuts → `LensPhotometryCatalog`
5. Shear calibration: compute m/c bias per tomographic bin

**Phase 2 — Tomography**
6. Assign objects to tomographic redshift bins
7. Produce `binned_shear_catalog` and `binned_lens_catalog`

**Phase 3 — Map Generation**
8. Source maps: g1, g2 shear HEALPix maps
9. Lens maps: galaxy number density maps
10. Auxiliary maps: PSF g1/g2, flags, depth maps
11. Mask creation: combine depth, PSF, bright-object masks
12. Noise maps: rotational noise (sources), half-split noise (lenses)

**Phase 4 — Measurement**
13. Real-space 2pt: TreeCorr → ξ+, ξ-, γ_t, ω written to `twopoint_data_real_raw`
14. Fourier 2pt: NaMaster → C_ell written to `twopoint_data_fourier`
15. Blinding: apply seed-based cosmological shift → `twopoint_data_real`

**Phase 5 — Analysis**
16. Theory: CCL predictions at fiducial cosmology
17. Covariance: Gaussian + mode-coupling matrix
18. Diagnostics: QA plots, PSF metrics, null tests
19. Final output: SACC file with measurements + covariance + theory + tracers

### Outputs (SACC format)
```
SACC file
├── Tracers:  source_0..N (shear bins), lens_0..M (density bins), each with n(z)
├── Data:     galaxy_shear_cl_ee, galaxy_shearDensity_cl_e, galaxy_density_cl
│             plus real-space ξ+, ξ-, γ_t, ω
├── Covariance matrix
├── Theory predictions
└── Metadata: blinding signature, git hash, module versions
```

---

## Key Stage Classes

| Stage | File | Purpose |
|---|---|---|
| `TXSourceSelectorMetadetect` | `source_selection/metadetect.py` | Select WL sources via MetaDetect |
| `TXSourceSelectorMetacal` | `source_selection/metacal.py` | Select WL sources via Metacal |
| `TXMeanLensSelector` | `lens_selector.py` | Select lens galaxies (magnitude cuts) |
| `TXShearCalibration` | `calibrate.py` | Apply m/c shear bias + bin assignment |
| `TXSourceMaps` | `maps.py` | Generate shear g1/g2 HEALPix maps |
| `TXLensMaps` | `maps.py` | Generate galaxy density HEALPix maps |
| `TXAuxiliarySourceMaps` | `auxiliary_maps.py` | Generate PSF/flag systematic maps |
| `TXSimpleMask` | `masks.py` | Combine depth/PSF/bright-object masks |
| `TXSourceNoiseMaps` | `noise_maps.py` | Shape noise via rotation |
| `TXLensNoiseMaps` | `noise_maps.py` | Density noise via half-splits |
| `TXTwoPoint` | `twopoint.py` | Real-space 2pt correlations (TreeCorr) |
| `TXTwoPointFourier` | `twopoint_fourier.py` | Fourier power spectra (NaMaster) |
| `TXBlinding` | `blinding.py` | Blind results (Muir et al. method) |
| `TXTwoPointTheoryReal` | `theory.py` | Real-space theory predictions (CCL) |
| `TXTwoPointTheoryFourier` | `theory.py` | Fourier theory predictions (CCL) |
| `TXFourierGaussianCovariance` | `covariance.py` | Gaussian covariance (TJPCov) |
| `TXPSFDiagnostics` | `psf_diagnostics.py` | PSF quality metrics (Rowe stats) |
| `TXTwoPointNullTests` | `twopoint_null_tests.py` | γ_t null tests (stars, randoms, fields) |

---

## Key Data Types (`txpipe/data_types.py`)

| Class | Format | Contents |
|---|---|---|
| `ShearCatalog` | HDF5 | Galaxy shear estimates; `catalog_type` = metacal/metadetect/lensfit |
| `PhotometryCatalog` | HDF5 | Photometric magnitudes and errors |
| `TomographyCatalog` | HDF5 | Per-object bin assignments |
| `MapsFile` | HDF5 | HEALPix maps via healsparse |
| `SACCFile` | FITS | 2-point data, covariance, theory, tracers |
| `FiducialCosmology` | YAML/HDF5 | CCL cosmological parameters |

---

## Key Configuration Parameters

```yaml
global:
  chunk_rows: 100000           # Memory chunk size for parallel iteration
  pixelization: healpix
  sparse: true                 # Use healsparse sparse maps

TXSourceSelectorMetadetect:
  s2n_cut: 10.0                # Signal-to-noise threshold
  T_cut: 0.5                   # Size cut (T > T_cut * T_psf)
  source_zbin_edges: [0.5, 0.7, 0.9, 1.1, 2.0]  # 4 tomographic bins

TXMeanLensSelector:
  lens_zbin_edges: [0.0, 0.2, 0.4]  # 3 lens bins
  r_lo_cut: 16.0
  r_hi_cut: 21.6

TXTwoPoint:
  min_sep: 2.5                 # arcmin
  max_sep: 60.0                # arcmin
  nbins: 10                    # Angular bins
  do_shear_shear: true
  do_shear_pos: true
  do_pos_pos: true

TXTwoPointFourier:
  ell_min: 30
  ell_max: 100
  bandwidth: 20                # NaMaster bandpower bandwidth

TXBlinding:
  seed: 1972
  Omega_b: [0.0485, 0.001]    # [fiducial, shift_sigma]
  Omega_c: [0.2545, 0.01]
  w0: [-1.0, 0.1]
```

---

## Major Dependencies

| Package | Role |
|---|---|
| **Ceci** | Pipeline DAG orchestration |
| **TreeCorr** | Real-space two-point correlations |
| **pymaster (NaMaster)** | Fourier power spectra with mode coupling |
| **healpy / healsparse** | HEALPix map I/O and manipulation |
| **sacc** | Standardized 2-point data format |
| **pyccl** | Cosmological theory calculations |
| **TJPCov** | Gaussian covariance estimation |
| **pz-rail** | Photometric redshift estimation framework |
| **dask / dask-mpi** | Lazy parallel array computation |
| **h5py (MPI)** | Parallel HDF5 I/O |
| **firecrown** | Downstream likelihood framework |
| **astropy, numpy, scipy** | General scientific computing |

---

## Running the Pipeline

```bash
# Run with a configuration file
ceci examples/metadetect/pipeline.yml

# Common launchers
ceci pipeline.yml --launcher mini          # Local serial/parallel
ceci pipeline.yml --launcher parsl         # Distributed via Parsl
ceci pipeline.yml --launcher cwl           # CWL workflow standard
```

---

## Supported Surveys & Shear Methods

| Survey | Shear Method | Example Config |
|---|---|---|
| Rubin LSST (simulated) | MetaDetect | `examples/metadetect/` |
| DES Y1/Y3 | Metacal | `examples/desy1/`, `examples/desy3/` |
| Hyper Suprime-Cam Y1/Y3 | HSC pipeline | `examples/hscy1/`, `examples/hscy3/` |
| KiDS | Lensfit | `examples/lensfit/` |
| Rubin DC2 / DP0.2 / DP1 | Metacal/MetaDetect | `examples/cosmodc2/`, `examples/dp0.2/` |
| Mock (lognormal/Gaussian) | Any | `examples/lognormal/`, `examples/gaussian_sims/` |

---

## Extensions

- **`extensions/cluster_counts/`**: Cluster lensing and number counts
- **`extensions/cmb_lensing/`**: Cross-correlations with CMB lensing maps
- **`extensions/hos/`**: Higher-order statistics
- **`extensions/fsb/`**: Filter-scale bispectrum (via pyfsb submodule)
- **`twopoint_scia.py`**: Self-calibration of intrinsic alignments
