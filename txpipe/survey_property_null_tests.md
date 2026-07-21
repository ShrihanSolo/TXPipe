# `survey_property_null_tests.py`

## Overview

This module defines a single TXPipe pipeline stage, **`TXMeanShearSurveyProperties`**,
which implements a *mean-shear-vs-survey-property* null test.

The physical idea: the mean weak-lensing shear of source galaxies should **not**
depend on the observing conditions in the part of the sky where each galaxy was
measured. If the mean measured shear varies systematically with a "survey
property" — PSF ellipticity, PSF size, depth, sky background, seeing, etc. —
that is evidence of a residual **additive shear systematic** that has leaked into
the catalog. This stage bins galaxies by the value of a survey property map at
their sky position and computes the mean *calibrated* shear (`g1`, `g2`) in each
bin, then reports the result numerically and as diagnostic plots with a
goodness-of-fit-to-zero (χ²/dof) statistic.

The stage is registered in `txpipe/__init__.py`:

```python
from .survey_property_null_tests import TXMeanShearSurveyProperties
```

---

## Inputs

| Tag | Type | Purpose |
|-----|------|---------|
| `shear_catalog` | `ShearCatalog` | Source galaxy shears (metacal / metadetect / lensfit / hsc) |
| `shear_tomography_catalog` | `TomographyCatalog` | Per-galaxy tomographic `bin` assignment (and metacal response) |
| `aux_source_maps` | `MapsFile` | Auxiliary maps derived from the source catalog (PSF, flags, …) |
| `aux_lens_maps` | `MapsFile` | Auxiliary maps derived from the lens catalog (depth, bright objects, …) |
| `mask` | `MapsFile` | Survey mask defining the footprint the test is restricted to |

The survey properties tested come primarily from the two auxiliary map files.
Additional maps can be supplied as external HealSparse files via the
`external_maps_dir` config option (see below).

The `mask` defines the survey footprint. Both the galaxy sample and the map
pixels used to choose bin edges are restricted to this footprint, so every
survey property is tested over the **same** region even when the individual
property maps (e.g. in different bands) have different coverage. See
[Footprint handling](#footprint-handling) below.

## Outputs

| Tag | Type | Contents |
|-----|------|----------|
| `shear_survey_property_plots` | `FileCollection` | One `<property>.png` diagnostic plot per survey property, plus a listing manifest |
| `shear_survey_property_null` | `HDFFile` | Numerical results: one HDF5 group per property with bin centers, `g1`/`g2`, and their errors |

## Configuration options

| Option | Type | Default | Meaning |
|--------|------|---------|---------|
| `nbins` | int | `20` | Number of bins along each survey property axis |
| `chunk_rows` | int | `100_000` | Rows read per catalog chunk (streaming) |
| `delta_gamma` | float | `0.02` | Metacal/metadetect shear-response step Δγ used for calibration |
| `outlier_fraction` | float | `0.05` | Tail fraction of map pixels trimmed at each end when setting bin edges |
| `mask_threshold` | float | `0.0` | Minimum fractional coverage for a mask pixel to count as inside the footprint |
| `properties` | list | `[]` | Restrict testing to these map names; empty means "all available maps" |
| `external_maps_dir` | str | `""` | Optional directory of extra `.hs` HealSparse maps to also test |

---

## How it works

### 1. Module-level helpers

**`_in_footprint(ra, dec, mask)`** — returns a boolean array flagging which
sky positions fall inside the mask footprint. `mask` is a boolean HealSparse
map; positions outside the footprint land on unobserved pixels, for which
HealSparse returns the `False` sentinel, so a direct lookup gives the in/out
flag. Used both for choosing bin edges and for cutting the galaxy sample.

**`_compute_bin_edges(hsp_map, nside, nbins, outlier_fraction, mask)`** —
computes the bin edges for a single survey property map, **restricted to the
mask footprint**:

1. Takes the map's valid (observed) pixels and keeps only those inside the mask
   footprint (via `_in_footprint` on the pixel centers).
2. Reads the values at those pixels, casts to float, keeps finite values.
3. Sets the lower and upper edges at the `outlier_fraction` and
   `1 - outlier_fraction` percentiles — i.e. it **trims the extreme tails**
   (default 5% at each end) so a handful of outlier pixels don't stretch the
   binning.
4. Returns `nbins + 1` **linearly spaced** edges between those percentiles.

Returns `None` if the map has no overlap with the footprint (or no finite
values there); such maps are skipped. Because the range is computed over the
footprint-restricted map pixels, per-band coverage extending beyond the shear
footprint does not skew the binning.

### 2. `TXMeanShearSurveyProperties.run()`

The stage runs in a single streaming pass over the shear catalog and is
MPI-parallel (`parallel = True`).

#### a. Determine the shear catalog type

`read_shear_catalog_type(self)` inspects the catalog and returns one of
`metacal`, `metadetect`, `lensfit`, or `hsc`. The catalog type controls:

- **which columns are read** (`shear_cols`),
- **where ra/dec live** — for metadetect these are under the unsheared `"00"`
  variant subgroup (`00/ra`, `00/dec`), otherwise plain `ra`/`dec`,
- **extra iterators** — metacal additionally streams the `R_gamma` response
  from the tomography catalog.

For metadetect, the sheared-variant columns are expanded via
`metadetect_variants("g1", "g2", "weight")`.

#### b. Read the mask

The mask is read once as a **boolean footprint map**:

```python
mask = f.read_mask(thresh=self.config["mask_threshold"], returnbool=True)
```

`mask_threshold` sets the minimum fractional coverage for a pixel to count as
inside the footprint. This footprint is used both to choose bin edges and to cut
the galaxy sample.

#### c. Load the survey property maps

A dictionary `all_maps` is built mapping `name -> (hsp_map, nside, bin_edges)`.
Each candidate map is passed through a local `_register_map` helper that applies
the `properties` filter, computes footprint-restricted bin edges, and **skips
the map if it does not overlap the footprint** (printing a note on rank 0):

1. **Auxiliary maps** — iterate over `aux_source_maps` then `aux_lens_maps`;
   for each map name returned by `f.list_maps()`, read the map with
   `f.read_map(name)` and its `nside` from `f.read_map_info(name)`.
2. **External maps** — if `external_maps_dir` is set, glob `*.hs` files in that
   directory (sorted), read each with `healsparse.HealSparseMap.read(path)`,
   take `nside` from `hsp_map.nside_sparse`, and derive the property name from the
   file stem.

If no maps survive, the stage prints a message (rank 0) and returns early.

#### d. Set up accumulators

One `MeanShearInBins` accumulator is created per map, all sharing:

- the property key `"survey_prop"` (the column the accumulator selects on),
- that map's `bin_edges` as bin limits,
- `delta_gamma`,
- `cut_source_bin=True` — only galaxies with a valid source-bin assignment
  (`bin != -1`) are counted,
- the detected `shear_catalog_type`.

`MeanShearInBins` (in `txpipe/shear_calibration/mean_shear_in_bins.py`) holds,
per bin, an appropriate calibration calculator (Metacal/MetaDetect/Lensfit/HSC)
plus a `ParallelMean` for the mean property value. Its `selector` picks the
galaxies whose `survey_prop` value falls in bin *i* (and, with
`cut_source_bin`, drops unassigned galaxies).

#### e. Single streaming pass over the catalog

Using `self.combined_iterators(...)`, the stage jointly iterates the shear
catalog, the tomography `bin` column, and (for metacal) the response, in chunks
of `chunk_rows`. For each chunk, a single footprint flag is computed for the
galaxies with `_in_footprint(ra, dec, mask)`, then for **every** map:

1. Convert galaxy `(ra, dec)` to HealPix pixel indices at that map's `nside`
   with `hp.ang2pix(..., lonlat=True, nest=True)` — **nested** ordering.
2. Look up the property value at each galaxy's pixel: `hsp_map[pixels]`.
3. Replace HealSparse's unobserved sentinel `hp.UNSEEN` with `NaN` (galaxies on
   pixels not covered by *this* map).
4. Set the value to `NaN` for galaxies **outside the mask footprint**, so they
   fall outside every bin and are dropped by `MeanShearInBins`.
5. Store the values as `data["survey_prop"]` and feed the chunk to that map's
   `MeanShearInBins.add_data(data)`.

Because all maps share the single catalog pass, adding more survey properties
costs extra per-chunk pixel lookups but not extra reads of the (large) catalog.

#### f. Collect and write outputs

After the pass, for each property the stage calls `binner.collect(self.comm)`,
which reduces across MPI ranks and returns:

- `mu` — mean survey-property value in each bin,
- `g1`, `g2` — mean calibrated shear components in each bin,
- `sigma1`, `sigma2` — their standard errors (`stats.sigma / sqrt(N_eff)`).

Only **rank 0** writes:

- **HDF5** (`shear_survey_property_null`): a group named after the property
  (with `/` replaced by `_`) containing datasets `bin_centers`, `g1`, `g2`,
  `sigma_g1`, `sigma_g2`.
- **Plot** (`shear_survey_property_plots`): a PNG per property showing `g1`
  (blue squares) and `g2` (orange circles) vs the property value, with error
  bars, a dashed zero line, and the χ²/dof of the points against zero in the
  title. The two series are slightly offset horizontally (`dx`) for legibility.

The χ²/dof is computed over finite bins with positive error bars:

```
chi2  = Σ (g / sigma)²    over g1 and g2 bins
n_dof = number of contributing bins
chi2_dof = chi2 / n_dof
```

A χ²/dof near 1 is consistent with no trend (null test passes); a large value
indicates the mean shear correlates with that survey property.

Finally rank 0 writes the PNG listing manifest and the HDF5 file is closed.

---

## Footprint handling

Survey property maps frequently cover **different regions**, especially across
bands, and the final shear catalog is usually measured on a subset of the full
survey-property footprint. Testing each property over its own coverage would
average the mean shear over inconsistent regions. To avoid this, the stage
restricts the test to the footprint defined by the `mask` input in two places:

1. **Bin edges** are computed only from map pixels inside the footprint
   (`_compute_bin_edges` → `_in_footprint`), so out-of-footprint (e.g.
   deeper/shallower per-band) coverage cannot skew the binning ranges. A map
   with no footprint overlap is skipped.
2. **The galaxy sample** is cut per chunk: galaxies outside the footprint have
   their property value set to `NaN` and therefore contribute to no bin.

`mask_threshold` controls the minimum fractional coverage for a mask pixel to be
counted as inside the footprint. The mask, each property map, and the galaxy
positions may all be at different `nside`; positions are matched to the mask via
`hp.ang2pix` at the mask's own resolution, so no explicit degrading is required.

The net effect is that every survey property is tested over the **same** footprint
∩ that map's own coverage, giving a consistent, apples-to-apples null test.

## Interpreting the results

- **Null test passes:** in every property's plot, `g1` and `g2` are consistent
  with zero across all bins and χ²/dof ≈ 1.
- **Null test fails:** a visible trend of mean shear with a survey property, and
  χ²/dof ≫ 1, flags a residual additive systematic tied to that observing
  condition — a candidate for further modelling or de-trending upstream.

## Related components

- `MeanShearInBins`, `metadetect_variants` — `txpipe/shear_calibration/`
- Auxiliary map producers (`aux_source_maps`, `aux_lens_maps`) —
  `txpipe/auxiliary_maps.py`
- External survey property maps are HealSparse `.hs` files (NEST-ordered);
  in TXPipe these are typically ingested from the LSST Butler by
  `TXIngestDataPreview1` in `txpipe/ingest/dp1.py`.
