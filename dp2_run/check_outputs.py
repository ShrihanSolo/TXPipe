"""
Summarise the TXMeanShearSurveyProperties output after a DP2 run.

Run from the TXPipe repository root:
    python dp2_run/check_outputs.py data/survey_property_rubin_metadetect
It reads the HDF5 summary and prints one line per survey property map,
then says whether anything looks wrong.
"""
import glob
import h5py
import numpy as np

import sys
# Pass the output directory, e.g. data/survey_property_rubin_metadetect
OUT = sys.argv[1]
path = f"{OUT}/shear_survey_property_null.hdf5"

problems = []
with h5py.File(path, "r") as f:
    names = list(f.keys())
    print(f"Maps tested: {len(names)}  (file: {path})\n")
    print(f"{'map':40s} {'bins used':>9s} {'chi2_null/dof':>14s} {'p':>8s} "
          f"{'slope g1 / err':>15s} {'slope g2 / err':>15s}")
    for name in names:
        a = f[name].attrs
        count = f[name]["count"][:]
        used = f[name]["used_in_statistics"][:]
        # Significance of each slope in units of its own error.
        s1 = a["slope_g1"] / a["slope_g1_err"]
        s2 = a["slope_g2"] / a["slope_g2_err"]
        print(f"{name[:40]:40s} {int(used.sum()):4d}/{used.size:<4d} "
              f"{a['chi2_null']:7.1f}/{a['n_dof_null']:<5d} {a['chi2_null_pvalue']:8.2g} "
              f"{s1:15.2f} {s2:15.2f}")
        # Checks: the new cuts and NaN handling should make these impossible.
        if used.any() and count[used].min() < a["min_count"]:
            problems.append(f"{name}: a used bin has fewer than min_count galaxies")
        if np.isfinite(a["slope_g1"]) and a["n_used_bins"] < 3:
            problems.append(f"{name}: a slope was fitted with fewer than 3 bins")
        if a["n_dof_null"] != 2 * a["n_used_bins"]:
            problems.append(f"{name}: n_dof_null is not 2 x bins used")

pngs = sorted(glob.glob(f"{OUT}/shear_survey_property_plots/*.png"))
print(f"\nPlots written: {len(pngs)}")
if len(pngs) != len(names):
    problems.append(f"{len(names)} maps in the HDF5 file but {len(pngs)} plots")

print("\nConclusion:", "no problems found." if not problems else "PROBLEMS:")
for p in problems:
    print("  -", p)
print("(Slope columns are slope / error, so |value| > 3 is a notable trend.)")
