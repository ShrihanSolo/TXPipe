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

from scipy.stats import chi2 as chi2_dist


def recompute_chi2_const(grp, used):
    """chi2 of the used bins' mean g1, g2 about each component's weighted mean,
    as computed in _null_test_statistics, from the saved per-bin values."""
    total = 0.0
    for comp in ("g1", "g2"):
        g = grp[comp][:][used]
        w = 1.0 / grp[f"sigma_{comp}"][:][used] ** 2
        total += np.sum(w * (g - np.sum(w * g) / np.sum(w)) ** 2)
    return total


problems = []
with h5py.File(path, "r") as f:
    # Only groups holding a map's results: the file also has other groups,
    # e.g. the provenance written by ceci, which have no binned data.
    names = [k for k in f.keys() if isinstance(f[k], h5py.Group) and "g1" in f[k]]
    others = [k for k in f.keys() if k not in names]
    print(f"Maps tested: {len(names)}  (file: {path})")
    if others:
        print(f"(other groups in the file, not maps: {', '.join(others)})")
    print()
    print(f"{'map':40s} {'bins used':>9s} {'chi2_null/dof':>14s} {'p':>8s} "
          f"{'chi2_const/dof':>14s} {'p':>8s} "
          f"{'slope g1 / err':>15s} {'slope g2 / err':>15s}")
    for name in names:
        a = f[name].attrs
        count = f[name]["count"][:]
        used = f[name]["used_in_statistics"][:]
        # Significance of each slope in units of its own error.
        s1 = a["slope_g1"] / a["slope_g1_err"]
        s2 = a["slope_g2"] / a["slope_g2_err"]
        # chi2_const (against the bins' weighted mean) is missing from outputs
        # written before it was added. It only needs the saved per-bin values,
        # so recompute it here the same way the stage does (marked with *).
        if "chi2_const" in a:
            c2, dof, p, mark = a["chi2_const"], a["n_dof_const"], a["chi2_const_pvalue"], " "
        else:
            c2, dof, p, mark = recompute_chi2_const(f[name], used), 2 * int(used.sum()) - 2, np.nan, "*"
            p = chi2_dist.sf(c2, dof) if dof > 0 else np.nan
        const_cols = f"{c2:7.1f}/{dof:<4d}{mark} {p:8.2g} "
        print(f"{name[:40]:40s} {int(used.sum()):4d}/{used.size:<4d} "
              f"{a['chi2_null']:7.1f}/{a['n_dof_null']:<5d} {a['chi2_null_pvalue']:8.2g} "
              f"{const_cols}"
              f"{s1:15.2f} {s2:15.2f}")
        # Checks: the new cuts and NaN handling should make these impossible.
        if used.any() and count[used].min() < a["min_count"]:
            problems.append(f"{name}: a used bin has fewer than min_count galaxies")
        if np.isfinite(a["slope_g1"]) and a["n_used_bins"] < 3:
            problems.append(f"{name}: a slope was fitted with fewer than 3 bins")
        if a["n_dof_null"] != 2 * a["n_used_bins"]:
            problems.append(f"{name}: n_dof_null is not 2 x bins used")
        if "chi2_const" in a and a["n_used_bins"] >= 2 and a["n_dof_const"] != 2 * a["n_used_bins"] - 2:
            problems.append(f"{name}: n_dof_const is not 2 x bins used - 2")

pngs = sorted(glob.glob(f"{OUT}/shear_survey_property_plots/*.png"))
print(f"\nPlots written: {len(pngs)}")
if len(pngs) != len(names):
    problems.append(f"{len(names)} maps in the HDF5 file but {len(pngs)} plots")

print("\nConclusion:", "no problems found." if not problems else "PROBLEMS:")
for p in problems:
    print("  -", p)
print("(Slope columns are slope / error, so |value| > 3 is a notable trend.)")
print("(chi2_null tests against zero shear, so a constant offset raises it;")
print(" chi2_const tests against each map's mean shear, so it only responds to")
print(" variation between bins. High null but normal const means an offset.")
print(" * = recomputed here, for outputs written before chi2_const was added.)")
