"""
Compare the DESC metadetect v1.1 masks in the data registry: full (361), wide (360), deep (359),
and the old mask (148) used by the current TXPipe DP2 example.

For each mask it prints the area, and how much of it lies inside each of the others; then it draws
one map with every pixel coloured by which masks contain it, plus one small panel per mask.

Run at NERSC from the repo root, in the DESC env (source bin/setup-nersc):
    python dp2_run/compare_masks.py                       # finds the files by registry name
    python dp2_run/compare_masks.py old=/path/mask.hdf5   # or give any path by hand (label=path)
Writes mask_comparison.png in the current directory.
"""
import glob
import os
import sys

import h5py
import healpy as hp
import healsparse as hsp
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

REGISTRY = "/global/cfs/cdirs/lsst/utilities/data-registry"
# label: (registry id, dataset name). The old mask's name is a guess from the old catalog's path.
MASKS = {
    "full": (361, "txpipe_desc_metadetect_v1.1_mask_full"),
    "wide": (360, "txpipe_desc_metadetect_v1.1_mask_wide"),
    "deep": (359, "txpipe_desc_metadetect_v1.1_mask_deep"),
    "old": (148, "txpipe_dp2_shear_catalog_desc_metaDetect_mask"),
}
PLOT_NSIDE = 512   # resolution for the picture only (about 0.11 deg pixels); areas use the native nside


def find_file(name):
    """The registry stores each dataset in <registry>/<schema>/<owner type>/<owner>/.gen_paths/<name>_<version>/."""
    dirs = []
    for depth in ("*/*/*/.gen_paths", "*/*/.gen_paths", "*/.gen_paths"):
        dirs += glob.glob(f"{REGISTRY}/{depth}/{name}_[0-9]*")
    dirs = [d for d in dirs if "1deg2" not in d]  # skip the 1 deg^2 mini versions
    if not dirs:
        return None
    newest = sorted(dirs, key=os.path.getmtime)[-1]
    files = sorted(glob.glob(f"{newest}/*.hdf5") + glob.glob(f"{newest}/*.hs"))
    if len(dirs) > 1:
        print(f"  note: {len(dirs)} versions of {name}; using the newest, {os.path.basename(newest)}")
    return files[0] if files else None


def read_mask(path):
    """Read a TXPipe mask file (maps/mask, healsparse or legacy pixel/value) or a plain healsparse file."""
    if path.endswith(".hs"):
        return hsp.HealSparseMap.read(path)
    with h5py.File(path, "r") as f:
        group = f["maps/mask"]
        if "healsparse" in group:
            pass  # read below with healsparse, after the file is closed
        else:
            nside, nest = group.attrs["nside"], group.attrs["nest"]
            pix, val = group["pixel"][:], group["value"][:]
            if not nest:
                pix = hp.ring2nest(nside, pix)
            m = hsp.HealSparseMap.make_empty(32, nside, dtype=val.dtype)
            m.update_values_pix(pix, val)
            return m
    return hsp.HealSparseMap.read(path, hdf5_group="maps/mask/healsparse")


def main():
    paths = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        paths[label] = path
    for label, (did, name) in MASKS.items():
        if label not in paths:
            paths[label] = find_file(name)

    masks = {}
    print("Masks found:")
    for label, path in paths.items():
        if path is None:
            print(f"  {label:5s}: NOT FOUND (give it by hand: {label}=/path/to/mask.hdf5)")
            continue
        m = read_mask(path)
        masks[label] = m
        print(f"  {label:5s}: nside {m.nside_sparse}, {path}")
    if not masks:
        sys.exit("No masks found.")

    # Areas at native resolution. A mask value is the fraction of the pixel that is usable,
    # so the area is the sum of the values times the pixel area.
    print("\nArea of each mask (deg^2):")
    print(f"  {'mask':5s} {'pixels set':>12s} {'area':>9s}  (area counting partial pixels as fractions)")
    for label, m in masks.items():
        pix = m.valid_pixels
        val = m[pix].astype(float)
        pixarea = hp.nside2pixarea(m.nside_sparse, degrees=True)
        print(f"  {label:5s} {pix.size:12d} {pix.size * pixarea:9.1f}  ({val.sum() * pixarea:.1f})")

    # Overlaps, compared at the finest common nside (exact if all masks have the same nside).
    nside = min(m.nside_sparse for m in masks.values())
    sets = {}
    for label, m in masks.items():
        pix = m.valid_pixels
        shift = 2 * int(np.log2(m.nside_sparse // nside))
        sets[label] = np.unique(pix >> shift)  # nest pixel at the coarser nside
    labels = list(sets)
    print(f"\nOverlap (at nside {nside}): row % of the ROW mask's area that lies inside the COLUMN mask")
    print("  " + " " * 6 + "".join(f"{b:>8s}" for b in labels))
    for a in labels:
        row = [100 * np.isin(sets[a], sets[b], assume_unique=True).mean() for b in labels]
        print(f"  {a:6s}" + "".join(f"{x:7.1f}%" for x in row))
    if {"full", "wide", "deep"} <= set(sets):
        union = np.union1d(sets["wide"], sets["deep"])
        both = np.intersect1d(sets["wide"], sets["deep"]).size
        same = np.array_equal(union, sets["full"])
        print(f"\n  Is full exactly wide + deep?  {'YES' if same else 'NO'}"
              f"  (wide and deep share {both} pixels at nside {nside})")

    # The picture: coarse pixels, coloured by which masks contain them.
    plot = {}
    for label, m in masks.items():
        shift = 2 * int(np.log2(m.nside_sparse // PLOT_NSIDE)) if m.nside_sparse > PLOT_NSIDE else 0
        plot[label] = np.unique(m.valid_pixels >> shift)
    allpix = np.unique(np.concatenate(list(plot.values())))
    ra, dec = hp.pix2ang(PLOT_NSIDE, allpix, nest=True, lonlat=True)
    if np.ptp(ra) > 180:  # the footprint crosses RA = 0: show RA from -180 to 180
        ra = np.where(ra > 180, ra - 360, ra)
    member = np.array([np.isin(allpix, plot[l]) for l in labels])  # one row per mask
    codes = [" + ".join(l for l, x in zip(labels, col) if x) for col in member.T]
    kinds = sorted(set(codes), key=lambda c: -codes.count(c))
    cmap = plt.get_cmap("tab10")

    n = len(labels)
    # Size the figure to the footprint's shape so the panels fill it.
    squash = np.cos(np.radians(np.median(dec)))
    shape = max(np.ptp(dec), 1) / max(np.ptp(ra) * squash, 1)  # height / width on the sky
    width = max(4 * n, 12)
    top_h = float(np.clip(width * shape + 1.2, 3, 10))
    row_h = float(np.clip(width / n * shape + 1.0, 2, 6))
    fig = plt.figure(figsize=(width, top_h + row_h), layout="constrained")
    sub = fig.subfigures(2, 1, height_ratios=[top_h, row_h])
    ax = sub[0].subplots()
    codes = np.array(codes)
    for i, k in enumerate(kinds):
        w = codes == k
        ax.scatter(ra[w], dec[w], s=1.5, marker="s", linewidths=0, color=cmap(i % 10),
                   label=f"{k}  ({w.sum()} px)")
    ax.legend(markerscale=6, fontsize=9, title="pixel is in", loc="best")
    ax.set_title("Which masks contain each pixel")
    axes = list(np.atleast_1d(sub[1].subplots(1, n, sharex=True, sharey=True)))
    for a, l, row in zip(axes, labels, member):
        a.scatter(ra[~row], dec[~row], s=0.5, marker="s", linewidths=0, color="0.85")
        a.scatter(ra[row], dec[row], s=0.5, marker="s", linewidths=0, color="C0")
        a.set_title(f"{l} (id {MASKS.get(l, ('?',))[0]})")
    for a in [ax] + axes:
        a.set_xlabel("RA [deg]")
        a.set_ylabel("Dec [deg]")
        a.set_aspect(1 / squash)
    ax.invert_xaxis()
    axes[0].invert_xaxis()  # the small panels share one x axis, so invert it once
    fig.savefig("mask_comparison.png", dpi=130)
    print("\nWrote mask_comparison.png")


if __name__ == "__main__":
    main()
