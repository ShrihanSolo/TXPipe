"""
Compare the DESC metadetect v1.1 masks in the data registry: full (361), wide (360), deep (359),
and the old mask (148) used by the current TXPipe DP2 example.

For each mask it prints the area, how much of it lies inside each of the others, and whether full is
exactly wide + deep; then it draws one map with every pixel coloured by which masks contain it, plus one
small panel per mask.

The masks are at very high resolution (nside 131072), far too big to hold in memory as pixel lists, so
each one is reduced to a coarse map of the covered fraction with healsparse's fracdet_map, as
TXPipe's MapsFile.read_mask(degrade_nside=...) does.

Run at NERSC from the repo root, in the DESC env (source bin/setup-nersc):
    python dp2_run/compare_masks.py                       # finds the files by registry name
    python dp2_run/compare_masks.py old=/path/mask.hdf5   # or give any path by hand (label=path)
Writes mask_comparison.png in the current directory.
"""
import glob
import os
import sys
import time

import h5py
import healpy as hp
import healsparse as hsp
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

REGISTRY = "/global/cfs/cdirs/lsst/utilities/data-registry"
# label: (registry id, dataset name or None). The old mask is looked up by id instead.
MASKS = {
    "full": (361, "txpipe_desc_metadetect_v1.1_mask_full"),
    "wide": (360, "txpipe_desc_metadetect_v1.1_mask_wide"),
    "deep": (359, "txpipe_desc_metadetect_v1.1_mask_deep"),
    "old": (148, None),
}
WORK_NSIDE = 512   # areas and overlaps (pixels of about 0.11 deg; edge pixels count as fractions)
PLOT_NSIDE = 512   # the picture only (about 0.11 deg pixels)


def find_by_name(name):
    """The registry keeps each dataset in <registry>/<schema>/<owner type>/<owner>/.gen_paths/<name>_<version>/."""
    dirs = []
    for depth in ("*/*/*/.gen_paths", "*/*/.gen_paths", "*/.gen_paths"):
        dirs += glob.glob(f"{REGISTRY}/{depth}/{name}_[0-9]*")
    dirs = [d for d in dirs if "1deg2" not in d]  # skip the 1 deg^2 mini versions
    if not dirs:
        return None
    newest = sorted(dirs, key=os.path.getmtime)[-1]
    files = sorted(glob.glob(f"{newest}/*.hdf5") + glob.glob(f"{newest}/*.hs"))
    return files[0] if files else None


def find_by_id(dataset_id):
    """Ask the data registry for a dataset's path. Returns None if the client isn't available."""
    try:
        from dataregistry import DataRegistry
        reg = DataRegistry()
        path = reg.Query.get_dataset_absolute_path(dataset_id)
    except Exception as e:  # no client, no access, or a different API version
        print(f"  (registry lookup of id {dataset_id} failed: {type(e).__name__}: {e})")
        return None
    if os.path.isdir(path):
        files = sorted(glob.glob(f"{path}/*.hdf5") + glob.glob(f"{path}/*.hs"))
        path = files[0] if files else None
    return path


def old_mask_candidates():
    """Older DP2 mask datasets in the registry, to help pick the old mask by hand."""
    out = []
    for depth in ("*/*/*/.gen_paths", "*/*/.gen_paths"):
        for d in glob.glob(f"{REGISTRY}/{depth}/*"):
            b = os.path.basename(d).lower()
            if "mask" in b and "dp2" in b and "v1.1" not in b and "1deg2" not in b:
                out.append(d)
    return sorted(out)


def coverage_fraction(path):
    """
    The mask summed into WORK_NSIDE pixels: returns (pixels, covered fraction), nest ordering.

    A mask value is the usable fraction of a native pixel, so a coarse pixel's fraction is the sum of
    its native values divided by the number of native pixels inside it.
    """
    if path.endswith(".hs"):
        group, legacy = None, False
    else:
        with h5py.File(path, "r") as f:
            g = f["maps/mask"]
            legacy = "healsparse" not in g
            group = "maps/mask/healsparse"
    total = {}

    def add(pix, val, nside):
        shift = 2 * int(round(np.log2(nside // WORK_NSIDE))) if nside > WORK_NSIDE else 0
        coarse = pix >> shift
        u, inv = np.unique(coarse, return_inverse=True)
        s = np.bincount(inv, weights=val.astype(float)) / 4 ** (shift // 2)
        for p, x in zip(u, s):
            total[p] = total.get(p, 0.0) + x

    if legacy:  # datasets "pixel" and "value": read them in slices
        with h5py.File(path, "r") as f:
            g = f["maps/mask"]
            nside, nest = int(g.attrs["nside"]), bool(g.attrs["nest"])
            n = g["pixel"].shape[0]
            for s in range(0, n, 20_000_000):
                pix, val = g["pixel"][s:s + 20_000_000], g["value"][s:s + 20_000_000]
                if not nest:
                    pix = hp.ring2nest(nside, pix)
                add(pix, val, nside)
    else:  # healsparse: load it whole and use fracdet_map, as TXPipe's MapsFile.read_mask does
        kw = {"hdf5_group": group} if group else {}
        m = hsp.HealSparseMap.read(path, **kw)
        nside = m.nside_sparse
        print(f"    loaded {os.path.basename(path)}: dtype {m.dtype}, nside {nside}, "
              f"coverage nside {m.nside_coverage}")
        if np.issubdtype(m.dtype, np.floating):
            print("    (float mask: the fraction below counts set pixels and ignores fractional values)")
        fd_nside = max(WORK_NSIDE, m.nside_coverage)  # fracdet_map can't go coarser than the coverage map
        fd = m.fracdet_map(fd_nside)
        del m
        pix = fd.valid_pixels
        add(pix, fd[pix], fd_nside)  # averages down to WORK_NSIDE if fd_nside is finer
    pix = np.array(sorted(total), dtype=np.int64)
    frac = np.array([total[p] for p in pix])
    return pix, frac, nside


def main():
    paths = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        paths[label] = path
    for label, (did, name) in MASKS.items():
        if label not in paths:
            paths[label] = find_by_name(name) if name else find_by_id(did)

    print("Masks found:")
    maps = {}
    for label, path in paths.items():
        if path is None:
            print(f"  {label:5s}: NOT FOUND (give it by hand: {label}=/path/to/mask.hdf5)")
            if label == "old":
                cands = old_mask_candidates()
                if cands:
                    print("         older DP2 mask datasets in the registry:")
                    for c in cands:
                        print(f"           {c}")
            continue
        t0 = time.time()
        pix, frac, nside = coverage_fraction(path)
        maps[label] = (pix, frac)
        print(f"  {label:5s}: native nside {nside}, read in {time.time() - t0:.0f} s, {path}")
    if not maps:
        sys.exit("No masks found.")
    labels = list(maps)

    # One table over all coarse pixels that any mask touches.
    allpix = np.unique(np.concatenate([p for p, f in maps.values()]))
    F = np.zeros((len(labels), allpix.size))
    for i, l in enumerate(labels):
        p, f = maps[l]
        F[i, np.searchsorted(allpix, p)] = f
    pixarea = hp.nside2pixarea(WORK_NSIDE, degrees=True)

    print("\nArea of each mask (deg^2, partial pixels counted as fractions):")
    for i, l in enumerate(labels):
        print(f"  {l:5s} {F[i].sum() * pixarea:9.1f}")

    print(f"\nOverlap: % of the ROW mask's area that is also in the COLUMN mask"
          f" (at nside {WORK_NSIDE}, about 0.11 deg; approximate only at the edges)")
    print("  " + " " * 6 + "".join(f"{b:>8s}" for b in labels))
    for i, a in enumerate(labels):
        row = [100 * np.minimum(F[i], F[j]).sum() / F[i].sum() for j in range(len(labels))]
        print(f"  {a:6s}" + "".join(f"{x:7.1f}%" for x in row))

    if {"full", "wide", "deep"} <= set(labels):
        f, w, d = (F[labels.index(k)] for k in ("full", "wide", "deep"))
        extra = np.clip(f - w - d, 0, None).sum() * pixarea  # in full but in neither wide nor deep
        missing = np.clip(w + d - f, 0, None).sum() * pixarea  # in wide or deep but not in full
        both = np.minimum(w, d).sum() * pixarea
        print(f"\nIs full = wide + deep?")
        print(f"  area in full but in neither wide nor deep: {extra:8.2f} deg^2")
        print(f"  area in wide or deep but not in full:      {missing:8.2f} deg^2")
        print(f"  area in both wide and deep:                {both:8.2f} deg^2")
        verdict = "YES (to within the edge pixels)" if max(extra, missing) < 0.01 * f.sum() * pixarea else "NO"
        print(f"  => {verdict}")

    # The picture: which masks contain each (coarser) pixel.
    shift = 2 * int(round(np.log2(WORK_NSIDE // PLOT_NSIDE)))
    ppix = allpix >> shift
    upix, inv = np.unique(ppix, return_inverse=True)
    inside = np.zeros((len(labels), upix.size), bool)
    for i in range(len(labels)):
        inside[i] = np.bincount(inv, weights=F[i], minlength=upix.size) > 0
    ra, dec = hp.pix2ang(PLOT_NSIDE, upix, nest=True, lonlat=True)
    if np.ptp(ra) > 180:  # the footprint crosses RA = 0: show RA from -180 to 180
        ra = np.where(ra > 180, ra - 360, ra)
    codes = np.array([" + ".join(l for l, x in zip(labels, col) if x) for col in inside.T])
    kinds = sorted(set(codes), key=lambda c: -(codes == c).sum())
    cmap = plt.get_cmap("tab10")
    pa = hp.nside2pixarea(PLOT_NSIDE, degrees=True)

    n = len(labels)
    squash = np.cos(np.radians(np.median(dec)))
    shape = max(np.ptp(dec), 1) / max(np.ptp(ra) * squash, 1)  # height / width on the sky
    width = max(4 * n, 12)
    top_h = float(np.clip(width * shape + 1.2, 3, 10))
    row_h = float(np.clip(width / n * shape + 1.0, 2, 6))
    fig = plt.figure(figsize=(width, top_h + row_h), layout="constrained")
    sub = fig.subfigures(2, 1, height_ratios=[top_h, row_h])
    ax = sub[0].subplots()
    msize = 4
    for i, k in enumerate(kinds):
        w = codes == k
        ax.scatter(ra[w], dec[w], s=msize, marker="s", linewidths=0, color=cmap(i % 10),
                   label=f"{k}  (~{w.sum() * pa:.0f} deg$^2$)")
    ax.legend(markerscale=3, fontsize=9, title="pixel is in", loc="best")
    ax.set_title("Which masks contain each pixel")
    axes = list(np.atleast_1d(sub[1].subplots(1, n, sharex=True, sharey=True)))
    for a, l, row in zip(axes, labels, inside):
        a.scatter(ra[~row], dec[~row], s=1, marker="s", linewidths=0, color="0.85")
        a.scatter(ra[row], dec[row], s=1, marker="s", linewidths=0, color="C0")
        a.set_title(f"{l} (id {MASKS.get(l, ('?',))[0]})")
    for a in [ax] + axes:
        a.set_xlabel("RA [deg]")
        a.set_ylabel("Dec [deg]")
        a.set_aspect(1 / squash)
    ax.invert_xaxis()
    axes[0].invert_xaxis()  # the small panels share one x axis, so invert it once
    fig.savefig("mask_comparison.png", dpi=130)
    print(f"\nWrote {os.path.abspath('mask_comparison.png')}")


if __name__ == "__main__":
    main()
