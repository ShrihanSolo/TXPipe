"""
Copy the DP2 survey property maps out of the DP2 butler repo at NERSC as
degraded .hs files that TXMeanShearSurveyProperties reads via external_maps_dir.

No LSST stack needed: the butler's file datastore holds each map as an
ordinary healsparse file, so this reads them directly with healsparse.
For each map it reads only the covered pixels, degraded to NSIDE, and writes
<dataset type>_<band>.hs (the DP1 naming the stage already parses).

Usage (TXPipe env; use a compute node for the real run):
    python convert_dp2_sp_maps.py --list          # show what would be converted
    python convert_dp2_sp_maps.py                 # convert (skips existing files)
    python convert_dp2_sp_maps.py --bands griz    # other bands
"""
import argparse
import glob
import os
import re
import time

import healsparse
import numpy as np

RUN = ("/global/cfs/cdirs/lsst/production/gen3/rubin/DP2/repo/"
       "LSSTCam/runs/DRP/DP2/v30_0_8/DM-55060/stage3/20260614T173751Z")
OUT = "/pscratch/sd/s/shrihan/dp2_survey_property_maps"
BANDS = "ugrizy"


def band_of(path):
    """The band a butler file belongs to, from a /<band>/ path component or a _<band>_ in its name."""
    parts = path.split(os.sep)
    for part in parts[:-1]:
        if len(part) == 1 and part in BANDS:
            return part
    found = re.findall(r"_([ugrizy])_", os.path.basename(path))
    return found[0] if len(found) == 1 else None


def find_maps(run, bands):
    """{(dataset type, band): file path} for every consolidated map file under run."""
    maps = {}
    for d in sorted(glob.glob(f"{run}/*consolidated_map*")):
        dtype = os.path.basename(d)
        for path in sorted(p for p in glob.glob(f"{d}/**/*", recursive=True) if os.path.isfile(p)):
            band = band_of(path)
            if band is None:
                print(f"  ?? cannot tell the band of {path}; skipped")
                continue
            if band not in bands:
                continue
            if (dtype, band) in maps:
                raise RuntimeError(f"two files for {dtype} {band}:\n  {maps[dtype, band]}\n  {path}")
            maps[dtype, band] = path
    return maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--bands", default="i")
    parser.add_argument("--nside", type=int, default=4096)
    parser.add_argument("--list", action="store_true", help="only list the files")
    args = parser.parse_args()

    maps = find_maps(args.run, args.bands)
    print(f"Found {len(maps)} map files for bands '{args.bands}' in\n  {args.run}\n")
    for (dtype, band), path in sorted(maps.items()):
        print(f"  {dtype:58s} {band}  {os.path.getsize(path) / 1e9:6.2f} GB  {os.path.basename(path)}")
    if args.list or not maps:
        print("\nList only; nothing written." if maps else "\nNothing found; check --run.")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"\nWriting nside {args.nside} maps to {args.out}")
    for (dtype, band), path in sorted(maps.items()):
        out = f"{args.out}/{dtype}_{band}.hs"
        if os.path.exists(out):
            print(f"  have  {os.path.basename(out)}")
            continue
        t0 = time.time()
        cov = healsparse.HealSparseCoverage.read(path)
        cov_pixels = np.where(cov.coverage_mask)[0]
        m = healsparse.HealSparseMap.read(path, pixels=cov_pixels, degrade_nside=args.nside)
        tmp = out + ".tmp"  # write then rename, so an interrupted run leaves no half file
        m.write(tmp, clobber=True)
        os.replace(tmp, out)
        vals = m.get_values_pix(m.valid_pixels)
        print(f"  wrote {os.path.basename(out)}: {m.n_valid:,} pixels, values "
              f"{np.nanmin(vals):.4g} to {np.nanmax(vals):.4g}, "
              f"{os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t0:.0f} s")
        del m, cov, vals

    print(f"\nDone. Set external_maps_dir: {args.out} in dp2_run/config_sp_dp2.yml")


if __name__ == "__main__":
    main()
