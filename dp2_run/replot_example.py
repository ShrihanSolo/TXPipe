"""
Draw the TXMeanShearSurveyProperties figure from a made-up footprint, to work on
the plot layout without running the stage or copying maps from NERSC.

The footprints are a ~2.4 deg field near ECDFS, a 100 x 10 deg band and
several separate small fields, each with a sky-background-like gradient. The binned shears are random with a small
trend, and the statistics come from the stage's own _null_test_statistics.

Usage (needs skyproj, e.g. the `plots` env):
    python dp2_run/replot_example.py [outdir] [scenario ...]
Scenarios are listed in SCENARIOS; with none given, all are drawn.
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STAGE_FILE = HERE.parent / "txpipe" / "survey_property_null_tests.py"
# The real DP2 footprint: nest nside-128 pixels holding kept stars in the
# fiducial rho run of the DP2 PSF-star catalog (inside the shear mask).
DP2_FOOTPRINT = "/workspace/TXPipe-notes/replot/dp2_footprint_nest128.npy"


def load_stage_module():
    """Import the stage file alone, with the TXPipe/ceci imports it needs faked."""
    class Anything:
        def __init__(self, *a, **k):
            pass

    def fake(name, **attrs):
        mod = types.ModuleType(name)
        mod.__path__ = []
        def getattr_(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return Anything
        mod.__getattr__ = getattr_
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    fake("ceci")
    fake("ceci.config", StageParameter=Anything)
    fake("txpipe")
    fake("txpipe.base_stage", PipelineStage=object)
    fake("txpipe.data_types")
    fake("txpipe.calibration")
    fake("txpipe.calibration.calibration_utils")
    fake("txpipe.calibration.calibration_utils.names", META_VARIANTS=[])
    fake("txpipe.utils")
    spec = importlib.util.spec_from_file_location("txpipe.survey_property_null_tests", STAGE_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def footprint_disc(nside, ra0, dec0, radius):
    import healpy as hp
    vec = hp.ang2vec(ra0, dec0, lonlat=True)
    return hp.query_disc(nside, vec, np.radians(radius), nest=True)


def footprint_band(nside, ra_min, ra_max, dec_min, dec_max):
    import healpy as hp
    theta = np.radians(90 - np.array([dec_max, dec_min]))
    pix = hp.query_strip(nside, theta[0], theta[1], nest=True)
    ra, _ = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    ra = np.where(ra > 180, ra - 360, ra)
    return pix[(ra >= ra_min) & (ra <= ra_max)]


def example_map(pix, nside):
    """A smooth gradient across the footprint plus noise, like a sky background map."""
    import healpy as hp
    import healsparse

    ra, dec = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    ra = np.where(ra > 180, ra - 360, ra)
    x = (ra - ra.mean()) / max(np.ptp(ra), 1e-3)
    y = (dec - dec.mean()) / max(np.ptp(dec), 1e-3)
    rng = np.random.default_rng(1)
    vals = 2000 + 300 * x + 150 * y + rng.normal(0, 20, pix.size)
    m = healsparse.HealSparseMap.make_empty(32, nside, dtype=np.float64)
    m[pix] = vals
    return m, vals


def example_bins(vals, nbins=20, min_count=1000, low_bins=True, outliers=False,
                 big_errors=False):
    rng = np.random.default_rng(2)
    edges = np.quantile(vals, np.linspace(0, 1, nbins + 1))
    mu = 0.5 * (edges[1:] + edges[:-1])
    counts = np.full(nbins, 40000)
    if low_bins:
        counts[0] = counts[-1] = 600  # two bins below min_count, drawn open grey
    if big_errors:
        counts[[3, 11]] = 1100  # used bins with ~6x the typical error
    sigma = 0.26 / np.sqrt(counts)
    g1 = 3e-4 + 2e-6 * (mu - mu.mean()) + rng.normal(0, sigma)
    g2 = -2e-4 + rng.normal(0, sigma)
    if outliers:
        g1[7] += 0.012  # ~9 sigma, used in the fit
        g2[15] -= 0.008
    return mu, g1, g2, sigma, sigma.copy(), counts, min_count


def footprint_from_file(nside, path=DP2_FOOTPRINT, nside_file=128):
    """Nest pixels at nside_file (e.g. the DP2 shear-mask footprint), upgraded to nside."""
    pix = np.load(path)
    k = (nside // nside_file) ** 2
    return (pix[:, None] * k + np.arange(k)).ravel()


# name: (footprint, nside, options for example_bins)
SCENARIOS = {
    "small_field": (lambda n: footprint_disc(n, 53.1, -28.1, 1.2), 4096, {}),
    "outliers": (lambda n: footprint_disc(n, 53.1, -28.1, 1.2), 4096, {"outliers": True}),
    "big_errors": (lambda n: footprint_disc(n, 53.1, -28.1, 1.2), 4096, {"big_errors": True}),
    "all_used": (lambda n: footprint_disc(n, 53.1, -28.1, 1.2), 4096, {"low_bins": False}),
    "band_100deg": (lambda n: footprint_band(n, -50, 50, -40, -30), 1024, {}),
    "dp2_footprint": (footprint_from_file, 1024, {}),
    "several_fields": (lambda n: np.concatenate([
        footprint_disc(n, 53.1, -28.1, 1.2), footprint_disc(n, 59.1, -48.7, 1.2),
        footprint_disc(n, 6.0, -72.1, 1.0), footprint_disc(n, 95.0, -25.0, 1.0)]), 2048, {}),
}


class FakeStage:
    config = {"map_plot_nside": 256, "projection": "McBryde"}


def main():
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    names = sys.argv[2:] or list(SCENARIOS)
    outdir.mkdir(parents=True, exist_ok=True)
    mod = load_stage_module()
    import matplotlib
    matplotlib.use("agg")

    for scenario in names:
        footprint, nside, opts = SCENARIOS[scenario]
        hsp_map, vals = example_map(footprint(nside), nside)
        mu, g1, g2, s1, s2, counts, min_count = example_bins(vals, **opts)
        usable, stats = mod._null_test_statistics(mu, g1, g2, s1, s2, counts, min_count)
        path = outdir / f"{scenario}.png"
        mod.TXMeanShearSurveyProperties.plot_property(
            FakeStage(), path, "sky_background_i", (hsp_map, nside),
            mu, g1, g2, s1, s2, usable, stats,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
