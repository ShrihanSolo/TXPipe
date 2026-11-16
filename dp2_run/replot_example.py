"""
Draw the TXMeanShearSurveyProperties figure from a made-up footprint, to work on
the plot layout without running the stage or copying maps from NERSC.

The footprint is a ~2.4 deg wide DP2-like field near ECDFS at nside 4096, with
a sky-background-like gradient. The binned shears are random with a small
trend, and the statistics come from the stage's own _null_test_statistics.

Usage (needs skyproj, e.g. the `plots` env):
    python dp2_run/replot_example.py [outdir]
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STAGE_FILE = HERE.parent / "txpipe" / "survey_property_null_tests.py"


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


def example_map(nside=4096, ra0=53.1, dec0=-28.1, radius=1.2):
    """A round field with a smooth gradient plus noise, like a sky background map."""
    import healpy as hp
    import healsparse

    vec = hp.ang2vec(ra0, dec0, lonlat=True)
    pix = hp.query_disc(nside, vec, np.radians(radius), nest=True)
    ra, dec = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    rng = np.random.default_rng(1)
    vals = 2000 + 150 * (ra - ra0) + 80 * (dec - dec0) + rng.normal(0, 20, pix.size)
    m = healsparse.HealSparseMap.make_empty(32, nside, dtype=np.float64)
    m[pix] = vals
    return m, vals


def example_bins(vals, nbins=20, min_count=1000):
    rng = np.random.default_rng(2)
    edges = np.quantile(vals, np.linspace(0, 1, nbins + 1))
    mu = 0.5 * (edges[1:] + edges[:-1])
    counts = np.full(nbins, 40000)
    counts[0] = counts[-1] = 600  # two bins below min_count, drawn open grey
    sigma = 0.26 / np.sqrt(counts)
    g1 = 3e-4 + 2e-6 * (mu - mu.mean()) + rng.normal(0, sigma)
    g2 = -2e-4 + rng.normal(0, sigma)
    return mu, g1, g2, sigma, sigma.copy(), counts, min_count


class FakeStage:
    config = {"map_plot_nside": 256, "projection": "McBryde"}


def main():
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    outdir.mkdir(parents=True, exist_ok=True)
    mod = load_stage_module()
    import matplotlib
    matplotlib.use("agg")

    hsp_map, vals = example_map()
    mu, g1, g2, s1, s2, counts, min_count = example_bins(vals)
    usable, stats = mod._null_test_statistics(mu, g1, g2, s1, s2, counts, min_count)
    name = "sky_background_i"
    path = outdir / f"{name}.png"
    mod.TXMeanShearSurveyProperties.plot_property(
        FakeStage(), path, name, (hsp_map, 4096), mu, g1, g2, s1, s2, usable, stats
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
