"""
Automatic layout checks for the TXMeanShearSurveyProperties figure.

Each case is drawn with the stage's own plotting code (as in replot_example.py)
and the drawn figure is measured:
  1. every axes, title, axis label, tick label, colour bar and legend lies
     inside the canvas;
  2. the map, colour bar, trend panel, titles and legend do not overlap;
  3. the map and its colour bar are centred on their target (over the trend
     panel when stacked, in their cell otherwise), within 0.2% of the width;
  4. the map fills its row: in the stacked layout it spans most of the width
     unless it is clipped by the height limit.

Usage (needs skyproj, e.g. the `plots` env):
    python dp2_run/check_plot_layout.py [outdir]
Prints PASS/FAIL per case and check, and a summary; exits 1 on any failure.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import replot_example as rx  # noqa: E402

TOL = 0.002  # figure fraction (~2 px) for the centring check


def hp_one_pixel(nside, ra=150.0, dec=2.0):
    import healpy as hp
    return np.array([hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)])


def edge_cases(mod):
    """name: (pixels function, nside, bins options, projection, map name, map values)."""
    disc = rx.footprint_disc
    return {
        "small_field": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {}, "McBryde", None, None),
        "band_100deg": (lambda n: rx.footprint_band(n, -50, 50, -40, -30), 1024, {}, "McBryde", None, None),
        "dp2_footprint": (rx.footprint_from_file, 1024, {}, "McBryde", None, None),
        "dp2_plus_field": (rx.SCENARIOS["dp2_plus_field"][0], 1024, {}, "McBryde", None, None),
        "several_fields": (rx.SCENARIOS["several_fields"][0], 2048, {}, "McBryde", None, None),
        "single_pixel": (lambda n: hp_one_pixel(n), 256, {}, "McBryde", None, None),
        "polar_cap": (lambda n: rx.footprint_band(n, -180, 180, -90, -75), 512, {}, "McBryde", None, None),
        "full_south": (lambda n: rx.footprint_band(n, -180, 180, -90, 0), 256, {}, "McBryde", None, None),
        "across_ra0_stripe": (lambda n: rx.footprint_band(n, -30, 30, -5, 5), 1024, {}, "McBryde", None, None),
        "mollweide": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {}, "Mollweide", None, None),
        "mollweide_band": (lambda n: rx.footprint_band(n, -50, 50, -40, -30), 1024, {}, "Mollweide", None, None),
        "constant_map": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {}, "McBryde", None, "constant"),
        "long_name": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {}, "McBryde",
                      "deepCoadd_psf_maglim_consolidated_map_weighted_mean_i", None),
        "outliers": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {"outliers": True}, "McBryde", None, None),
        "big_errors": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {"big_errors": True}, "McBryde", None, None),
        "two_bins": (lambda n: disc(n, 53.1, -28.1, 1.2), 4096, {"nbins": 2, "low_bins": False}, "McBryde", None, None),
        "empty_map": (lambda n: np.empty(0, int), 1024, {}, "McBryde", None, None),
    }


def artists(fig):
    """Named boxes (display coords) of everything drawn, grouped by owner."""
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison and not ax.get_title():
            continue
        out.append((ax, "tight", ax.get_tightbbox(r)))
        if ax.get_title():
            out.append((ax, "title", ax.title.get_window_extent(r)))
        leg = ax.get_legend()
        if leg is not None:
            out.append((ax, "legend", leg.get_window_extent(r)))
    return out


def overlap(a, b, pad=1.0):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > pad) and (min(a.y1, b.y1) - max(a.y0, b.y0) > pad)


def check(fig, info):
    """Return a list of (check, ok, detail)."""
    r = fig.canvas.get_renderer()
    fig.canvas.draw()
    W, H = fig.bbox.width, fig.bbox.height
    results = []

    # 1. Inside the canvas (allow 2 px for antialiasing).
    bad = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        bb = ax.get_tightbbox(r)
        if bb is None:
            continue
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
            bad.append(f"{info['role'].get(ax, '?')} [{bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}] of {W:.0f}x{H:.0f}")
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(r)
            if lb.x0 < -2 or lb.x1 > W + 2 or lb.y0 < -2:
                bad.append(f"legend [{lb.x0:.0f},{lb.x1:.0f}] of {W:.0f}")
    results.append(("inside canvas", not bad, "; ".join(bad)))

    # 2. No overlaps between the panels' drawn areas (axes frames, not their
    #    labels) and between titles/legend and other panels.
    role = info["role"]
    boxes = {}
    for ax, name in role.items():
        if name == "placeholder":
            continue
        boxes[name] = ax.get_window_extent(r)
        if ax.get_title():
            boxes[name + " title"] = ax.title.get_window_extent(r)
        if ax.get_legend() is not None:
            boxes[name + " legend"] = ax.get_legend().get_window_extent(r)
        if name in ("map", "trend"):
            boxes[name + " xlabel"] = ax.xaxis.label.get_window_extent(r)
            for t in ax.get_xticklabels():
                if t.get_text():
                    boxes.setdefault(name + " xticks", t.get_window_extent(r))
    clashes = []
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.split()[0] == b.split()[0]:
                continue  # an axes and its own title/labels
            if overlap(boxes[a], boxes[b]):
                clashes.append(f"{a} / {b}")
    # skyproj's tick labels are drawn by its own artists; check the map's
    # tight box (which includes them) against the trend panel's tight box.
    if "map" in boxes and "trend" in boxes:
        mt = next(a for a, n in role.items() if n == "map").get_tightbbox(r)
        tt = next(a for a, n in role.items() if n == "trend").get_tightbbox(r)
        if overlap(mt, tt):
            clashes.append("map (with labels) / trend (with labels)")
    results.append(("no overlaps", not clashes, "; ".join(clashes)))

    # 2b. Tick labels of the colour bar and the trend panel do not run into
    #     each other.
    crowded = []
    for ax, name in role.items():
        if name not in ("colorbar", "trend"):
            continue
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_window_extent(r) for t in axis.get_ticklabels()
                      if t.get_visible() and t.get_text()]
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if overlap(labels[i], labels[j], pad=0.5):
                        crowded.append(f"{name} {axis.axis_name}-ticks")
                        break
                else:
                    continue
                break
    results.append(("tick labels clear", not crowded, "; ".join(crowded)))

    # 3. Centred.
    if "map" in role.values():
        group = [a for a, n in role.items() if n in ("map", "colorbar")]
        x0 = min(a.get_position().x0 for a in group)
        x1 = max(a.get_position().x1 for a in group)
        target = info["centre_target"]
        off = 0.5 * (x0 + x1) - target
        results.append(("centred", abs(off) < TOL, f"offset {off:+.3f} of width"))

        # 4. Fill (stacked only): the map spans the width, or is at the height cap.
        if info["wide"]:
            m = next(a for a, n in role.items() if n == "map").get_position()
            t = next(a for a, n in role.items() if n == "trend").get_position()
            frac = m.width / t.width
            capped = m.height * H / fig.dpi >= 4.95
            results.append(("map fills width", frac > 0.85 or capped,
                            f"map width {frac:.0%} of trend width" + (" (height cap)" if capped else "")))
    return results


def build(mod, scen):
    import healpy as hp
    footprint, nside, opts, projection, name, values = scen
    pix = footprint(nside)
    if pix.size:
        hsp_map, vals = rx.example_map(pix, nside)
        if values == "constant":
            hsp_map[pix] = 1.5
            vals = np.full(pix.size, 1.5) + np.random.default_rng(0).normal(0, 1e-3, pix.size)
    else:
        import healsparse
        hsp_map = healsparse.HealSparseMap.make_empty(32, nside, dtype=np.float64)
        vals = np.random.default_rng(0).normal(2000, 50, 1000)
    mu, g1, g2, s1, s2, counts, min_count = rx.example_bins(vals, **opts)
    usable, stats = mod._null_test_statistics(mu, g1, g2, s1, s2, counts, min_count)
    stage = rx.FakeStage()
    stage.config = dict(stage.config, projection=projection)
    import matplotlib.pyplot as plt
    with plt.rc_context(mod.PAPER_STYLE):
        fig = stage._property_figure(name or "sky_background_i", (hsp_map, nside),
                                     mu, g1, g2, s1, s2, usable, stats)
    return fig, hsp_map


def roles(fig, mod, hsp_map):
    """
    Name the axes (map, colorbar, trend) and find the centring target. skyproj
    replaces the placeholder axes with its own SkyAxes, so the map cell is
    taken from the trend panel's gridspec.
    """
    role = {}
    trend = None
    for ax in fig.axes:
        if ax.get_ylabel() == "Mean shear":
            role[ax] = "trend"
            trend = ax
        elif type(ax).__name__ == "SkyAxes":
            role[ax] = "map"
        elif ax.get_label() == "<colorbar>":
            role[ax] = "colorbar"
        else:
            role[ax] = "placeholder"
    wide = mod._footprint_aspect(hsp_map) > 2.5 if hsp_map.n_valid else False
    cell = trend.get_subplotspec().get_gridspec()[0].get_position(fig)
    target_box = trend.get_position() if wide else cell
    return {"role": role, "wide": wide,
            "centre_target": 0.5 * (target_box.x0 + target_box.x1)}


def main():
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "layout_check")
    outdir.mkdir(parents=True, exist_ok=True)
    mod = rx.load_stage_module()
    rx.FakeStage._property_figure = mod.TXMeanShearSurveyProperties._property_figure

    failures = 0
    for case, scen in edge_cases(mod).items():
        try:
            fig, hsp_map = build(mod, scen)
        except Exception as e:  # a crash is a failure of the case
            print(f"FAIL  {case:18s} crashed: {type(e).__name__}: {e}")
            failures += 1
            continue
        # Measure and save inside the stage's style, as plot_property does:
        # some settings (fonts for maths, the dpi) apply only when drawing.
        with plt.rc_context(mod.PAPER_STYLE):
            info = roles(fig, mod, hsp_map)
            res = check(fig, info)
            fig.savefig(outdir / f"{case}.png")
        plt.close(fig)
        for name, ok, detail in res:
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {case:18s} {name:16s} {detail}")
    print(f"\n{failures} failed check(s). Figures in {outdir}/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
