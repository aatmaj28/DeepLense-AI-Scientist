"""Check the flip/rotation label maps against lenstronomy itself (runs INSIDE the dlens-lenstronomy container).

For random lens systems, renders the noiseless image, applies each of the 16 augmentations
the training loop can draw (column flip, row flip, rot90 k=0..3) to the IMAGE, then
re-renders with the labels moved by src/dlens/data/_lens_labels.py. The two must agree
to numerical precision; the source's position and ellipticity are moved the same way.
A deliberately wrong map (e2 left alone under flips) must be caught, so the test can
fail at all. Also checks theta_E_IV = theta_E (1+q) / (2 sqrt(q)) against lenstronomy's
SIE convergence.

Usage (in container): python check_label_transforms.py /path/to/_lens_labels.py
Prints a JSON summary; exit status 0 only if everything passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel

import lens_regression_sim as sim

spec = importlib.util.spec_from_file_location("lens_labels", sys.argv[1])
ll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ll)

TOL = 1e-6  # max |difference| relative to the image peak


def moved(system: dict, fc: bool, fr: bool, k: int, wrong: bool = False) -> tuple[list, list]:
    lens, src = dict(system["kwargs_lens"][0]), dict(system["kwargs_source"][0])
    for d in (lens, src):
        d["center_x"], d["center_y"] = ll.transform_position(d["center_x"], d["center_y"], fc, fr, k)
        if wrong:  # a plausible sign error: forget that reflections flip e2
            d["e1"], d["e2"] = ll.transform_ellipticity(d["e1"], d["e2"], False, False, k)
        else:
            d["e1"], d["e2"] = ll.transform_ellipticity(d["e1"], d["e2"], fc, fr, k)
    return [lens], [src]


def main() -> int:
    rng = np.random.default_rng(12345)
    systems = [sim.sample(rng) for _ in range(6)]
    worst, failures, caught = 0.0, [], 0
    for s, system in enumerate(systems):
        base = sim.render(system)  # sets the source amp, which the re-renders below reuse
        peak = float(np.abs(base).max())
        for fc, fr, k in ll.ALL_TRANSFORMS:
            expected = ll.transform_image(base, fc, fr, k)
            got = sim.image_model(*moved(system, fc, fr, k), system["exposure_time"])
            err = float(np.abs(got - expected).max()) / peak
            worst = max(worst, err)
            if err > TOL:
                failures.append({"system": s, "flip_cols": fc, "flip_rows": fr, "k": k, "rel_err": err})
            if fc != fr:  # the wrong map differs from the right one only when exactly one flip is on
                bad = sim.image_model(*moved(system, fc, fr, k, wrong=True), system["exposure_time"])
                caught += float(np.abs(bad - expected).max()) / peak > 100 * TOL

    # theta_E_IV: lenstronomy's SIE convergence vs IV's GLEE form, major axis along x.
    sie, conv_err = LensModel(["SIE"]), 0.0
    for q in (0.6, 0.8, 0.95):
        e1, e2 = param_util.phi_q2_ellipticity(0.0, q)
        x, y = np.array([0.7, 0.0, 0.4]), np.array([0.0, 0.9, -0.3])
        kappa = sie.kappa(x, y, [{"theta_E": 1.3, "e1": e1, "e2": e2, "center_x": 0, "center_y": 0}])
        theta_iv = 1.3 * (1 + q) / (2 * np.sqrt(q))
        kappa_iv = theta_iv / ((1 + q) * np.sqrt(x ** 2 + y ** 2 / q ** 2))
        conv_err = max(conv_err, float(np.max(np.abs(kappa / kappa_iv - 1))))

    wrong_cases = len(systems) * sum(1 for fc, fr, _ in ll.ALL_TRANSFORMS if fc != fr)
    ok = not failures and caught == wrong_cases and conv_err < 1e-3
    print(json.dumps({"passed": ok, "systems": len(systems), "transforms": len(ll.ALL_TRANSFORMS),
                      "worst_rel_err": worst, "tolerance": TOL, "failures": failures[:5],
                      "negative_control_caught": f"{caught}/{wrong_cases}",
                      "theta_E_iv_max_rel_err": conv_err}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
