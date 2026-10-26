"""Sampling and rendering for the SIE regression dataset (runs INSIDE the dlens-lenstronomy container).

The renderer, grid, PSF and noise are DeepLenseSim Model_I's ``simple_sim`` (150 px at
0.05"/px, Gaussian PSF FWHM 0.087", background rms 0.01, exposure 10**U(3, 3.5), Poisson
noise), so images look like Model_I. Only the lens and source are varied, with the v1
ranges from the regression design (research/regression-design.md, section 1):

    theta_E      U(0.5, 2.5)"              lens: SIE only (no shear, no lens light,
    q, phi       U(0.6, 1.0), U(0, pi)            no substructure)
    center_x/y   U(-0.3, 0.3)" each
    source pos   uniform in a disk of radius 0.5*theta_E around the lens centre
    source       SERSIC_ELLIPSE: R U(0.1, 0.5)", n U(0.5, 2.5), q U(0.5, 1), phi U(0, pi)
    brightness   peak surface brightness of the lensed, PSF-convolved source,
                 log-uniform in [90, 450] counts/s/arcsec^2

Brightness is set at the image's brightest pixel rather than through amp, so the peak
SNR stays in the DeepLense band [14, 60] for every Sersic index. lenstronomy's amp is
the brightness at R_sersic: with amp drawn directly (the design's U(10, 40)) the peak
per unit amp varied 13x (p1-p99), mostly with n, and 20% of images fell below a
noiseless peak SNR of 14. The profile's central
brightness amp * exp(b_n) over-corrects, because a high-n cusp is narrower than a
pixel and the PSF: its rendered peak is 26% of it at n ~ 2.25 against 93% at n < 1.
The image is linear in amp, so render() draws a unit-amp image and scales it.

Python 3.10 / lenstronomy 1.9.2 (the sandbox image); numpy only beyond that.
"""

from __future__ import annotations

import numpy as np
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.param_util as param_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util

NUM_PIX = 150
DELTA_PIX = 0.05
FWHM = 0.087
BACKGROUND_RMS = 1.0e-2

RANGES = {
    "theta_E": (0.5, 2.5), "q": (0.6, 1.0), "phi": (0.0, np.pi), "center": (-0.3, 0.3),
    "src_radius_frac": 0.5, "src_R_sersic": (0.1, 0.5), "src_n_sersic": (0.5, 2.5),
    "src_q": (0.5, 1.0), "src_phi": (0.0, np.pi), "src_peak_sb": (90.0, 450.0),
    "log10_exposure": (3.0, 3.5),
}
SNR_CORNER = 10  # px; side of the four corner boxes measured_snr takes the background from

_LENS = LensModel(["SIE"])
_SOURCE = LightModel(["SERSIC_ELLIPSE"])
_, _, _RA0, _DEC0, _, _, _PIX2ANGLE, _ = util.make_grid_with_coordtransform(
    numPix=NUM_PIX, deltapix=DELTA_PIX, center_ra=0, center_dec=0, subgrid_res=1, inverse=False)


def sample(rng: np.random.Generator) -> dict:
    """Draw one lens system. Returns lenstronomy kwargs plus the derived labels.

    The source amp is a placeholder of 1 until render() sets it from src_peak_sb.
    """
    r = RANGES
    theta_e = rng.uniform(*r["theta_E"])
    q, phi = rng.uniform(*r["q"]), rng.uniform(*r["phi"])
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    cx, cy = rng.uniform(*r["center"], size=2)
    rad = r["src_radius_frac"] * theta_e * np.sqrt(rng.uniform())
    ang = rng.uniform(0.0, 2 * np.pi)
    src_q, src_phi = rng.uniform(*r["src_q"]), rng.uniform(*r["src_phi"])
    se1, se2 = param_util.phi_q2_ellipticity(src_phi, src_q)
    return {
        "kwargs_lens": [{"theta_E": theta_e, "e1": e1, "e2": e2, "center_x": cx, "center_y": cy}],
        "kwargs_source": [{"amp": 1.0, "R_sersic": rng.uniform(*r["src_R_sersic"]),
                           "n_sersic": rng.uniform(*r["src_n_sersic"]), "e1": se1, "e2": se2,
                           "center_x": cx + rad * np.cos(ang), "center_y": cy + rad * np.sin(ang)}],
        "q": q, "phi": phi,
        "src_peak_sb": float(np.exp(rng.uniform(*np.log(r["src_peak_sb"])))),
        "exposure_time": 10 ** rng.uniform(*r["log10_exposure"]),
    }


def image_model(kwargs_lens: list, kwargs_source: list, exposure_time: float) -> np.ndarray:
    """Noiseless model image (counts per second per pixel), exactly as Model_I renders it."""
    data = ImageData(background_rms=BACKGROUND_RMS, exposure_time=exposure_time,
                     ra_at_xy_0=_RA0, dec_at_xy_0=_DEC0, transform_pix2angle=_PIX2ANGLE,
                     image_data=np.zeros((NUM_PIX, NUM_PIX)))
    psf = PSF(psf_type="GAUSSIAN", fwhm=FWHM, pixel_size=DELTA_PIX, truncation=3)
    model = ImageModel(data, psf, lens_model_class=_LENS, source_model_class=_SOURCE,
                       kwargs_numerics={"supersampling_factor": 1, "supersampling_convolution": False},
                       lens_light_model_class=None)
    return model.image(kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None)


def render(system: dict) -> np.ndarray:
    """Noiseless image with the source scaled to its sampled peak brightness.

    Sets the system's source amp (in place) to the value that gives that peak.
    """
    src = system["kwargs_source"][0]
    unit = image_model(system["kwargs_lens"], [{**src, "amp": 1.0}], system["exposure_time"])
    src["amp"] = system["src_peak_sb"] * DELTA_PIX ** 2 / float(unit.max())
    return unit * src["amp"]


def observe(model: np.ndarray, exposure_time: float) -> np.ndarray:
    """Model_I's noise: Poisson + Gaussian background on the model, then a Poisson draw.

    Uses numpy's global RandomState, as lenstronomy's image_util does; seed it first.
    """
    poisson = image_util.add_poisson(model, exp_time=exposure_time)
    bkg = image_util.add_background(model, sigma_bkd=BACKGROUND_RMS)
    return np.random.poisson((exposure_time * (model + poisson + bkg)).clip(min=0))


def peak_snr(model: np.ndarray, exposure_time: float) -> float:
    """Peak-pixel SNR of the noiseless model: signal over Poisson plus background noise."""
    peak = float(model.max()) * exposure_time
    return peak / np.sqrt(peak + (BACKGROUND_RMS * exposure_time) ** 2)


def measured_snr(img: np.ndarray) -> float:
    """Peak SNR of a noisy image, on the scale the validator's [14, 60] band came from.

    (max - background mean) / sqrt(max + background variance), with the background
    from the four SNR_CORNER-px corner boxes. On 900 Model_I images this gives
    p5/median/p95 14.8/18.0/21.8 against the recorded calibration 14.8/18.1/22.0.
    Corners rather than the whole border, because arcs of theta_E > ~2" reach the
    border and inflate its variance.
    """
    b = SNR_CORNER
    bkg = np.concatenate([img[:b, :b].ravel(), img[:b, -b:].ravel(), img[-b:, :b].ravel(), img[-b:, -b:].ravel()])
    peak = float(img.max())
    return (peak - float(bkg.mean())) / np.sqrt(peak + float(bkg.var()))


def labels(system: dict) -> dict:
    """The labels.csv row for one system (lenstronomy conventions, plus IV's for comparison).

    HOLISMOKES IV (Schuldt+2021) uses e_IV = (1-q^2)/(1+q^2) (cos 2phi, sin 2phi) and an
    SIE normalised by the axes' arithmetic mean, so theta_E_IV = theta_E (1+q) / (2 sqrt(q)).
    """
    lens, src = system["kwargs_lens"][0], system["kwargs_source"][0]
    q, phi = system["q"], system["phi"]
    e_iv = (1 - q ** 2) / (1 + q ** 2)
    return {
        "theta_E": lens["theta_E"], "e1": lens["e1"], "e2": lens["e2"],
        "center_x": lens["center_x"], "center_y": lens["center_y"],
        "q": q, "phi": phi,
        "e_x_iv": e_iv * np.cos(2 * phi), "e_y_iv": e_iv * np.sin(2 * phi),
        "theta_E_iv": lens["theta_E"] * (1 + q) / (2 * np.sqrt(q)),
        "gamma1": 0.0, "gamma2": 0.0,
        "src_x": src["center_x"], "src_y": src["center_y"], "src_R_sersic": src["R_sersic"],
        "src_n_sersic": src["n_sersic"], "src_e1": src["e1"], "src_e2": src["e2"], "src_amp": src["amp"],
        "src_peak_sb": system["src_peak_sb"],
        "src_central_sb": src["amp"] * float(np.exp(SersicUtil.b_n(src["n_sersic"]))),
        "exposure_time": system["exposure_time"],
    }
