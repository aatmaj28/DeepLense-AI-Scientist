"""Write a tiny HDF5 catalog with the GalaxiesML fields RealSky reads, for smoke tests.

RealSky reads specz_redshift, image (N, 5, 127, 127) float32, {g,r,i,z}_cmodel_mag,
g_half_light_radius (pixels), g_ellipticity and g_pos_angle (degrees) from
GalaxiesML (Zenodo 10.5281/zenodo.11117528). This file has the same names, dtypes,
shapes and units, filled with smooth synthetic galaxies, so the pipeline can run
without the 13 GB download. It is not science data.

Usage: python make_synthetic_catalog.py OUT.hdf5 [--n 300] [--seed 0]
"""
import argparse

import h5py
import numpy as np

NPIX = 127   # GalaxiesML cutouts are 127x127 at 0.168"/pixel
NBANDS = 5   # g, r, i, z, y


def exponential_galaxy(r_half, ellipticity, pos_angle_deg):
    """An n=1 Sersic (exponential) profile centred on the middle pixel."""
    y, x = np.mgrid[:NPIX, :NPIX].astype(float)
    c = (NPIX - 1) / 2
    theta = np.deg2rad(pos_angle_deg)
    dx, dy = x - c, y - c
    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)
    r = np.hypot(xr, yr / (1.0 - ellipticity))
    return np.exp(-1.678 * r / r_half)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--n", type=int, default=300, help="number of galaxies (default 300)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.n

    # Redshifts spanning what RealSky samples (lens and source, z_source <= 4).
    z = np.sort(rng.uniform(0.05, 4.0, n))
    g = rng.uniform(21.5, 24.5, n)
    r = g - rng.uniform(0.2, 0.8, n)
    i = r - rng.uniform(0.1, 0.5, n)
    zmag = i - rng.uniform(0.0, 0.3, n)
    r_half = rng.uniform(2.5, 7.0, n)          # pixels, as in GalaxiesML (median ~4.3)
    ellipticity = rng.uniform(0.0, 0.5, n)
    pos_angle = rng.uniform(-90.0, 90.0, n)    # degrees, as in GalaxiesML
    mags = np.stack([g, r, i, zmag, zmag - 0.1], axis=1)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("object_id", data=np.arange(n, dtype=np.int64))
        f.create_dataset("specz_redshift", data=z)
        for band, col in zip("griz", range(4)):
            f.create_dataset(f"{band}_cmodel_mag", data=mags[:, col])
        f.create_dataset("g_half_light_radius", data=r_half)
        f.create_dataset("g_ellipticity", data=ellipticity)
        f.create_dataset("g_pos_angle", data=pos_angle)
        images = f.create_dataset("image", shape=(n, NBANDS, NPIX, NPIX), dtype="float32",
                                  chunks=(1, NBANDS, NPIX, NPIX), compression="gzip", compression_opts=4)
        for k in range(n):
            profile = exponential_galaxy(r_half[k], ellipticity[k], pos_angle[k])
            images[k] = (profile[None] * 10 ** (-0.4 * (mags[k][:, None, None] - 25.0))).astype("float32")
    print(f"wrote {n} synthetic galaxies to {args.out}")


if __name__ == "__main__":
    main()
