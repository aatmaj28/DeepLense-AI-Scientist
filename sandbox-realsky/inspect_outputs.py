"""Check and summarize RealSky output files.

For every simulation: its dark-matter type, instrument, light profile and bands,
and for each band whether the lensed, no-substructure and unlensed images exist,
are 127x127 and finite, and are not all zero.

Usage: python inspect_outputs.py OUTPUT_DIR
Exit status: 0 if every expected image is present and valid, 1 otherwise.
"""
import glob
import os
import sys

import h5py
import numpy as np


def values(ds):
    """RealSky stores metadata as [value..., comment]; return the values."""
    vals = [v.decode() if isinstance(v, bytes) else str(v) for v in ds[()]]
    return vals[:-1]


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "*.h5"), recursive=True))
    if not files:
        print("no RealSky output files found")
        return 1
    problems = []
    for path in files:
        with h5py.File(path, "r") as f:
            for name in sorted(f["images"]):
                g = f["images"][name]
                idx = name.rsplit("_", 1)[1]
                meta = {k: values(g[k]) for k in ("DM_type", "instrument", "light_profile", "bands",
                                                  "num_subhalos", "theta_e", "z_lens", "z_source", "snr")
                        if k in g}
                peaks = []
                for band in meta.get("bands", []):
                    for suffix in ("", "_nss", "_unlensed"):
                        key = f"exposure_{idx}_{band}{suffix}"
                        if key not in g:
                            problems.append(f"{os.path.basename(path)}:{name}: missing {key}")
                            continue
                        img = g[key][()]
                        if img.shape != (127, 127) or not np.isfinite(img).all() or not img.any():
                            problems.append(f"{os.path.basename(path)}:{name}: bad {key} "
                                            f"(shape {img.shape}, finite {np.isfinite(img).mean():.3f})")
                        if suffix == "":
                            peaks.append(f"{band}:{img.max():.3g}")
                kappa = g.get(f"convergence_field_{idx}")
                print(f"{meta.get('DM_type', ['?'])[0]:6s} {meta.get('instrument', ['?'])[0]:10s} "
                      f"{meta.get('light_profile', ['?'])[0]:9s} subhalos={meta.get('num_subhalos', ['?'])[0]:>6s} "
                      f"theta_E={meta.get('theta_e', ['?'])[0]:>5s} z=({meta.get('z_lens', ['?'])[0]}, "
                      f"{meta.get('z_source', ['?'])[0]}) peak counts {' '.join(peaks)}"
                      + (f" kappa finite {np.isfinite(kappa[()]).mean():.4f}" if kappa is not None else ""))
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
