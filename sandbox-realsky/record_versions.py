"""Print the image's installed versions as JSON (run at build time into $REALSKY_VERSIONS)."""
import importlib.metadata as metadata
import json
import os
import platform

PACKAGES = ("lenstronomy", "pyHalo", "colossus", "mcfit", "nflows", "torch", "numpy", "scipy",
            "astropy", "h5py", "numba", "matplotlib", "pandas", "scikit-learn", "pyyaml", "RealSky")

versions = {}
for name in PACKAGES:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None

print(json.dumps({
    "python": platform.python_version(),
    "machine": platform.machine(),
    "realsky_commit": os.environ.get("REALSKY_COMMIT"),
    "packages": versions,
}, indent=1))
