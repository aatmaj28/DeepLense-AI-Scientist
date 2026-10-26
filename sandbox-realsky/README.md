# RealSky sandbox (Model A)

An isolated Python 3.12 environment for running Lucca Paris's
[RealSky](https://github.com/lparis3/deeplens_update) pipeline, which produced
the DeepLense Model A dataset. It sits alongside the Model I image in
[`../sandbox`](../sandbox), which stays on lenstronomy 1.9.2 for DeepLenseSim.
Nothing in the agent uses this image yet.

**RealSky has no license**, so none of its code is in this repository. The
Dockerfile clones it at build time, pinned to commit
`429bebe0e0280973f50d4315d18651ebde53d16f`. For the same reason, build the
image locally and don't publish it: it contains RealSky's code.

## Build

```bash
docker build -t dlens-realsky:latest sandbox-realsky/
```

## Versions

| Package | Version | Why |
|---|---|---|
| lenstronomy | 1.13.6 | RealSky needs Roman `time_domain_wide` (≥ 1.13.5) and `SimAPI(numpix=)` plus the Euclid `VIS_obs` global (≤ 1.14.0) |
| pyHalo | 1.4.1 | Newest release whose SIDM preset works with RealSky's arguments; 1.4.2–1.4.6 fail on `x_core_halo`, 1.4.7+ need a lenstronomy fork |
| colossus, mcfit | 1.4.0, 0.0.22 | pyHalo imports them but does not declare them |
| torch | 2.14.1+cpu | CPU-only wheel from the PyTorch index |
| nflows | 0.14 | RealSky's normalizing flows |

RealSky never pinned its versions, so this stack is reconstructed from the API
it calls, not confirmed by its author.

Every build records what it installed, where a run can read it:

- `$REALSKY_VERSIONS` (`/opt/realsky-env/versions.json`): Python, the RealSky
  commit and the key packages. `docker run --rm dlens-realsky` prints it.
- `/opt/realsky-env/pip-freeze.txt`: every installed package.

`requirements.txt` lists the top-level packages. `constraints.txt` pins every
transitive dependency to the first verified build's `pip freeze`, so rebuilds
install identical versions. To refresh it after changing `requirements.txt`, empty
it, build, and regenerate it from `/opt/realsky-env/pip-freeze.txt` (dropping the
`-e` RealSky line and `torch`).

## Running a simulation

RealSky reads its config from `/opt/realsky/configs/default.yaml` and its
GalaxiesML catalog from `/opt/realsky/input_data/RealSky.hdf5`, and writes to
`./output_data` in the working directory. Mount all three, and use the same
limits as the Model I sandbox:

```bash
docker run --rm --network=none --memory=2g --cpus=1 --pids-limit=256 \
  -v "$CATALOG:/opt/realsky/input_data/RealSky.hdf5:ro" \
  -v "$CONFIG:/opt/realsky/configs/default.yaml:ro" \
  -v "$OUT:/work" -w /work \
  dlens-realsky:latest timeout 1200 python /opt/realsky/scripts/run_pipeline.py
```

- Paths must be under `$HOME` for Colima to mount them.
- Always wrap the run in `timeout`: RealSky resamples and retries any simulation
  that raises, forever, so a systematic error never exits on its own.
- The catalog is one of GalaxiesML's `5x127x127_*_with_morphology.hdf5` files
  (Zenodo 10.5281/zenodo.11117528, CC-BY-4.0). RealSky's 2024 prototype used the
  13.2 GB testing file.

## Smoke test

```bash
bash sandbox-realsky/smoke_test.sh
```

This needs no download. `make_synthetic_catalog.py` writes 300 smooth synthetic
galaxies with the GalaxiesML fields, dtypes and units RealSky reads. The script
then runs one simulation per dark-matter type (CDM, WDM, SIDM, Axion) for LSST,
which uses real-image INTERPOL sources, and for Roman_VIS (Model A, which forces
Sersic sources). It runs one container at a time and checks every output image with
`inspect_outputs.py`. Set `INSTRUMENTS`, `TIMEOUT` or `WORK` to change it.

Result on 2026-10-26 (Colima, aarch64), run twice: all 8 simulations completed with
no retries, in about 90 s in total. Every lensed, no-substructure and unlensed image
was present, 127×127, finite and non-zero, and every convergence map was finite. The
slowest run was SIDM with ~35,000 subhalos (52 s). Nothing exceeded the 2 GB memory
cap, and no run tried to reach the network. The only warning is an
`AstropyDeprecationWarning` from lenstronomy under astropy 8.

## RealSky behaviour to be aware of

These come from an audit of commit `429bebe` and are RealSky's own choices, not this
image's:

- Roman_VIS always uses Sersic sources, so Model A's sources are not real galaxy images.
- Noise is Poisson only: it calls `noise_for_model(..., background_noise=False)`.
- Images are scaled by an extra `pixel_scale²`, which makes them 14–98× fainter than
  their catalog magnitudes. Exposure counts are scaled up to compensate for SNR.
- Euclid VIS adds the r and i renders, which doubles the flux.
- Random seeds come from process IDs, so runs are not reproducible.
- Within one run, every instrument gets the same sampled lens system: the smoke
  test's LSST and Roman_VIS simulations had identical redshifts, θ_E and masses.
  The likely cause is that each instrument's worker is forked from the same parent
  and inherits its torch random state. Separate runs draw different systems.
