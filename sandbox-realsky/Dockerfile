# Sandbox image for RealSky, Lucca Paris's Model A simulation pipeline
# (github.com/lparis3/deeplens_update). It sits alongside the Model I image in
# ../sandbox, which stays on lenstronomy 1.9.2 for DeepLenseSim.
#
# RealSky has no license, so none of its code is copied into this repository:
# it is cloned at build time, pinned to one commit.
#
# Versions: lenstronomy 1.13.6 + pyHalo 1.4.1 is the newest released stack that
# runs every call RealSky makes, including all four dark-matter presets.
# lenstronomy 1.13.5-1.14.0 have the API it uses (Roman time_domain_wide,
# SimAPI(numpix=), the Euclid VIS_obs global); pyHalo 1.4.2-1.4.6 break its
# SIDM preset, and 1.4.7+ need profiles that exist only in a lenstronomy fork.
# RealSky never pinned versions, so this stack is reconstructed, not confirmed.

FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016

# git is needed only to fetch RealSky. Patch base-image CVEs where possible.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch from the PyTorch index; PyPI's Linux wheels pull in CUDA.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"

# Pinned science stack. constraints.txt pins every transitive dependency to the
# versions of the first verified build, so rebuilds resolve the same packages.
COPY requirements.txt constraints.txt /opt/realsky-env/
RUN pip install --no-cache-dir -r /opt/realsky-env/requirements.txt -c /opt/realsky-env/constraints.txt

# RealSky at a fixed commit. Editable install, because RealSky locates its
# configs and network checkpoints relative to its own source tree.
ARG REALSKY_COMMIT=429bebe0e0280973f50d4315d18651ebde53d16f
RUN git clone --quiet https://github.com/lparis3/deeplens_update.git /opt/realsky \
    && git -C /opt/realsky checkout --quiet --detach "$REALSKY_COMMIT" \
    && test "$(git -C /opt/realsky rev-parse HEAD)" = "$REALSKY_COMMIT" \
    && rm -rf /opt/realsky/.git \
    && pip install --no-cache-dir --no-deps -e /opt/realsky

# Record what is installed, where any run can read it: $REALSKY_VERSIONS.
COPY record_versions.py make_synthetic_catalog.py inspect_outputs.py /opt/realsky-env/
RUN REALSKY_COMMIT="$REALSKY_COMMIT" python /opt/realsky-env/record_versions.py > /opt/realsky-env/versions.json \
    && pip freeze > /opt/realsky-env/pip-freeze.txt
ENV REALSKY_VERSIONS=/opt/realsky-env/versions.json
LABEL org.opencontainers.image.title="dlens-realsky" \
      realsky.commit="$REALSKY_COMMIT"

# Generated code and simulations run as a non-root user.
RUN useradd -m -u 10001 sandbox
USER sandbox
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /work

# Build-time check that the whole pipeline imports.
RUN python -c "import RealSky_pipeline.pipeline.parent_pipeline, lenstronomy, pyHalo, nflows, torch; print('RealSky sandbox OK')"

CMD ["python", "-c", "import json, os; print(json.dumps(json.load(open(os.environ['REALSKY_VERSIONS'])), indent=1))"]
