#!/usr/bin/env bash
# Smoke test for dlens-realsky without the 13 GB GalaxiesML download.
#
# Builds a tiny synthetic catalog with the GalaxiesML fields RealSky reads, then
# runs RealSky's own run_pipeline.py once per dark-matter type (CDM, WDM, SIDM,
# Axion), one simulation per instrument, one container at a time, with the same
# hardening as the Model I sandbox. Finally checks every output image.
#
# Usage: bash smoke_test.sh            (env: IMAGE, INSTRUMENTS, TIMEOUT, WORK)
set -euo pipefail

IMAGE=${IMAGE:-dlens-realsky:latest}
INSTRUMENTS=${INSTRUMENTS:-"LSST Roman_VIS"}
TIMEOUT=${TIMEOUT:-1200}  # seconds per container; RealSky retries failed sims forever
# Under $HOME, so Colima and Docker Desktop can bind-mount it.
WORK=${WORK:-$HOME/.cache/dlens/realsky-smoke/$(date +%Y%m%d-%H%M%S)}
HARDEN=(--rm --network=none --memory=2g --cpus=1 --pids-limit=256)

mkdir -p "$WORK/catalog"
docker run "${HARDEN[@]}" -v "$WORK/catalog:/data" "$IMAGE" \
  python /opt/realsky-env/make_synthetic_catalog.py /data/RealSky.hdf5

read -r -a instruments <<< "$INSTRUMENTS"
instruments_yaml=$(printf '    - %s\n' "${instruments[@]}")
status=0
for dm in CDM WDM SIDM Axion; do
  mkdir -p "$WORK/out/$dm"
  cat > "$WORK/config_$dm.yaml" <<EOF
run:
  simulations_per_permutation: 1
  n_workers: 1
  light_profile: INTERPOL
  instruments:
$instruments_yaml
  dm_types:
    - $dm
  nss: true
  output_dir: output_data
data:
  hsc_catalog: input_data/RealSky.hdf5
EOF
  start=$(date +%s)
  set +e
  docker run "${HARDEN[@]}" \
    -v "$WORK/catalog/RealSky.hdf5:/opt/realsky/input_data/RealSky.hdf5:ro" \
    -v "$WORK/config_$dm.yaml:/opt/realsky/configs/default.yaml:ro" \
    -v "$WORK/out/$dm:/work" -w /work \
    "$IMAGE" timeout "$TIMEOUT" python /opt/realsky/scripts/run_pipeline.py > "$WORK/log_$dm.txt" 2>&1
  rc=$?
  set -e
  retries=$(grep -c "Trying a new simulation due to error" "$WORK/log_$dm.txt" || true)
  note=""
  [ "$rc" -eq 124 ] && note=" (timed out)"
  [ "$rc" -eq 137 ] && note=" (killed: out of memory?)"
  echo "$dm: exit $rc$note after $(( $(date +%s) - start ))s; resampled after an error $retries time(s); log $WORK/log_$dm.txt"
  [ "$rc" -eq 0 ] || status=1
done

docker run "${HARDEN[@]}" -v "$WORK/out:/out:ro" "$IMAGE" python /opt/realsky-env/inspect_outputs.py /out || status=1
echo "outputs in $WORK/out"
exit $status
