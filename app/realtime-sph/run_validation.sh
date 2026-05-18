#!/usr/bin/env bash
set -euo pipefail

app="${1:-./build/ds_vk_realtime_sph_app}"
fixed_dt="0.0166667"

echo "[realtime-sph] 20k dam-break"
"${app}" --smoke-frames 300 --fixed-frame-dt "${fixed_dt}" --hide-ui --print-stats --fail-on-invalid --max-z-span 1.4

echo "[realtime-sph] 50k dam-break"
"${app}" --particles 50000 --smoke-frames 300 --fixed-frame-dt "${fixed_dt}" --hide-ui --print-stats --fail-on-invalid --max-z-span 2.4

echo "[realtime-sph] 4096 no-gravity cube invariant"
"${app}" --scenario no-gravity-cube --particles 4096 --smoke-frames 300 --fixed-frame-dt "${fixed_dt}" --hide-ui --print-stats --fail-on-invalid --max-center-delta 0.01 --max-speed 1.0

echo "[realtime-sph] 8192 opposing-cubes invariant"
"${app}" --scenario opposing-cubes --particles 8192 --smoke-frames 300 --fixed-frame-dt "${fixed_dt}" --hide-ui --print-stats --fail-on-invalid --max-speed 3.0
