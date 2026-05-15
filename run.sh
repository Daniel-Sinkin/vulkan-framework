#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

build_dir="${DS_VK_BUILD_DIR:-build}"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    cmake -S . -B "${build_dir}"
fi

cmake --build "${build_dir}" --target ds_vk_basic_app
exec "${build_dir}/ds_vk_basic_app" "$@"
