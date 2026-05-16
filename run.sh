#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

build_dir="${DS_VK_BUILD_DIR:-build}"
app_name="${DS_VK_APP:-basic}"

if [[ "${1:-}" == "--app" ]]; then
    app_name="${2:-basic}"
    shift 2
fi

case "${app_name}" in
    basic)
        target="ds_vk_basic_app"
        ;;
    vectorfield)
        target="ds_vk_vectorfield_app"
        ;;
    dfsph)
        target="ds_vk_dfsph_app"
        ;;
    pba)
        target="ds_vk_pba_app"
        ;;
    *)
        echo "unknown app '${app_name}' (expected basic, vectorfield, dfsph, or pba)" >&2
        exit 2
        ;;
esac

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    cmake -S . -B "${build_dir}"
fi

cmake --build "${build_dir}" --target "${target}"
exec "${build_dir}/${target}" "$@"
