#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

build_dir="${DS_VK_BUILD_DIR:-build}"
app_name="${DS_VK_APP:-basic}"

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --app)
            app_name="${2:-basic}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

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
    quake | vk_quake)
        target="ds_vk_quake_app"
        ;;
    *)
        echo "unknown app '${app_name}' (expected basic, vectorfield, dfsph, pba, or quake)" >&2
        exit 2
        ;;
esac

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    cmake -S . -B "${build_dir}"
fi

build_log="$(mktemp "${TMPDIR:-/tmp}/ds_vk_build.XXXXXX")"
cleanup() {
    rm -f "${build_log}"
}
trap cleanup EXIT

if ! cmake --build "${build_dir}" --target "${target}" -j >"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

printf '\033[H\033[2J'
exec "${build_dir}/${target}" "$@"
