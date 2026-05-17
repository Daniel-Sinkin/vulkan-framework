#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

app_name="${DS_VK_APP:-quake}"

if [[ "${1:-}" == "--app" ]]; then
    app_name="${2:-quake}"
    shift 2
fi

if ! command -v watchexec >/dev/null 2>&1; then
    echo "watcher.sh requires watchexec. Install it with: brew install watchexec" >&2
    exit 127
fi

watch_args=(
    --restart
    --quiet
    --clear=reset
    --postpone
    --stop-signal SIGTERM
    --stop-timeout 2s
    --watch app
    --watch ds_vk
    --watch assets
    --ignore 'build/**'
    --ignore 'build-*/**'
    --ignore 'run/**'
    --ignore 'imgui.ini'
    --ignore '**/*.swp'
    --ignore '**/*.swo'
    --ignore '**/*~'
    --ignore '**/.DS_Store'
)

if [[ "${app_name}" == "quake" || "${app_name}" == "vk_quake" ]]; then
    watch_args=(
        --restart
        --quiet
        --clear=reset
        --postpone
        --stop-signal SIGTERM
        --stop-timeout 2s
        --watch app/quake_main.cpp
        --watch app/quake
        --watch ds_vk
        --ignore 'build/**'
        --ignore 'build-*/**'
        --ignore 'run/**'
        --ignore 'imgui.ini'
        --ignore '**/*.swp'
        --ignore '**/*.swo'
        --ignore '**/*~'
        --ignore '**/.DS_Store'
    )
fi

exec watchexec "${watch_args[@]}" -- ./run.sh --app "${app_name}" -- "$@"
