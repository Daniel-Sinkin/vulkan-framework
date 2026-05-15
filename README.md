# ds_vk Vulkan Framework

Personal Vulkan helper/runtime for C++ visualization experiments.

This is deliberately not a game engine and not a cross-API abstraction. The
runtime owns recurring setup work, while app code can still access raw Vulkan
handles through `ds_vk::FrameContext`.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The app target is `ds_vk_basic_app` when Vulkan and `glslc` are available.

```sh
./run.sh
./run.sh --smoke-frames 20 --screenshot run/basic.png --hide-ui
./build/ds_vk_basic_app
./build/ds_vk_basic_app --smoke-frames 20 --screenshot run/basic.png --hide-ui
python3 scripts/validate_screenshot.py run/basic.png
```

`scripts/validate_screenshot.py` uses Pillow. A local `.venv` is fine for that:

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install pillow
./.venv/bin/python scripts/validate_screenshot.py run/basic.png
```

## Current App

The first full user lives in `app/`, not `examples/`. It shows:

- z-up orbit camera with pivot pan and wheel zoom
- ImGui camera controls
- generated floor quad
- generated cube
- generated smooth UV sphere with adjustable slices/stacks
- debug line/arrow/sphere grid and axes
- swapchain screenshot capture

Research and design notes live in `docs/`.

## Editor Notes

The repo-local `.clangd` points clangd at `build/compile_commands.json`, disables
all-scope completion, and skips background indexing for `external/`. Vendored
headers are still parsed when framework/app code includes them, but SDL/ImGui/VMA
sources should not dominate Neovim indexing.
