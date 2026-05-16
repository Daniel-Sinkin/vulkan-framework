# ds_vk Vulkan Framework

Personal Vulkan helper/runtime for C++ visualization experiments.

This is deliberately not a game engine and not a cross-API abstraction. The
runtime owns recurring setup work, while app code can still access raw Vulkan
handles through `ds_vk::FrameContext`.

The primary runtime API is explicit frame driving, not an inherited app runner:

```cpp
ds_vk::Runtime runtime{cfg};
runtime.initialize();
app.setup(runtime);

while (auto* frame = runtime.begin_frame())
{
    app.update(*frame, frame->dt_seconds);

    runtime.render_shadow_pass();
    runtime.begin_main_pass();
    runtime.render_draw_list();
    runtime.render_imgui();
    runtime.end_main_pass();
    runtime.end_frame();
}
```

Raw Vulkan commands can be recorded through `frame->command_buffer` between those
phase calls. `Runtime::run_prototype(app)` exists only as a quick MVP wrapper
around the same protocol.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The default app target is `ds_vk_quake_app` when Vulkan and `glslc` are available.

```sh
./run.sh
./run.sh --app basic
./run.sh --app vectorfield
./run.sh --app dfsph
./run.sh --app pba
./run.sh --app basic --smoke-frames 20 --screenshot run/basic.png --hide-ui
./run.sh --app vectorfield --smoke-frames 20 --screenshot run/vectorfield.png --hide-ui
./run.sh --app dfsph --smoke-frames 20 --screenshot run/dfsph.png --hide-ui
./run.sh --app pba --smoke-frames 20 --screenshot run/pba.png --hide-ui
./build/ds_vk_quake_app
./build/ds_vk_basic_app
./build/ds_vk_vectorfield_app
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

The first full user lives in `app/`, not `examples/`. `ds_vk_basic_app` shows:

- z-up orbit camera with pivot pan and wheel zoom
- ImGui camera controls
- generated floor quad
- generated cube
- generated smooth UV sphere with adjustable slices/stacks
- CC0 Poly Haven HDRI background and approximate environment lighting
- debug line/arrow/sphere grid and axes
- swapchain screenshot capture

`ds_vk_vectorfield_app` is the first migrated-style standalone user. It owns its
own vectorfield examples, time controls, trace seeds, and selection state while
using `ds_vk::viz` trails/vector arrows and `ds_vk::Picker` screen-segment
picking.

`ds_vk_dfsph_app` is an MVP fixed-data migration of the old DFSPH viewer. It
loads the vendored small dambreak SPlisHSPlasH VTK history from
`assets/dfsph/dambreak_small_iisph_v1/vtk`, plays it back, and uses `ds_vk::viz`
for optional velocity arrows.

`ds_vk_pba_app` is an MVP realtime rigid-body visualization user. It owns its
physics state locally, uses Space to pause/resume simulation while camera input
continues working, and routes speed coloring plus velocity arrows through
`ds_vk::viz`.

`ds_vk_quake_app` is currently an experimental stub for future Quake asset/render
work.

`ds_vk/assets.hpp` currently provides a small CPU-side glTF/GLB mesh loader into
`MeshData`. It is intentionally not a runtime resource manager.

Research and design notes live in `docs/`.

## Editor Notes

The repo-local `.clangd` points clangd at `build/compile_commands.json`, disables
all-scope completion, and skips background indexing for `external/`. Vendored
headers are still parsed when framework/app code includes them, but SDL/ImGui/VMA
sources should not dominate Neovim indexing.
