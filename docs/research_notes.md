# Research Notes

Start time for this framework pass: 2026-05-15 19:29:00 CEST.
Minimum handoff time requested by Daniel:

- Initially 2026-05-15 21:29:00 CEST.
- Revised during the run to 2026-05-15 20:29:00 CEST.

## Source Project: `dfsph_viewer`

Canonical local reference:
`/Users/danielsinkin/GitHub_private/SPH-Seminar/dfsph_viewer`.

Useful files read:

- `CMakeLists.txt`
- `.clang-format`
- `docs/research_notes.md`
- `docs/future_work.md`
- `src/dfsph_viewer/gfx/vulkan_app.cpp`
- `src/dfsph_viewer/gfx/vulkan_particle_renderer.cpp`
- `src/dfsph_viewer/core/camera.{hpp,cpp}`
- `scripts/visual_regression.py`
- `shaders/rigid_body.*`, `shaders/line_segments.vert`, `shaders/lines.frag`

Transferable decisions:

- Keep C++23, strict warnings, `RelWithDebInfo` as the default build type, and
  the same Allman/trailing-return style.
- Vendor dependencies locally. The SPH viewer currently vendors SDL3, GLM, Dear
  ImGui docking, and VMA; this framework should do the same rather than
  cross-referencing that project.
- On macOS/MoltenVK, instance creation needs
  `VK_KHR_portability_enumeration` when available and device creation needs
  `VK_KHR_portability_subset` when exposed. SDL's drawable pixel size must drive
  swapchain size and camera aspect for Retina correctness.
- The proven camera interaction is z-up orbit around a pivot: right mouse
  rotates, middle mouse pans the pivot in camera plane, mouse wheel zooms.
- Screenshot/readback is a first-class runtime responsibility, not an app-level
  afterthought. The SPH viewer copies the swapchain image into a VMA buffer and
  writes a PNG through `stb_image_write`.
- Debug visualization should use a dedicated batched path. The vectorfield pass
  found that line/arrow segment instances expanded in the vertex shader are a
  good upgrade over CPU-billboarded quads.

Things intentionally *not* copied verbatim:

- The SPH viewer's global ImGui helper state is useful for bootstrapping, but
  this repo should shape it into a small `Runtime` object so app code can get a
  `FrameContext` with raw Vulkan handles.
- The SPH viewer's domain state is particle/cache-specific. This repo starts
  with a generic mesh/debug draw list and lets future apps add their own data.
- The conservative particle occlusion culling work is valuable but premature for
  the first framework skeleton.

## External Vulkan References

- Vulkan Guide, memory allocation:
  https://docs.vulkan.org/guide/latest/memory_allocation.html
  - Sub-allocation is the first-class approach in Vulkan.
  - OS/driver memory allocation can be slow and is capped by
    `maxMemoryAllocationCount`.
  - UMA systems expose host-visible device-local memory, which matters on Apple
    Silicon/MoltenVK and supports cheap dynamic uploads when memory pressure is
    reasonable.

- Vulkan Memory Allocator, memory mapping:
  https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/memory_mapping.html
  - Use `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT` or
    `VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT` with `VMA_MEMORY_USAGE_AUTO`
    for mappable allocations.
  - `vmaCopyMemoryToAllocation` is safe for one-shot uploads; persistently
    mapped per-frame buffers are better for dynamic draw/debug data.

- Vulkan Guide, synchronization:
  https://docs.vulkan.org/guide/latest/synchronization.html
  - Synchronization remains the app developer's responsibility.
  - A personal framework should provide helpers and clear frame boundaries, but
    should not pretend compute/graphics hazards can be hidden in a universal
    abstraction.

- Khronos Vulkan Samples, command buffer usage:
  https://github.khronos.org/Vulkan-Site/samples/latest/samples/performance/command_buffer_usage/README.html
  - Per-frame command buffers should be recycled via command pool reset rather
    than allocated/freed on hot paths.
  - Secondary command buffers help only when work is large enough and split
    sensibly. Starting with one primary command buffer per frame is correct for
    this lightweight framework.
  - Use `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` for frame command buffers.

- Vulkan Guide, descriptor indexing:
  https://docs.vulkan.org/guide/latest/extensions/VK_EXT_descriptor_indexing.html
  and descriptor arrays:
  https://docs.vulkan.org/guide/latest/descriptor_arrays.html
  - Descriptor indexing is promoted to Vulkan 1.2, but the actual feature bits
    still need to be queried and enabled.
  - `descriptorBindingPartiallyBound`, `runtimeDescriptorArray`, and
    non-uniform indexing are the key pieces for later bindless-ish resources.
  - The first framework pass should query and record support instead of forcing
    every pipeline through a bindless model immediately.
  - If a physical device is below Vulkan 1.2, `VK_EXT_descriptor_indexing` must be
    exposed and enabled before the descriptor-indexing feature struct is used for
    device creation. On Vulkan 1.2+ the same struct is available through core
    Vulkan.

- Khronos descriptor indexing sample:
  https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
  - Treating descriptors as a large indexed array is the actual "bindless"
    direction, but update-after-bind has its own lifetime hazards. For this
    framework, that argues for querying/enabling the feature now and adding the
    descriptor heap only when a real texture/storage-buffer-heavy app appears.

- Khronos Vulkan Samples framework:
  https://github.khronos.org/Vulkan-Site/samples/latest/framework/README.html
  - It has both a high-level sample base class and a lower-level API sample base
    class. The lower-level model matches this repo better because Daniel wants
    to work with Vulkan internals directly.

- Granite renderer:
  https://github.com/Themaister/Granite
  - Useful as a personal Vulkan renderer precedent. Its README explicitly says
    the backend focuses entirely on Vulkan and reuses Vulkan enums/data where
    appropriate.
  - It validates the idea of simplifying painful points without pretending to be
    a cross-API engine.
  - Its listed features are a useful future checklist: deferred destruction,
    linear allocators, automatic pipeline creation, shader reload, pipeline cache
    save/reload. This repo intentionally starts smaller.

- vkguide descriptor notes:
  https://vkguide.dev/docs/chapter-4/descriptors/
  - Descriptor sets are best grouped by update frequency when not bindless.
    This remains the fallback plan if descriptor indexing is unavailable on a
    target device.
  - Per-frame descriptor pools should generally be reset whole rather than
    freeing individual per-frame descriptor sets.

- Sascha Willems Vulkan examples:
  https://github.com/SaschaWillems/Vulkan
  - Useful for concrete Vulkan feature examples and mesh/shader patterns.
  - Too sample-oriented to copy as a framework shape, but good as future
    reference when adding glTF, compute, and specialized pipelines.

## Framework Shape Chosen For This Pass

The repo starts as a "Vulkan runtime plus helpers", not an engine:

- The library is grouped under `ds_vk/`: headers, `.cpp` files, and built-in
  shaders live together because this repo is the library, not an installed
  package split into `include/` and `src/`.
- `ds_vk::Runtime` owns SDL, Vulkan instance/device/swapchain, command buffers,
  frame sync, VMA, ImGui, depth attachment, screenshots, and built-in pipelines.
- `Runtime::run(app)` accepts a normal app object with matching hook methods:
  `setup`, `update`, `draw_ui`, and `shutdown`. This avoids an inheritance
  requirement while preserving a small callback surface for the runtime loop.
- `ds_vk::FrameContext` exposes raw `VkInstance`, `VkPhysicalDevice`,
  `VkDevice`, `VkQueue`, `VkCommandBuffer`, `VmaAllocator`, the current
  `Camera`, and a `DrawList`.
- `ds_vk::DrawList` is immediate-mode from app code: `draw_mesh`,
  `draw_basic_mesh`, `debug_line`, `debug_arrow`, `debug_sphere`. Internally
  the renderer batches debug segments and records mesh draw commands for the
  frame.
- Mesh generation is framework code: quad, cube, UV sphere.
- `.clang-tidy` is intentionally strict on bugprone/analyzer/modernize checks,
  but it does not require private-member suffixes for plain data aggregates and
  does not flag Vulkan's standard zero-initialized C structs as invalid enum
  initialization. Those checks created noise rather than useful pressure.

This intentionally leaves a compute escape hatch: app code can record raw Vulkan
commands with `frame.command_buffer` before or after using helper draw calls, and
future synchronization helpers should be thin wrappers over `vkCmdPipelineBarrier2`.

## Known Risks / Future Research

- Bindless is not implemented in the first graphics pipeline. The runtime should
  still query descriptor indexing support now so the next pass can add a
  descriptor heap without changing the app-facing concept of material/resource
  handles.
- This pass keeps mesh draws direct and simple. For the fifth-project goal,
  repeated mesh draws should later bucket by `(pipeline, mesh)` and use an
  instance buffer or indirect draw path.
- Shader hot reload is deferred. CMake-compiled GLSL is the fastest reliable
  first step; hot reload can reuse the pipeline rebuild boundaries once more
  material types exist.
- Static mesh upload currently uses VMA-managed mappable buffers directly. That
  is simple and good enough for the first app; a later resource pass should add
  staging uploads into device-preferred buffers for larger meshes/scenes.
- Mesh replacement currently uses `vkDeviceWaitIdle` for old-resource safety.
  Deferred destruction per frame is the right upgrade once more dynamic resource
  churn exists.
- Visual correctness is harder than pure unit correctness. The runtime therefore
  needs CLI screenshot capture and a Python pixel-analysis script so automated
  smoke tests can check "nonblank and roughly expected" output.

## Validation Log

### 2026-05-15 First Build/Smoke

- `cmake -S . -B build` configured successfully with the Homebrew LLVM toolchain
  and vendored SDL3/GLM/ImGui/VMA.
- Initial build failed because `ds_vk/mesh.cpp` included `glm/gtx/quaternion.hpp`.
  GLM's GTX headers are experimental unless `GLM_ENABLE_EXPERIMENTAL` is set.
  The implementation did not need GTX functionality; switching to stable
  `glm/gtc/quaternion.hpp` and `glm::mat4_cast` fixed the model.
- `cmake --build build` then completed successfully.
- `ctest --test-dir build --output-on-failure` passed the CPU-side mesh and
  camera tests.
- `./build/ds_vk_basic_app --smoke-frames 20 --screenshot run/basic.png --hide-ui`
  ran the Vulkan app and wrote a PNG screenshot.
- `scripts/validate_screenshot.py` initially failed because Pillow was missing.
  A local `.venv` was created and `pillow` was installed for screenshot pixel
  validation.
- `./.venv/bin/python scripts/validate_screenshot.py run/basic.png` passed with
  `2560x1600`, mean RGB around `35`, and standard deviation around `23`.
- A UI-included smoke screenshot also passed. The first UI capture revealed
  clipped ImGui labels in the camera/basic panels; the panels were widened and
  widget widths constrained. `run/basic_ui_wide.png` passed pixel validation and
  visually confirmed the labels fit.

### 2026-05-15 Warning Baseline

- `clang-tidy ds_vk/camera.cpp ds_vk/mesh.cpp ds_vk/runtime.cpp ds_vk/vma_support.cpp
  app/main.cpp tests/test_main.cpp -p build` initially found a mix of useful
  issues and noisy graphics-API/style friction.
- Real fixes kept:
  - Moved CLI runtime configuration inside `main`'s exception boundary.
  - Removed a `std::move` on a trivially copyable mesh resource.
  - Added explicit vector byte-size conversion to avoid multiplication-width
    mistakes in upload paths.
  - Added trailing return types on local lambdas.
  - Made descriptor-indexing enablement depend on Vulkan 1.2 support or the
    `VK_EXT_descriptor_indexing` extension.
- Tidy profile decisions:
  - Disabled `bugprone-easily-swappable-parameters` because graphics/math helper
    APIs naturally contain adjacent `Vec3`, `u32`, and `f32` parameters.
  - Disabled `bugprone-invalid-enum-default-initialization` because Vulkan C
    structs are conventionally value-initialized and then filled field by field.
  - Removed the member suffix rule because the framework uses many public data
    aggregates where suffixes made the style worse rather than clearer.
- After fixes, clang-tidy exits successfully. Its only output is dependency
  warnings suppressed by the header filter.
- `cmake --build build` and `ctest --test-dir build --output-on-failure` still
  pass after the warning cleanup.
- `./build/ds_vk_basic_app --smoke-frames 20 --screenshot
  run/basic_descriptor_gate.png --hide-ui` wrote a screenshot after the
  descriptor-indexing feature-gate change.
- `./.venv/bin/python scripts/validate_screenshot.py
  run/basic_descriptor_gate.png` passed with `2560x1600`, mean RGB around `35`,
  and standard deviation around `23`.

### 2026-05-15 Debug Sphere/Test Harness

- Added `DrawList::debug_sphere`, implemented as three great circles through the
  same debug segment path as lines/arrows. This keeps the first debug primitive
  expansion small and avoids adding another pipeline.
- Added `tests/test_runtime.cpp` so draw-list behavior is tested without
  initializing Vulkan.
- A parallel build/test command produced a false `ctest` failure because CTest
  tried to run `ds_vk_runtime_tests` before the new executable had finished
  building. The correction is sequencing, not implementation: build first, then
  run tests.
- The first build of the new runtime test failed because `{1.0f}` cannot be
  passed as a `Vec4` argument through copy-initialization; GLM's scalar vector
  constructor is explicit. The test now uses `ds_vk::Vec4{1.0f}`, matching app
  code style.
- After the fix, `cmake --build build` and `ctest --test-dir build
  --output-on-failure` pass with both `ds_vk_tests` and `ds_vk_runtime_tests`.
- `./build/ds_vk_basic_app --smoke-frames 20 --screenshot
  run/basic_debug_sphere.png --hide-ui` wrote a PNG with the wire sphere
  visible over the shaded sphere.
- `./.venv/bin/python scripts/validate_screenshot.py
  run/basic_debug_sphere.png` passed with `2560x1600`, mean RGB around `35`, and
  standard deviation around `24`.

### 2026-05-15 Push Constant Audit

- The first mesh shader interface used `view_projection`, `model`, `color`, and
  `light_direction_ambient` in push constants. That is 160 bytes, which works on
  this Mac but exceeds Vulkan's 128-byte guaranteed minimum.
- The mesh push block became 128 bytes: model-view-projection, three packed
  normal-matrix columns, and material base color. Lighting constants lived in
  the fragment shader until a real material/descriptor path existed.
- The runtime checks `maxPushConstantsSize` before creating the device and has
  `static_assert`s for the mesh/debug push block sizes.
- This also fixed normal transformation for non-uniform scaling by using the
  inverse-transpose of the model matrix instead of `mat3(model)`.
- `cmake --build build`, `ctest --test-dir build --output-on-failure`, and
  clang-tidy still pass after the push-constant change.
- `./build/ds_vk_basic_app --smoke-frames 20 --screenshot run/basic_push128.png
  --hide-ui` wrote a PNG after recompiling the shaders.
- `./.venv/bin/python scripts/validate_screenshot.py run/basic_push128.png`
  passed with `2560x1600`, mean RGB around `37`, and standard deviation around
  `28`.

### 2026-05-15 Resize/Swapchain Edge

- While reviewing the frame loop, I found that `VK_ERROR_OUT_OF_DATE_KHR` from
  `vkAcquireNextImageKHR` could continue the loop after starting an ImGui frame
  but before `ImGui::Render`.
- The loop now calls `ImGui::EndFrame()` before continuing in that path. This is
  a small correctness fix for resize/minimize churn and should prevent ImGui
  frame-lifecycle assertions later.
- `cmake --build build`, `ctest --test-dir build --output-on-failure`, and a
  targeted tidy pass over runtime/app/runtime tests pass after the change.

### 2026-05-15 Core-Only Build Check

- Configured `build-core` with `-DDS_VK_BUILD_APP=OFF
  -DDS_VK_USE_SYSTEM_VULKAN=OFF` to verify the CPU-side core and mesh/camera
  tests can still build without the Vulkan app target.
- I initially repeated the build/test race by launching CTest beside the first
  build of that tree. After the build completed, `ctest --test-dir build-core
  --output-on-failure` passed.

### 2026-05-15 Camera Basis Robustness

- The runtime input path clamps orbit pitch away from vertical, but app code can
  still assign camera values directly.
- `Camera::right()` and `Camera::up()` now use a small safe-normalize helper so
  a vertical pitch does not turn a zero cross product into NaNs.
- Added a CPU test that sets pitch to 90 degrees and checks the camera basis is
  still finite.
- `cmake --build build`, `cmake --build build-core`, both CTest suites, and a
  targeted tidy pass over camera/core tests pass after the change.
- A final UI-visible screenshot, `run/basic_ui_final.png`, passed pixel
  validation and visual review. The camera/basic viewer labels fit after adding
  the sphere-wire checkbox.

### 2026-05-15 Mesh Replacement

- The first sphere smoothness UI called `upload_mesh` on every rebuild, which
  meant dragging the sliders accumulated old GPU buffers until shutdown.
- Added `Runtime::replace_mesh`; it allocates the replacement first, swaps the
  mesh resource into the existing handle slot, and then destroys the old buffers.
  If the handle is invalid, it falls back to `upload_mesh`.
- Mesh resource creation now cleans up partially allocated vertex/index buffers
  if any allocation or upload step throws.
- `upload_mesh` also cleans up the newly created buffers if storing the resource
  handle in the mesh vector throws.
- Mesh upload/replacement now errors clearly if called before the runtime is
  initialized.
- `replace_mesh` currently waits for the device to go idle before destroying old
  mesh buffers. This is conservative but correct for slider-driven rebuilds; a
  later renderer can replace it with deferred destruction keyed to frame fences.
- The basic app now uses `replace_mesh` for generated sphere rebuilds.
- `cmake --build build`, `ctest --test-dir build --output-on-failure`, and a
  targeted tidy pass over runtime/app pass after the change.
- `./build/ds_vk_basic_app --smoke-frames 20 --screenshot
  run/basic_replace_mesh.png --hide-ui` wrote a screenshot, and
  `./.venv/bin/python scripts/validate_screenshot.py run/basic_replace_mesh.png`
  passed with `2560x1600`, mean RGB around `37`, and standard deviation around
  `28`.

### 2026-05-15 Screenshot Readback Guard

- `write_capture_png` now creates directories and allocates the RGBA staging
  vector before mapping the VMA readback buffer.
- The mapped readback buffer is wrapped in a `try`/`catch` so invalidation or
  conversion failures unmap before rethrowing.
- `cmake --build build`, `ctest --test-dir build --output-on-failure`, and
  clang-tidy over `ds_vk/runtime.cpp` pass after the change.
- `./build/ds_vk_basic_app --smoke-frames 20 --screenshot
  run/basic_capture_guard.png --hide-ui` wrote a screenshot, and
  `./.venv/bin/python scripts/validate_screenshot.py
  run/basic_capture_guard.png` passed with `2560x1600`, mean RGB around `37`,
  and standard deviation around `28`.

### 2026-05-15 Runtime Stats Timing

- `last_render_ms` originally started after render command recording and mostly
  measured command-buffer end/submit plus screenshot work.
- The timer now starts before the render pass begins so the stat covers mesh,
  debug, and ImGui command recording as well.
- `cmake --build build`, `ctest --test-dir build --output-on-failure`, and
  clang-tidy over `ds_vk/runtime.cpp` pass after the change.

### 2026-05-15 Phong-Style Shading Check

- The first mesh fragment shader was ambient/diffuse with a back-light term, but
  Daniel explicitly asked for a Phong-shaded sphere.
- Added a small fixed-view Blinn-Phong specular term while keeping the compact
  128-byte mesh push-constant interface. A later material/camera descriptor path
  can make the view/light inputs fully dynamic.
- `cmake --build build` regenerated `mesh.frag.spv`, `ctest --test-dir build
  --output-on-failure` passed, and `run/basic_phong.png` passed screenshot
  validation with `2560x1600`, mean RGB around `39`, and standard deviation
  around `29`.

### 2026-05-15 Helper API Version

- VMA and ImGui initialization now use the selected physical device API version,
  capped at Vulkan 1.2, instead of blindly passing `VK_API_VERSION_1_2`.
- This does not change behavior on the current macOS/MoltenVK path, but keeps
  the nice-to-have Linux/Windows path less brittle on devices exposing a lower
  Vulkan version.
- `cmake --build build`, `ctest --test-dir build --output-on-failure`, and
  clang-tidy over `ds_vk/runtime.cpp` pass after the change.

### 2026-05-15 Final Verification Before Handoff

- Daniel allowed stopping early at 2026-05-15 20:21:03 CEST, before the revised
  20:29 CEST minimum.
- Final checks run:
  - `cmake --build build`
  - `ctest --test-dir build --output-on-failure`
  - `ctest --test-dir build-core --output-on-failure`
  - `clang-tidy ds_vk/camera.cpp ds_vk/mesh.cpp ds_vk/runtime.cpp ds_vk/vma_support.cpp
    app/main.cpp tests/test_main.cpp tests/test_runtime.cpp -p build`
  - `./build/ds_vk_basic_app --smoke-frames 20 --screenshot
    run/basic_final.png --hide-ui`
  - `./.venv/bin/python scripts/validate_screenshot.py run/basic_final.png`
- Final screenshot validation passed with `2560x1600`, mean RGB around `39`,
  and standard deviation around `29`.

### 2026-05-15 Library Layout Correction

- Daniel pointed out that `include/ds_vk` plus `src` is a weird split for this
  repository because this repo is the library itself.
- Moved framework headers, implementation files, and built-in shaders into one
  `ds_vk/` tree.
- `app/`, `tests/`, `external/`, `docs/`, and `scripts/` remain top-level
  project support/user areas.
- Updated CMake and `.clang-tidy` for the new paths.
- Verified after the move with `cmake -S . -B build`, `cmake -S . -B
  build-core -DDS_VK_BUILD_APP=OFF -DDS_VK_USE_SYSTEM_VULKAN=OFF`, both build
  trees, both CTest suites, clang-tidy on the new `ds_vk/...` paths, and
  `run/basic_layout.png` screenshot validation.

### 2026-05-15 Plain App Hooks

- Daniel preferred not having an inheritance-based app interface.
- Removed `ds_vk::Application` and the virtual `setup/update/draw_ui/shutdown`
  surface.
- `Runtime::run(app)` now builds a small callback table from a normal app object
  with matching hook methods. `setup`, `draw_ui`, and `shutdown` are optional,
  while a class with no recognized hook fails compilation.
- Added compile-time runtime tests that a plain app object is accepted, a
  setup-only object is accepted, an object with no hooks is rejected by the
  concept, and the app hook type is not polymorphic.
- Verified with `cmake --build build`, `cmake --build build-core`,
  `ctest --test-dir build --output-on-failure`, `ctest --test-dir build-core
  --output-on-failure`, and a full clang-tidy pass over framework/app/tests.
- The sandboxed `./run.sh` execution could build but could not access a macOS
  display. Running it outside the sandbox wrote `run/no_inheritance.png`, and
  screenshot validation passed with `2560x1600`, mean RGB around `39`, and
  standard deviation around `29`.

### 2026-05-15 Neovim / clangd Index Scope

- `build/compile_commands.json` currently has 252 translation units, 245 of
  which are vendored dependency files under `external/`.
- Added a repo-local `.clangd` that points clangd at `build`, disables all-scope
  completion, and sets `Index: Background: Skip` for `external/.*`.
- This does not make Vulkan/SDL/ImGui headers disappear from normal project
  files, because clangd still has to parse headers included by the current
  translation unit. It should, however, stop background indexing every vendored
  SDL/ImGui/VMA/GLM source file as if it were project code.
- `clangd --check=app/main.cpp --compile-commands-dir=build` initially exposed
  Daniel's in-progress `debug_axis_cfg_` declaration as an editor-facing parse
  error. The member is now typed as `DebugAxisConfig debug_axis_cfg_{}` and the
  axis draw code uses it.
- Verified `clangd --check=app/main.cpp --compile-commands-dir=build`, `cmake
  --build build`, `ctest --test-dir build --output-on-failure`, and
  `run/clangd_external_skip.png` screenshot validation.

### 2026-05-15 Config Struct Experiments

- Added `CameraConfig` as a plain aggregate so app code can use C++ designated
  initializers for setup:
  `runtime.camera({.pivot = 0.7f * k_axis_z, .distance = 5.4f, ...});`.
- `Camera::configure` and `Runtime::camera(config)` set the full camera state
  from the config values. Omitted fields use `CameraConfig` defaults rather than
  preserving previous camera state.
- The app floor grid config now separates `height` from `line_width`; the
  migration had used the same value for z-offset and rendered line width.
- Restored the floor grid loop to sweep `-count_per_side` through
  `+count_per_side`.
- Added a CPU test for `CameraConfig` application and default retention.
- Verified with both build trees, both CTest suites, clangd checks for
  `app/main.cpp` and `ds_vk/camera.cpp`, clang-tidy over touched files, and
  `run/camera_config_grid.png` screenshot validation.

### 2026-05-15 Config-Based Draw Calls And Materials

- Added config structs for immediate draw calls: `MeshDrawConfig`,
  `BasicMeshDrawConfig`, `DebugLineConfig`, `DebugArrowConfig`, and
  `DebugSphereConfig`.
- `draw_mesh` now takes a mesh, transform, and `Material`. The first material is
  deliberately small: `Material{.base_color = ...}` feeding the existing
  Blinn-Phong-style mesh shader.
- `draw_basic_mesh` keeps the former convenience behavior by mapping a color
  into `Material::base_color`.
- Debug draw calls also have config overloads; old positional overloads remain
  as convenience wrappers for now.
- The shader push block was still 128 bytes, so material expansion beyond
  `base_color` should go through a descriptor/material table rather than growing
  push constants past the Vulkan guaranteed minimum.
- Added runtime tests that `draw_mesh` records material color and
  `draw_basic_mesh` maps color into the material.

### 2026-05-15 PBR Materials, Debug Modes, And Selection

- Replaced the interim Blinn-Phong material with a small Cook-Torrance
  metallic/roughness material:
  `Material{.base_color, .emissive_color, .metallic, .roughness, .ambient_occlusion}`.
- Kept this deliberately below Blender BSDF complexity. The shader has one
  fixed directional light and fixed view approximation for now, but the
  roughness/metallic knobs are already the right user-facing material language
  for future environment lighting or camera/light descriptors.
- The material storage-buffer element now also carries debug parameters. This
  lets app code request selected pulse, color override, scalar heatmap, normal
  visualization, or object-id color without growing the push-constant block.
- Added `ObjectId` and `MeshDebugConfig` to draw configs. The draw list culls
  `.debug = {.hidden = true}` before recording, and selected objects default to
  the pulsing debug shader if no explicit debug mode is set.
- Selection is modeled as a helper layer instead of callbacks on meshes. A mesh
  resource is just geometry; an app draw is what has object identity, transform,
  material, hidden state, and selected/debug state.
- Added CPU tests for PBR material fields surviving draw recording, hidden mesh
  draw culling, object IDs, debug config propagation, and selection helper
  intersections. AABB picking normalizes reversed bounds so apps can build boxes
  from arbitrary point pairs.

### 2026-05-15 Picker Plugin And Click Selection

- Promoted the ad hoc selection helpers into a static optional picker module:
  `ds_vk_picker` builds from `ds_vk/plugins/picker.cpp` and links into apps or
  tests that want object selection.
- The runtime exposes per-frame mouse state through `FrameContext::input`.
  A left click is recorded only when ImGui does not want the mouse, and SDL
  window coordinates are converted to framebuffer pixels so Retina swapchains
  and `make_pick_ray` agree.
- `ds_vk::Picker` has two public query paths:
  - `click({.camera, .mouse_px, .viewport_px, .layer_mask})`, for normal app UI
    mouse picking;
  - `raycast({.ray, .layer_mask})`, for code that already has a world-space ray.
- The picker currently supports sphere, AABB, OBB, capsule, and screen-space
  segment targets. Triangle-mesh picking is intentionally not included because
  general triangle colliders need a separate acceleration-structure design and
  are too easy to misuse in visualization apps.
- Targets carry app-owned `ObjectId`, optional layer masks, `sub_index`, and
  `user_bits`. The framework never pretends that `MeshHandle` is a scene object;
  the app creates one pick target per selectable thing and keeps transforms,
  physics IDs, neighborhood state, and UI data in its own model.
- The basic app registers one sphere target and one cube OBB target each frame.
  Clicking either opens a small selection window with object information.
  Selecting the sphere or cube applies the existing selected-pulse debug mode.

### 2026-05-16 Viz Plugin And Vector Field Demo

- Added a second static plugin target, `ds_vk_viz`, under `ds_vk/plugins`.
  This is intentionally a reusable visualization vocabulary, not an SPH viewer
  or app framework. It depends only on `ds_vk_core` and can be used in the
  core-only build.
- `ds_vk::viz` currently provides:
  - `ColorRamp` with grayscale, blue-red, viridis, magma, and turbo-style
    presets;
  - `range_from_values` for finite scalar ranges;
  - `draw_vector_field` for spans of positions/vectors and any draw sink that
    supports `debug_arrow`;
  - `draw_cross_marker`, a camera-facing X marker for selected points.
- The app now generates a small swirling vector field over the floor and draws
  it through `viz::draw_vector_field` with magnitude coloring. This keeps the
  demo app useful as a visual smoke test for the plugin without baking field
  semantics into the plugin.
- Added tests for color-ramp clamping/normalization, scalar range scanning,
  vector-field draw emission, and cross markers.

### 2026-05-16 Picker Y Convention Fix

- Daniel noticed that picker selection was not accurate. Comparing against
  `dfsph_viewer` showed the bug: the framework used the usual OpenGL-style
  screen Y conversion in `make_pick_ray`, but the camera projection already
  contains the Vulkan Y flip (`proj[1][1] *= -1`). The correct mapping for this
  projection is therefore `y_ndc = 2 * y / height - 1`, matching
  `dfsph::ray_from_screen`.
- The picker screen-segment projection had the matching inverse error. It now
  maps projected NDC Y back to framebuffer pixels with
  `(ndc.y * 0.5 + 0.5) * height`, again matching the DFSPh viewer's screen-space
  picking code.
- Retina/high-DPI scaling was already using the same shape as DFSPh:
  SDL window coordinates are scaled by `SDL_GetWindowSizeInPixels /
  SDL_GetWindowSize` before they are passed to the picker. No change was needed
  there.
- Removed the right-click line-connection experiment from the app and removed
  the extra `FrameContext::input.right_click` field. Right click is back to
  camera orbit only while picker correctness is being stabilized.
- Added CPU tests that would have caught the mirror:
  - upper framebuffer pixels ray toward world up;
  - lower framebuffer pixels ray toward world down;
  - left/right framebuffer pixels map to camera left/right;
  - screen-segment picking uses the same Vulkan framebuffer Y convention.

### 2026-05-16 PBR View Vector And Normal Debug

- Daniel found a hard diagonal/geodesic-like shading split on the sphere when
  metallic and roughness were both around 0.5.
- The sphere vertex normals were not the main culprit: added CPU invariants now
  verify UV sphere normals are radial, finite, unit length, and that
  non-degenerate generated triangles wind outward.
- The actual shader issue was that the interim PBR fragment shader used a fixed
  hard-coded view vector. That makes metallic/specular response split across a
  world-space plane instead of following the camera and fragment position.
- The mesh vertex push block is now exactly 128 bytes containing view-projection
  and model matrices. The vertex shader emits world position and computes its
  normal matrix from the model matrix. The material storage-buffer record now
  also carries camera position, and the fragment shader uses
  `camera_position - world_position` for the view direction.
- Added a global normal-color debug path in the app. It can be toggled in ImGui
  with "Normal debug" or started from the CLI with `--normal-debug`, and it
  applies `MeshDebugMode::normal` to all mesh draws.
- Verified with normal and normal-debug screenshots:
  `run/pbr_view_fix.png` and `run/normal_debug.png`.

### 2026-05-16 Strong Color Types

- Daniel pointed out that using `Vec4` as the public color type lets app code do
  nonsensical vector arithmetic such as adding colors directly.
- Mirrored the older `physically-based-animations` direction: colors are now
  standalone array-backed types with accessors, while math-like color operations
  are named helpers.
- Added `ds_vk::Color` for float RGBA and `ds_vk::ColorU8` for packed 8-bit
  RGBA, plus `to_vec4`, `to_color`, `to_color_u8`, `with_alpha`, and
  `mix_color`.
- Public mesh/material/debug/viz/runtime config APIs now use `Color`. Internal
  Vulkan shader ABI structs still pack colors into `Vec4` because SPIR-V,
  vertex input, and storage-buffer layout want four-component values.
- Added tests for `mix_color`, float/u8 conversion, and a compile-time concept
  check that `Color + Color` is not a valid expression.

### 2026-05-16 Material Base-Color Textures

- Added `TextureHandle`, `TextureLoadConfig`, and `MaterialTextures` so material
  configs can use the designated-initializer shape:
  `Material{.textures = {.base_color = texture}}`.
- Vendored `stb_image.h` from `nothings/stb` for PNG/JPG texture loading. The
  runtime now creates a 1x1 white fallback texture at slot 0 and exposes
  `Runtime::load_texture(path, {.srgb = true})`.
- The first material texture implementation uses a fixed combined-image-sampler
  array rather than fully bindless descriptors. The table is currently 16
  textures because the MoltenVK device used for validation reported
  `maxPerStageDescriptorSamplers = 16`; an initial 32-slot table triggered
  validation VUID `VkPipelineLayoutCreateInfo-descriptorType-03016`. That was a
  design assumption bug, not a flaky test.
- The runtime now guards startup with `maxPerStageDescriptorSamplers >= 16` so
  the shader ABI and descriptor layout cannot silently exceed the device limit.
  A future descriptor-indexing/bindless pass should make this configurable after
  querying limits and feature bits.
- Added UV coordinates to generated quads, cubes, and UV spheres. CPU tests now
  check that generated mesh UVs are finite and normalized.
- Integrated two CC0 Poly Haven diffuse maps into the app:
  `concrete_floor_diff_1k.jpg` for the floor and
  `wood_table_001_diff_1k.png` for the cube. The asset README records source
  URLs and authorship/license notes.
- Verified by rebuilding, running CPU tests, and rendering
  `run/texture_materials.png`. The first screenshot run caught the sampler-limit
  validation error above; after reducing the table to 16 the screenshot run was
  clean and nonblank.
