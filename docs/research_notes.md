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

## Lighting / Shadow Research Pass

Sources checked for this pass:

- Khronos Vulkan Samples, multithreaded render passes:
  https://docs.vulkan.org/samples/latest/samples/performance/multithreading_render_passes/README.html
  - The sample describes classic shadow mapping as two render passes: first
    render depth from the light's view, then render the camera view and sample
    that depth texture in the fragment shader to decide whether the fragment is
    occluded from the light.
  - This maps well to this framework because it keeps the renderer shader based
    and does not require acceleration structures or ray queries.

- Vulkan Guide, depth:
  https://docs.vulkan.org/guide/latest/depth.html
  - Vulkan depth buffers are ordinary `VkImage` / `VkImageView` resources used
    by framebuffers. Shadow maps therefore need both
    `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT` and
    `VK_IMAGE_USAGE_SAMPLED_BIT`.
  - Depth-only shadow passes should clear through the attachment load op.
  - Depth writes belong to the early/late fragment test pipeline stages, not the
    fragment shader stage. Synchronization from shadow rendering to shadow
    sampling must use depth-stencil attachment write access as the source and
    shader-read access as the destination.

- Vulkan Guide, synchronization examples:
  https://docs.vulkan.org/guide/latest/synchronization_examples.html
  - The guide has a specific "depth attachment then fragment-shader sampled"
    example for shadow maps. The important shape is depth-stencil attachment
    write in early/late fragment tests to fragment shader read.
  - `ds_vk` currently uses legacy `vkCmdPipelineBarrier` because the rest of the
    runtime already uses Vulkan 1.2-era render passes. A later cleanup can move
    this to `vkCmdPipelineBarrier2`.

- Vulkan spec, render passes:
  https://docs.vulkan.org/spec/latest/chapters/renderpass.html
  - Read-only depth/stencil layouts are intended for resources that are sampled
    or otherwise read after being used as attachments.
  - For this pass the shadow attachment final layout is
    `VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL` and the descriptor uses
    that same layout.

- Khronos glTF `KHR_lights_punctual`:
  https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md
  - Useful light semantics even though this repo is not a glTF renderer.
  - Adopted the same three conceptual light families: directional, point
    (called radial in `ds_vk`), and spot.
  - Adopted inverse-square point/spot attenuation with a smooth range cutoff.
  - Adopted the common CPU-precomputed spot cone scale/offset form so the shader
    only does a dot product, clamp, and square.

- Sascha Willems / Vulkan sample descriptions:
  https://github.com/Voultapher/Vulkan-Samples
  - Useful as a sanity check for the raster path: directional shadow mapping is
    a projective two-pass depth texture; omnidirectional point-light shadows use
    a cube map and are materially more work.

Implementation decisions:

- Add explicit `LightConfig` data to the draw list rather than faking light from
  a hard-coded shader coordinate.
- Keep emission as surface color only. It does not generate lighting because
  that would imply global illumination or at least many extra approximations.
- Store per-frame lights in a storage buffer. This keeps the mesh pipeline
  descriptor layout stable while the app changes light count and parameters.
- Limit built-in lights to 16 for now. That is enough for visual experiments and
  keeps the shader/data layout simple.
- Support one active shadow-casting directional or spot light per frame. The
  first shadow-enabled supported light wins. This is intentionally a first
  working shape, not a final multi-shadow atlas.
- Do not implement radial/point shadows yet. Proper radial shadows need an
  omnidirectional cube shadow map, six light-space renders or layered rendering,
  and different sampling/comparison math. That should be its own focused pass.
- Add `MeshRenderMask` with camera visibility, shadow producer, shadow consumer,
  and light receiver booleans. These are per draw/object semantics and should
  not be tied to `MeshHandle`, because the same mesh resource can represent many
  app-owned objects.

## DFSPH / PBA Migration MVP Pass

Start time for this migration pass: 2026-05-16 07:56:29 CEST.
Earliest stopping time requested by Daniel after revision:
2026-05-16 08:56:29 CEST.

Local projects scoped for migration pressure:

- `/Users/danielsinkin/GitHub_private/SPH-Seminar/dfsph_viewer`
- `/Users/danielsinkin/GitHub_private/physically-based-animation`

Migration decisions:

- Treat DFSPH and PBA as full `app/` users, not examples and not framework
  modules. Their job is to pull on the framework from real use cases.
- Keep scene theory out of `ds_vk`. Scene graphs, object ownership, playback
  timelines, and domain state remain app concerns until repeated projects prove
  a smaller reusable shape.
- Add only narrow reusable framework pieces:
  - `ds_vk::assets` for minimal glTF/GLB CPU mesh loading.
  - `ds_vk::viz::draw_aabb` so DFSPH/PBA bounds visualization does not duplicate
    line-corner code.
  - `InputState::space_pressed` so interactive apps can pause simulations
    without freezing camera/runtime input.
- DFSPH MVP uses the vendored small-dambreak VTK history only. It intentionally
  does not vendor SPlisHSPlasH or recreate the original scene/cache machinery.
- PBA MVP keeps the physics in `app/pba/physics.*` as a headless app-owned
  library. The current solver is an intentionally small realtime AABB pyramid
  demo, not a migration of every old PBA scene.
- PBA exposes bodies through `std::span<Body>` / `std::span<const Body>` rather
  than returning the internal vector. The app can edit body state, but cannot
  accidentally resize the simulation storage through the accessor.
- GLB unknown chunks are intentionally skipped after length validation. That is
  covered by a regression fixture because GLB is extension-friendly; malformed
  headers, missing JSON, invalid accessors, and out-of-buffer reads still fail.
  Khronos glTF 2.0 specifies that clients must ignore unknown GLB chunk types so
  extensions can add later chunks:
  https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#chunks-overview.

## Environment HDRI Pass

Start time for this pass: 2026-05-16.

Source asset:

- Poly Haven `studio_small_01`, CC0:
  https://polyhaven.com/a/studio_small_01
- Vendored file:
  `assets/hdri/polyhaven/studio_small_01_1k.hdr`

Implementation decisions:

- Use a 2D equirectangular HDR texture for the first pass, not a cubemap or
  prefiltered irradiance/specular map. This keeps the implementation small and
  shader-based while still making roughness/metallic changes respond to the
  environment.
- Reuse the existing material texture descriptor array for the HDRI instead of
  adding a new sampler binding. The earlier MoltenVK validation work showed the
  fragment-stage sampler limit is tight; using the table slot keeps the layout
  at 15 material/HDRI textures plus one shadow sampler.
- Add `Runtime::load_hdr_texture()` using `stbi_loadf` and
  `VK_FORMAT_R32G32B32A32_SFLOAT`. Regular LDR material loading remains through
  `Runtime::load_texture()`.
- Add `DrawList::set_environment()` as per-frame draw data. The environment is
  app-owned state, while the runtime owns the Vulkan texture and descriptor
  table.
- Background rendering uses a small fullscreen-triangle pipeline sampling the
  same equirectangular texture. Mesh lighting uses a deliberately approximate
  shader-side term: sample the environment along the normal for diffuse and
  along the reflection vector for specular. A later pass can replace this with
  irradiance and prefiltered reflection maps once the framework needs higher
  fidelity.

Useful follow-up pressure points:

- DFSPH particle rendering now reaches Vulkan as batched mesh instances, but the
  app still builds generic per-particle draw/material commands on the CPU. If
  the larger scenes become CPU-bound, the real reusable unlock is a compact
  particle/viz upload path that bypasses that generic command construction.
- PBA collision is app-owned for now. The manipulator plugin owns only
  transient interaction state, with app callbacks for transform get/set.
- Capture/video export from the old projects is still a good candidate for a
  `ds_vk::capture` module once screenshot validation stabilizes.
- Reduce the fixed material texture table from 16 to 15 slots because the shadow
  map sampler shares the same fragment shader stage and MoltenVK commonly
  exposes a 16-sampler per-stage floor on Apple hardware.

## PBA Comparison Pass

Checked local source:
`/Users/danielsinkin/GitHub_private/physically-based-animations`.

Important discrepancies from the proper PBA implementation:

- The proper PBA has SOA rigid-body storage, force and torque accumulators,
  angular velocity, inertia tensors, sleeping, grabbed-body flags, sweep-and-
  prune broadphase, OBB contact generation, warm starting, position/velocity
  constraint solvers, and a contact cache.
- The `ds_vk` PBA user remains deliberately smaller: app-owned `Body` data,
  linear velocity only, no torque/angular solver, no sleeping, no contact cache,
  and an AABB-style penetration resolver.
- This pass integrated the reusable parts that fit the MVP without turning
  `ds_vk` into a physics engine:
  - `Body::force_accum`;
  - `GravityForce`, `AttractorForce`, `RepulsionForce`, and `NBodyForce`;
  - grabbed-body skipping for force integration and collision resolution;
  - a sweep-and-prune broadphase over world AABBs;
  - per-step broadphase stats for checking candidate count against all-pairs;
  - `ds_vk::Manipulator`, a static plugin with callback-owned transforms.
- Performance judgment: for the current 56-body pyramid this app should be
  roughly comparable or faster per step than the proper PBA because the solver
  is much simpler. It is not equivalent for larger scenes. The proper PBA's SOA
  layout, parallel force/inertia passes, cached contacts, OBB contacts, and
  solver structure remain the better design for the old repo's large scenes.
  The new sweep-and-prune pass removes the worst all-pairs behavior for sparse
  scenes, but the MVP is still not a replacement for the full PBA solver.

## DFSPH Renderer / Vulkan Performance Pass

Start time for this pass: 2026-05-16.

Old DFSPH agent research checked:

- `/Users/danielsinkin/GitHub_private/SPH-Seminar/dfsph_viewer/docs/research_notes.md`
- `/Users/danielsinkin/GitHub_private/SPH-Seminar/dfsph_viewer/docs/future_work.md`
- `src/dfsph_viewer/core/culling.{hpp,cpp}`
- `src/dfsph_viewer/gfx/vulkan_particle_renderer.cpp`

Useful old-implementation findings:

- The old DFSPH particle path rendered particles as one sphere mesh plus a
  per-particle instance buffer. That is the important scalability property: 50k
  particles should not mean 50k Vulkan draw calls.
- The old renderer used persistently mapped VMA buffers and explicitly called
  `vmaFlushAllocation` after host writes. This matters for non-coherent
  host-visible memory and is still correct when the flush becomes a no-op on a
  coherent memory type.
- Conservative frustum culling and optional front-to-back/screen-space
  occlusion were app-level decisions with debug counters. The measured old
  screen-space occlusion path could reduce GPU sphere work in overdraw-heavy
  views, but it cost a CPU proof pass and was slower in forced cache-miss
  playback cases. That should remain an opt-in DFSPH-specific/debug feature
  rather than a core-runtime default.
- The old renderer split dynamic cache keys by camera, frame, debug settings,
  surface mesh, visualization lines, and rigid-body mesh data. That is a useful
  future shape for app-level large-data caches, but it would be premature in the
  generic `DrawList` right now.

External references checked:

- Khronos command-buffer sample:
  https://github.khronos.org/Vulkan-Site/samples/latest/samples/performance/command_buffer_usage/README.html
  - Recycle command buffers through command-pool reset rather than
    allocate/free on hot paths. Secondary command buffers are only worth it when
    there is enough recorded work per buffer.
- Vulkan Guide memory allocation:
  https://docs.vulkan.org/guide/latest/memory_allocation.html
  - Suballocation is the normal Vulkan shape. UMA devices can expose memory that
    is both device-local and host-visible, which is relevant for Apple
    Silicon/MoltenVK dynamic uploads.
- VMA memory mapping and usage-pattern docs:
  https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/memory_mapping.html
  and
  https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
  - Persistently mapped buffers are acceptable. Host writes to non-coherent
    mapped memory need an explicit flush; VMA handles atom-size alignment in
    `vmaFlushAllocation`.
- NVIDIA Vulkan Do's and Don'ts:
  https://developer.nvidia.com/blog/?p=14696
  - Minimize pipeline binds and group draw calls by shader/pipeline/material
    family where practical. Keep barriers precise and do not assume the driver
    will hide command-recording cost on worker threads.
- Khronos pipeline-barrier sample:
  https://docs.vulkan.org/samples/latest/samples/performance/pipeline_barriers/README.html
  - Broad barriers can force unnecessary pipeline flushes. Correctness comes
    first, but performance-sensitive barriers should name the actual producer
    and consumer stages.

Implemented here:

- Generic mesh rendering now collapses visible `DrawList` mesh commands into
  instanced batches grouped by `MeshHandle`. The app still uses an
  immediate-mode-ish `draw_mesh` / `draw_basic_mesh` call per object, but the
  Vulkan side binds each mesh once per contiguous batch and uses
  `vkCmdDrawIndexed(..., instance_count, ..., first_instance)`.
- Per-instance model matrix, normal matrix, and material index moved to a
  storage buffer consumed by `mesh.vert`. This is enough for the DFSPH particle
  use case: many sphere instances with per-particle color/material differences.
- Runtime stats now report both submitted mesh draw commands and emitted mesh
  batches so CPU-side draw-call pressure is visible in the UI.
- Mapped dynamic uploads now flush through a small `Runtime::Impl::flush_buffer`
  helper after writes to texture staging, material, instance, lighting, and
  debug segment buffers.

Not copied yet:

- DFSPH-specific frustum/screen-occlusion culling. The framework now removes the
  worst Vulkan submission issue, but the app still builds per-particle commands
  and per-particle materials on the CPU. If the 130k dambreak scene is still too
  heavy, the next targeted step is a DFSPH particle/viz upload path that builds
  a compact sphere-instance buffer directly instead of going through generic
  `MeshDrawCommand` objects.
- Indirect draws and GPU-driven culling. They are interesting, but too much
  machinery for the current scale and would make the framework less pleasant to
  modify while the API is still young.

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
- Generic mesh draws are now batched by mesh through an instance buffer. The
  next performance question is not Vulkan draw-call count, but whether very
  large app data sets need app-specific compact upload paths and cache keys.
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
- The first implementation waited for the device to go idle before destroying old
  mesh buffers. That was conservative but too coarse for later streamed-surface
  playback; see the 2026-05-16 surface streaming note below.
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

### 2026-05-16 Lights, Masks, And First Shadow Map

- Replaced the hard-coded shader light direction with explicit per-frame lights
  recorded through `DrawList`.
- Added directional, radial, and spot light configs. Directional lights have a
  direction and no attenuation; radial and spot lights have position, range, and
  inverse-square-ish attenuation with a smooth cutoff. Spot lights also use
  CPU-precomputed cone scale/offset from the Khronos punctual-lights reference.
- Added `MeshRenderMask`:
  - `visible_to_camera` controls camera-pass rendering;
  - `shadow_producer` controls participation in the shadow depth pass;
  - `shadow_consumer` controls whether the object samples the active shadow map;
  - `light_receiver` controls whether direct lights are applied.
- Added a depth-only shadow pipeline and `shadow.vert`. The runtime renders the
  first shadow-enabled directional or spot light into a sampled depth texture,
  then the mesh fragment shader applies a small 3x3 PCF kernel when evaluating
  that light.
- The shadow-map synchronization barrier uses depth-stencil attachment write
  access from both early and late fragment-test stages to fragment-shader
  sampled-image reads. This follows the Vulkan Guide shape for depth attachment
  to shadow-map sampling.
- Reduced material texture slots from 16 to 15 because the mesh fragment shader
  also binds one shadow-map sampler. This keeps the current fixed descriptor set
  below the 16-sampler MoltenVK limit found earlier.
- App scene now contains the textured floor, large sphere, cube, column, small
  metallic sphere, vector field, grid, and light gizmos. The floor is configured
  as a shadow consumer but not a shadow producer, which exercises the render
  masks without making the ground self-cast.
- Verification:
  - `cmake --build build-core`
  - `cmake --build build`
  - `ctest --test-dir build-core --output-on-failure`
  - `ctest --test-dir build --output-on-failure`
  - `clang-tidy -p build ds_vk/runtime.cpp app/main.cpp tests/test_runtime.cpp tests/test_main.cpp`
  - `./run.sh --smoke-frames 8 --screenshot run/lights_shadows.png --hide-ui`
  - `./.venv/bin/python scripts/validate_screenshot.py run/lights_shadows.png --min-stddev 4 --min-brightness 5`

### 2026-05-16 Camera Depth Debug View

- Added `MeshDebugMode::camera_depth`. It shades mesh fragments as grayscale
  linear camera-space depth using the material's debug scalar range as near/far
  visualization bounds.
- This is intentionally not yet a sampled swapchain-depth attachment preview.
  It still uses the normal camera depth test for visibility, so it gives the
  same "fog-like depth buffer" read on visible mesh surfaces without adding
  another sampled-depth descriptor, render-pass dependency, or post-process
  fullscreen pipeline.
- The app exposes this as "Camera depth debug", plus `--depth-debug` for
  screenshot/smoke runs. When enabled, colored debug overlays are suppressed so
  the view stays grayscale.
- The app now labels the three light toggles explicitly as Directional, Radial,
  and Spot, and light gizmos follow the same per-light `enabled` flags.

### 2026-05-16 DFSPH Renderer Performance Backport

- Checked the old DFSPH agent research notes and renderer/culling source, then
  backported the broadly useful parts: instanced mesh batching for repeated
  mesh draws and explicit VMA flushes after mapped-buffer writes.
- Verification:
  - `cmake --build build`
  - `ctest --test-dir build --output-on-failure`
  - `clang-format --dry-run --Werror ds_vk/runtime.cpp ds_vk/runtime.hpp ds_vk/shaders/mesh.vert`
  - `clang-tidy -p build ds_vk/runtime.cpp app/dfsph_main.cpp`
  - `git diff --check`
  - `./run.sh --app dfsph --scene-id dambreak_small_iisph_v1 --show-mesh --show-particles --hide-ui --smoke-frames 8 --screenshot run/dfsph_perf_batch_smoke.png`
  - `./.venv/bin/python scripts/validate_screenshot.py run/dfsph_perf_batch_smoke.png`

### 2026-05-16 DFSPH Surface Mesh CPU Preload

- Profiling the 50k DFSPH surface mesh showed the playback hot path was CPU
  limited by per-frame `.mesh.gz` read/decompress/decode work, not by particle
  rendering. The surface cache is about 1.07 GiB compressed, about 2.89 GiB as
  decompressed quantized payloads, and about 9.38 GiB as fully expanded
  `MeshData` CPU vectors before allocator overhead. That is large, but
  acceptable as an opt-in DFSPH mesh-viewer path on the development machine.
- `ds_vk_dfsph_app` now preloads decoded CPU `MeshData` for every available
  surface frame when the surface mesh view is enabled. This deliberately keeps
  the renderer resource model unchanged: playback still uploads the active
  frame through `Runtime::replace_mesh`, but file IO, gzip, and decode are paid
  once up front.
- The preload uses an explicit `std::jthread` manager plus worker `std::jthread`s
  instead of nested `std::async`, so ownership, joining, and stop requests are
  visible. The worker count is capped at 8 for now. The fullscreen ImGui loading
  overlay dims and captures the UI, then shows elapsed time, worker count,
  checked/decoded frame counts, and a determinate progress bar.
- Crash note: the first threaded preload attempt put a 1 MiB gzip read buffer on
  each worker stack and crashed on macOS with `SIGBUS` against the stack guard.
  `read_gzip_file` now uses a heap `std::vector<u8>` chunk buffer instead.
- Before CPU preload on
  `dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles --hide-ui
  --smoke-frames 120 --playback-speed 20 --profile`, active surface frames
  averaged about 20.45 ms total: 17.77 ms read/decompress/decode and 2.68 ms
  upload.
- Single-threaded CPU preload initially took 12882.841 ms for all 600 frames.
  The explicit 8-worker preload reduced that to 4223.951 ms on the same 50k
  scene. Runtime surface read/gzip/decode counters were 0.000 ms after preload.
- The remaining cost is now the upload/replacement path. Before the next change,
  `replace_mesh` still used `vkDeviceWaitIdle` before destroying the previous
  mesh buffers, so fast mesh playback could stall on synchronization and buffer
  upload. A bigger fix would use staged per-frame uploads plus deferred resource
  destruction tied to frame fences, or a more specialized streaming surface
  buffer path. That is outside this CPU preload change, but it is now the clear
  next bottleneck instead of gzip/decode.
- Verification:
  - `cmake --build build --target ds_vk_dfsph_app`
  - `./run.sh --app dfsph --scene-id dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles --smoke-frames 12 --screenshot run/dfsph_surface_preload_overlay.png`
  - `./.venv/bin/python scripts/validate_screenshot.py run/dfsph_surface_preload_overlay.png`
  - `./run.sh --app dfsph --scene-id dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles --hide-ui --smoke-frames 1200 --playback-speed 20 --profile`
  - `clang-format --dry-run --Werror app/dfsph_main.cpp`
  - `clang-tidy -p build app/dfsph_main.cpp`
  - `ctest --test-dir build --output-on-failure`
  - `git diff --check`

### 2026-05-16 DFSPH Surface Streaming Follow-up

- Rechecked the old DFSPH viewer performance notes and source. The old renderer
  avoided device-wide waits in normal playback, cached per-swapchain surface
  buffers, applied back-face culling where valid, and had a separate culling path
  for particle spheres. The immediately applicable lesson for this framework was
  not to treat a high-frequency mesh replacement like an editor slider rebuild.
- The new stutter source was `Runtime::replace_mesh`: every DFSPH surface frame
  created replacement vertex/index buffers and then called `vkDeviceWaitIdle`
  before destroying the previous buffers. On the 50k scene this creates exactly
  the intermittent playback hitch that remains after CPU preload removes gzip and
  decode from the hot path.
- `replace_mesh` now retires the old mesh buffers into a small queue and destroys
  them only after enough swapchain frames have passed. The frame loop collects
  retired meshes after waiting the current swapchain image fence and resetting
  that command pool, which keeps old `VkBuffer` handles alive for command buffers
  that may still reference them without stopping the whole device.
- The first profile after that still showed upload spikes because the DFSPH app
  was allocating replacement vertex/index buffers every surface frame. The app
  now keeps one streamed surface `MeshHandle` per swapchain image and updates the
  current image's handle in place once capacity permits. That is safe because the
  runtime has already waited that swapchain image's fence before app update code
  runs. Larger frames still fall back to `replace_mesh`, so capacity grows lazily.
- The reusable framework piece is `Runtime::update_mesh`. It is intentionally a
  sharp tool: it reuses existing buffers when there is enough capacity, but the
  caller owns synchronization and should use per-frame/per-swapchain handles for
  continuously streamed geometry.
- Mesh resources are now created persistently mapped when possible. Streamed
  updates therefore use direct `memcpy` plus `vmaFlushAllocation` instead of
  repeatedly entering VMA's map/copy/unmap path. This does not remove the raw
  bandwidth cost of uploading large surface meshes, but it removes avoidable
  allocation and mapping overhead from the steady-state loop.
- After CPU preload completes, the DFSPH app reserves each swapchain surface
  mesh handle to the largest decoded vertex/index count in the history. That
  pays the worst allocation/capacity growth cost once near the loading phase
  instead of letting the first large frame encountered during playback hitch the
  interactive path. This now uses explicit `Runtime::reserve_mesh_capacity`
  rather than uploading a real frame solely to grow the buffers.
- Final 50k looped smoke profile
  (`--scene-id dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles
  --hide-ui --smoke-frames 8000 --playback-speed 20 --loop --profile`) rendered
  7406 surface frames after preload. `surface_read`, `surface_gzip`, and
  `surface_decode` stayed at 0.000 ms. `surface_upload` averaged 5.549 ms with
  a 62.859 ms max, and preload was 5071.832 ms. The remaining steady cost is raw
  CPU writes of large surface vertex/index buffers, not file IO or Vulkan
  device-idle synchronization.
- The DFSPH surface path now uses a compact `PositionNormalMeshData` format
  instead of the generic `MeshData` vertex. Surface playback needs position and
  normal only; vertex color and UV are constant/unused. This cuts the streamed
  vertex stride from the generic 48-byte `Vertex` to a 24-byte
  `PositionNormalVertex`. The index buffer is unchanged, so the total upload is
  not exactly halved, but the large per-frame write is substantially smaller.
- Compact surface vertices use a separate Vulkan graphics pipeline with the same
  mesh descriptor layout and fragment shader. The compact vertex shader emits a
  constant white vertex color and zero UV, which is correct for the current
  material-colored DFSPH surface path. Textured meshes should stay on the
  generic `MeshData` path.
- After compacting the 50k looped smoke profile
  (`--scene-id dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles
  --hide-ui --smoke-frames 8000 --playback-speed 20 --loop --profile`) rendered
  7594 surface frames after preload. `surface_upload` averaged 3.403 ms with a
  47.707 ms max. That is a clear improvement from the reserved generic-vertex
  path's 5.549 ms mean / 62.859 ms max, and confirms the bottleneck had become
  raw CPU-to-mapped-buffer write volume.
- The DFSPH surface path now preserves cache quantization through preload and
  upload when the cache version provides it: positions stay as `u16` values in
  per-frame bounds, and cache normals are repacked to octahedral `u8x2` during
  CPU preload. The runtime exposes this as `QuantizedPositionNormalMeshData`.
  Its vertex stride is 12 bytes: four `u16` position channels, two `u8`
  oct-normal channels, and two reserved bytes.
- Decode is done by Vulkan normalized vertex fetch plus the existing model path.
  The per-mesh decode transform is pre-composed into the instance/model matrix,
  so local position decoding remains `bounds_min + extent * q / 65535` without
  increasing push constant size. Positions use `R16G16B16A16_UNORM`; normals use
  `R8G8_UNORM` and are octahedrally decoded and normalized in the vertex shader.
  Position playback accuracy stays at the cache's quantization level; normal
  precision is now oct8, which is visually unchanged in the current DFSPH smoke
  comparison.
- The first quantized shader version used integer vertex attributes and explicit
  shader divides. That cut upload cost but moved too much work into the render
  bucket. Switching to `UNORM`/`SNORM` vertex formats moved conversion back into
  vertex fetch and made the whole frame faster.
- Final quantized 50k looped smoke profile
  (`--scene-id dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles
  --hide-ui --smoke-frames 8000 --playback-speed 20 --loop --profile`) rendered
  6215 surface frames after preload. `surface_upload` averaged 0.427 ms with a
  10.795 ms max; frame mean was 1.927 ms. Preload was 3325.309 ms because CPU
  preload no longer expands all vertices to f32.
- Oct8 follow-up with playback stalling enabled rendered 5312 surface uploads in
  the same smoke command. This run was display/frame-paced (`frame` mean 16.825
  ms, `render` mean 15.421 ms), so it is not directly comparable to the earlier
  uncapped-looking timing. It still validated the packed path: preload was
  3270.184 ms and `surface_upload` averaged 1.097 ms with a 25.527 ms max.
  Screenshot comparison against the previous quantized surface smoke was exact:
  `mean_abs_rgb [0.0, 0.0, 0.0]`, `max_channel 0`.
- The closest available source scene to the earlier "130k" dambreak discussion
  is `dambreak_150k_300f_dfsph_v1` with 148,877 particles. It was migrated from
  the DFSPH viewer showcase exports as 300 particle VTK frames plus 300
  quantized surface frames. A mesh-only screenshot smoke passed:
  `./run.sh --app dfsph --scene-id dambreak_150k_300f_dfsph_v1 --show-mesh
  --hide-particles --hide-ui --smoke-frames 16 --screenshot
  run/dfsph_150k_surface_smoke.png`.
- The 150k looped mesh-only profile
  (`--scene-id dambreak_150k_300f_dfsph_v1 --show-mesh --hide-particles
  --hide-ui --smoke-frames 2400 --playback-speed 20 --loop --profile`) rendered
  2040 surface uploads after preload. Maximum surface size was 2,067,538
  vertices and 691,112 triangles. Preload took 6298.399 ms. `surface_upload`
  averaged 6.238 ms with a 78.367 ms max; frame mean was 17.688 ms with a
  167.857 ms max. This is workable enough for inspection, but not smooth enough
  to call done. The next obvious improvement is an app-specific surface stream
  ring/prefetcher so the visible frame usually draws an already-uploaded mesh.
- The first 8-slot surface stream ring experiment filled too aggressively and
  made burst stalls worse. The tuned policy keeps 8 reusable slots, targets
  `current + 1` through `current + 4`, uploads at most two missing future frames
  per sync step while warming the window, and stalls playback if the next
  surface frame is not cached. The screenshot smoke stayed pixel-identical to
  the previous surface path.
- The tuned 150k stream profile
  (`--scene-id dambreak_150k_300f_dfsph_v1 --show-mesh --hide-particles
  --hide-ui --smoke-frames 2400 --playback-speed 20 --loop --profile`) rendered
  2097 surface uploads. `frame` averaged 16.870 ms with a 99.498 ms max,
  `render` averaged 11.029 ms with a 38.870 ms max, and `surface_upload`
  averaged 6.402 ms with a 68.350 ms max. The ring reduced the worst frame and
  render spikes, but it does not change the fundamental one-new-surface-frame
  write volume during sequential playback.
- Follow-up profiling separated startup warmup from steady playback. The app now
  pre-fills the eight surface stream slots after CPU preload, reports
  `surface_warmup` separately, and skips the runtime stat sample that contains
  that warmup. In the measured 150k runs, warmup was not the source of the large
  upload max: warmup stayed around 5-11 ms, while steady uploads still produced
  larger spikes. A noisy short run saw the largest steady upload on frame 88
  (1,871,332 vertices / 625,312 triangles), so the spike was not simply frame 0
  initialization.
- `Runtime::update_mesh` no longer validates every index by default. That scan is
  useful for debugging generated meshes, but it is pure CPU work on streaming
  frames that were already validated during preload. Callers can opt back in with
  `MeshUpdateConfig{.validate_indices = true}`.
- The 150k surface CPU cache is large enough to matter: reading only the gzip
  payload headers gives about 5.054 GiB of resident decoded mesh data for 300
  frames. Each max-sized uploaded frame is about 31.57 MiB. If the machine is
  under memory pressure, those first-use copies can still show page/cache stalls
  even though the renderer is not close to raw hardware bandwidth.
- Rough transfer math for the 150k max surface: `2,067,538 * 12` vertex bytes
  plus `691,112 * 3 * 4` index bytes is about 33.1 MB, or 31.6 MiB, for a full
  max-sized surface update. Using the 6.402 ms mean upload as a rough upper-bound
  byte rate gives about 5.2 GB/s effective CPU write throughput. This is far
  below Apple M2's advertised 100 GB/s unified memory bandwidth
  (https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/),
  which strongly suggests the bottleneck is not a PCIe-style CPU-to-discrete-GPU
  transfer. In this app the buffers are already VMA host-visible mapped buffers
  (`VMA_MEMORY_USAGE_AUTO_PREFER_HOST` plus `HOST_ACCESS_SEQUENTIAL_WRITE`), so
  on Apple Silicon/MoltenVK this should be shared-memory writing plus coherency
  flush/cache behavior, not a copy into separate VRAM.
- Mesh smoothing was intentionally removed from this pass. The CPU render-copy
  smoothing experiment was the wrong shape for packed playback; proper smoothing
  needs its own design, either as cache preprocessing or a GPU compute path.
- No explicit `vkCmdPipelineBarrier2` is needed for this specific upload path:
  the CPU writes happen before command-buffer recording/submission, and VMA's
  copy helper maps, copies, unmaps, and flushes non-coherent memory as required.
  The synchronization issue here is lifetime of old buffers, not shader/transfer
  visibility between queued GPU operations.
- Verification after the compact surface format:
  - `cmake --build build`
  - `ctest --test-dir build --output-on-failure`
  - `clang-format --dry-run --Werror app/dfsph_main.cpp ds_vk/mesh.hpp ds_vk/mesh.cpp ds_vk/runtime.hpp ds_vk/runtime.cpp ds_vk/shaders/mesh_position_normal.vert ds_vk/shaders/mesh_quantized_position_normal.vert ds_vk/shaders/shadow_quantized_position.vert`
  - `clang-tidy -p build ds_vk/mesh.cpp`
  - `clang-tidy -p build ds_vk/runtime.cpp`
  - `clang-tidy -p build app/dfsph_main.cpp`
  - `clang-tidy -p build tests/test_main.cpp`
  - `git diff --check`
  - `./run.sh --app dfsph --scene-id dambreak_50k_600f_dfsph_v2 --show-mesh --hide-particles --hide-ui --smoke-frames 8000 --playback-speed 20 --loop --profile`
  - `./run.sh --app dfsph --scene-id dambreak_small_iisph_v1 --show-mesh --hide-particles --hide-ui --smoke-frames 16 --screenshot run/dfsph_compact_surface_smoke.png`
  - `./.venv/bin/python scripts/validate_screenshot.py run/dfsph_compact_surface_smoke.png`
  - `./run.sh --app dfsph --scene-id dambreak_small_iisph_v1 --show-mesh --hide-particles --hide-ui --smoke-frames 16 --screenshot run/dfsph_quantized_surface_smoke.png`
  - `./.venv/bin/python scripts/validate_screenshot.py run/dfsph_quantized_surface_smoke.png`
  - pixel comparison between compact decoded and quantized screenshot:
    `mean_abs_rgb [0.0, 0.0, 0.0]`, `max_channel 0`.

## 2026-05-17 Explicit Frame API Cut

- Changed the framework center from `Runtime::run(app)` to explicit frame
  driving. The intended app shape is now `initialize()`, `begin_frame()`, raw
  Vulkan/app work, `render_shadow_pass()`, `begin_main_pass()`,
  `render_draw_list()`, optional raw in-pass work, `render_imgui()`,
  `end_main_pass()`, and `end_frame()`.
- This is a deliberate Vulkan-mental-model choice. The runtime still owns SDL,
  swapchain acquisition, per-frame command buffers, fences/semaphores, common
  render passes, ImGui setup/rendering, and presentation, but app code chooses
  command order when it needs compute, barriers, custom transfer work, or custom
  render commands.
- Kept `Runtime::run_prototype(app)` as a thin baby-mode wrapper around the
  manual protocol for quick CPU-side MVPs. It is no longer the conceptual core.
- Full app users (`basic`, `vectorfield`, `pba`, `dfsph`) were moved to the
  explicit loop so the repo itself demonstrates the preferred API.
