# Architecture

`ds_vk` is a small personal Vulkan framework for visualization experiments. It
is not a renderer abstraction layer, game engine, or cross-API project.

## Goals

- Make a new C++/Vulkan visualization app reach "camera, mesh, debug lines,
  ImGui controls, screenshot" quickly.
- Keep raw Vulkan handles visible so app code can record custom graphics or
  compute work without fighting the framework.
- Centralize recurring Vulkan setup: SDL window, MoltenVK portability extensions,
  swapchain/depth images, command buffers, frame sync, VMA, ImGui, shader
  loading, and screenshot readback.
- Grow from real app pressure. The fifth project matters more than the first.

## Non-Goals

- No OpenGL/Metal/DirectX abstraction.
- No stable public API promise.
- No scene graph in the first shape.
- No attempt to hide Vulkan synchronization for arbitrary compute/render
  interaction.

## Current Runtime Loop

The primary API is explicit frame driving. `Runtime` owns the window, swapchain,
per-frame command buffer, synchronization, ImGui frame setup, common render
passes, and presentation. The app owns the order in which it asks those pieces to
record work:

```cpp
ds_vk::Runtime runtime{cfg};
runtime.initialize();

while (auto* frame = runtime.begin_frame())
{
    app.update(*frame, frame->dt_seconds);

    // Optional raw Vulkan outside the main color/depth render pass.
    vkCmdBindPipeline(frame->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdDispatch(frame->command_buffer, groups_x, groups_y, groups_z);

    if (runtime.ui_visible())
    {
        runtime.draw_runtime_ui();
        app.draw_ui(*frame);
    }

    runtime.render_shadow_pass();
    runtime.begin_main_pass();
    runtime.render_draw_list();

    // Optional raw Vulkan inside the main render pass can be recorded here.

    runtime.render_imgui();
    runtime.end_main_pass();
    runtime.end_frame();
}
```

`begin_frame()` polls SDL input, updates the shared camera controls, starts a new
ImGui frame, acquires a swapchain image, waits/resets the image fence and command
pool, begins the primary command buffer, clears the frame `DrawList`, and returns
the active `FrameContext`. A `nullptr` return means the app should exit.

The built-in pass helpers then map closely to the raster frame:

1. `render_shadow_pass()` records the optional depth-only shadow pass.
2. `begin_main_pass()` begins the swapchain color/depth render pass.
3. `render_draw_list()` records built-in mesh, environment, and debug pipelines.
4. `render_imgui()` finalizes ImGui draw data and records it into the active pass.
5. `end_main_pass()` closes the render pass and records optional screenshot copy.
6. `end_frame()` ends/submits the command buffer, writes pending screenshots,
   presents the swapchain image, and updates runtime stats.

`Runtime::run_prototype(app)` still exists as a thin baby-mode wrapper around
that protocol for quick CPU-heavy MVPs, but full apps in `app/` use the explicit
loop directly.

## Current Shader Interface

- Public color data uses `ds_vk::Color` and `ds_vk::ColorU8`, not `Vec4`.
  Colors are array-backed standalone types with named conversion helpers such as
  `to_vec4`, `to_color_u8`, `with_alpha`, and `mix_color`; generic vector
  arithmetic is intentionally unavailable for colors.
- Mesh draws use a 128-byte vertex-stage push block: view-projection and model
  matrices. The vertex shader derives world position and a normal matrix from
  the model matrix.
- The first mesh pipeline binds a per-frame material storage buffer and a
  per-frame lighting storage buffer. The mesh fragment shader uses
  Cook-Torrance metallic/roughness lighting from explicit directional, radial,
  and spot lights plus a lightweight equirectangular HDR environment term; view
  direction comes from the actual camera position stored in the material buffer.
  Each non-instanced draw passes its material index
  through `firstInstance`/`gl_InstanceIndex`.
- Materials currently carry base color, emissive color, metallic, roughness,
  ambient occlusion, and an optional base-color texture handle. Emission only
  contributes to the emitting surface; it does not spawn lights.
- Material textures are bound through a fixed 15-slot combined-image-sampler
  table. Slot 0 is a generated white fallback; app-loaded texture handles occupy
  later slots and materials opt into them with
  `.textures = {.base_color = handle}`. The HDR environment texture also lives
  in this same table, and the selected slot is passed through the lighting data.
  That avoids adding another fragment sampler binding and keeps the current
  layout under the validated MoltenVK per-stage sampler limit before a later
  bindless pass.
- Mesh draw configs also carry an `ObjectId` and `MeshDebugConfig`. Hidden draws
  are culled before recording; selected/color-override/scalar-heatmap/normal/id
  views are applied in the mesh fragment shader. `camera_depth` debug mode
  renders visible mesh surfaces as grayscale linear camera-space depth using the
  configured debug scalar range; this is the lightweight depth-buffer inspection
  path before adding a true sampled-depth preview pass.
- Mesh draw configs carry a `MeshRenderMask` with separate switches for camera
  visibility, shadow production, shadow consumption, and light reception. A
  draw can therefore be hidden from the camera while still casting a shadow, or
  visible but unlit for debug/material inspection.
- Shadow mapping currently supports one active shadow-casting directional or
  spot light per frame. Directional lights use an orthographic light projection
  centered around the camera pivot; spot lights use a square perspective
  projection from the light. Radial light shadows need an omnidirectional cube
  shadow map and are intentionally left for a separate pass.
- Debug segments use a separate 96-byte push block and one per-frame mapped
  segment buffer.
- `ds_vk::viz` builds visual helpers on top of that debug segment path: color
  ramps, scalar ranges, vector fields, and camera-facing cross markers.

## Repo Layout

- `ds_vk/` is the framework/library: public headers, implementation files, and
  built-in shaders live together there.
- `app/` contains full app users of the framework.
- `app/pba/` is an app-owned headless physics library used by the PBA user. It
  is intentionally not a framework scene or simulation module.
- `tests/` contains executable test harnesses.
- `external/` contains vendored dependencies.
- `docs/` and `scripts/` support research and validation.

## Current App Users

- `ds_vk_basic_app` is the small interactive material/light/picker playground.
- `ds_vk_vectorfield_app` is the vector-field visualization user that matures
  `ds_vk::viz` without making vector fields part of the runtime core.
- `ds_vk_dfsph_app` is a fixed-data DFSPH playback user. It loads the vendored
  small-dambreak VTK history from `assets/dfsph/.../vtk`, renders particles as
  mesh draws, can preload decoded CPU surface meshes for the mesh view, and uses
  `ds_vk::viz` for velocity arrows and bounds markers. It does not vendor
  SPlisHSPlasH or generate scenes on demand.
- `ds_vk_pba_app` is a realtime rigid-body visualization user. Space toggles
  simulation pause while camera controls continue to work. Its pyramid physics
  is an app-side MVP AABB solver with force accumulators, a small force set,
  grabbed-body handling, and sweep-and-prune broadphase stats. Speed coloring
  and velocity arrows are routed through `ds_vk::viz`; selection uses
  `ds_vk::picker`, and object manipulation uses the callback-based
  `ds_vk::Manipulator` plugin.

## Asset Loading

`ds_vk::assets` currently contains a small CPU-side glTF/GLB mesh loader for the
framework's own visualization needs. It supports triangle primitives with
positions, normals, texcoords, indices, embedded data URIs, external buffers, and
GLB BIN chunks. It can generate smooth normals when a test mesh omits them.

This is not yet a full asset system. It does not try to own scene hierarchy,
animations, material graphs, skinning, or arbitrary glTF extensions. GLB unknown
chunks are ignored after bounds validation, matching the extension-friendly GLB
shape; missing required mesh data remains an error.

## App Surface

Apps are normal C++ objects. The framework does not require inheritance or an
interface base class. A full app usually still has functions like this, but
`main()` calls them explicitly:

```cpp
class MyApp final
{
  public:
    auto setup(ds_vk::Runtime& runtime) -> void;
    auto update(ds_vk::FrameContext& frame, ds_vk::f32 dt_seconds) -> void;
    auto draw_ui(ds_vk::FrameContext& frame) -> void;
    auto shutdown(ds_vk::Runtime& runtime) -> void;
};
```

The app can stay high-level for simple work:

```cpp
runtime.camera({
    .pivot = 0.7f * ds_vk::k_axis_z,
    .distance = 5.4f,
    .yaw = glm::radians(42.0f),
    .pitch = glm::radians(25.0f),
});

frame.draw.draw_basic_mesh({
    .mesh = mesh,
    .color = color,
});
frame.draw.draw_mesh({
    .mesh = mesh,
    .transform = transform,
    .object_id = {.value = 17u},
    .material = {.base_color = color, .metallic = 0.2f, .roughness = 0.5f},
    .debug = {.mode = ds_vk::MeshDebugMode::selected_pulse, .selected = is_selected},
});
frame.draw.debug_arrow({
    .origin = origin,
    .vector = direction,
    .color = color,
});
frame.draw.debug_sphere({
    .center = center,
    .radius = radius,
    .color = color,
});
```

It can also drop to Vulkan directly:

```cpp
vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdDispatch(frame.command_buffer, groups_x, groups_y, groups_z);
```

Mesh upload/replacement requires an initialized runtime, so app code should do it
after `runtime.initialize()`, usually from `setup`, `update`, or UI callbacks.

## Selection Shape

Selection is deliberately not an `on_click` callback on a mesh resource. A mesh
can be rendered many times with different transforms, materials, masks, and app
meaning, so selection is modeled as app-owned object IDs plus an optional static
picker module. `ds_vk::Picker` owns only per-frame picking targets, not scene
objects:

```cpp
ds_vk::Picker picker;

picker.add_sphere({
    .object_id = sphere_id,
    .center = sphere_position,
    .radius = sphere_radius,
});
picker.add_obb({
    .object_id = cube_id,
    .center = cube_position,
    .half_extent = 0.5F * glm::abs(cube_scale),
    .rotation = cube_rotation,
});

const auto hit = picker.click({
    .camera = frame.camera,
    .mouse_px = frame.input.left_click.position_px,
    .viewport_px = {
        static_cast<ds_vk::f32>(frame.extent.width),
        static_cast<ds_vk::f32>(frame.extent.height),
    },
});
```

The app rebuilds picker targets from whatever collider shape makes sense for the
current visualization, maps the returned `ObjectId` to its own state, and then
passes `.debug = {.selected = true}` or `.debug = {.hidden = true}` through draw
configs on later frames. The picker supports `click` for screen-space mouse
input and `raycast` for code that already has a world ray. Current target shapes
are sphere, AABB, OBB, capsule, and screen-space segment.

The picker expects framebuffer pixel coordinates. The runtime converts SDL
window coordinates to framebuffer coordinates with `SDL_GetWindowSizeInPixels /
SDL_GetWindowSize`, which keeps Retina/high-DPI picking aligned with the
swapchain. Because `Camera::projection_matrix` already applies Vulkan's Y flip,
the pick ray maps screen Y using `y_ndc = 2 * y / height - 1`.

The `viz` plugin deliberately remains visual grammar rather than app semantics.
It does not know what SPH, rigid-body velocity, or neighborhood membership means;
app code computes those values and passes positions, vectors, scalar ranges, and
selected endpoints into reusable helpers:

```cpp
viz::draw_vector_field(frame.draw, {
    .positions = std::span<const ds_vk::Vec3>{positions},
    .vectors = std::span<const ds_vk::Vec3>{velocities},
    .scale = 0.04f,
    .color_by_magnitude = true,
    .color_ramp = speed_ramp,
});

viz::draw_cross_marker(frame.draw, {
    .camera = frame.camera,
    .center = selected_hit_position,
});
```

## Near-Term Roadmap

- Descriptor indexing/bindless support path: feature-gated query/enablement
  exists now; replace the fixed 15-slot material texture table once an app needs
  larger texture/storage-buffer sets.
- Pipeline cache and shader reload.
- Instanced mesh buckets for repeated cube/sphere visualization.
- A render-target object-id picking path for dense scenes where CPU
  sphere/AABB candidates are not enough.
- More debug primitives: box, basis triad, text labels.
- glTF mesh loading, likely from the SPH viewer once this base runtime is stable.
