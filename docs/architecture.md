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

1. SDL polls input and ImGui consumes UI events.
2. Camera controls update the shared `Camera`.
3. The app receives `update(FrameContext&, dt)` if that method exists.
4. The app appends mesh/debug commands to `FrameContext::draw`.
5. Built-in camera UI and the app's optional `draw_ui(FrameContext&)` build
   ImGui draw data.
6. Built-in pipelines render meshes, debug segments, and ImGui.
7. Optional screenshot copy reads the swapchain into a VMA buffer and writes PNG.

## Current Shader Interface

- Mesh draws use a 128-byte vertex-stage push block: model-view-projection,
  packed normal matrix, and per-draw color.
- The first mesh fragment shader uses fixed light/view vectors with a small
  Blinn-Phong specular term.
- The first mesh pipeline intentionally has no descriptors. That keeps the first
  app fast to iterate and leaves descriptor-indexing work for the material/resource
  table pass.
- Debug segments use a separate 96-byte push block and one per-frame mapped
  segment buffer.

## Repo Layout

- `ds_vk/` is the framework/library: public headers, implementation files, and
  built-in shaders live together there.
- `app/` contains full app users of the framework.
- `tests/` contains executable test harnesses.
- `external/` contains vendored dependencies.
- `docs/` and `scripts/` support research and validation.

## App Surface

```cpp
class MyApp final
{
  public:
    auto setup(ds_vk::Runtime& runtime) -> void;
    auto update(ds_vk::FrameContext& frame, float dt_seconds) -> void;
    auto draw_ui(ds_vk::FrameContext& frame) -> void;
};
```

`Runtime::run(app)` uses compile-time detection for these hook names instead of
an inherited interface. `setup`, `draw_ui`, and `shutdown` are optional; a typo
that leaves the app with no recognized hook fails to compile.

The app can stay high-level for simple work:

```cpp
frame.draw.draw_mesh(mesh, transform, color);
frame.draw.debug_arrow(origin, direction, color);
frame.draw.debug_sphere(center, radius, color);
```

It can also drop to Vulkan directly:

```cpp
vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdDispatch(frame.command_buffer, groups_x, groups_y, groups_z);
```

Mesh upload/replacement requires an initialized runtime, so app code should do it
from `setup`, `update`, or UI callbacks during `run`, not before calling
`Runtime::run`.

## Near-Term Roadmap

- Descriptor indexing support path: feature-gated query/enablement exists now;
  add a resource table for textures/storage buffers once a real app needs it.
- Pipeline cache and shader reload.
- Instanced mesh buckets for repeated cube/sphere visualization.
- More debug primitives: box, basis triad, text labels.
- glTF mesh loading, likely from the SPH viewer once this base runtime is stable.
