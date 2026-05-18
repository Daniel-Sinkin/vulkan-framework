# Realtime SPH

GPU-side realtime SPH/PBF fluid demo for the Vulkan framework.

## Run

```bash
./build/ds_vk_realtime_sph_app
```

Useful validation options:

```bash
./build/ds_vk_realtime_sph_app --smoke-frames 300 --fixed-frame-dt 0.0166667 --hide-ui --print-stats --fail-on-invalid --max-z-span 1.4
./build/ds_vk_realtime_sph_app --particles 50000 --smoke-frames 300 --fixed-frame-dt 0.0166667 --hide-ui --print-stats --fail-on-invalid --max-z-span 2.4
./build/ds_vk_realtime_sph_app --scenario no-gravity-cube --particles 4096 --smoke-frames 300 --fixed-frame-dt 0.0166667 --hide-ui --print-stats --fail-on-invalid --max-center-delta 0.01 --max-speed 1.0
./build/ds_vk_realtime_sph_app --scenario opposing-cubes --particles 8192 --smoke-frames 300 --fixed-frame-dt 0.0166667 --hide-ui --print-stats --fail-on-invalid --max-speed 3.0
```

Or run the full local suite:

```bash
app/realtime-sph/run_validation.sh ./build/ds_vk_realtime_sph_app
```

## Solver

- Position Based Fluids style SPH density constraints.
- Dense uniform grid with fixed per-cell particle slots.
- Separate correction and apply compute passes to avoid same-dispatch position read/write races.
- Particle rendering uses GPU sphere impostors from the same particle SSBO; no CPU particle readback is needed for rendering.
- The dam-break domain is smaller for the default 20k scene and wider for 50k so the settled fluid retains visible depth.
- Rigid bodies are deliberately disabled until the fluid-only solver is visually stable.

## Validation Targets

The local validation script currently covers:

- 20k dam-break scene.
- 50k dam-break scene.
- 4096-particle no-gravity cube invariant.
- 8192-particle opposing-cubes collision invariant.
