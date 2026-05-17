# DFSPH VTK assets

These directories are copied particle VTK exports from:

`/Users/danielsinkin/GitHub_private/SPH-Seminar/dfsph_viewer/run/showcase/_splishsplash_runs`

The `dambreak_small_iisph_v1` VTK export comes from the older exact IISPH run:

`/Users/danielsinkin/GitHub_private/SPH-Seminar/dfsph_viewer/run/_splishsplash_runs/dambreak_small_iisph_v1`

They intentionally do not vendor SPlisHSPlasH. The `ds_vk_dfsph_app` reads the exported VTK
histories directly for playback and scene switching.

Only the scene exports referenced by `app/dfsph_main.cpp` are mirrored here:

- `dambreak_small_iisph_v1`
- `dambreak_20k_600f_v1`
- `dambreak_50k_600f_dfsph_v2`
- `dambreak_150k_300f_dfsph_v1`
- `twoway_rigidbody_50k_4bodies_dfsph_v1`
- `viscous_bunny_dfsph_bender2017_nu_000_v2`
- `viscous_bunny_dfsph_bender2017_nu_025_v2`
- `viscous_bunny_dfsph_bender2017_nu_050_v2`

For each scene, copy the `vtk/` particle and rigid-body exports plus the matching
`surface/` mesh cache when one exists. Other showcase runs are intentionally
omitted. The small IISPH scene reuses the `dambreak_small_v1/surface` cache from
the proper viewer because both small runs share the same `dambreak_small` scene
setup.

The large stress-test dambreak scene is the source `dambreak_150k_300f_dfsph_v1`
showcase export. It has 148,877 particles and is the closest available source
scene to the earlier "130k" discussion.
