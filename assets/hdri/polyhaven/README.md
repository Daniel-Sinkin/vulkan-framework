# Poly Haven HDRI Assets

Vendored on 2026-05-16 for the `ds_vk_basic_app` environment lighting smoke test.

All assets in this directory are from Poly Haven and marked CC0 on their asset
pages.

- `studio_small_01_1k.hdr`: 1K HDR environment from
  <https://polyhaven.com/a/studio_small_01>
- `qwantani_puresky_1k.hdr`: local-only 1K HDR sky environment from
  <https://polyhaven.com/a/qwantani_puresky>, used by the DFSPH viewer. This
  file is ignored by git; download it from Poly Haven when recreating local
  assets.

The runtime consumes these as equirectangular HDR textures for both visible
background rendering and approximate shader-side image based lighting.
