#pragma once

#include <filesystem>

#ifndef DS_VK_QUAKE_ASSET_DIR
#    define DS_VK_QUAKE_ASSET_DIR "app/quake/assets"
#endif

namespace ds_vk_quake::paths
{
inline const std::filesystem::path asset_root{DS_VK_QUAKE_ASSET_DIR};
inline auto quake_root = asset_root.parent_path();
inline auto progs_dir = asset_root / "progs";
inline auto maps_dir = asset_root / "maps";
inline auto sounds_dir = asset_root / "sound";
inline auto music_dir = asset_root / "music";
inline auto texel_index_output_dir = quake_root / "outputs" / "texel_indices";
}  // namespace ds_vk_quake::paths
