#pragma once

#include "ds_vk/mesh.hpp"
#include "ds_vk/types.hpp"

#include <filesystem>

namespace ds_vk
{
struct GltfMeshLoadConfig
{
    Transform transform{};
    Color color{Color::white};
    bool generate_normals_if_missing{true};
};

[[nodiscard]] auto load_gltf_mesh(const std::filesystem::path&, const GltfMeshLoadConfig& = {})
    -> MeshData;
}  // namespace ds_vk
