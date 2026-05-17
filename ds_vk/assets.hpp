#pragma once

#include "ds_vk/mesh.hpp"
#include "ds_vk/types.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace ds_vk
{
struct GltfMeshLoadConfig
{
    Transform transform{};
    Color color{Color::white};
    bool generate_normals_if_missing{true};
};

struct GltfImageData
{
    std::string name{};
    u32 width{};
    u32 height{};
    std::vector<ColorU8> pixels{};
};

struct GltfMaterialData
{
    std::string name{};
    Color base_color{Color::white};
    usize base_color_image{k_invalid_index};
    f32 metallic{};
    f32 roughness{1.0f};
    bool double_sided{true};
};

struct GltfMeshData
{
    std::string name{};
    MeshData mesh{};
    usize material{k_invalid_index};
    Transform transform{};
    bool visible_in_default_scene{true};
};

struct GltfWriteConfig
{
    std::string generator{"ds_vk"};
    bool pretty{true};
};

[[nodiscard]] auto load_gltf_mesh(const std::filesystem::path&, const GltfMeshLoadConfig& = {})
    -> MeshData;
auto write_gltf_scene(
    const std::filesystem::path&,
    std::span<const GltfMeshData>,
    std::span<const GltfMaterialData> = {},
    std::span<const GltfImageData> = {},
    const GltfWriteConfig& = {}
) -> void;
}  // namespace ds_vk
