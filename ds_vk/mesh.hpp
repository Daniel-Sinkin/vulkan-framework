#pragma once

#include "ds_vk/geometry.hpp"
#include "ds_vk/types.hpp"

#include <vector>

namespace ds_vk
{
struct Vertex
{
    Vec3 position{};
    Vec3 normal{k_axis_z};
    Color color{};
    Vec2 texcoord{};
};

struct MeshData
{
    std::vector<Vertex> vertices{};
    std::vector<u32> indices{};
};

struct Transform
{
    Vec3 translation{};
    Quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    Vec3 scale{1.0f};

    [[nodiscard]] auto matrix() const noexcept -> Mat4;
};

struct UvSphereConfig
{
    f32 radius{1.0f};
    u32 slices{32u};
    u32 stacks{16u};
    Color color{Color::white};
};

// clang-format off
[[nodiscard]] auto make_quad(f32 side_length = 1.0f, Color color = Color::white)                                      -> MeshData;
[[nodiscard]] auto make_cube(f32 side_length = 1.0f, Color color = Color::white)                                      -> MeshData;
[[nodiscard]] auto make_uv_sphere(const UvSphereConfig& config = UvSphereConfig{})                                    -> MeshData;
[[nodiscard]] auto aabb_of(const MeshData& mesh)                                                                      -> Aabb;
[[nodiscard]] auto triangle_count(const MeshData& mesh) noexcept                                                      -> usize;
[[nodiscard]] auto has_valid_indices(const MeshData& mesh) noexcept                                                   -> bool;
// clang-format on
}  // namespace ds_vk
