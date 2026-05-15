#pragma once

#include "ds_vk/types.hpp"

#include <vector>

namespace ds_vk
{
struct Vertex
{
    Vec3 position{};
    Vec3 normal{0.0f, 0.0f, 1.0f};
    Vec4 color{1.0f};
};

struct MeshData
{
    std::vector<Vertex> vertices{};
    std::vector<u32> indices{};
};

struct Bounds
{
    Vec3 min{};
    Vec3 max{};
};

struct Transform
{
    Vec3 translation{};
    Quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    Vec3 scale{1.0f};

    [[nodiscard]] auto matrix() const noexcept -> Mat4;
};

[[nodiscard]] auto make_quad(f32 side_length = 1.0f, Vec4 color = Vec4{1.0f}) -> MeshData;
[[nodiscard]] auto make_cube(f32 side_length = 1.0f, Vec4 color = Vec4{1.0f}) -> MeshData;
[[nodiscard]] auto
make_uv_sphere(f32 radius = 1.0f, u32 slices = 32, u32 stacks = 16, Vec4 color = Vec4{1.0f})
    -> MeshData;
[[nodiscard]] auto bounds_of(const MeshData& mesh) -> Bounds;
[[nodiscard]] auto triangle_count(const MeshData& mesh) noexcept -> usize;
[[nodiscard]] auto has_valid_indices(const MeshData& mesh) noexcept -> bool;
}  // namespace ds_vk
