#pragma once

#include "ds_vk/geometry.hpp"
#include "ds_vk/types.hpp"

#include <array>
#include <cstddef>
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

struct PositionNormalVertex
{
    Vec3 position{};
    Vec3 normal{k_axis_z};
};

static_assert(sizeof(PositionNormalVertex) == 24zu);
static_assert(offsetof(PositionNormalVertex, normal) == 12zu);

struct QuantizedPositionNormalVertex
{
    std::array<u16, 4> position{};
    std::array<u8, 2> normal_oct{};
    std::array<u8, 2> reserved{};
};

static_assert(sizeof(QuantizedPositionNormalVertex) == 12zu);
static_assert(offsetof(QuantizedPositionNormalVertex, normal_oct) == 8zu);
static_assert(offsetof(QuantizedPositionNormalVertex, reserved) == 10zu);

struct MeshData
{
    std::vector<Vertex> vertices{};
    std::vector<u32> indices{};
};

struct PositionNormalMeshData
{
    std::vector<PositionNormalVertex> vertices{};
    std::vector<u32> indices{};
};

struct QuantizedPositionNormalMeshData
{
    Vec3 decode_origin{};
    Vec3 decode_extent{1.0f};
    std::vector<QuantizedPositionNormalVertex> vertices{};
    std::vector<u32> indices{};
};

enum class MeshVertexFormat : u8
{
    standard = 0,
    position_normal = 1,
    quantized_position_normal = 2,
};

struct Transform
{
    Vec3 translation{};
    Quat rotation{k_quat_identity};
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
[[nodiscard]] auto make_uv_sphere(const UvSphereConfig& = {})                                                        -> MeshData;
[[nodiscard]] auto aabb_of(const MeshData&)                                                                          -> Aabb;
[[nodiscard]] auto aabb_of(const PositionNormalMeshData&)                                                            -> Aabb;
[[nodiscard]] auto aabb_of(const QuantizedPositionNormalMeshData&)                                                   -> Aabb;
[[nodiscard]] auto triangle_count(const MeshData&) noexcept                                                          -> usize;
[[nodiscard]] auto triangle_count(const PositionNormalMeshData&) noexcept                                            -> usize;
[[nodiscard]] auto triangle_count(const QuantizedPositionNormalMeshData&) noexcept                                   -> usize;
[[nodiscard]] auto has_valid_indices(const MeshData&) noexcept                                                       -> bool;
[[nodiscard]] auto has_valid_indices(const PositionNormalMeshData&) noexcept                                         -> bool;
[[nodiscard]] auto has_valid_indices(const QuantizedPositionNormalMeshData&) noexcept                                -> bool;
// clang-format on
}  // namespace ds_vk
