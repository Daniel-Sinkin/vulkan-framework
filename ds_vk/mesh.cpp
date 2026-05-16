#include "ds_vk/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <numbers>

namespace ds_vk
{
namespace
{
auto push_face(MeshData& mesh, Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 normal, Color color) -> void
{
    const auto base = static_cast<u32>(mesh.vertices.size());
    mesh.vertices.push_back(
        Vertex{.position = a, .normal = normal, .color = color, .texcoord = {0.0f, 0.0f}}
    );
    mesh.vertices.push_back(
        Vertex{.position = b, .normal = normal, .color = color, .texcoord = {1.0f, 0.0f}}
    );
    mesh.vertices.push_back(
        Vertex{.position = c, .normal = normal, .color = color, .texcoord = {1.0f, 1.0f}}
    );
    mesh.vertices.push_back(
        Vertex{.position = d, .normal = normal, .color = color, .texcoord = {0.0f, 1.0f}}
    );
    mesh.indices.insert(
        mesh.indices.end(), {base + 0u, base + 1u, base + 2u, base + 0u, base + 2u, base + 3u}
    );
}
}  // namespace

auto Transform::matrix() const noexcept -> Mat4
{
    const auto translation_matrix = glm::translate(Mat4{1.0f}, translation);
    const auto rotation_matrix = glm::mat4_cast(rotation);
    const auto scale_matrix = glm::scale(Mat4{1.0f}, scale);
    return translation_matrix * rotation_matrix * scale_matrix;
}

auto make_quad(f32 side_length, Color color) -> MeshData
{
    const auto half = 0.5f * std::max(0.0f, side_length);
    MeshData mesh{};
    mesh.vertices = {
        Vertex{
            .position = {-half, -half, 0.0f},
            .normal = k_axis_z,
            .color = color,
            .texcoord = {0.0f, 0.0f},
        },
        Vertex{
            .position = {half, -half, 0.0f},
            .normal = k_axis_z,
            .color = color,
            .texcoord = {1.0f, 0.0f},
        },
        Vertex{
            .position = {half, half, 0.0f},
            .normal = k_axis_z,
            .color = color,
            .texcoord = {1.0f, 1.0f},
        },
        Vertex{
            .position = {-half, half, 0.0f},
            .normal = k_axis_z,
            .color = color,
            .texcoord = {0.0f, 1.0f},
        },
    };
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

auto make_cube(f32 side_length, Color color) -> MeshData
{
    const auto half = 0.5f * std::max(0.0f, side_length);
    MeshData mesh{};
    mesh.vertices.reserve(24zu);
    mesh.indices.reserve(36zu);

    push_face(
        mesh,
        {-half, -half, half},
        {half, -half, half},
        {half, half, half},
        {-half, half, half},
        k_axis_z,
        color
    );
    push_face(
        mesh,
        {half, -half, -half},
        {-half, -half, -half},
        {-half, half, -half},
        {half, half, -half},
        -k_axis_z,
        color
    );
    push_face(
        mesh,
        {half, -half, half},
        {half, -half, -half},
        {half, half, -half},
        {half, half, half},
        k_axis_x,
        color
    );
    push_face(
        mesh,
        {-half, -half, -half},
        {-half, -half, half},
        {-half, half, half},
        {-half, half, -half},
        -k_axis_x,
        color
    );
    push_face(
        mesh,
        {-half, half, half},
        {half, half, half},
        {half, half, -half},
        {-half, half, -half},
        k_axis_y,
        color
    );
    push_face(
        mesh,
        {-half, -half, -half},
        {half, -half, -half},
        {half, -half, half},
        {-half, -half, half},
        -k_axis_y,
        color
    );

    return mesh;
}

auto make_uv_sphere(const UvSphereConfig& config) -> MeshData
{
    const auto slices = std::max(3u, config.slices);
    const auto stacks = std::max(2u, config.stacks);
    const auto safe_radius = std::max(0.0f, config.radius);
    MeshData mesh{};

    const auto n_vertices = static_cast<usize>(slices + 1u) * static_cast<usize>(stacks + 1u);
    const auto n_indices = static_cast<usize>(slices) * static_cast<usize>(stacks) * 6zu;

    mesh.vertices.reserve(n_vertices);
    mesh.indices.reserve(n_indices);

    for (u32 stack = 0; stack <= stacks; ++stack)
    {
        const auto v = static_cast<f32>(stack) / static_cast<f32>(stacks);
        const auto phi = std::numbers::pi_v<f32> * v;
        const auto sin_phi = std::sin(phi);
        const auto cos_phi = std::cos(phi);
        for (u32 slice = 0; slice <= slices; ++slice)
        {
            const auto u = static_cast<f32>(slice) / static_cast<f32>(slices);
            const auto theta = 2.0f * std::numbers::pi_v<f32> * u;
            const Vec3 normal{
                sin_phi * std::cos(theta),
                sin_phi * std::sin(theta),
                cos_phi,
            };
            mesh.vertices.push_back(
                Vertex{
                    .position = normal * safe_radius,
                    .normal = normal,
                    .color = config.color,
                    .texcoord = {u, v},
                }
            );
        }
    }

    for (u32 stack = 0; stack < stacks; ++stack)
    {
        for (u32 slice = 0; slice < slices; ++slice)
        {
            const auto row0 = stack * (slices + 1u);
            const auto row1 = (stack + 1u) * (slices + 1u);
            const auto a = row0 + slice;
            const auto b = row0 + slice + 1u;
            const auto c = row1 + slice;
            const auto d = row1 + slice + 1u;
            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
        }
    }

    return mesh;
}

auto aabb_of(const MeshData& mesh) -> Aabb
{
    if (mesh.vertices.empty())
    {
        return {};
    }

    Vec3 min_value{std::numeric_limits<f32>::max()};
    Vec3 max_value{std::numeric_limits<f32>::lowest()};
    for (const auto& vertex : mesh.vertices)
    {
        min_value = glm::min(min_value, vertex.position);
        max_value = glm::max(max_value, vertex.position);
    }
    return Aabb{.min = min_value, .max = max_value};
}

auto aabb_of(const PositionNormalMeshData& mesh) -> Aabb
{
    if (mesh.vertices.empty())
    {
        return {};
    }

    Vec3 min_value{std::numeric_limits<f32>::max()};
    Vec3 max_value{std::numeric_limits<f32>::lowest()};
    for (const auto& vertex : mesh.vertices)
    {
        min_value = glm::min(min_value, vertex.position);
        max_value = glm::max(max_value, vertex.position);
    }
    return Aabb{.min = min_value, .max = max_value};
}

auto aabb_of(const QuantizedPositionNormalMeshData& mesh) -> Aabb
{
    if (mesh.vertices.empty())
    {
        return {};
    }
    return Aabb{.min = mesh.decode_origin, .max = mesh.decode_origin + mesh.decode_extent};
}

auto triangle_count(const MeshData& mesh) noexcept -> usize
{
    return mesh.indices.size() / 3zu;
}

auto triangle_count(const PositionNormalMeshData& mesh) noexcept -> usize
{
    return mesh.indices.size() / 3zu;
}

auto triangle_count(const QuantizedPositionNormalMeshData& mesh) noexcept -> usize
{
    return mesh.indices.size() / 3zu;
}

auto has_valid_indices(const MeshData& mesh) noexcept -> bool
{
    return std::ranges::all_of(
        mesh.indices,
        [&](u32 index) -> bool { return static_cast<usize>(index) < mesh.vertices.size(); }
    );
}

auto has_valid_indices(const PositionNormalMeshData& mesh) noexcept -> bool
{
    return std::ranges::all_of(
        mesh.indices,
        [&](u32 index) -> bool { return static_cast<usize>(index) < mesh.vertices.size(); }
    );
}

auto has_valid_indices(const QuantizedPositionNormalMeshData& mesh) noexcept -> bool
{
    return std::ranges::all_of(
        mesh.indices,
        [&](u32 index) -> bool { return static_cast<usize>(index) < mesh.vertices.size(); }
    );
}
}  // namespace ds_vk
