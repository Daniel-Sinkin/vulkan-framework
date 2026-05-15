#include "ds_vk/camera.hpp"
#include "ds_vk/mesh.hpp"

#include <cmath>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <string_view>

namespace
{
auto g_failures = 0;

auto check(const bool condition, const std::string_view message) -> void
{
    if (!condition)
    {
        ++g_failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

auto near(const ds_vk::f32 a, const ds_vk::f32 b, const ds_vk::f32 eps = 1.0e-5f) -> bool
{
    return std::abs(a - b) <= eps;
}

auto finite_vec3(const ds_vk::Vec3 value) -> bool
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

auto test_quad_mesh() -> void
{
    const auto mesh = ds_vk::make_quad(2.0f);
    check(mesh.vertices.size() == 4u, "quad has four vertices");
    check(mesh.indices.size() == 6u, "quad has two triangles");
    check(ds_vk::has_valid_indices(mesh), "quad indices are valid");
    check(ds_vk::triangle_count(mesh) == 2u, "quad triangle count");
    const auto bounds = ds_vk::bounds_of(mesh);
    check(near(bounds.min.x, -1.0f) && near(bounds.max.x, 1.0f), "quad x bounds");
    check(near(bounds.min.y, -1.0f) && near(bounds.max.y, 1.0f), "quad y bounds");
    check(near(bounds.min.z, 0.0f) && near(bounds.max.z, 0.0f), "quad lies on z=0");
}

auto test_cube_mesh() -> void
{
    const auto mesh = ds_vk::make_cube(2.0f);
    check(mesh.vertices.size() == 24u, "cube has per-face vertices");
    check(mesh.indices.size() == 36u, "cube has 12 triangles");
    check(ds_vk::has_valid_indices(mesh), "cube indices are valid");
    for (const auto& vertex : mesh.vertices)
    {
        check(near(glm::length(vertex.normal), 1.0f), "cube normals are unit length");
    }
}

auto test_sphere_mesh() -> void
{
    constexpr auto slices = 12u;
    constexpr auto stacks = 6u;
    const auto mesh = ds_vk::make_uv_sphere(2.0f, slices, stacks);
    const auto expected_vertices =
        static_cast<ds_vk::usize>(slices + 1u) * static_cast<ds_vk::usize>(stacks + 1u);
    const auto expected_indices =
        static_cast<ds_vk::usize>(slices) * static_cast<ds_vk::usize>(stacks) * 6u;
    check(mesh.vertices.size() == expected_vertices, "sphere vertex count");
    check(mesh.indices.size() == expected_indices, "sphere index count");
    check(ds_vk::has_valid_indices(mesh), "sphere indices are valid");
    for (const auto& vertex : mesh.vertices)
    {
        check(finite_vec3(vertex.position), "sphere positions are finite");
        check(finite_vec3(vertex.normal), "sphere normals are finite");
        check(near(glm::length(vertex.normal), 1.0f, 1.0e-4f), "sphere normals are unit length");
        check(near(glm::length(vertex.position), 2.0f, 1.0e-4f), "sphere radius is respected");
    }
}

auto test_camera_projection() -> void
{
    auto camera = ds_vk::Camera{};
    camera.distance = 3.0f;
    const auto position = camera.position();
    check(finite_vec3(position), "camera position is finite");
    check(near(glm::length(position - camera.pivot), 3.0f, 1.0e-4f), "camera distance");
    const auto view = camera.view_matrix();
    const auto projection = camera.projection_matrix(16.0f / 9.0f);
    check(std::isfinite(view[0][0]), "view matrix is finite");
    check(std::isfinite(projection[0][0]), "projection matrix is finite");
    check(projection[1][1] < 0.0f, "projection matrix uses Vulkan inverted Y");

    camera.pitch = glm::half_pi<ds_vk::f32>();
    check(finite_vec3(camera.right()), "camera right vector is finite at vertical pitch");
    check(finite_vec3(camera.up()), "camera up vector is finite at vertical pitch");
}
}  // namespace

auto main() -> int
{
    test_quad_mesh();
    test_cube_mesh();
    test_sphere_mesh();
    test_camera_projection();
    if (g_failures != 0)
    {
        std::cerr << g_failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "all ds_vk tests passed\n";
    return 0;
}
