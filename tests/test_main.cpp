#include "ds_vk/assets.hpp"
#include "ds_vk/camera.hpp"
#include "ds_vk/geometry.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/manipulator.hpp"
#include "ds_vk/plugins/picker.hpp"
#include "ds_vk/plugins/viz.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace
{
auto g_failures = 0;

#ifndef DS_VK_TEST_ASSET_DIR
#    define DS_VK_TEST_ASSET_DIR "assets"
#endif

auto check(bool condition, const std::string_view message) -> void
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
    return std::isfinite(value.x) and std::isfinite(value.y) and std::isfinite(value.z);
}

auto finite_vec2(const ds_vk::Vec2 value) -> bool
{
    return std::isfinite(value.x) and std::isfinite(value.y);
}

template <typename T>
auto append_pod(std::vector<ds_vk::u8>& bytes, T value) -> void
{
    const auto offset = bytes.size();
    bytes.resize(offset + sizeof(T));
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

auto append_bytes(std::vector<ds_vk::u8>& bytes, std::span<const ds_vk::u8> data) -> void
{
    bytes.insert(bytes.end(), data.begin(), data.end());
}

auto pad_to_glb_alignment(std::vector<ds_vk::u8>& bytes, ds_vk::u8 pad) -> void
{
    while ((bytes.size() % 4zu) != 0zu)
    {
        bytes.push_back(pad);
    }
}

[[nodiscard]] auto make_triangle_glb_with_unknown_chunk() -> std::vector<ds_vk::u8>
{
    constexpr ds_vk::u32 k_glb_magic{0x46546c67u};
    constexpr ds_vk::u32 k_glb_version_2{2u};
    constexpr ds_vk::u32 k_glb_json_chunk_type{0x4e4f534au};
    constexpr ds_vk::u32 k_glb_binary_chunk_type{0x004e4942u};
    constexpr ds_vk::u32 k_unknown_chunk_type{0x54534554u};
    constexpr ds_vk::usize k_binary_byte_length{102zu};

    std::vector<ds_vk::u8> bin{};
    bin.reserve(k_binary_byte_length);
    const auto append_vec3 = [&](ds_vk::Vec3 value) -> void
    {
        append_pod(bin, value.x);
        append_pod(bin, value.y);
        append_pod(bin, value.z);
    };
    const auto append_vec2 = [&](ds_vk::Vec2 value) -> void
    {
        append_pod(bin, value.x);
        append_pod(bin, value.y);
    };
    append_vec3({0.0f, 0.0f, 0.0f});
    append_vec3({1.0f, 0.0f, 0.0f});
    append_vec3({0.0f, 1.0f, 0.0f});
    append_vec3(ds_vk::k_axis_z);
    append_vec3(ds_vk::k_axis_z);
    append_vec3(ds_vk::k_axis_z);
    append_vec2({0.0f, 0.0f});
    append_vec2({1.0f, 0.0f});
    append_vec2({0.0f, 1.0f});
    append_pod(bin, ds_vk::u16{0u});
    append_pod(bin, ds_vk::u16{1u});
    append_pod(bin, ds_vk::u16{2u});
    check(bin.size() == k_binary_byte_length, "glb fixture binary layout");
    pad_to_glb_alignment(bin, ds_vk::u8{0u});

    std::vector<ds_vk::u8> json_chunk{};
    constexpr std::string_view json_text{
        R"({"asset":{"version":"2.0"},"buffers":[{"byteLength":102}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":36},{"buffer":0,"byteOffset":72,"byteLength":24},{"buffer":0,"byteOffset":96,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"mode":4}]}]})"
    };
    json_chunk.assign(json_text.begin(), json_text.end());
    pad_to_glb_alignment(json_chunk, ds_vk::u8{' '});

    std::vector<ds_vk::u8> unknown_chunk{7u, 5u, 3u, 1u};
    pad_to_glb_alignment(unknown_chunk, ds_vk::u8{0u});

    constexpr ds_vk::usize k_glb_header_bytes{12zu};
    constexpr ds_vk::usize k_glb_chunk_header_bytes{8zu};
    const auto total_length = k_glb_header_bytes + 3zu * k_glb_chunk_header_bytes
                              + json_chunk.size() + bin.size() + unknown_chunk.size();

    std::vector<ds_vk::u8> glb{};
    append_pod(glb, k_glb_magic);
    append_pod(glb, k_glb_version_2);
    append_pod(glb, static_cast<ds_vk::u32>(total_length));
    append_pod(glb, static_cast<ds_vk::u32>(json_chunk.size()));
    append_pod(glb, k_glb_json_chunk_type);
    append_bytes(glb, json_chunk);
    append_pod(glb, static_cast<ds_vk::u32>(bin.size()));
    append_pod(glb, k_glb_binary_chunk_type);
    append_bytes(glb, bin);
    append_pod(glb, static_cast<ds_vk::u32>(unknown_chunk.size()));
    append_pod(glb, k_unknown_chunk_type);
    append_bytes(glb, unknown_chunk);
    return glb;
}

auto replace_first_glb_chunk_type(std::vector<ds_vk::u8>& glb, ds_vk::u32 chunk_type) -> void
{
    constexpr auto k_glb_first_chunk_type_offset = 16zu;
    std::memcpy(glb.data() + k_glb_first_chunk_type_offset, &chunk_type, sizeof(chunk_type));
}

auto set_glb_container_length(std::vector<ds_vk::u8>& glb, ds_vk::u32 byte_count) -> void
{
    constexpr auto k_glb_length_offset = 8zu;
    std::memcpy(glb.data() + k_glb_length_offset, &byte_count, sizeof(byte_count));
}

struct FakeDrawSink
{
    ds_vk::usize line_count{};
    ds_vk::usize arrow_count{};
    ds_vk::Color first_line_color{};
    ds_vk::Color last_line_color{};
    ds_vk::Vec3 last_arrow_origin{};
    ds_vk::Vec3 last_arrow_vector{};
    ds_vk::Color last_arrow_color;

    auto debug_line(const auto& config) noexcept -> void
    {
        if (line_count == 0zu)
        {
            first_line_color = config.color;
        }
        last_line_color = config.color;
        ++line_count;
    }

    auto debug_arrow(const auto& config) noexcept -> void
    {
        ++arrow_count;
        last_arrow_origin = config.origin;
        last_arrow_vector = config.vector;
        last_arrow_color = config.color;
    }
};

template <typename T>
concept Addable = requires(T a, T b) { a + b; };

static_assert(!Addable<ds_vk::Color>);
static_assert(sizeof(ds_vk::ProjectionMode) == sizeof(ds_vk::u8));
static_assert(sizeof(ds_vk::PickerShapeType) == sizeof(ds_vk::u8));

auto test_color_types() -> void
{
    const ds_vk::Color low{0.0f, 0.25f, 0.5f, 1.0f};
    const ds_vk::Color high{1.0f, 0.75f, 0.0f, 0.5f};
    const auto mixed = ds_vk::mix_color(low, high, 0.25f);
    check(near(mixed.r(), 0.25f), "color mix red channel");
    check(near(mixed.g(), 0.375f), "color mix green channel");
    check(near(mixed.b(), 0.375f), "color mix blue channel");
    check(near(mixed.a(), 0.875f), "color mix alpha channel");

    const auto u8 = ds_vk::to_color_u8(ds_vk::Color{1.0f, 0.5f, -1.0f, 2.0f});
    check(u8.r() == 255u, "u8 color clamps high channel");
    check(u8.g() == 128u, "u8 color rounds half channel");
    check(u8.b() == 0u, "u8 color clamps low channel");
    check(u8.a() == 255u, "u8 color clamps alpha channel");
    const auto f32 = ds_vk::to_color(ds_vk::ColorU8{255u, 128u, 0u, 64u});
    check(near(f32.r(), 1.0f), "u8 color converts red to float");
    check(near(f32.g(), 128.0f / 255.0f), "u8 color converts green to float");
    check(near(f32.a(), 64.0f / 255.0f), "u8 color converts alpha to float");
}

auto test_quad_mesh() -> void
{
    const auto mesh = ds_vk::make_quad(2.0f);
    check(mesh.vertices.size() == 4u, "quad has four vertices");
    check(mesh.indices.size() == 6u, "quad has two triangles");
    check(ds_vk::has_valid_indices(mesh), "quad indices are valid");
    check(ds_vk::triangle_count(mesh) == 2u, "quad triangle count");
    check(near(mesh.vertices[0].texcoord.x, 0.0f), "quad has uvs");
    check(near(mesh.vertices[2].texcoord.y, 1.0f), "quad uv max y");
    const auto aabb = ds_vk::aabb_of(mesh);
    check(near(aabb.min.x, -1.0f) and near(aabb.max.x, 1.0f), "quad x bounds");
    check(near(aabb.min.y, -1.0f) and near(aabb.max.y, 1.0f), "quad y bounds");
    check(near(aabb.min.z, 0.0f) and near(aabb.max.z, 0.0f), "quad lies on z=0");
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
        check(finite_vec2(vertex.texcoord), "cube uvs are finite");
        check(vertex.texcoord.x >= 0.0f and vertex.texcoord.x <= 1.0f, "cube uv x is normalized");
        check(vertex.texcoord.y >= 0.0f and vertex.texcoord.y <= 1.0f, "cube uv y is normalized");
    }
}

auto test_quantized_position_normal_mesh() -> void
{
    ds_vk::QuantizedPositionNormalMeshData mesh{
        .decode_origin = {1.0f, 2.0f, 3.0f},
        .decode_extent = {4.0f, 5.0f, 6.0f},
        .vertices =
            {
                {.position = {0u, 0u, 0u, 0u}, .normal_oct = {128u, 128u}, .reserved = {}},
                {.position = {65535u, 0u, 0u, 0u}, .normal_oct = {128u, 128u}, .reserved = {}},
                {.position = {0u, 65535u, 0u, 0u}, .normal_oct = {128u, 128u}, .reserved = {}},
            },
        .indices = {0u, 1u, 2u},
    };
    check(ds_vk::has_valid_indices(mesh), "quantized mesh indices are valid");
    check(ds_vk::triangle_count(mesh) == 1zu, "quantized mesh triangle count");
    const auto aabb = ds_vk::aabb_of(mesh);
    check(near(aabb.min.x, 1.0f) and near(aabb.max.x, 5.0f), "quantized mesh x bounds");
    check(near(aabb.min.y, 2.0f) and near(aabb.max.y, 7.0f), "quantized mesh y bounds");
    check(near(aabb.min.z, 3.0f) and near(aabb.max.z, 9.0f), "quantized mesh z bounds");

    mesh.indices.push_back(7u);
    check(!ds_vk::has_valid_indices(mesh), "quantized mesh rejects invalid index");
}

auto test_sphere_mesh() -> void
{
    constexpr auto slices = 12u;
    constexpr auto stacks = 6u;
    const auto mesh = ds_vk::make_uv_sphere(
        ds_vk::UvSphereConfig{
            .radius = 2.0f,
            .slices = slices,
            .stacks = stacks,
        }
    );
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
        check(finite_vec2(vertex.texcoord), "sphere uvs are finite");
        check(vertex.texcoord.x >= 0.0f and vertex.texcoord.x <= 1.0f, "sphere uv x is normalized");
        check(vertex.texcoord.y >= 0.0f and vertex.texcoord.y <= 1.0f, "sphere uv y is normalized");
        check(near(glm::length(vertex.normal), 1.0f, 1.0e-4f), "sphere normals are unit length");
        check(near(glm::length(vertex.position), 2.0f, 1.0e-4f), "sphere radius is respected");
        check(
            glm::dot(glm::normalize(vertex.position), vertex.normal) > 0.999f,
            "sphere normals are radial"
        );
    }
    for (auto i = 0zu; i + 2u < mesh.indices.size(); i += 3u)
    {
        const auto a = mesh.vertices[mesh.indices[i + 0u]].position;
        const auto b = mesh.vertices[mesh.indices[i + 1u]].position;
        const auto c = mesh.vertices[mesh.indices[i + 2u]].position;
        const auto face_normal = glm::cross(b - a, c - a);
        if (glm::dot(face_normal, face_normal) > 1.0e-8f)
        {
            const auto center = (a + b + c) / 3.0f;
            check(
                glm::dot(glm::normalize(face_normal), glm::normalize(center)) > 0.95f,
                "sphere triangle winding is outward"
            );
        }
    }
}

auto test_gltf_assets() -> void
{
    const auto mesh = ds_vk::load_gltf_mesh(
        std::filesystem::path{DS_VK_TEST_ASSET_DIR} / "test/triangle.gltf",
        ds_vk::GltfMeshLoadConfig{.color = ds_vk::Color::cyan}
    );
    check(mesh.vertices.size() == 3zu, "gltf loader reads vertex count");
    check(mesh.indices.size() == 3zu, "gltf loader reads index count");
    check(ds_vk::has_valid_indices(mesh), "gltf loader reads valid indices");
    check(near(mesh.vertices[1].position.x, 1.0f), "gltf loader reads positions");
    check(near(mesh.vertices[2].texcoord.y, 1.0f), "gltf loader reads texcoords");
    check(near(glm::length(mesh.vertices[0].normal), 1.0f), "gltf loader reads normals");
    check(near(mesh.vertices[0].color.g(), 1.0f), "gltf loader applies mesh color");

    const auto generated_normal_mesh = ds_vk::load_gltf_mesh(
        std::filesystem::path{DS_VK_TEST_ASSET_DIR} / "test/triangle_no_normals.gltf"
    );
    check(generated_normal_mesh.vertices.size() == 3zu, "gltf loader reads missing-normal mesh");
    for (const auto& vertex : generated_normal_mesh.vertices)
    {
        check(near(glm::length(vertex.normal), 1.0f), "gltf loader generates unit normals");
        check(vertex.normal.z > 0.99f, "gltf loader generated normal points along +z");
    }

    const std::hash<std::string> path_hash{};
    const auto cwd_hash = path_hash(std::filesystem::current_path().string());
    const auto glb_path = std::filesystem::temp_directory_path()
                          / std::format("ds_vk_unknown_chunk_test_{}.glb", cwd_hash);
    {
        std::ofstream out{glb_path, std::ios::binary};
        const auto glb_bytes = make_triangle_glb_with_unknown_chunk();
        out.write(
            reinterpret_cast<const char*>(glb_bytes.data()),
            static_cast<std::streamsize>(glb_bytes.size())
        );
    }
    const auto glb_mesh = ds_vk::load_gltf_mesh(glb_path);
    check(glb_mesh.vertices.size() == 3zu, "glb loader reads vertex count");
    check(glb_mesh.indices.size() == 3zu, "glb loader reads index count");
    check(near(glb_mesh.vertices[2].texcoord.y, 1.0f), "glb loader ignores unknown chunks");

    {
        auto invalid_glb_bytes = make_triangle_glb_with_unknown_chunk();
        replace_first_glb_chunk_type(invalid_glb_bytes, 0x54534554u);
        std::ofstream out{glb_path, std::ios::binary};
        out.write(
            reinterpret_cast<const char*>(invalid_glb_bytes.data()),
            static_cast<std::streamsize>(invalid_glb_bytes.size())
        );
    }
    auto rejected_non_json_first_chunk = false;
    try
    {
        (void) ds_vk::load_gltf_mesh(glb_path);
    }
    catch (const std::runtime_error&)
    {
        rejected_non_json_first_chunk = true;
    }
    check(rejected_non_json_first_chunk, "glb loader requires JSON as first chunk");

    {
        auto invalid_glb_bytes = make_triangle_glb_with_unknown_chunk();
        constexpr ds_vk::u32 k_partial_chunk_header{0x12345678u};
        append_pod(invalid_glb_bytes, k_partial_chunk_header);
        set_glb_container_length(
            invalid_glb_bytes, static_cast<ds_vk::u32>(invalid_glb_bytes.size())
        );
        std::ofstream out{glb_path, std::ios::binary};
        out.write(
            reinterpret_cast<const char*>(invalid_glb_bytes.data()),
            static_cast<std::streamsize>(invalid_glb_bytes.size())
        );
    }
    auto rejected_truncated_chunk_header = false;
    try
    {
        (void) ds_vk::load_gltf_mesh(glb_path);
    }
    catch (const std::runtime_error&)
    {
        rejected_truncated_chunk_header = true;
    }
    check(rejected_truncated_chunk_header, "glb loader rejects truncated chunk header");
    std::filesystem::remove(glb_path);
}

auto test_camera_projection() -> void
{
    ds_vk::Camera camera{};
    camera.distance() = 3.0f;
    const auto position = camera.position();
    check(finite_vec3(position), "camera position is finite");
    check(near(glm::length(position - camera.pivot()), 3.0f, 1.0e-4f), "camera distance");
    const auto view = camera.view_matrix();
    const auto projection = camera.projection_matrix(16.0f / 9.0f);
    check(std::isfinite(view[0][0]), "view matrix is finite");
    check(std::isfinite(projection[0][0]), "projection matrix is finite");
    check(projection[1][1] < 0.0f, "projection matrix uses Vulkan inverted Y");

    camera.pitch() = glm::half_pi<ds_vk::f32>();
    check(finite_vec3(camera.right()), "camera right vector is finite at vertical pitch");
    check(finite_vec3(camera.up()), "camera up vector is finite at vertical pitch");
}

auto test_camera_config() -> void
{
    ds_vk::Camera camera{};
    auto& configured = camera.configure({
        .pivot = 0.7f * ds_vk::k_axis_z,
        .distance = 5.4f,
        .yaw = glm::radians(42.0f),
        .pitch = glm::radians(25.0f),
    });

    check(&configured == &camera, "camera config returns the configured camera");
    check(near(camera.pivot().z, 0.7f), "camera config pivot");
    check(near(camera.distance(), 5.4f), "camera config distance");
    check(near(camera.yaw(), glm::radians(42.0f)), "camera config yaw");
    check(near(camera.pitch(), glm::radians(25.0f)), "camera config pitch");
    check(near(camera.fov_y(), glm::radians(55.0f)), "camera config preserves defaults");
}

auto test_geometry_helpers() -> void
{
    const ds_vk::Ray ray{
        .origin = {0.0f, 0.0f, 0.0f},
        .direction = ds_vk::k_axis_x,
    };
    const auto sphere_hit = ds_vk::intersect_sphere(
        ray,
        ds_vk::Sphere{
            .center = {3.0f, 0.0f, 0.0f},
            .radius = 1.0f,
        }
    );
    check(sphere_hit.has_value() and near(*sphere_hit, 2.0f), "pick ray intersects sphere");

    const auto sphere_miss = ds_vk::intersect_sphere(
        ray,
        ds_vk::Sphere{
            .center = {0.0f, 3.0f, 0.0f},
            .radius = 0.5f,
        }
    );
    check(!sphere_miss.has_value(), "pick ray misses sphere");

    const auto aabb_hit = ds_vk::intersect_aabb(
        ray,
        ds_vk::Aabb{
            .min = {4.0f, -1.0f, -1.0f},
            .max = {5.0f, 1.0f, 1.0f},
        }
    );
    check(aabb_hit.has_value() and near(*aabb_hit, 4.0f), "pick ray intersects aabb");

    const auto reversed_aabb_hit = ds_vk::intersect_aabb(
        ray,
        ds_vk::Aabb{
            .min = {5.0f, 1.0f, 1.0f},
            .max = {4.0f, -1.0f, -1.0f},
        }
    );
    check(
        reversed_aabb_hit.has_value() and near(*reversed_aabb_hit, 4.0f),
        "pick ray normalizes reversed aabb bounds"
    );

    const auto obb_hit = ds_vk::hit_obb(
        ds_vk::Ray{.origin = {0.0f, 0.0f, 0.0f}, .direction = ds_vk::k_axis_y},
        ds_vk::Obb{.center = {0.0f, 3.0f, 0.0f}, .half_extent = {0.5f, 0.5f, 0.5f}}
    );
    check(obb_hit.has_value() and near(obb_hit->distance, 2.5f), "ray hits obb");

    const auto capsule_hit = ds_vk::hit_capsule(
        ds_vk::Ray{.origin = {-2.0f, 0.0f, 0.0f}, .direction = ds_vk::k_axis_x},
        ds_vk::Capsule{.a = -ds_vk::k_axis_z, .b = ds_vk::k_axis_z, .radius = 0.25f}
    );
    check(capsule_hit.has_value(), "ray hits capsule");

    ds_vk::Camera camera{};
    camera.configure({
        .pivot = {0.0f, 0.0f, 0.0f},
        .distance = 4.0f,
        .yaw = 0.0f,
        .pitch = 0.0f,
    });
    const auto center_ray = ds_vk::make_camera_ray(camera, {400.0f, 300.0f}, {800.0f, 600.0f});
    check(finite_vec3(center_ray.origin), "pick ray origin is finite");
    check(finite_vec3(center_ray.direction), "pick ray direction is finite");
    check(near(glm::length(center_ray.direction), 1.0f, 1.0e-4f), "pick ray direction is unit");
    const auto upper_ray = ds_vk::make_camera_ray(camera, {400.0f, 200.0f}, {800.0f, 600.0f});
    const auto lower_ray = ds_vk::make_camera_ray(camera, {400.0f, 400.0f}, {800.0f, 600.0f});
    const auto left_ray = ds_vk::make_camera_ray(camera, {300.0f, 300.0f}, {800.0f, 600.0f});
    const auto right_ray = ds_vk::make_camera_ray(camera, {500.0f, 300.0f}, {800.0f, 600.0f});
    check(upper_ray.direction.z > 0.0f, "pick ray maps upper screen pixels toward world up");
    check(lower_ray.direction.z < 0.0f, "pick ray maps lower screen pixels toward world down");
    check(left_ray.direction.y < 0.0f, "pick ray maps left screen pixels to camera left");
    check(right_ray.direction.y > 0.0f, "pick ray maps right screen pixels to camera right");
}

auto test_picker_plugin() -> void
{
    ds_vk::Picker picker{};
    const ds_vk::ObjectId sphere_id{.value = 11u};
    const ds_vk::ObjectId aabb_id{.value = 12u};
    (void) picker.add_sphere({
        .object_id = sphere_id,
        .sphere = {.center = {3.0f, 0.0f, 0.0f}, .radius = 1.0f},
    });
    (void) picker.add_aabb({
        .object_id = aabb_id,
        .aabb = {.min = {5.0f, -1.0f, -1.0f}, .max = {6.0f, 1.0f, 1.0f}},
    });
    const ds_vk::Ray ray{
        .origin = {0.0f, 0.0f, 0.0f},
        .direction = ds_vk::k_axis_x,
    };
    const auto first_hit = picker.raycast(ray);
    check(first_hit.has_value(), "picker raycast returns nearest hit");
    if (first_hit.has_value())
    {
        check(first_hit->object_id.value == sphere_id.value, "picker raycast picks sphere first");
        check(near(first_hit->distance, 2.0f), "picker raycast records hit distance");
    }

    picker.clear();
    (void) picker.add_sphere({
        .object_id = {.value = 20u},
        .layer = ds_vk::Layer{1u << 2u},
        .sphere = {.center = {2.0f, 0.0f, 0.0f}, .radius = 0.5f},
    });
    check(
        !picker.raycast({.ray = ray, .layer_mask = ds_vk::LayerMask{1u << 1u}}).has_value(),
        "picker raycast respects layer mask misses"
    );
    check(
        picker.raycast({.ray = ray, .layer_mask = ds_vk::LayerMask{1u << 2u}}).has_value(),
        "picker raycast respects layer mask hits"
    );

    picker.clear();
    (void) picker.add_obb({
        .object_id = {.value = 30u},
        .obb = {.center = {0.0f, 3.0f, 0.0f}, .half_extent = {0.5f, 0.5f, 0.5f}},
    });
    const auto obb_hit = picker.raycast({
        .origin = {0.0f, 0.0f, 0.0f},
        .direction = ds_vk::k_axis_y,
    });
    check(obb_hit.has_value() and obb_hit->object_id.value == 30u, "picker supports obb targets");

    picker.clear();
    (void) picker.add_capsule({
        .object_id = {.value = 40u},
        .capsule = {.a = -ds_vk::k_axis_z, .b = ds_vk::k_axis_z, .radius = 0.25f},
    });
    const auto capsule_hit = picker.raycast({
        .origin = {-2.0f, 0.0f, 0.0f},
        .direction = ds_vk::k_axis_x,
    });
    check(
        capsule_hit.has_value() and capsule_hit->object_id.value == 40u,
        "picker supports capsule targets"
    );

    ds_vk::Camera camera{};
    camera.configure({
        .pivot = {0.0f, 0.0f, 0.0f},
        .distance = 4.0f,
        .yaw = 0.0f,
        .pitch = 0.0f,
    });
    picker.clear();
    (void) picker.add_sphere({
        .object_id = {.value = 50u},
        .sphere = {.center = {0.0f, 0.0f, 0.0f}, .radius = 1.0f},
    });
    const auto click_hit = picker.click({
        .camera = camera,
        .mouse_px = {400.0f, 300.0f},
        .viewport_px = {800.0f, 600.0f},
    });
    check(
        click_hit.has_value() and click_hit->object_id.value == 50u, "picker click uses camera ray"
    );

    picker.clear();
    (void) picker.add_screen_segment({
        .object_id = {.value = 60u},
        .segment = {.start = {0.0f, -0.5f, 0.0f}, .end = {0.0f, 0.5f, 0.0f}},
        .radius_px = 12.0f,
    });
    const auto segment_hit = picker.click({
        .camera = camera,
        .mouse_px = {400.0f, 300.0f},
        .viewport_px = {800.0f, 600.0f},
    });
    check(
        segment_hit.has_value() and segment_hit->object_id.value == 60u,
        "picker click supports screen segment targets"
    );

    const auto view_projection = camera.view_projection_matrix(800.0f / 600.0f);
    auto clip = view_projection * ds_vk::Vec4{0.0f, 0.0f, 1.0f, 1.0f};
    clip /= clip.w;
    const auto above_center_px =
        ds_vk::Vec2{(clip.x * 0.5f + 0.5f) * 800.0f, (clip.y * 0.5f + 0.5f) * 600.0f};
    picker.clear();
    (void) picker.add_screen_segment({
        .object_id = {.value = 61u},
        .segment = {.start = {0.0f, -0.2f, 1.0f}, .end = {0.0f, 0.2f, 1.0f}},
        .radius_px = 8.0f,
    });
    const auto above_segment_hit = picker.click({
        .camera = camera,
        .mouse_px = above_center_px,
        .viewport_px = {800.0f, 600.0f},
    });
    check(
        above_segment_hit.has_value() and above_segment_hit->object_id.value == 61u,
        "picker screen segment projection uses Vulkan framebuffer y convention"
    );
}

auto test_manipulator_plugin() -> void
{
    ds_vk::Camera camera{};
    camera.configure({
        .pivot = {0.0f, 0.0f, 0.0f},
        .distance = 4.0f,
        .yaw = 0.0f,
        .pitch = 0.0f,
    });
    const ds_vk::ObjectId id{.value = 88u};
    std::array selected{id};
    ds_vk::Transform transform{};
    ds_vk::Manipulator manipulator{};
    const ds_vk::ManipulatorCallbacks callbacks{
        .get_transform = [&](ds_vk::ObjectId object_id) -> std::optional<ds_vk::Transform>
        {
            if (object_id.value != id.value)
            {
                return std::nullopt;
            }
            return transform;
        },
        .set_transform = [&](ds_vk::ObjectId object_id, const ds_vk::Transform& updated) -> void
        {
            if (object_id.value == id.value)
            {
                transform = updated;
            }
        },
    };

    manipulator.update({
        .input =
            ds_vk::ManipulatorInput{
                .camera = camera,
                .mouse_px = {400.0f, 300.0f},
                .viewport_px = {800.0f, 600.0f},
                .translate_pressed = true,
            },
        .selected_ids = std::span<const ds_vk::ObjectId>{selected},
        .callbacks = callbacks,
    });
    check(manipulator.active(), "manipulator starts with selected target");
    manipulator.update({
        .input =
            ds_vk::ManipulatorInput{
                .camera = camera,
                .mouse_px = {500.0f, 300.0f},
                .viewport_px = {800.0f, 600.0f},
            },
        .selected_ids = std::span<const ds_vk::ObjectId>{selected},
        .callbacks = callbacks,
    });
    check(transform.translation.y > 0.0f, "manipulator translates in camera plane");

    manipulator.update({
        .input =
            ds_vk::ManipulatorInput{
                .camera = camera,
                .mouse_px = {500.0f, 300.0f},
                .viewport_px = {800.0f, 600.0f},
                .cancel_pressed = true,
            },
        .selected_ids = std::span<const ds_vk::ObjectId>{selected},
        .callbacks = callbacks,
    });
    check(!manipulator.active(), "manipulator cancel exits active mode");
    check(near(glm::length(transform.translation), 0.0f), "manipulator cancel restores transform");

    manipulator.update({
        .input =
            ds_vk::ManipulatorInput{
                .camera = camera,
                .mouse_px = {400.0f, 300.0f},
                .viewport_px = {800.0f, 600.0f},
                .scale_pressed = true,
            },
        .selected_ids = std::span<const ds_vk::ObjectId>{selected},
        .callbacks = callbacks,
    });
    manipulator.update({
        .input =
            ds_vk::ManipulatorInput{
                .camera = camera,
                .mouse_px = {520.0f, 300.0f},
                .viewport_px = {800.0f, 600.0f},
                .confirm_pressed = true,
            },
        .selected_ids = std::span<const ds_vk::ObjectId>{selected},
        .callbacks = callbacks,
    });
    check(!manipulator.active(), "manipulator confirm exits active mode");
    check(transform.scale.x > 1.0f, "manipulator scales selected target");
}

auto test_viz_plugin() -> void
{
    const ds_vk::viz::ColorRamp ramp{
        ds_vk::viz::ColorRampConfig{
            .preset = ds_vk::viz::ColorPreset::blue_red,
            .range = {.min = 0.0f, .max = 10.0f},
        },
    };
    const auto low = ramp.sample(0.0f);
    const auto high = ramp.sample(20.0f);
    check(low.b() > low.r(), "blue-red ramp starts blue");
    check(high.r() > high.b(), "blue-red ramp clamps high values to red");
    check(near(ramp.normalized_value(5.0f), 0.5f), "color ramp normalizes scalar values");

    const std::array values{3.0f, -1.0f, 9.0f};
    const auto range = ds_vk::viz::range_from_values(values);
    check(near(range.min, -1.0f) and near(range.max, 9.0f), "viz range scans values");

    ds_vk::Camera camera{};
    camera.configure({
        .pivot = {0.0f, 0.0f, 0.0f},
        .distance = 4.0f,
    });

    FakeDrawSink draw{};
    const std::array positions{
        ds_vk::Vec3{0.0f, 0.0f, 0.0f},
        ds_vk::k_axis_x,
    };
    const std::array vectors{
        ds_vk::k_axis_y,
        2.0f * ds_vk::k_axis_y,
    };
    const auto arrows = ds_vk::viz::draw_vector_field(
        draw,
        ds_vk::viz::VectorFieldConfig{
            .positions = std::span<const ds_vk::Vec3>{positions},
            .vectors = std::span<const ds_vk::Vec3>{vectors},
            .scale = 0.5f,
            .color_by_magnitude = true,
            .color_ramp = ramp,
            .max_vectors = 1zu,
        }
    );
    check(arrows == 1u and draw.arrow_count == 1u, "viz vector field respects max_vectors");
    check(near(draw.last_arrow_vector.y, 0.5f), "viz vector field scales vectors");

    const auto cross_lines = ds_vk::viz::draw_cross_marker(
        draw,
        ds_vk::viz::CrossMarkerConfig{
            .camera = camera,
            .center = {0.0f, 0.0f, 0.0f},
            .radius = 0.2f,
        }
    );
    check(cross_lines == 2u, "viz cross marker draws two lines");

    FakeDrawSink trail_draw{};
    const std::array trail_points{
        ds_vk::Vec3{0.0f, 0.0f, 0.0f},
        ds_vk::k_axis_x,
        ds_vk::k_axis_x + ds_vk::k_axis_y,
    };
    const auto trail_segments = ds_vk::viz::draw_trail(
        trail_draw,
        ds_vk::viz::TrailConfig{
            .points = std::span<const ds_vk::Vec3>{trail_points},
            .color = ds_vk::Color{1.0f, 0.7f, 0.2f, 1.0f},
            .tail_alpha = 0.25f,
            .head_alpha = 0.85f,
        }
    );
    check(trail_segments == 2u and trail_draw.line_count == 2u, "viz trail draws line segments");
    check(near(trail_draw.first_line_color.a(), 0.25f), "viz trail applies tail alpha");
    check(near(trail_draw.last_line_color.a(), 0.85f), "viz trail applies head alpha");

    FakeDrawSink aabb_draw{};
    const auto aabb_segments = ds_vk::viz::draw_aabb(
        aabb_draw,
        ds_vk::viz::AabbMarkerConfig{
            .aabb =
                {
                    .min = {1.0f, 1.0f, 1.0f},
                    .max = {-1.0f, -2.0f, 0.5f},
                },
            .color = ds_vk::Color{0.2f, 0.4f, 0.8f, 1.0f},
        }
    );
    check(aabb_segments == 12zu and aabb_draw.line_count == 12zu, "viz aabb draws twelve edges");
    check(near(aabb_draw.last_line_color.b(), 0.8f), "viz aabb forwards line color");
}
}  // namespace

auto main() -> int
{
    try
    {
        test_color_types();
        test_quad_mesh();
        test_cube_mesh();
        test_quantized_position_normal_mesh();
        test_sphere_mesh();
        test_gltf_assets();
        test_camera_projection();
        test_camera_config();
        test_geometry_helpers();
        test_picker_plugin();
        test_manipulator_plugin();
        test_viz_plugin();
    }
    catch (const std::exception& error)
    {
        std::cerr << "[FAIL] unhandled exception: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0)
    {
        std::cerr << g_failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "all ds_vk tests passed\n";
    return 0;
}
