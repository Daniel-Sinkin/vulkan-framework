#include "app/realtime-sph/realtime_sph_app.hpp"

#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/manipulator.hpp"
#include "ds_vk/plugins/picker.hpp"
#include "ds_vk/plugins/viz.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <iostream>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ds_vk_app::realtime_sph
{
namespace
{
using namespace ds_vk;

#ifndef DS_VK_SHADER_DIR
#    define DS_VK_SHADER_DIR "build/shaders"
#endif

constexpr u32 k_workgroup_size{128u};
constexpr u32 k_default_particles{20'000u};
constexpr u32 k_max_particles_per_cell{96u};
constexpr u32 k_max_cached_neighbors{96u};
constexpr u32 k_max_profile_iterations{12u};
constexpr u32 k_profile_query_count{8u + 3u * k_max_profile_iterations};
constexpr u32 k_kernel_lut_size{4096u};
constexpr u32 k_max_rigid_bodies{8u};
constexpr f32 k_rigid_impulse_fixed_scale{100000.0f};
constexpr f32 k_rigid_buoyancy_contact_reference{520.0f};

struct Buffer
{
    VkBuffer handle{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    void* mapped{};
    VkDeviceSize size{};
};

struct GpuParticle
{
    Vec4 position_radius{};
    Vec4 previous_density{};
    Vec4 velocity_lambda{};
    Vec4 delta_neighbors{};
};

struct GpuSimParams
{
    Vec4 domain_min_dt{};
    Vec4 domain_max_radius{};
    Vec4 gravity_rest_density{};
    Vec4 kernel_poly6_spiky{};
    Vec4 pbf_params{};
    Vec4 viscosity_vorticity{};
    glm::uvec4 counts{};
    glm::uvec4 grid_size{};
};

struct GpuKernelLutSample
{
    f32 poly6{};
    f32 spiky_gradient_factor{};
    f32 pad0{};
    f32 pad1{};
};

struct GpuRenderParams
{
    Mat4 view_projection{1.0f};
    Mat4 view{1.0f};
    Mat4 projection{1.0f};
    Vec4 camera_right{};
    Vec4 camera_up{};
    Vec4 camera_forward{};
    Vec4 base_color{};
    Vec4 light_direction{};
    Vec4 options{};
};

struct GpuRigidBody
{
    Vec4 center_type{};
    Vec4 extent_radius{};
    Vec4 velocity_inv_mass{};
    Vec4 orientation{};
};

struct GpuRigidImpulse
{
    glm::ivec4 correction_contacts{};
    glm::ivec4 torque_pad{};
};

struct GpuStats
{
    u32 escaped{};
    u32 overflow{};
    u32 active{};
    u32 nan_count{};
    u32 max_neighbors{};
    u32 max_speed_milli{};
    i32 sum_position_x_milli{};
    i32 sum_position_y_milli{};
    i32 sum_position_z_milli{};
    i32 sum_velocity_x_milli{};
    i32 sum_velocity_y_milli{};
    i32 sum_velocity_z_milli{};
    i32 min_position_x_milli{};
    i32 min_position_y_milli{};
    i32 min_position_z_milli{};
    i32 max_position_x_milli{};
    i32 max_position_y_milli{};
    i32 max_position_z_milli{};
    u32 invalid_events{};
    u32 overflow_events{};
};

static_assert(sizeof(GpuParticle) == 64zu);
static_assert(sizeof(GpuSimParams) == 128zu);
static_assert(sizeof(GpuKernelLutSample) == 16zu);
static_assert(sizeof(GpuRigidBody) == 64zu);
static_assert(sizeof(GpuRigidImpulse) == 32zu);
static_assert(sizeof(GpuStats) == 80zu);

struct GpuProfileBreakdown
{
    double total_ms{};
    double reset_ms{};
    double predict_ms{};
    double build_grid_ms{};
    double build_neighbors_ms{};
    double lambda_ms{};
    double delta_ms{};
    double apply_delta_ms{};
    double velocity_ms{};
    double neighbor_scan_lambda_ms{};
    double neighbor_scan_delta_ms{};
    u32 iterations{};
};

struct FrameResources
{
    Buffer sim_params{};
    Buffer render_params{};
    Buffer stats{};
    Buffer rigid_impulses{};
    VkQueryPool profile_queries{VK_NULL_HANDLE};
    u32 profile_query_count_written{};
    u32 profile_iterations_written{};
    bool profile_queries_written{};
    bool rigid_impulses_consumed{true};
    VkDescriptorSet compute_descriptor{VK_NULL_HANDLE};
    VkDescriptorSet render_descriptor{VK_NULL_HANDLE};
};

enum class Scenario : u8
{
    dam_break = 0u,
    no_gravity_cube = 1u,
    opposing_cubes = 2u,
};

enum class SolverProfileMode : u32
{
    full = 0u,
    neighbor_scan = 1u,
};

enum class RigidBodyShape : u32
{
    sphere = 0u,
    box = 1u,
};

struct RigidBody
{
    ObjectId object_id{};
    RigidBodyShape shape{RigidBodyShape::box};
    Vec3 position{};
    Vec3 velocity{};
    Quat orientation{k_quat_identity};
    Vec3 angular_velocity{};
    Vec3 half_extent{0.18f};
    f32 radius{0.18f};
    f32 inv_mass{1.0f};
    Color color{0.86f, 0.78f, 0.58f, 1.0f};
    u32 contact_count{};
    bool grabbed{};
};

struct SimulationSettings
{
    u32 target_particles{k_default_particles};
    f32 particle_radius{0.025f};
    f32 support_radius{0.10f};
    f32 fixed_dt{0.005f};
    u32 solver_iterations{5u};
    u32 max_substeps_per_frame{8u};
    f32 simulation_speed{1.0f};
    f32 speed_color_mix{0.55f};
    f32 render_radius_scale{1.0f};
    u32 neighbor_rebuild_interval{1u};
    f32 rigid_feedback_strength{0.006f};
    f32 rigid_torque_strength{1.0f};
    f32 rigid_buoyancy_strength{1.65f};
    Vec3 gravity{0.0f, 0.0f, -9.81f};
    Scenario scenario{Scenario::dam_break};
    SolverProfileMode profile_mode{SolverProfileMode::full};
    bool force_neighbor_rebuilds{true};
    bool rigid_bodies_enabled{true};
    bool paused{};
};

struct Domain
{
    Vec3 min{};
    Vec3 max{};
};

[[nodiscard]] auto ceil_div(u32 value, u32 divisor) noexcept -> u32
{
    return (value + divisor - 1u) / divisor;
}

[[nodiscard]] auto byte_size(usize count, usize element_size) -> VkDeviceSize
{
    return static_cast<VkDeviceSize>(count * element_size);
}

[[nodiscard]] constexpr auto profile_query_count_for_iterations(u32 iterations) noexcept -> u32
{
    return 8u + 3u * std::min(iterations, k_max_profile_iterations);
}

auto check_vk(VkResult result, std::string_view label) -> void
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(
            std::string{label} + " failed with VkResult " + std::to_string(static_cast<int>(result))
        );
    }
}

[[nodiscard]] auto read_spirv(const std::filesystem::path& path) -> std::vector<u32>
{
    auto file = std::ifstream{path, std::ios::binary | std::ios::ate};
    if (!file)
    {
        throw std::runtime_error("failed to open shader: " + path.string());
    }
    const auto end = file.tellg();
    if (end < std::streamoff{0})
    {
        throw std::runtime_error("failed to query shader size: " + path.string());
    }
    const auto bytes = static_cast<usize>(end);
    if (bytes % sizeof(u32) != 0zu)
    {
        throw std::runtime_error("shader byte size is not SPIR-V aligned: " + path.string());
    }
    auto code = std::vector<u32>(bytes / sizeof(u32));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(bytes));
    if (!file)
    {
        throw std::runtime_error("failed to read shader: " + path.string());
    }
    return code;
}

[[nodiscard]] auto create_shader_module(VkDevice device, const std::filesystem::path& path)
    -> VkShaderModule
{
    const auto code = read_spirv(path);
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = byte_size(code.size(), sizeof(u32));
    create_info.pCode = code.data();

    VkShaderModule module{VK_NULL_HANDLE};
    check_vk(vkCreateShaderModule(device, &create_info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

[[nodiscard]] auto shader_path(std::string_view filename) -> std::filesystem::path
{
    return std::filesystem::path{DS_VK_SHADER_DIR} / filename;
}

auto destroy_buffer(VmaAllocator allocator, Buffer& buffer) noexcept -> void
{
    if (buffer.handle != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(allocator, buffer.handle, buffer.allocation);
    }
    buffer = {};
}

[[nodiscard]] auto create_buffer(
    VmaAllocator allocator,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    bool mapped,
    const char* label
) -> Buffer
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
    if (mapped)
    {
        allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                                | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }

    VmaAllocationInfo allocation_result{};
    auto buffer = Buffer{.size = size};
    check_vk(
        vmaCreateBuffer(
            allocator,
            &buffer_info,
            &allocation_info,
            &buffer.handle,
            &buffer.allocation,
            &allocation_result
        ),
        label
    );
    buffer.mapped = allocation_result.pMappedData;
    return buffer;
}

template <typename T>
auto write_mapped(VmaAllocator allocator, Buffer& buffer, const T& value) -> void
{
    if (buffer.mapped == nullptr)
    {
        throw std::runtime_error("attempted to write an unmapped buffer");
    }
    std::memcpy(buffer.mapped, &value, sizeof(T));
    check_vk(vmaFlushAllocation(allocator, buffer.allocation, 0, sizeof(T)), "vmaFlushAllocation");
}

auto write_mapped_bytes(VmaAllocator allocator, Buffer& buffer, const void* data, VkDeviceSize size)
    -> void
{
    if (buffer.mapped == nullptr)
    {
        throw std::runtime_error("attempted to write an unmapped buffer");
    }
    if (size > buffer.size)
    {
        throw std::runtime_error("mapped buffer write exceeds allocation");
    }
    std::memcpy(buffer.mapped, data, static_cast<usize>(size));
    check_vk(vmaFlushAllocation(allocator, buffer.allocation, 0, size), "vmaFlushAllocation");
}

[[nodiscard]] auto poly6_constant(f32 h) noexcept -> f32
{
    return 315.0f / (64.0f * std::numbers::pi_v<f32> * std::pow(h, 9.0f));
}

[[nodiscard]] auto spiky_gradient_constant(f32 h) noexcept -> f32
{
    return -45.0f / (std::numbers::pi_v<f32> * std::pow(h, 6.0f));
}

[[nodiscard]] auto spiky_gradient_factor(f32 r2, f32 h) noexcept -> f32
{
    if (r2 <= 1.0e-12f)
    {
        return 0.0f;
    }
    const auto r = std::sqrt(r2);
    if (r >= h)
    {
        return 0.0f;
    }
    const auto x = h - r;
    return spiky_gradient_constant(h) * x * x / r;
}

[[nodiscard]] auto poly6_value(f32 r2, f32 h) noexcept -> f32
{
    const auto h2 = h * h;
    if (r2 >= h2)
    {
        return 0.0f;
    }
    const auto x = h2 - r2;
    return poly6_constant(h) * x * x * x;
}

[[nodiscard]] auto generate_kernel_lut(f32 h) -> std::array<GpuKernelLutSample, k_kernel_lut_size>
{
    auto lut = std::array<GpuKernelLutSample, k_kernel_lut_size>{};
    const auto h2 = h * h;
    for (auto i = 0u; i < k_kernel_lut_size; ++i)
    {
        const auto t = (static_cast<f32>(i) + 0.5f) / static_cast<f32>(k_kernel_lut_size);
        const auto r2 = t * h2;
        lut[i].poly6 = poly6_value(r2, h);
        lut[i].spiky_gradient_factor = spiky_gradient_factor(r2, h);
    }
    return lut;
}

[[nodiscard]] auto lattice_rest_density(f32 spacing, f32 h) noexcept -> f32
{
    auto density = 0.0f;
    const auto radius = static_cast<int>(std::ceil(h / spacing));
    for (auto z = -radius; z <= radius; ++z)
    {
        for (auto y = -radius; y <= radius; ++y)
        {
            for (auto x = -radius; x <= radius; ++x)
            {
                const auto offset = spacing
                                    * Vec3{
                                        static_cast<f32>(x),
                                        static_cast<f32>(y),
                                        static_cast<f32>(z),
                                    };
                density += poly6_value(glm::dot(offset, offset), h);
            }
        }
    }
    return density;
}

[[nodiscard]] auto make_grid_size(const Domain& domain, f32 support_radius) -> glm::uvec3
{
    const auto span = domain.max - domain.min;
    return glm::uvec3{
        static_cast<u32>(std::max(1.0f, std::ceil(span.x / support_radius))),
        static_cast<u32>(std::max(1.0f, std::ceil(span.y / support_radius))),
        static_cast<u32>(std::max(1.0f, std::ceil(span.z / support_radius))),
    };
}

[[nodiscard]] auto flatten_grid_size(glm::uvec3 grid) noexcept -> u32
{
    return grid.x * grid.y * grid.z;
}

[[nodiscard]] auto generate_dam_break_particles(const SimulationSettings& settings)
    -> std::vector<GpuParticle>
{
    const auto spacing = 2.0f * settings.particle_radius;
    auto particles = std::vector<GpuParticle>{};
    particles.reserve(settings.target_particles);

    const auto push_particle = [&](Vec3 position, Vec3 velocity)
    {
        particles.push_back(
            GpuParticle{
                .position_radius = Vec4{position, settings.particle_radius},
                .previous_density = Vec4{position, 0.0f},
                .velocity_lambda = Vec4{velocity, 0.0f},
                .delta_neighbors = Vec4{0.0f},
            }
        );
    };
    const auto push_block = [&](Vec3 block_min, u32 nx, u32 ny, u32 target, Vec3 velocity)
    {
        const auto nz = ceil_div(target, nx * ny);
        for (auto z = 0u; z < nz && particles.size() < settings.target_particles; ++z)
        {
            for (auto y = 0u; y < ny && particles.size() < settings.target_particles; ++y)
            {
                for (auto x = 0u; x < nx && particles.size() < settings.target_particles; ++x)
                {
                    const auto jitter = Vec3{
                        static_cast<f32>((x * 17u + y * 3u + z * 11u) % 7u) * 0.00035f,
                        static_cast<f32>((x * 5u + y * 13u + z * 2u) % 5u) * 0.00030f,
                        0.0f,
                    };
                    const auto position = block_min + spacing * Vec3{
                                                           static_cast<f32>(x),
                                                           static_cast<f32>(y),
                                                           static_cast<f32>(z),
                                                       }
                                          + jitter;
                    push_particle(position, velocity);
                }
            }
        }
    };

    switch (settings.scenario)
    {
        case Scenario::dam_break:
            push_block(
                Vec3{-2.55f, -0.95f, 0.08f}, 28u, 38u, settings.target_particles, Vec3{0.0f}
            );
            break;
        case Scenario::no_gravity_cube:
            {
                const auto n = static_cast<u32>(
                    std::ceil(std::cbrt(static_cast<f32>(settings.target_particles)))
                );
                const auto width = spacing * static_cast<f32>(std::max(1u, n) - 1u);
                push_block(
                    Vec3{-0.5f * width, -0.5f * width, 0.25f},
                    n,
                    n,
                    settings.target_particles,
                    Vec3{0.0f}
                );
                break;
            }
        case Scenario::opposing_cubes:
            {
                const auto half = settings.target_particles / 2u;
                const auto n = static_cast<u32>(
                    std::ceil(std::cbrt(static_cast<f32>(std::max(1u, half))))
                );
                const auto width = spacing * static_cast<f32>(std::max(1u, n) - 1u);
                constexpr auto gap = 0.12f;
                push_block(
                    Vec3{-0.5f * gap - width, -0.5f * width, 0.25f},
                    n,
                    n,
                    half,
                    Vec3{1.25f, 0.0f, 0.0f}
                );
                push_block(
                    Vec3{0.5f * gap, -0.5f * width, 0.25f},
                    n,
                    n,
                    settings.target_particles - half,
                    Vec3{-1.25f, 0.0f, 0.0f}
                );
                break;
            }
    }
    return particles;
}

[[nodiscard]] auto center_of_mass_from_stats(const GpuStats& stats) noexcept -> Vec3
{
    const auto inv_count = stats.active == 0u ? 0.0f : 1.0f / static_cast<f32>(stats.active);
    return Vec3{
        static_cast<f32>(stats.sum_position_x_milli) * 0.001f * inv_count,
        static_cast<f32>(stats.sum_position_y_milli) * 0.001f * inv_count,
        static_cast<f32>(stats.sum_position_z_milli) * 0.001f * inv_count,
    };
}

[[nodiscard]] auto min_position_from_stats(const GpuStats& stats) noexcept -> Vec3
{
    if (stats.active == 0u || stats.min_position_x_milli == 2147483647)
    {
        return Vec3{0.0f};
    }
    return Vec3{
        static_cast<f32>(stats.min_position_x_milli) * 0.001f,
        static_cast<f32>(stats.min_position_y_milli) * 0.001f,
        static_cast<f32>(stats.min_position_z_milli) * 0.001f,
    };
}

[[nodiscard]] auto max_position_from_stats(const GpuStats& stats) noexcept -> Vec3
{
    if (stats.active == 0u || stats.max_position_x_milli == -2147483647)
    {
        return Vec3{0.0f};
    }
    return Vec3{
        static_cast<f32>(stats.max_position_x_milli) * 0.001f,
        static_cast<f32>(stats.max_position_y_milli) * 0.001f,
        static_cast<f32>(stats.max_position_z_milli) * 0.001f,
    };
}

[[nodiscard]] auto scenario_name(Scenario scenario) noexcept -> const char*
{
    switch (scenario)
    {
        case Scenario::dam_break:
            return "Dam break";
        case Scenario::no_gravity_cube:
            return "No-gravity cube";
        case Scenario::opposing_cubes:
            return "Opposing cubes";
    }
    return "Unknown";
}

[[nodiscard]] auto scenario_cli_name(Scenario scenario) noexcept -> const char*
{
    switch (scenario)
    {
        case Scenario::dam_break:
            return "dam-break";
        case Scenario::no_gravity_cube:
            return "no-gravity-cube";
        case Scenario::opposing_cubes:
            return "opposing-cubes";
    }
    return "unknown";
}

[[nodiscard]] auto solver_profile_mode_name(SolverProfileMode mode) noexcept -> const char*
{
    switch (mode)
    {
        case SolverProfileMode::full:
            return "full";
        case SolverProfileMode::neighbor_scan:
            return "neighbor-scan";
    }
    return "unknown";
}

auto apply_scenario_defaults(SimulationSettings& settings) noexcept -> void
{
    if (settings.scenario == Scenario::no_gravity_cube
        || settings.scenario == Scenario::opposing_cubes)
    {
        settings.gravity = Vec3{0.0f};
        return;
    }
    settings.gravity = Vec3{0.0f, 0.0f, -9.81f};
}

[[nodiscard]] auto make_default_domain(const SimulationSettings& settings) noexcept -> Domain
{
    if (settings.target_particles <= 25000u)
    {
        return Domain{
            .min = Vec3{-2.75f, -1.15f, 0.0f},
            .max = Vec3{1.15f, 1.15f, 2.4f},
        };
    }
    return Domain{
        .min = Vec3{-2.75f, -1.35f, 0.0f},
        .max = Vec3{2.25f, 1.35f, 2.8f},
    };
}

[[nodiscard]] auto artificial_pressure_strength(Scenario scenario) noexcept -> f32
{
    return scenario == Scenario::dam_break ? 0.001f : 0.0f;
}

[[nodiscard]] auto clamp_length(Vec3 value, f32 max_length) noexcept -> Vec3
{
    const auto length = glm::length(value);
    if (length <= max_length || length <= 1.0e-6f)
    {
        return value;
    }
    return value * (max_length / length);
}

[[nodiscard]] auto rigid_shape_id(RigidBodyShape shape) noexcept -> f32
{
    return static_cast<f32>(static_cast<u32>(shape));
}

[[nodiscard]] auto quat_to_gpu(Quat quat) noexcept -> Vec4
{
    return Vec4{quat.x, quat.y, quat.z, quat.w};
}

[[nodiscard]] auto oriented_box_aabb_extent(const RigidBody& body) noexcept -> Vec3
{
    if (body.shape == RigidBodyShape::sphere)
    {
        return Vec3{body.radius};
    }
    const auto rotation = glm::mat3_cast(body.orientation);
    const auto axis_x = Vec3{rotation[0][0], rotation[0][1], rotation[0][2]};
    const auto axis_y = Vec3{rotation[1][0], rotation[1][1], rotation[1][2]};
    const auto axis_z = Vec3{rotation[2][0], rotation[2][1], rotation[2][2]};
    return glm::abs(axis_x) * body.half_extent.x + glm::abs(axis_y) * body.half_extent.y
           + glm::abs(axis_z) * body.half_extent.z;
}

[[nodiscard]] auto box_inverse_inertia_world(const RigidBody& body) noexcept -> glm::mat3
{
    if (body.inv_mass <= 0.0f)
    {
        return glm::mat3{0.0f};
    }
    const auto size = 2.0f * glm::max(body.half_extent, Vec3{0.02f});
    const auto inverse_inertia = Vec3{
        12.0f * body.inv_mass / (size.y * size.y + size.z * size.z),
        12.0f * body.inv_mass / (size.x * size.x + size.z * size.z),
        12.0f * body.inv_mass / (size.x * size.x + size.y * size.y),
    };
    const auto local_inverse_inertia = glm::mat3{
        Vec3{inverse_inertia.x, 0.0f, 0.0f},
        Vec3{0.0f, inverse_inertia.y, 0.0f},
        Vec3{0.0f, 0.0f, inverse_inertia.z},
    };
    const auto rotation = glm::mat3_cast(body.orientation);
    return rotation * local_inverse_inertia * glm::transpose(rotation);
}

class RealtimeSphApp
{
  public:
    explicit RealtimeSphApp(SimulationSettings settings)
        : settings_(settings), domain_(make_default_domain(settings))
    {
    }

    auto setup(Runtime& runtime) -> void
    {
        rigid_box_mesh_ = runtime.upload_mesh(make_cube(1.0f, Color::white));
        rigid_sphere_mesh_ = runtime.upload_mesh(make_uv_sphere({
            .radius = 1.0f,
            .slices = 32u,
            .stacks = 16u,
            .color = Color::white,
        }));
        runtime.camera({
            .pivot = Vec3{0.0f, 0.0f, 1.05f},
            .distance = 6.7f,
            .yaw = glm::radians(42.0f),
            .pitch = glm::radians(24.0f),
            .z_near = 0.015f,
        });
    }

    auto update(FrameContext& frame, f32 dt_seconds) -> void
    {
        ensure_resources(frame);
        read_stats(frame);
        read_gpu_profile(frame);
        read_rigid_impulses(frame);
        rigid_frame_dt_ = std::max(settings_.fixed_dt, std::min(dt_seconds, 0.08f));

        if (frame.input.space_pressed)
        {
            settings_.paused = !settings_.paused;
        }
        if (frame.input.key_r_pressed)
        {
            reset_requested_ = true;
        }
        if (reset_requested_)
        {
            reset_particles();
            reset_rigid_bodies();
            reset_requested_ = false;
        }

        picker_.clear();
        register_pick_targets();
        const auto was_manipulating = manipulator_.active();
        update_manipulator(frame);
        if (!was_manipulating && !manipulator_.active())
        {
            handle_selection_click(frame);
        }
        sync_grabbed_flags();

        accumulator_ += std::min(dt_seconds, 0.08f) * settings_.simulation_speed;
        auto substeps = 0u;
        auto profiled_step = false;
        while (!settings_.paused && accumulator_ >= settings_.fixed_dt
               && substeps < settings_.max_substeps_per_frame)
        {
            const auto rebuild_decision = decide_neighbor_rebuild();
            integrate_rigid_bodies(settings_.fixed_dt);
            write_rigid_bodies();
            write_sim_params(frame, settings_.fixed_dt);
            record_simulation_step(frame, !profiled_step, rebuild_decision.rebuild);
            commit_neighbor_rebuild_decision(rebuild_decision);
            profiled_step = true;
            accumulator_ -= settings_.fixed_dt;
            ++substeps;
            ++sim_step_index_;
        }
        if (substeps == settings_.max_substeps_per_frame)
        {
            accumulator_ = std::min(accumulator_, settings_.fixed_dt);
        }

        configure_scene_draw(frame);
        draw_rigid_bodies(frame);
    }

    auto render(FrameContext& frame) -> void
    {
        if (!resources_ready_)
        {
            return;
        }
        ensure_render_pipeline(frame);
        write_render_params(frame);

        const auto frame_slot = frame_slot_for(frame);
        auto* cmd = frame.command_buffer;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline_);
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            render_pipeline_layout_,
            0,
            1,
            &frames_[frame_slot].render_descriptor,
            0,
            nullptr
        );
        const VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<f32>(frame.extent.width),
            .height = static_cast<f32>(frame.extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const VkRect2D scissor{.offset = {}, .extent = frame.extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdDraw(cmd, 6u, particle_count_, 0u, 0u);
    }

    auto draw_ui(FrameContext& frame) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{18.0f, 280.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{410.0f, 440.0f}, ImGuiCond_Once);
        if (ImGui::Begin("Realtime SPH"))
        {
            ImGui::PushItemWidth(210.0f);
            const auto fps = ImGui::GetIO().Framerate;
            const auto frame_ms =
                fps > 0.0f ? 1000.0f / fps : std::max(0.0f, frame.dt_seconds * 1000.0f);
            ImGui::Text(
                "frame: %.2f ms (%.1f fps)", static_cast<double>(frame_ms), static_cast<double>(fps)
            );
            ImGui::Checkbox("Paused", &settings_.paused);
            if (ImGui::Button("Reset"))
            {
                reset_requested_ = true;
            }
            ImGui::SameLine();
            ImGui::Text("%u particles", particle_count_);
            if (ImGui::BeginCombo("Scenario", scenario_name(settings_.scenario)))
            {
                constexpr auto scenarios = std::array{
                    Scenario::dam_break,
                    Scenario::no_gravity_cube,
                    Scenario::opposing_cubes,
                };
                for (const auto scenario : scenarios)
                {
                    const auto selected = settings_.scenario == scenario;
                    if (ImGui::Selectable(scenario_name(scenario), selected))
                    {
                        settings_.scenario = scenario;
                        apply_scenario_defaults(settings_);
                        reset_requested_ = true;
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SliderFloat("Simulation speed", &settings_.simulation_speed, 0.0f, 2.0f, "%.2f");
            auto iterations = static_cast<int>(settings_.solver_iterations);
            if (ImGui::SliderInt("PBF iterations", &iterations, 1, 8))
            {
                settings_.solver_iterations = static_cast<u32>(std::max(1, iterations));
            }
            auto rebuild_interval = static_cast<int>(settings_.neighbor_rebuild_interval);
            if (ImGui::SliderInt("Neighbor rebuild interval", &rebuild_interval, 1, 8))
            {
                settings_.neighbor_rebuild_interval =
                    static_cast<u32>(std::max(1, rebuild_interval));
            }
            ImGui::Checkbox("Forced neighbor rebuilds", &settings_.force_neighbor_rebuilds);
            ImGui::Checkbox("Rigid bodies", &settings_.rigid_bodies_enabled);
            ImGui::SliderFloat(
                "Rigid feedback", &settings_.rigid_feedback_strength, 0.0f, 0.08f, "%.3f"
            );
            ImGui::SliderFloat(
                "Rigid torque", &settings_.rigid_torque_strength, 0.0f, 4.0f, "%.2f"
            );
            ImGui::SliderFloat(
                "Rigid buoyancy", &settings_.rigid_buoyancy_strength, 0.0f, 3.0f, "%.2f"
            );
            ImGui::Text("profile mode: %s", solver_profile_mode_name(settings_.profile_mode));
            ImGui::SliderFloat(
                "Render radius", &settings_.render_radius_scale, 0.55f, 1.6f, "%.2f"
            );
            ImGui::SliderFloat("Speed coloring", &settings_.speed_color_mix, 0.0f, 1.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("GPU stats");
            ImGui::Text("step: %llu", static_cast<unsigned long long>(sim_step_index_));
            ImGui::Text("grid overflow: %u", last_stats_.overflow);
            ImGui::Text("escaped/teleported: %u", last_stats_.escaped);
            ImGui::Text("max neighbors: %u", last_stats_.max_neighbors);
            ImGui::Text(
                "neighbor rebuilds: %llu (%llu scheduled, %llu forced)",
                static_cast<unsigned long long>(neighbor_rebuilds_),
                static_cast<unsigned long long>(scheduled_neighbor_rebuilds_),
                static_cast<unsigned long long>(forced_neighbor_rebuilds_)
            );
            ImGui::Text(
                "max speed: %.3f", static_cast<double>(last_stats_.max_speed_milli) / 1000.0
            );
            ImGui::Text(
                "rigid bodies: %zu contacts: %u",
                rigid_bodies_.size(),
                last_rigid_contact_count_
            );
            ImGui::Text("max angular speed: %.3f", static_cast<double>(max_rigid_angular_speed()));
            if (timestamp_supported_)
            {
                ImGui::Text("GPU step: %.3f ms", last_gpu_sim_ms_);
                ImGui::Text(
                    "  setup: %.3f ms (reset %.3f, predict %.3f, grid %.3f, nbh %.3f)",
                    last_gpu_profile_.reset_ms + last_gpu_profile_.predict_ms
                        + last_gpu_profile_.build_grid_ms + last_gpu_profile_.build_neighbors_ms,
                    last_gpu_profile_.reset_ms,
                    last_gpu_profile_.predict_ms,
                    last_gpu_profile_.build_grid_ms,
                    last_gpu_profile_.build_neighbors_ms
                );
                ImGui::Text(
                    "  solve: %.3f ms (lambda %.3f, delta %.3f, apply %.3f)",
                    last_gpu_profile_.lambda_ms + last_gpu_profile_.delta_ms
                        + last_gpu_profile_.apply_delta_ms,
                    last_gpu_profile_.lambda_ms,
                    last_gpu_profile_.delta_ms,
                    last_gpu_profile_.apply_delta_ms
                );
                ImGui::Text("  velocity: %.3f ms", last_gpu_profile_.velocity_ms);
                if (settings_.profile_mode == SolverProfileMode::neighbor_scan)
                {
                    ImGui::Text(
                        "  neighbor scan: %.3f ms (lambda %.3f, delta %.3f)",
                        last_gpu_profile_.neighbor_scan_lambda_ms
                            + last_gpu_profile_.neighbor_scan_delta_ms,
                        last_gpu_profile_.neighbor_scan_lambda_ms,
                        last_gpu_profile_.neighbor_scan_delta_ms
                    );
                }
            }
            const auto com = center_of_mass_from_stats(last_stats_);
            ImGui::Text(
                "center: %.3f %.3f %.3f",
                static_cast<double>(com.x),
                static_cast<double>(com.y),
                static_cast<double>(com.z)
            );
            const auto min_pos = min_position_from_stats(last_stats_);
            const auto max_pos = max_position_from_stats(last_stats_);
            const auto span = max_pos - min_pos;
            ImGui::Text(
                "span: %.3f %.3f %.3f",
                static_cast<double>(span.x),
                static_cast<double>(span.y),
                static_cast<double>(span.z)
            );
            ImGui::Text("nan count: %u", last_stats_.nan_count);
            ImGui::Text("invalid events: %u", last_stats_.invalid_events);
            ImGui::Text("overflow events: %u", last_stats_.overflow_events);
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }

    [[nodiscard]] auto final_stats() const noexcept -> GpuStats
    {
        return last_stats_;
    }

    [[nodiscard]] auto sim_step_count() const noexcept -> u64
    {
        return sim_step_index_;
    }

    [[nodiscard]] auto last_gpu_sim_ms() const noexcept -> double
    {
        return last_gpu_sim_ms_;
    }

    [[nodiscard]] auto last_gpu_profile() const noexcept -> GpuProfileBreakdown
    {
        return last_gpu_profile_;
    }

    [[nodiscard]] auto observed_invalid_state() const noexcept -> bool
    {
        return observed_invalid_state_;
    }

    [[nodiscard]] auto neighbor_rebuild_count() const noexcept -> u64
    {
        return neighbor_rebuilds_;
    }

    [[nodiscard]] auto scheduled_neighbor_rebuild_count() const noexcept -> u64
    {
        return scheduled_neighbor_rebuilds_;
    }

    [[nodiscard]] auto forced_neighbor_rebuild_count() const noexcept -> u64
    {
        return forced_neighbor_rebuilds_;
    }

    [[nodiscard]] auto last_rigid_contact_count() const noexcept -> u32
    {
        return last_rigid_contact_count_;
    }

    [[nodiscard]] auto max_rigid_angular_speed() const noexcept -> f32
    {
        auto max_speed = 0.0f;
        for (const auto& body : rigid_bodies_)
        {
            max_speed = std::max(max_speed, glm::length(body.angular_velocity));
        }
        return max_speed;
    }

    [[nodiscard]] auto initial_center() const noexcept -> Vec3
    {
        return initial_center_;
    }

    auto finalize_gpu_results() -> void
    {
        if (device_ == VK_NULL_HANDLE || frames_.empty() || !has_recorded_frame_slot_)
        {
            return;
        }
        vkDeviceWaitIdle(device_);
        read_stats_for_slot(last_recorded_frame_slot_);
        read_gpu_profile_for_slot(last_recorded_frame_slot_);
        read_rigid_impulses_for_slot(last_recorded_frame_slot_);
    }

    auto shutdown(Runtime&) noexcept -> void
    {
        destroy();
    }

  private:
    struct NeighborRebuildDecision
    {
        bool rebuild{};
        bool scheduled{};
        bool forced{};
    };

    auto ensure_resources(FrameContext& frame) -> void
    {
        if (device_ == VK_NULL_HANDLE)
        {
            device_ = frame.device;
            allocator_ = frame.allocator;
            auto properties = VkPhysicalDeviceProperties{};
            vkGetPhysicalDeviceProperties(frame.physical_device, &properties);
            timestamp_period_ns_ = properties.limits.timestampPeriod;
            timestamp_supported_ = properties.limits.timestampComputeAndGraphics == VK_TRUE
                                   && timestamp_period_ns_ > 0.0f;
        }
        if (!resources_ready_)
        {
            create_static_resources(frame);
            create_frame_resources(frame.swapchain_image_count);
            create_descriptor_resources();
            create_compute_pipelines();
            reset_particles();
            reset_rigid_bodies();
            resources_ready_ = true;
        }
        if (frames_.size() != static_cast<usize>(frame.swapchain_image_count))
        {
            vkDeviceWaitIdle(device_);
            destroy_frame_resources();
            create_frame_resources(frame.swapchain_image_count);
            create_descriptor_resources();
        }
    }

    [[nodiscard]] auto decide_neighbor_rebuild() const noexcept -> NeighborRebuildDecision
    {
        const auto interval = std::max(1u, settings_.neighbor_rebuild_interval);
        const auto scheduled =
            !neighbor_cache_valid_ || interval == 1u || neighbor_cache_age_steps_ >= interval;
        auto forced = false;
        if (settings_.force_neighbor_rebuilds && neighbor_cache_valid_ && !scheduled)
        {
            const auto max_speed =
                static_cast<f32>(static_cast<double>(last_stats_.max_speed_milli) / 1000.0);
            const auto traveled = max_speed * neighbor_cache_elapsed_dt_;
            forced = traveled > settings_.support_radius * 0.5f;
        }
        return NeighborRebuildDecision{
            .rebuild = scheduled || forced,
            .scheduled = scheduled,
            .forced = forced,
        };
    }

    auto commit_neighbor_rebuild_decision(const NeighborRebuildDecision& decision) noexcept -> void
    {
        if (decision.rebuild)
        {
            neighbor_cache_valid_ = true;
            neighbor_cache_age_steps_ = 1u;
            neighbor_cache_elapsed_dt_ = settings_.fixed_dt;
            ++neighbor_rebuilds_;
            if (decision.scheduled)
            {
                ++scheduled_neighbor_rebuilds_;
            }
            if (decision.forced)
            {
                ++forced_neighbor_rebuilds_;
            }
            return;
        }
        ++neighbor_cache_age_steps_;
        neighbor_cache_elapsed_dt_ += settings_.fixed_dt;
    }

    auto create_static_resources(FrameContext& frame) -> void
    {
        grid_size_ = make_grid_size(domain_, settings_.support_radius);
        cell_count_ = flatten_grid_size(grid_size_);
        particle_capacity_ = settings_.target_particles;

        position_buffer_ = create_buffer(
            frame.allocator,
            byte_size(particle_capacity_, sizeof(Vec4)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph positions"
        );
        previous_density_buffer_ = create_buffer(
            frame.allocator,
            byte_size(particle_capacity_, sizeof(Vec4)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph previous density"
        );
        velocity_lambda_buffer_ = create_buffer(
            frame.allocator,
            byte_size(particle_capacity_, sizeof(Vec4)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph velocity lambda"
        );
        delta_neighbors_buffer_ = create_buffer(
            frame.allocator,
            byte_size(particle_capacity_, sizeof(Vec4)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph delta neighbors"
        );
        cell_counts_buffer_ = create_buffer(
            frame.allocator,
            byte_size(cell_count_, sizeof(u32)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            false,
            "realtime sph cell counts"
        );
        cell_particles_buffer_ = create_buffer(
            frame.allocator,
            byte_size(static_cast<usize>(cell_count_) * k_max_particles_per_cell, sizeof(u32)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            false,
            "realtime sph cell particles"
        );
        neighbor_counts_buffer_ = create_buffer(
            frame.allocator,
            byte_size(particle_capacity_, sizeof(u32)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            false,
            "realtime sph neighbor counts"
        );
        neighbor_ids_buffer_ = create_buffer(
            frame.allocator,
            byte_size(static_cast<usize>(particle_capacity_) * k_max_cached_neighbors, sizeof(u32)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            false,
            "realtime sph neighbor ids"
        );
        rigid_bodies_buffer_ = create_buffer(
            frame.allocator,
            byte_size(k_max_rigid_bodies, sizeof(GpuRigidBody)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph rigid bodies"
        );
        kernel_lut_buffer_ = create_buffer(
            frame.allocator,
            byte_size(k_kernel_lut_size, sizeof(GpuKernelLutSample)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph kernel lut"
        );
        const auto kernel_lut = generate_kernel_lut(settings_.support_radius);
        write_mapped_bytes(
            frame.allocator,
            kernel_lut_buffer_,
            kernel_lut.data(),
            byte_size(kernel_lut.size(), sizeof(GpuKernelLutSample))
        );
    }

    auto create_frame_resources(u32 frame_count) -> void
    {
        frames_.resize(frame_count);
        for (auto& frame : frames_)
        {
            frame.sim_params = create_buffer(
                allocator_,
                sizeof(GpuSimParams),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true,
                "realtime sph params"
            );
            frame.render_params = create_buffer(
                allocator_,
                sizeof(GpuRenderParams),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true,
                "realtime sph render params"
            );
            frame.stats = create_buffer(
                allocator_,
                sizeof(GpuStats),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true,
                "realtime sph stats"
            );
            frame.rigid_impulses = create_buffer(
                allocator_,
                byte_size(k_max_rigid_bodies, sizeof(GpuRigidImpulse)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true,
                "realtime sph rigid impulses"
            );
            if (timestamp_supported_)
            {
                VkQueryPoolCreateInfo query_info{};
                query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
                query_info.queryCount = k_profile_query_count;
                check_vk(
                    vkCreateQueryPool(device_, &query_info, nullptr, &frame.profile_queries),
                    "vkCreateQueryPool realtime sph profile"
                );
            }
            const auto zero = GpuStats{};
            write_mapped(allocator_, frame.stats, zero);
            const auto zero_impulses = std::array<GpuRigidImpulse, k_max_rigid_bodies>{};
            write_mapped_bytes(
                allocator_,
                frame.rigid_impulses,
                zero_impulses.data(),
                byte_size(zero_impulses.size(), sizeof(GpuRigidImpulse))
            );
        }
    }

    auto create_descriptor_resources() -> void
    {
        destroy_descriptor_resources();

        const std::array compute_bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 5,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 6,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 7,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 8,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 9,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 10,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 11,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 12,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
        };
        VkDescriptorSetLayoutCreateInfo compute_layout_info{};
        compute_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        compute_layout_info.bindingCount = static_cast<u32>(compute_bindings.size());
        compute_layout_info.pBindings = compute_bindings.data();
        check_vk(
            vkCreateDescriptorSetLayout(
                device_, &compute_layout_info, nullptr, &compute_set_layout_
            ),
            "vkCreateDescriptorSetLayout compute"
        );

        const std::array render_bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
        };
        VkDescriptorSetLayoutCreateInfo render_layout_info{};
        render_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        render_layout_info.bindingCount = static_cast<u32>(render_bindings.size());
        render_layout_info.pBindings = render_bindings.data();
        check_vk(
            vkCreateDescriptorSetLayout(device_, &render_layout_info, nullptr, &render_set_layout_),
            "vkCreateDescriptorSetLayout render"
        );

        const auto frame_count = static_cast<u32>(frames_.size());
        const std::array pool_sizes{
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = frame_count * 16u,
            },
        };
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = frame_count * 2u;
        pool_info.poolSizeCount = static_cast<u32>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        check_vk(
            vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
            "vkCreateDescriptorPool realtime sph"
        );

        auto compute_layouts =
            std::vector<VkDescriptorSetLayout>(frames_.size(), compute_set_layout_);
        auto render_layouts =
            std::vector<VkDescriptorSetLayout>(frames_.size(), render_set_layout_);
        auto compute_sets = std::vector<VkDescriptorSet>(frames_.size());
        auto render_sets = std::vector<VkDescriptorSet>(frames_.size());

        VkDescriptorSetAllocateInfo compute_alloc{};
        compute_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        compute_alloc.descriptorPool = descriptor_pool_;
        compute_alloc.descriptorSetCount = static_cast<u32>(compute_sets.size());
        compute_alloc.pSetLayouts = compute_layouts.data();
        check_vk(
            vkAllocateDescriptorSets(device_, &compute_alloc, compute_sets.data()),
            "vkAllocateDescriptorSets compute"
        );

        VkDescriptorSetAllocateInfo render_alloc{};
        render_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        render_alloc.descriptorPool = descriptor_pool_;
        render_alloc.descriptorSetCount = static_cast<u32>(render_sets.size());
        render_alloc.pSetLayouts = render_layouts.data();
        check_vk(
            vkAllocateDescriptorSets(device_, &render_alloc, render_sets.data()),
            "vkAllocateDescriptorSets render"
        );

        for (auto i = 0zu; i < frames_.size(); ++i)
        {
            frames_[i].compute_descriptor = compute_sets[i];
            frames_[i].render_descriptor = render_sets[i];
            update_descriptors(i);
        }
    }

    auto update_descriptors(usize frame_index) -> void
    {
        const std::array compute_infos{
            VkDescriptorBufferInfo{
                .buffer = position_buffer_.handle, .offset = 0, .range = position_buffer_.size
            },
            VkDescriptorBufferInfo{
                .buffer = previous_density_buffer_.handle,
                .offset = 0,
                .range = previous_density_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = velocity_lambda_buffer_.handle,
                .offset = 0,
                .range = velocity_lambda_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = delta_neighbors_buffer_.handle,
                .offset = 0,
                .range = delta_neighbors_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = cell_counts_buffer_.handle, .offset = 0, .range = cell_counts_buffer_.size
            },
            VkDescriptorBufferInfo{
                .buffer = cell_particles_buffer_.handle,
                .offset = 0,
                .range = cell_particles_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = frames_[frame_index].sim_params.handle,
                .offset = 0,
                .range = frames_[frame_index].sim_params.size,
            },
            VkDescriptorBufferInfo{
                .buffer = frames_[frame_index].stats.handle,
                .offset = 0,
                .range = frames_[frame_index].stats.size,
            },
            VkDescriptorBufferInfo{
                .buffer = kernel_lut_buffer_.handle,
                .offset = 0,
                .range = kernel_lut_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = neighbor_counts_buffer_.handle,
                .offset = 0,
                .range = neighbor_counts_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = neighbor_ids_buffer_.handle,
                .offset = 0,
                .range = neighbor_ids_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = rigid_bodies_buffer_.handle,
                .offset = 0,
                .range = rigid_bodies_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = frames_[frame_index].rigid_impulses.handle,
                .offset = 0,
                .range = frames_[frame_index].rigid_impulses.size,
            },
        };
        auto compute_writes = std::array<VkWriteDescriptorSet, compute_infos.size()>{};
        for (auto binding = 0zu; binding < compute_infos.size(); ++binding)
        {
            compute_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compute_writes[binding].dstSet = frames_[frame_index].compute_descriptor;
            compute_writes[binding].dstBinding = static_cast<u32>(binding);
            compute_writes[binding].descriptorCount = 1;
            compute_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            compute_writes[binding].pBufferInfo = &compute_infos[binding];
        }

        const std::array render_infos{
            VkDescriptorBufferInfo{
                .buffer = position_buffer_.handle, .offset = 0, .range = position_buffer_.size
            },
            VkDescriptorBufferInfo{
                .buffer = velocity_lambda_buffer_.handle,
                .offset = 0,
                .range = velocity_lambda_buffer_.size,
            },
            VkDescriptorBufferInfo{
                .buffer = frames_[frame_index].render_params.handle,
                .offset = 0,
                .range = frames_[frame_index].render_params.size,
            },
        };
        auto render_writes = std::array<VkWriteDescriptorSet, render_infos.size()>{};
        for (auto binding = 0zu; binding < render_infos.size(); ++binding)
        {
            render_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            render_writes[binding].dstSet = frames_[frame_index].render_descriptor;
            render_writes[binding].dstBinding = static_cast<u32>(binding);
            render_writes[binding].descriptorCount = 1;
            render_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            render_writes[binding].pBufferInfo = &render_infos[binding];
        }

        vkUpdateDescriptorSets(
            device_, static_cast<u32>(compute_writes.size()), compute_writes.data(), 0, nullptr
        );
        vkUpdateDescriptorSets(
            device_, static_cast<u32>(render_writes.size()), render_writes.data(), 0, nullptr
        );
    }

    auto create_compute_pipelines() -> void
    {
        const auto profile_push_range = VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0u,
            .size = sizeof(u32),
        };
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &compute_set_layout_;
        layout_info.pushConstantRangeCount = 1u;
        layout_info.pPushConstantRanges = &profile_push_range;
        check_vk(
            vkCreatePipelineLayout(device_, &layout_info, nullptr, &compute_pipeline_layout_),
            "vkCreatePipelineLayout compute"
        );

        reset_pipeline_ = create_compute_pipeline("realtime_sph_reset.comp.spv");
        predict_pipeline_ = create_compute_pipeline("realtime_sph_predict.comp.spv");
        build_grid_pipeline_ = create_compute_pipeline("realtime_sph_build_grid.comp.spv");
        build_neighbors_pipeline_ = create_compute_pipeline("realtime_sph_build_neighbors.comp.spv");
        lambda_pipeline_ = create_compute_pipeline("realtime_sph_lambda.comp.spv");
        delta_pipeline_ = create_compute_pipeline("realtime_sph_delta.comp.spv");
        apply_delta_pipeline_ = create_compute_pipeline("realtime_sph_apply_delta.comp.spv");
        velocity_pipeline_ = create_compute_pipeline("realtime_sph_velocity.comp.spv");
    }

    [[nodiscard]] auto create_compute_pipeline(std::string_view shader_name) -> VkPipeline
    {
        const auto module = create_shader_module(device_, shader_path(shader_name));
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage;
        pipeline_info.layout = compute_pipeline_layout_;

        VkPipeline pipeline{VK_NULL_HANDLE};
        check_vk(
            vkCreateComputePipelines(
                device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline
            ),
            "vkCreateComputePipelines"
        );
        vkDestroyShaderModule(device_, module, nullptr);
        return pipeline;
    }

    auto ensure_render_pipeline(FrameContext& frame) -> void
    {
        if (render_pipeline_ != VK_NULL_HANDLE && render_pass_ == frame.main_render_pass)
        {
            return;
        }
        if (render_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, render_pipeline_, nullptr);
            render_pipeline_ = VK_NULL_HANDLE;
        }
        if (render_pipeline_layout_ == VK_NULL_HANDLE)
        {
            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &render_set_layout_;
            check_vk(
                vkCreatePipelineLayout(device_, &layout_info, nullptr, &render_pipeline_layout_),
                "vkCreatePipelineLayout render"
            );
        }

        const auto vert =
            create_shader_module(device_, shader_path("realtime_sph_particles.vert.spv"));
        const auto frag =
            create_shader_module(device_, shader_path("realtime_sph_particles.frag.spv"));
        const auto make_stage = [](VkShaderModule module, VkShaderStageFlagBits stage)
        {
            auto create_info = VkPipelineShaderStageCreateInfo{};
            create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            create_info.stage = stage;
            create_info.module = module;
            create_info.pName = "main";
            return create_info;
        };
        const std::array stages{
            make_stage(vert, VK_SHADER_STAGE_VERTEX_BIT),
            make_stage(frag, VK_SHADER_STAGE_FRAGMENT_BIT),
        };

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState color_attachment{};
        color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments = &color_attachment;

        const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<u32>(dynamic_states.size());
        dynamic.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = static_cast<u32>(stages.size());
        pipeline_info.pStages = stages.data();
        pipeline_info.pVertexInputState = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterization;
        pipeline_info.pMultisampleState = &multisample;
        pipeline_info.pDepthStencilState = &depth;
        pipeline_info.pColorBlendState = &blend;
        pipeline_info.pDynamicState = &dynamic;
        pipeline_info.layout = render_pipeline_layout_;
        pipeline_info.renderPass = frame.main_render_pass;
        pipeline_info.subpass = 0;
        check_vk(
            vkCreateGraphicsPipelines(
                device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &render_pipeline_
            ),
            "vkCreateGraphicsPipelines realtime sph particles"
        );
        render_pass_ = frame.main_render_pass;
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    auto reset_particles() -> void
    {
        if (device_ != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_);
        }
        const auto particles = generate_dam_break_particles(settings_);
        auto position_sum = Vec3{0.0f};
        for (const auto& particle : particles)
        {
            position_sum += Vec3{particle.position_radius};
        }
        initial_center_ =
            particles.empty() ? Vec3{0.0f} : position_sum / static_cast<f32>(particles.size());
        particle_count_ = static_cast<u32>(particles.size());
        rest_density_ =
            lattice_rest_density(2.0f * settings_.particle_radius, settings_.support_radius);
        auto positions = std::vector<Vec4>{};
        auto previous_density = std::vector<Vec4>{};
        auto velocity_lambda = std::vector<Vec4>{};
        auto delta_neighbors = std::vector<Vec4>{};
        positions.reserve(particles.size());
        previous_density.reserve(particles.size());
        velocity_lambda.reserve(particles.size());
        delta_neighbors.reserve(particles.size());
        for (const auto& particle : particles)
        {
            positions.push_back(particle.position_radius);
            previous_density.push_back(particle.previous_density);
            velocity_lambda.push_back(particle.velocity_lambda);
            delta_neighbors.push_back(particle.delta_neighbors);
        }
        write_mapped_bytes(
            allocator_, position_buffer_, positions.data(), byte_size(positions.size(), sizeof(Vec4))
        );
        write_mapped_bytes(
            allocator_,
            previous_density_buffer_,
            previous_density.data(),
            byte_size(previous_density.size(), sizeof(Vec4))
        );
        write_mapped_bytes(
            allocator_,
            velocity_lambda_buffer_,
            velocity_lambda.data(),
            byte_size(velocity_lambda.size(), sizeof(Vec4))
        );
        write_mapped_bytes(
            allocator_,
            delta_neighbors_buffer_,
            delta_neighbors.data(),
            byte_size(delta_neighbors.size(), sizeof(Vec4))
        );
        const auto zero_stats = GpuStats{};
        for (auto& frame : frames_)
        {
            write_mapped(allocator_, frame.stats, zero_stats);
        }
        last_stats_ = {};
        last_gpu_profile_ = {};
        last_gpu_sim_ms_ = 0.0;
        observed_invalid_state_ = false;
        accumulator_ = 0.0f;
        sim_step_index_ = 0u;
        neighbor_cache_valid_ = false;
        neighbor_cache_age_steps_ = 0u;
        neighbor_cache_elapsed_dt_ = 0.0f;
        neighbor_rebuilds_ = 0u;
        scheduled_neighbor_rebuilds_ = 0u;
        forced_neighbor_rebuilds_ = 0u;
    }

    auto reset_rigid_bodies() -> void
    {
        rigid_bodies_.clear();
        last_rigid_contact_count_ = 0u;
        manipulator_.cancel();
        selected_rigid_ids_.clear();

        const auto add_body = [this](RigidBody body)
        {
            if (rigid_bodies_.size() >= k_max_rigid_bodies)
            {
                return;
            }
            rigid_bodies_.push_back(body);
        };

        if (settings_.scenario == Scenario::dam_break)
        {
            add_body(RigidBody{
                .object_id = ObjectId{.value = 1001u},
                .shape = RigidBodyShape::box,
                .position = Vec3{-0.20f, -0.42f, 0.50f},
                .velocity = Vec3{0.0f},
                .orientation = glm::angleAxis(glm::radians(7.0f), k_axis_z),
                .half_extent = Vec3{0.20f, 0.16f, 0.18f},
                .radius = 0.20f,
                .inv_mass = 0.75f,
                .color = Color{0.88f, 0.70f, 0.46f, 1.0f},
            });
            add_body(RigidBody{
                .object_id = ObjectId{.value = 1002u},
                .shape = RigidBodyShape::sphere,
                .position = Vec3{0.56f, 0.36f, 0.64f},
                .velocity = Vec3{0.0f},
                .half_extent = Vec3{0.17f},
                .radius = 0.17f,
                .inv_mass = 0.95f,
                .color = Color{0.68f, 0.84f, 0.96f, 1.0f},
            });
        }
        else
        {
            add_body(RigidBody{
                .object_id = ObjectId{.value = 1001u},
                .shape = RigidBodyShape::box,
                .position = Vec3{1.02f, 0.84f, 0.95f},
                .velocity = Vec3{0.0f},
                .orientation = glm::angleAxis(glm::radians(7.0f), k_axis_z),
                .half_extent = Vec3{0.18f, 0.16f, 0.18f},
                .radius = 0.18f,
                .inv_mass = 0.75f,
                .color = Color{0.88f, 0.70f, 0.46f, 1.0f},
            });
            add_body(RigidBody{
                .object_id = ObjectId{.value = 1002u},
                .shape = RigidBodyShape::sphere,
                .position = Vec3{1.02f, -0.84f, 0.95f},
                .velocity = Vec3{0.0f},
                .half_extent = Vec3{0.17f},
                .radius = 0.17f,
                .inv_mass = 0.95f,
                .color = Color{0.68f, 0.84f, 0.96f, 1.0f},
            });
        }

        if (!rigid_bodies_.empty())
        {
            selected_rigid_ids_ = {rigid_bodies_.front().object_id};
        }
        write_rigid_bodies();
    }

    [[nodiscard]] auto find_rigid_body(ObjectId id) noexcept -> RigidBody*
    {
        const auto iter = std::ranges::find_if(
            rigid_bodies_, [id](const RigidBody& body) noexcept
            { return body.object_id.value == id.value; }
        );
        return iter == rigid_bodies_.end() ? nullptr : &*iter;
    }

    [[nodiscard]] auto find_rigid_body(ObjectId id) const noexcept -> const RigidBody*
    {
        const auto iter = std::ranges::find_if(
            rigid_bodies_, [id](const RigidBody& body) noexcept
            { return body.object_id.value == id.value; }
        );
        return iter == rigid_bodies_.end() ? nullptr : &*iter;
    }

    [[nodiscard]] auto is_rigid_body_selected(ObjectId id) const noexcept -> bool
    {
        return std::ranges::any_of(
            selected_rigid_ids_,
            [id](ObjectId selected) noexcept { return selected.value == id.value; }
        );
    }

    auto toggle_rigid_body_selection(ObjectId id) -> void
    {
        const auto iter = std::ranges::find_if(
            selected_rigid_ids_,
            [id](ObjectId selected) noexcept { return selected.value == id.value; }
        );
        if (iter == selected_rigid_ids_.end())
        {
            selected_rigid_ids_.push_back(id);
            return;
        }
        selected_rigid_ids_.erase(iter);
    }

    auto register_pick_targets() -> void
    {
        if (!settings_.rigid_bodies_enabled)
        {
            return;
        }
        for (const auto& body : rigid_bodies_)
        {
            if (body.shape == RigidBodyShape::sphere)
            {
                (void) picker_.add_sphere({
                    .object_id = body.object_id,
                    .sphere = Sphere{.center = body.position, .radius = body.radius},
                });
                continue;
            }
            (void) picker_.add_obb({
                .object_id = body.object_id,
                .obb =
                    Obb{
                        .center = body.position,
                        .half_extent = body.half_extent,
                        .rotation = body.orientation,
                    },
            });
        }
    }

    auto handle_selection_click(const FrameContext& frame) -> void
    {
        if (!settings_.rigid_bodies_enabled || !frame.input.left_click.occurred
            || frame.input.mouse_captured_by_ui)
        {
            return;
        }
        const auto hit = picker_.click({
            .camera = frame.camera,
            .mouse_px = frame.input.left_click.position_px,
            .viewport_px =
                Vec2{
                    static_cast<f32>(frame.extent.width),
                    static_cast<f32>(frame.extent.height),
                },
        });
        if (!hit.has_value())
        {
            if (!frame.input.left_click.modifiers.shift)
            {
                selected_rigid_ids_.clear();
            }
            return;
        }
        if (frame.input.left_click.modifiers.shift)
        {
            toggle_rigid_body_selection(hit->object_id);
            return;
        }
        selected_rigid_ids_ = {hit->object_id};
    }

    [[nodiscard]] auto transform_for_rigid_body(ObjectId id) const -> std::optional<Transform>
    {
        const auto* body = find_rigid_body(id);
        if (body == nullptr)
        {
            return std::nullopt;
        }
        const auto scale = body->shape == RigidBodyShape::sphere ? Vec3{2.0f * body->radius}
                                                                 : 2.0f * body->half_extent;
        return Transform{
            .translation = body->position,
            .rotation = body->orientation,
            .scale = scale,
        };
    }

    auto set_transform_for_rigid_body(ObjectId id, const Transform& transform) -> void
    {
        auto* body = find_rigid_body(id);
        if (body == nullptr)
        {
            return;
        }
        const auto dt = std::max(rigid_frame_dt_, 1.0e-4f);
        body->velocity = clamp_length((transform.translation - body->position) / dt, 12.0f);
        body->position = transform.translation;
        body->orientation = glm::normalize(transform.rotation);
        body->angular_velocity = Vec3{0.0f};
        if (body->shape == RigidBodyShape::sphere)
        {
            body->radius = std::max(0.06f, 0.5f * std::max({
                                                std::abs(transform.scale.x),
                                                std::abs(transform.scale.y),
                                                std::abs(transform.scale.z),
                                            }));
            body->half_extent = Vec3{body->radius};
        }
        else
        {
            body->half_extent = 0.5f * glm::max(glm::abs(transform.scale), Vec3{0.06f});
            body->radius = std::max({body->half_extent.x, body->half_extent.y, body->half_extent.z});
        }
        body->grabbed = true;
    }

    auto update_manipulator(const FrameContext& frame) -> void
    {
        manipulator_.update({
            .input =
                ManipulatorInput{
                    .camera = frame.camera,
                    .mouse_px = frame.input.mouse_px,
                    .viewport_px =
                        Vec2{
                            static_cast<f32>(frame.extent.width),
                            static_cast<f32>(frame.extent.height),
                        },
                    .mouse_captured_by_ui = frame.input.mouse_captured_by_ui,
                    .translate_pressed = frame.input.key_g_pressed,
                    .rotate_pressed = false,
                    .scale_pressed = false,
                    .x_pressed = frame.input.key_x_pressed,
                    .y_pressed = frame.input.key_y_pressed,
                    .z_pressed = frame.input.key_z_pressed,
                    .confirm_pressed =
                        frame.input.left_click.occurred || frame.input.key_enter_pressed,
                    .cancel_pressed = frame.input.key_c_pressed,
                },
            .selected_ids = std::span<const ObjectId>{selected_rigid_ids_},
            .callbacks = ManipulatorCallbacks{
                .get_transform = [this](ObjectId id) -> std::optional<Transform>
                { return transform_for_rigid_body(id); },
                .set_transform = [this](ObjectId id, const Transform& transform) -> void
                { set_transform_for_rigid_body(id, transform); },
            },
        });
    }

    auto sync_grabbed_flags() -> void
    {
        for (auto& body : rigid_bodies_)
        {
            body.grabbed = settings_.rigid_bodies_enabled && manipulator_.active()
                           && is_rigid_body_selected(body.object_id);
            if (body.grabbed)
            {
                body.angular_velocity = Vec3{0.0f};
            }
        }
    }

    auto integrate_rigid_bodies(f32 dt) -> void
    {
        if (!settings_.rigid_bodies_enabled)
        {
            return;
        }
        const auto lower = domain_.min + Vec3{settings_.particle_radius};
        const auto upper = domain_.max - Vec3{settings_.particle_radius};
        for (auto& body : rigid_bodies_)
        {
            if (body.grabbed || body.inv_mass <= 0.0f)
            {
                continue;
            }
            const auto contact_ratio = std::clamp(
                static_cast<f32>(body.contact_count) / k_rigid_buoyancy_contact_reference,
                0.0f,
                1.0f
            );
            const auto buoyant_gravity_scale =
                1.0f - settings_.rigid_buoyancy_strength * contact_ratio;
            body.velocity += settings_.gravity * buoyant_gravity_scale * dt;
            if (contact_ratio > 0.0f)
            {
                body.velocity.z *= 1.0f - 0.16f * contact_ratio;
            }
            body.velocity *= 0.995f;
            body.angular_velocity *= 0.985f;
            body.velocity = clamp_length(body.velocity, 8.0f);
            body.angular_velocity = clamp_length(body.angular_velocity, 9.0f);
            body.position += body.velocity * dt;
            const auto angular_speed = glm::length(body.angular_velocity);
            if (angular_speed > 1.0e-5f)
            {
                body.orientation = glm::normalize(
                    glm::angleAxis(angular_speed * dt, body.angular_velocity / angular_speed)
                    * body.orientation
                );
            }

            const auto aabb_extent =
                glm::max(oriented_box_aabb_extent(body), Vec3{settings_.particle_radius});
            const auto min_allowed = lower + aabb_extent;
            const auto max_allowed = upper - aabb_extent;
            for (auto axis = 0; axis < 3; ++axis)
            {
                if (body.position[axis] < min_allowed[axis])
                {
                    body.position[axis] = min_allowed[axis];
                    body.velocity[axis] = std::abs(body.velocity[axis]) * 0.08f;
                    body.angular_velocity *= 0.92f;
                }
                if (body.position[axis] > max_allowed[axis])
                {
                    body.position[axis] = max_allowed[axis];
                    body.velocity[axis] = -std::abs(body.velocity[axis]) * 0.08f;
                    body.angular_velocity *= 0.92f;
                }
            }
        }
    }

    auto write_rigid_bodies() -> void
    {
        if (rigid_bodies_buffer_.mapped == nullptr)
        {
            return;
        }
        auto gpu_bodies = std::array<GpuRigidBody, k_max_rigid_bodies>{};
        const auto count = std::min<usize>(rigid_bodies_.size(), k_max_rigid_bodies);
        for (auto i = 0zu; i < count; ++i)
        {
            const auto& body = rigid_bodies_[i];
            gpu_bodies[i] = GpuRigidBody{
                .center_type = Vec4{body.position, rigid_shape_id(body.shape)},
                .extent_radius = Vec4{body.half_extent, body.radius},
                .velocity_inv_mass =
                    Vec4{
                        body.velocity,
                        body.grabbed || !settings_.rigid_bodies_enabled ? 0.0f : body.inv_mass,
                    },
                .orientation = quat_to_gpu(body.orientation),
            };
        }
        write_mapped_bytes(
            allocator_,
            rigid_bodies_buffer_,
            gpu_bodies.data(),
            byte_size(gpu_bodies.size(), sizeof(GpuRigidBody))
        );
    }

    auto read_rigid_impulses(FrameContext& frame) -> void
    {
        if (frames_.empty())
        {
            return;
        }
        read_rigid_impulses_for_slot(frame_slot_for(frame));
    }

    auto read_rigid_impulses_for_slot(usize frame_slot) -> void
    {
        if (frame_slot >= frames_.size() || frames_[frame_slot].rigid_impulses_consumed)
        {
            return;
        }
        auto& impulse_buffer = frames_[frame_slot].rigid_impulses;
        check_vk(
            vmaInvalidateAllocation(allocator_, impulse_buffer.allocation, 0, impulse_buffer.size),
            "vmaInvalidateAllocation rigid impulses"
        );
        auto impulses = std::array<GpuRigidImpulse, k_max_rigid_bodies>{};
        std::memcpy(
            impulses.data(),
            impulse_buffer.mapped,
            byte_size(impulses.size(), sizeof(GpuRigidImpulse))
        );
        frames_[frame_slot].rigid_impulses_consumed = true;

        last_rigid_contact_count_ = 0u;
        if (!settings_.rigid_bodies_enabled)
        {
            for (auto& body : rigid_bodies_)
            {
                body.contact_count = 0u;
            }
            return;
        }
        const auto dt = std::max(settings_.fixed_dt, 1.0e-4f);
        const auto count = std::min<usize>(rigid_bodies_.size(), impulses.size());
        for (auto i = 0zu; i < count; ++i)
        {
            auto& body = rigid_bodies_[i];
            const auto packed = impulses[i].correction_contacts;
            const auto contacts = static_cast<u32>(std::max(0, packed.w));
            body.contact_count = contacts;
            last_rigid_contact_count_ += contacts;
            if (contacts == 0u || body.grabbed || body.inv_mass <= 0.0f)
            {
                continue;
            }
            const auto body_correction = Vec3{
                static_cast<f32>(packed.x),
                static_cast<f32>(packed.y),
                static_cast<f32>(packed.z),
            } / k_rigid_impulse_fixed_scale;
            const auto packed_torque = impulses[i].torque_pad;
            const auto body_torque = Vec3{
                static_cast<f32>(packed_torque.x),
                static_cast<f32>(packed_torque.y),
                static_cast<f32>(packed_torque.z),
            } / k_rigid_impulse_fixed_scale;
            const auto contact_scale = 1.0f / std::sqrt(std::max(1.0f, static_cast<f32>(contacts)));
            const auto delta_v = clamp_length(
                body_correction
                    * (settings_.rigid_feedback_strength * body.inv_mass * contact_scale / dt),
                2.5f
            );
            body.velocity = clamp_length(body.velocity + delta_v, 8.0f);
            if (body.shape == RigidBodyShape::box)
            {
                const auto delta_angular_velocity = clamp_length(
                    box_inverse_inertia_world(body) * body_torque
                        * (settings_.rigid_feedback_strength * settings_.rigid_torque_strength
                           * contact_scale / dt),
                    3.5f
                );
                body.angular_velocity =
                    clamp_length(body.angular_velocity + delta_angular_velocity, 9.0f);
            }
        }
    }

    auto read_stats(FrameContext& frame) -> void
    {
        if (frames_.empty())
        {
            return;
        }
        read_stats_for_slot(frame_slot_for(frame));
    }

    auto read_stats_for_slot(usize frame_slot) -> void
    {
        if (frame_slot >= frames_.size())
        {
            return;
        }
        auto& stats_buffer = frames_[frame_slot].stats;
        check_vk(
            vmaInvalidateAllocation(allocator_, stats_buffer.allocation, 0, sizeof(GpuStats)),
            "vmaInvalidateAllocation stats"
        );
        std::memcpy(&last_stats_, stats_buffer.mapped, sizeof(GpuStats));
        observed_invalid_state_ = observed_invalid_state_ || last_stats_.escaped != 0u
                                  || last_stats_.overflow != 0u || last_stats_.nan_count != 0u
                                  || last_stats_.invalid_events != 0u;
    }

    auto read_gpu_profile(FrameContext& frame) -> void
    {
        if (frames_.empty())
        {
            return;
        }
        read_gpu_profile_for_slot(frame_slot_for(frame));
    }

    auto read_gpu_profile_for_slot(usize frame_slot) -> void
    {
        if (!timestamp_supported_ || frame_slot >= frames_.size())
        {
            return;
        }
        const auto& resources = frames_[frame_slot];
        const auto query_pool = resources.profile_queries;
        if (query_pool == VK_NULL_HANDLE || !resources.profile_queries_written)
        {
            return;
        }
        const auto query_count =
            std::min(resources.profile_query_count_written, k_profile_query_count);
        if (query_count < 2u)
        {
            return;
        }
        auto timestamps = std::array<u64, k_profile_query_count>{};
        const auto result = vkGetQueryPoolResults(
            device_,
            query_pool,
            0u,
            query_count,
            byte_size(query_count, sizeof(u64)),
            timestamps.data(),
            sizeof(u64),
            VK_QUERY_RESULT_64_BIT
        );
        if (result == VK_SUCCESS && timestamps[query_count - 1u] > timestamps[0])
        {
            const auto to_ms = [this](u64 start, u64 end) -> double
            {
                if (end <= start)
                {
                    return 0.0;
                }
                return static_cast<double>(end - start)
                       * static_cast<double>(timestamp_period_ns_) / 1.0e6;
            };

            auto profile = GpuProfileBreakdown{};
            profile.iterations = resources.profile_iterations_written;
            profile.reset_ms = query_count > 1u ? to_ms(timestamps[0], timestamps[1]) : 0.0;
            profile.predict_ms = query_count > 2u ? to_ms(timestamps[1], timestamps[2]) : 0.0;
            profile.build_grid_ms = query_count > 3u ? to_ms(timestamps[2], timestamps[3]) : 0.0;
            profile.build_neighbors_ms =
                query_count > 4u ? to_ms(timestamps[3], timestamps[4]) : 0.0;

            auto query = 4u;
            for (auto i = 0u; i < profile.iterations && query + 3u < query_count; ++i)
            {
                profile.lambda_ms += to_ms(timestamps[query], timestamps[query + 1u]);
                profile.delta_ms += to_ms(timestamps[query + 1u], timestamps[query + 2u]);
                profile.apply_delta_ms += to_ms(timestamps[query + 2u], timestamps[query + 3u]);
                query += 3u;
            }
            if (query + 1u < query_count)
            {
                profile.velocity_ms = to_ms(timestamps[query], timestamps[query + 1u]);
                ++query;
            }
            profile.total_ms = to_ms(timestamps[0], timestamps[query]);
            if (query + 2u < query_count)
            {
                profile.neighbor_scan_lambda_ms = to_ms(timestamps[query], timestamps[query + 1u]);
                profile.neighbor_scan_delta_ms =
                    to_ms(timestamps[query + 1u], timestamps[query + 2u]);
            }
            last_gpu_profile_ = profile;
            last_gpu_sim_ms_ = profile.total_ms;
        }
        else if (result != VK_NOT_READY)
        {
            check_vk(result, "vkGetQueryPoolResults realtime sph profile");
        }
    }

    auto write_sim_params(FrameContext& frame, f32 dt) -> void
    {
        const auto params = GpuSimParams{
            .domain_min_dt = Vec4{domain_.min, dt},
            .domain_max_radius = Vec4{domain_.max, settings_.particle_radius},
            .gravity_rest_density = Vec4{settings_.gravity, rest_density_},
            .kernel_poly6_spiky =
                Vec4{
                    poly6_constant(settings_.support_radius),
                    spiky_gradient_constant(settings_.support_radius),
                    settings_.support_radius,
                    1.0f,
                },
            .pbf_params =
                Vec4{
                    1.0e-6f,
                    artificial_pressure_strength(settings_.scenario),
                    static_cast<f32>(k_kernel_lut_size)
                        / (settings_.support_radius * settings_.support_radius),
                    std::max(
                        poly6_value(
                            0.30f * settings_.support_radius * 0.30f
                                * settings_.support_radius,
                            settings_.support_radius
                        ),
                        1.0e-6f
                    ),
                },
            .viscosity_vorticity = Vec4{0.0f, 0.02f, 1.0f, 15.0f},
            .counts =
                glm::uvec4{
                    particle_count_,
                    cell_count_,
                    k_max_particles_per_cell,
                    k_max_cached_neighbors,
                },
            .grid_size = glm::uvec4{
                grid_size_,
                settings_.rigid_bodies_enabled ? static_cast<u32>(rigid_bodies_.size()) : 0u,
            },
        };
        write_mapped(allocator_, frames_[frame_slot_for(frame)].sim_params, params);
    }

    auto write_render_params(FrameContext& frame) -> void
    {
        const auto aspect = frame.extent.height == 0u ? 1.0f
                                                      : static_cast<f32>(frame.extent.width)
                                                            / static_cast<f32>(frame.extent.height);
        const auto params = GpuRenderParams{
            .view_projection = frame.camera.view_projection_matrix(aspect),
            .view = frame.camera.view_matrix(),
            .projection = frame.camera.projection_matrix(aspect),
            .camera_right = Vec4{frame.camera.right(), 0.0f},
            .camera_up = Vec4{frame.camera.up(), 0.0f},
            .camera_forward =
                Vec4{glm::normalize(frame.camera.pivot() - frame.camera.position()), 0.0f},
            .base_color = Vec4{0.05f, 0.58f, 0.85f, 1.0f},
            .light_direction = Vec4{-0.42f, -0.32f, -0.84f, 0.0f},
            .options = Vec4{
                settings_.render_radius_scale,
                5.0f,
                settings_.speed_color_mix,
                0.0f,
            },
        };
        write_mapped(allocator_, frames_[frame_slot_for(frame)].render_params, params);
    }

    auto record_simulation_step(FrameContext& frame, bool profile_step, bool rebuild_neighbors)
        -> void
    {
        const auto frame_slot = frame_slot_for(frame);
        last_recorded_frame_slot_ = frame_slot;
        has_recorded_frame_slot_ = true;
        frames_[frame_slot].rigid_impulses_consumed = false;
        auto* cmd = frame.command_buffer;
        const auto iterations = std::min(settings_.solver_iterations, k_max_profile_iterations);
        const auto profile_query_count = profile_query_count_for_iterations(iterations);
        const auto query_pool = frames_[frame_slot].profile_queries;
        const auto profile_enabled = profile_step && query_pool != VK_NULL_HANDLE;
        auto query_index = 0u;
        const auto write_profile_timestamp = [&](VkPipelineStageFlagBits stage =
                                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        {
            if (!profile_enabled)
            {
                return;
            }
            if (query_index < profile_query_count)
            {
                vkCmdWriteTimestamp(cmd, stage, query_pool, query_index);
            }
            ++query_index;
        };
        if (profile_enabled)
        {
            vkCmdResetQueryPool(cmd, query_pool, 0u, profile_query_count);
            write_profile_timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        }
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            compute_pipeline_layout_,
            0,
            1,
            &frames_[frame_slot].compute_descriptor,
            0,
            nullptr
        );

        bind_and_dispatch(cmd, reset_pipeline_, std::max(cell_count_, particle_count_));
        compute_barrier(cmd);
        write_profile_timestamp();
        bind_and_dispatch(cmd, predict_pipeline_, particle_count_);
        compute_barrier(cmd);
        write_profile_timestamp();
        if (rebuild_neighbors)
        {
            bind_and_dispatch(cmd, build_grid_pipeline_, particle_count_);
            compute_barrier(cmd);
        }
        write_profile_timestamp();
        if (rebuild_neighbors)
        {
            bind_and_dispatch(cmd, build_neighbors_pipeline_, particle_count_);
            compute_barrier(cmd);
        }
        write_profile_timestamp();
        for (auto i = 0u; i < iterations; ++i)
        {
            bind_and_dispatch(cmd, lambda_pipeline_, particle_count_);
            compute_barrier(cmd);
            write_profile_timestamp();
            // Keep correction and application separate so neighbors are read from one coherent
            // position field.
            bind_and_dispatch(cmd, delta_pipeline_, particle_count_);
            compute_barrier(cmd);
            write_profile_timestamp();
            bind_and_dispatch(cmd, apply_delta_pipeline_, particle_count_);
            compute_barrier(cmd);
            write_profile_timestamp();
        }
        bind_and_dispatch(cmd, velocity_pipeline_, particle_count_);
        compute_barrier(cmd);
        if (profile_enabled)
        {
            write_profile_timestamp();
            if (settings_.profile_mode == SolverProfileMode::neighbor_scan)
            {
                bind_and_dispatch(
                    cmd, lambda_pipeline_, particle_count_, SolverProfileMode::neighbor_scan
                );
                compute_barrier(cmd);
                write_profile_timestamp();
                bind_and_dispatch(
                    cmd, delta_pipeline_, particle_count_, SolverProfileMode::neighbor_scan
                );
                compute_barrier(cmd);
                write_profile_timestamp();
            }
            frames_[frame_slot].profile_query_count_written = query_index;
            frames_[frame_slot].profile_iterations_written = iterations;
            frames_[frame_slot].profile_queries_written = true;
        }
        compute_to_render_barrier(cmd);
    }

    auto bind_and_dispatch(
        VkCommandBuffer cmd,
        VkPipeline pipeline,
        u32 item_count,
        SolverProfileMode solver_profile_mode = SolverProfileMode::full
    ) const -> void
    {
        const auto mode = static_cast<u32>(solver_profile_mode);
        vkCmdPushConstants(
            cmd,
            compute_pipeline_layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0u,
            sizeof(mode),
            &mode
        );
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdDispatch(cmd, ceil_div(std::max(1u, item_count), k_workgroup_size), 1u, 1u);
    }

    static auto compute_barrier(VkCommandBuffer cmd) -> void
    {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            nullptr,
            0,
            nullptr
        );
    }

    static auto compute_to_render_barrier(VkCommandBuffer cmd) -> void
    {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            nullptr,
            0,
            nullptr
        );
    }

    auto configure_scene_draw(FrameContext& frame) -> void
    {
        frame.draw.set_environment(
            EnvironmentConfig{
                .background_color = Color{0.17f, 0.19f, 0.20f, 1.0f},
                .background_top_color = Color{0.43f, 0.54f, 0.62f, 1.0f},
                .gradient_background = true,
            }
        );
        frame.draw.set_ambient_light(Color{0.16f, 0.18f, 0.20f, 1.0f});
        frame.draw.directional_light(
            DirectionalLightConfig{
                .direction = Vec3{-0.42f, -0.32f, -0.84f},
                .color = Color{0.92f, 0.96f, 1.0f, 1.0f},
                .intensity = 1.65f,
                .shadow = {.enabled = false},
            }
        );
        (void) viz::draw_aabb(
            frame.draw,
            viz::AabbMarkerConfig{
                .aabb = Aabb{.min = domain_.min, .max = domain_.max},
                .color = Color{0.48f, 0.58f, 0.62f, 0.62f},
                .width = 0.006f,
                .draw_on_top = true,
            }
        );
        frame.draw.debug_line({
            .start = Vec3{domain_.min.x, domain_.min.y, 0.0f},
            .end = Vec3{domain_.max.x, domain_.min.y, 0.0f},
            .color = Color{0.24f, 0.30f, 0.32f, 0.70f},
            .width = 0.004f,
        });
    }

    auto draw_rigid_bodies(FrameContext& frame) const -> void
    {
        if (!settings_.rigid_bodies_enabled)
        {
            return;
        }
        for (const auto& body : rigid_bodies_)
        {
            const auto selected = is_rigid_body_selected(body.object_id);
            const auto base_color =
                selected ? mix_color(body.color, Color{1.0f, 0.96f, 0.72f, 1.0f}, 0.35f)
                         : body.color;
            const auto material = Material{
                .base_color = base_color,
                .emissive_color = selected ? Color{0.10f, 0.08f, 0.03f, 1.0f} : Color::black,
                .metallic = 0.0f,
                .roughness = body.grabbed ? 0.38f : 0.58f,
            };
            if (body.shape == RigidBodyShape::sphere)
            {
                frame.draw.draw_mesh({
                    .mesh = rigid_sphere_mesh_,
                    .object_id = body.object_id,
                    .transform =
                        Transform{
                            .translation = body.position,
                            .rotation = body.orientation,
                            .scale = Vec3{body.radius},
                        },
                    .material = material,
                });
                continue;
            }
            frame.draw.draw_mesh({
                .mesh = rigid_box_mesh_,
                .object_id = body.object_id,
                .transform =
                    Transform{
                        .translation = body.position,
                        .rotation = body.orientation,
                        .scale = 2.0f * body.half_extent,
                    },
                .material = material,
            });
        }
    }

    [[nodiscard]] auto frame_slot_for(FrameContext& frame) const -> usize
    {
        return static_cast<usize>(frame.swapchain_image_index)
               % std::max<usize>(1zu, frames_.size());
    }

    auto destroy_descriptor_resources() noexcept -> void
    {
        if (descriptor_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            descriptor_pool_ = VK_NULL_HANDLE;
        }
        if (compute_set_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, compute_set_layout_, nullptr);
            compute_set_layout_ = VK_NULL_HANDLE;
        }
        if (render_set_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, render_set_layout_, nullptr);
            render_set_layout_ = VK_NULL_HANDLE;
        }
    }

    auto destroy_frame_resources() noexcept -> void
    {
        for (auto& frame : frames_)
        {
            if (frame.profile_queries != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(device_, frame.profile_queries, nullptr);
                frame.profile_queries = VK_NULL_HANDLE;
            }
            destroy_buffer(allocator_, frame.sim_params);
            destroy_buffer(allocator_, frame.render_params);
            destroy_buffer(allocator_, frame.stats);
            destroy_buffer(allocator_, frame.rigid_impulses);
            frame.compute_descriptor = VK_NULL_HANDLE;
            frame.render_descriptor = VK_NULL_HANDLE;
        }
        frames_.clear();
    }

    auto destroy() noexcept -> void
    {
        if (device_ == VK_NULL_HANDLE)
        {
            return;
        }
        vkDeviceWaitIdle(device_);
        if (render_pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, render_pipeline_, nullptr);
            render_pipeline_ = VK_NULL_HANDLE;
        }
        if (render_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, render_pipeline_layout_, nullptr);
            render_pipeline_layout_ = VK_NULL_HANDLE;
        }
        for (auto* pipeline :
             {&reset_pipeline_,
              &predict_pipeline_,
              &build_grid_pipeline_,
              &build_neighbors_pipeline_,
              &lambda_pipeline_,
              &delta_pipeline_,
              &apply_delta_pipeline_,
              &velocity_pipeline_})
        {
            if (*pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device_, *pipeline, nullptr);
                *pipeline = VK_NULL_HANDLE;
            }
        }
        if (compute_pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, compute_pipeline_layout_, nullptr);
            compute_pipeline_layout_ = VK_NULL_HANDLE;
        }
        destroy_descriptor_resources();
        destroy_frame_resources();
        destroy_buffer(allocator_, kernel_lut_buffer_);
        destroy_buffer(allocator_, rigid_bodies_buffer_);
        destroy_buffer(allocator_, neighbor_ids_buffer_);
        destroy_buffer(allocator_, neighbor_counts_buffer_);
        destroy_buffer(allocator_, cell_particles_buffer_);
        destroy_buffer(allocator_, cell_counts_buffer_);
        destroy_buffer(allocator_, delta_neighbors_buffer_);
        destroy_buffer(allocator_, velocity_lambda_buffer_);
        destroy_buffer(allocator_, previous_density_buffer_);
        destroy_buffer(allocator_, position_buffer_);
        resources_ready_ = false;
    }

    SimulationSettings settings_{};
    Domain domain_{};
    VkDevice device_{VK_NULL_HANDLE};
    VmaAllocator allocator_{VK_NULL_HANDLE};
    Buffer position_buffer_{};
    Buffer previous_density_buffer_{};
    Buffer velocity_lambda_buffer_{};
    Buffer delta_neighbors_buffer_{};
    Buffer cell_counts_buffer_{};
    Buffer cell_particles_buffer_{};
    Buffer neighbor_counts_buffer_{};
    Buffer neighbor_ids_buffer_{};
    Buffer rigid_bodies_buffer_{};
    Buffer kernel_lut_buffer_{};
    std::vector<FrameResources> frames_{};
    VkDescriptorSetLayout compute_set_layout_{VK_NULL_HANDLE};
    VkDescriptorSetLayout render_set_layout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    VkPipelineLayout compute_pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline reset_pipeline_{VK_NULL_HANDLE};
    VkPipeline predict_pipeline_{VK_NULL_HANDLE};
    VkPipeline build_grid_pipeline_{VK_NULL_HANDLE};
    VkPipeline build_neighbors_pipeline_{VK_NULL_HANDLE};
    VkPipeline lambda_pipeline_{VK_NULL_HANDLE};
    VkPipeline delta_pipeline_{VK_NULL_HANDLE};
    VkPipeline apply_delta_pipeline_{VK_NULL_HANDLE};
    VkPipeline velocity_pipeline_{VK_NULL_HANDLE};
    VkPipelineLayout render_pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline render_pipeline_{VK_NULL_HANDLE};
    VkRenderPass render_pass_{VK_NULL_HANDLE};
    MeshHandle rigid_box_mesh_{};
    MeshHandle rigid_sphere_mesh_{};
    Picker picker_{};
    Manipulator manipulator_{};
    std::vector<RigidBody> rigid_bodies_{};
    std::vector<ObjectId> selected_rigid_ids_{};
    glm::uvec3 grid_size_{};
    u32 cell_count_{};
    u32 particle_capacity_{};
    u32 particle_count_{};
    u32 last_rigid_contact_count_{};
    f32 rest_density_{1.0f};
    f32 accumulator_{};
    f32 rigid_frame_dt_{1.0f / 60.0f};
    u64 sim_step_index_{};
    u64 neighbor_rebuilds_{};
    u64 scheduled_neighbor_rebuilds_{};
    u64 forced_neighbor_rebuilds_{};
    GpuStats last_stats_{};
    Vec3 initial_center_{};
    f32 timestamp_period_ns_{};
    double last_gpu_sim_ms_{};
    GpuProfileBreakdown last_gpu_profile_{};
    usize last_recorded_frame_slot_{};
    bool timestamp_supported_{};
    bool has_recorded_frame_slot_{};
    bool observed_invalid_state_{};
    bool resources_ready_{};
    bool reset_requested_{};
    bool neighbor_cache_valid_{};
    u32 neighbor_cache_age_steps_{};
    f32 neighbor_cache_elapsed_dt_{};
};

[[nodiscard]] auto parse_u32(std::string_view text, u32 fallback) noexcept -> u32
{
    try
    {
        return static_cast<u32>(std::stoul(std::string{text}));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

[[nodiscard]] auto parse_f32(std::string_view text, f32 fallback) noexcept -> f32
{
    try
    {
        return std::stof(std::string{text});
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

[[nodiscard]] auto scenario_from_text(std::string_view text) noexcept -> std::optional<Scenario>
{
    if (text == "dam-break")
    {
        return Scenario::dam_break;
    }
    if (text == "no-gravity-cube")
    {
        return Scenario::no_gravity_cube;
    }
    if (text == "opposing-cubes")
    {
        return Scenario::opposing_cubes;
    }
    return std::nullopt;
}

[[nodiscard]] auto solver_profile_mode_from_text(std::string_view text) noexcept
    -> std::optional<SolverProfileMode>
{
    if (text == "full")
    {
        return SolverProfileMode::full;
    }
    if (text == "neighbor-scan")
    {
        return SolverProfileMode::neighbor_scan;
    }
    return std::nullopt;
}

auto print_usage(const char* program) -> void
{
    std::cerr << "usage: " << program
              << " [--particles N] [--paused] [--smoke-frames N] [--screenshot PATH]"
                 " [--fixed-frame-dt SECONDS] [--iterations N] [--max-substeps N]"
                 " [--scenario dam-break|no-gravity-cube|opposing-cubes]"
                 " [--solver-profile full|neighbor-scan] [--neighbor-rebuild-interval N]"
                 " [--no-forced-neighbor-rebuild]"
                 " [--no-rigid-bodies] [--rigid-feedback N] [--rigid-torque N]"
                 " [--rigid-buoyancy N]"
                 " [--max-center-delta N] [--max-speed N] [--max-z-span N]"
                 " [--print-stats] [--fail-on-invalid] [--transparent-screenshot] [--hide-ui]\n";
}

}  // namespace

auto run_realtime_sph_app(int argc, char** argv) -> int
{
    auto settings = SimulationSettings{};
    auto print_stats = false;
    auto fail_on_invalid = false;
    auto fixed_frame_dt = std::optional<f32>{};
    auto max_center_delta = std::optional<f32>{};
    auto max_speed = std::optional<f32>{};
    auto max_z_span = std::optional<f32>{};
    auto runtime_config = RuntimeConfig{
        .window_title = "Realtime GPU SPH",
        .initial_width = 1440u,
        .initial_height = 920u,
        .clear_color = Color{0.17f, 0.19f, 0.20f, 1.0f},
    };

    for (auto i = 1; i < argc; ++i)
    {
        const auto arg = std::string_view{argv[i]};
        if (arg == "--particles" && i + 1 < argc)
        {
            settings.target_particles =
                std::clamp(parse_u32(argv[++i], k_default_particles), 1'000u, 50'000u);
        }
        else if (arg == "--paused")
        {
            settings.paused = true;
        }
        else if (arg == "--smoke-frames" && i + 1 < argc)
        {
            runtime_config.smoke_frames = parse_u32(argv[++i], 0u);
        }
        else if (arg == "--screenshot" && i + 1 < argc)
        {
            runtime_config.screenshot_path = argv[++i];
        }
        else if (arg == "--fixed-frame-dt" && i + 1 < argc)
        {
            fixed_frame_dt = std::clamp(parse_f32(argv[++i], 1.0f / 60.0f), 0.0f, 0.08f);
        }
        else if (arg == "--iterations" && i + 1 < argc)
        {
            settings.solver_iterations =
                std::clamp(parse_u32(argv[++i], settings.solver_iterations), 1u, 12u);
        }
        else if (arg == "--max-substeps" && i + 1 < argc)
        {
            settings.max_substeps_per_frame =
                std::clamp(parse_u32(argv[++i], settings.max_substeps_per_frame), 1u, 8u);
        }
        else if (arg == "--max-center-delta" && i + 1 < argc)
        {
            max_center_delta = std::max(0.0f, parse_f32(argv[++i], 0.0f));
        }
        else if (arg == "--max-speed" && i + 1 < argc)
        {
            max_speed = std::max(0.0f, parse_f32(argv[++i], 0.0f));
        }
        else if (arg == "--max-z-span" && i + 1 < argc)
        {
            max_z_span = std::max(0.0f, parse_f32(argv[++i], 0.0f));
        }
        else if (arg == "--scenario" && i + 1 < argc)
        {
            const auto scenario = scenario_from_text(argv[++i]);
            if (!scenario.has_value())
            {
                print_usage(argv[0]);
                return 2;
            }
            settings.scenario = *scenario;
        }
        else if (arg == "--solver-profile" && i + 1 < argc)
        {
            const auto mode = solver_profile_mode_from_text(argv[++i]);
            if (!mode.has_value())
            {
                print_usage(argv[0]);
                return 2;
            }
            settings.profile_mode = *mode;
        }
        else if (arg == "--neighbor-rebuild-interval" && i + 1 < argc)
        {
            settings.neighbor_rebuild_interval =
                std::clamp(parse_u32(argv[++i], settings.neighbor_rebuild_interval), 1u, 32u);
        }
        else if (arg == "--no-forced-neighbor-rebuild")
        {
            settings.force_neighbor_rebuilds = false;
        }
        else if (arg == "--no-rigid-bodies")
        {
            settings.rigid_bodies_enabled = false;
        }
        else if (arg == "--rigid-feedback" && i + 1 < argc)
        {
            settings.rigid_feedback_strength =
                std::clamp(parse_f32(argv[++i], settings.rigid_feedback_strength), 0.0f, 0.2f);
        }
        else if (arg == "--rigid-torque" && i + 1 < argc)
        {
            settings.rigid_torque_strength =
                std::clamp(parse_f32(argv[++i], settings.rigid_torque_strength), 0.0f, 8.0f);
        }
        else if (arg == "--rigid-buoyancy" && i + 1 < argc)
        {
            settings.rigid_buoyancy_strength =
                std::clamp(parse_f32(argv[++i], settings.rigid_buoyancy_strength), 0.0f, 4.0f);
        }
        else if (arg == "--print-stats")
        {
            print_stats = true;
        }
        else if (arg == "--fail-on-invalid")
        {
            fail_on_invalid = true;
        }
        else if (arg == "--transparent-screenshot")
        {
            runtime_config.transparent_screenshot = true;
        }
        else if (arg == "--hide-ui")
        {
            runtime_config.hide_ui = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            print_usage(argv[0]);
            return 2;
        }
    }

    apply_scenario_defaults(settings);

    auto runtime = Runtime{runtime_config};
    auto app = RealtimeSphApp{settings};

    runtime.initialize();
    app.setup(runtime);
    const auto run_started = std::chrono::steady_clock::now();
    auto rendered_frames = 0u;
    while (auto* frame = runtime.begin_frame())
    {
        app.update(*frame, fixed_frame_dt.value_or(frame->dt_seconds));
        if (runtime.ui_visible())
        {
            runtime.draw_runtime_ui();
            app.draw_ui(*frame);
        }
        runtime.render_shadow_pass();
        runtime.begin_main_pass();
        runtime.render_draw_list();
        app.render(*frame);
        runtime.render_imgui();
        runtime.end_main_pass();
        runtime.end_frame();
        ++rendered_frames;
    }
    const auto elapsed =
        std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - run_started);
    app.finalize_gpu_results();
    const auto stats = app.final_stats();
    const auto com = center_of_mass_from_stats(stats);
    const auto center_delta = com - app.initial_center();
    const auto center_delta_length = glm::length(center_delta);
    const auto min_pos = min_position_from_stats(stats);
    const auto max_pos = max_position_from_stats(stats);
    const auto span = max_pos - min_pos;
    const auto inv_count = stats.active == 0u ? 0.0 : 1.0 / static_cast<double>(stats.active);
    const auto mean_velocity = Vec3{
        static_cast<f32>(static_cast<double>(stats.sum_velocity_x_milli) * 0.001 * inv_count),
        static_cast<f32>(static_cast<double>(stats.sum_velocity_y_milli) * 0.001 * inv_count),
        static_cast<f32>(static_cast<double>(stats.sum_velocity_z_milli) * 0.001 * inv_count),
    };
    const auto max_speed_value = static_cast<double>(stats.max_speed_milli) / 1000.0;
    const auto gpu_profile = app.last_gpu_profile();
    if (print_stats)
    {
        const auto fps = elapsed.count() <= 0.0
                             ? 0.0
                             : 1000.0 * static_cast<double>(rendered_frames) / elapsed.count();
        std::cout << "scenario=" << scenario_cli_name(settings.scenario)
                  << " solver_profile=" << solver_profile_mode_name(settings.profile_mode)
                  << " neighbor_rebuild_interval=" << settings.neighbor_rebuild_interval
                  << " forced_neighbor_rebuild_enabled="
                  << (settings.force_neighbor_rebuilds ? 1 : 0)
                  << " rigid_bodies_enabled=" << (settings.rigid_bodies_enabled ? 1 : 0)
                  << " rigid_torque=" << settings.rigid_torque_strength
                  << " rigid_buoyancy=" << settings.rigid_buoyancy_strength
                  << " frames=" << rendered_frames
                  << " sim_steps=" << static_cast<unsigned long long>(app.sim_step_count())
                  << " wall_ms=" << elapsed.count() << " fps=" << fps
                  << " frame_ms=" << (fps <= 0.0 ? 0.0 : 1000.0 / fps)
                  << " gpu_step_ms=" << app.last_gpu_sim_ms() << " particles=" << stats.active
                  << " gpu_reset_ms=" << gpu_profile.reset_ms
                  << " gpu_predict_ms=" << gpu_profile.predict_ms
                  << " gpu_build_grid_ms=" << gpu_profile.build_grid_ms
                  << " gpu_build_neighbors_ms=" << gpu_profile.build_neighbors_ms
                  << " gpu_lambda_ms=" << gpu_profile.lambda_ms
                  << " gpu_delta_ms=" << gpu_profile.delta_ms
                  << " gpu_apply_delta_ms=" << gpu_profile.apply_delta_ms
                  << " gpu_velocity_ms=" << gpu_profile.velocity_ms
                  << " gpu_neighbor_scan_lambda_ms=" << gpu_profile.neighbor_scan_lambda_ms
                  << " gpu_neighbor_scan_delta_ms=" << gpu_profile.neighbor_scan_delta_ms
                  << " gpu_profile_iterations=" << gpu_profile.iterations
                  << " neighbor_rebuilds="
                  << static_cast<unsigned long long>(app.neighbor_rebuild_count())
                  << " scheduled_neighbor_rebuilds="
                  << static_cast<unsigned long long>(app.scheduled_neighbor_rebuild_count())
                  << " forced_neighbor_rebuilds="
                  << static_cast<unsigned long long>(app.forced_neighbor_rebuild_count())
                  << " rigid_contacts=" << app.last_rigid_contact_count()
                  << " rigid_max_angular_speed=" << app.max_rigid_angular_speed()
                  << " escaped=" << stats.escaped << " overflow=" << stats.overflow
                  << " nan=" << stats.nan_count << " invalid_events=" << stats.invalid_events
                  << " overflow_events=" << stats.overflow_events
                  << " observed_invalid=" << (app.observed_invalid_state() ? 1 : 0)
                  << " max_neighbors=" << stats.max_neighbors << " max_speed=" << max_speed_value
                  << " center=(" << static_cast<double>(com.x) << "," << static_cast<double>(com.y)
                  << "," << static_cast<double>(com.z) << ")"
                  << " center_delta=(" << static_cast<double>(center_delta.x) << ","
                  << static_cast<double>(center_delta.y) << ","
                  << static_cast<double>(center_delta.z) << ")"
                  << " center_delta_len=" << static_cast<double>(center_delta_length)
                  << " bounds_min=(" << static_cast<double>(min_pos.x) << ","
                  << static_cast<double>(min_pos.y) << "," << static_cast<double>(min_pos.z) << ")"
                  << " bounds_max=(" << static_cast<double>(max_pos.x) << ","
                  << static_cast<double>(max_pos.y) << "," << static_cast<double>(max_pos.z) << ")"
                  << " span=(" << static_cast<double>(span.x) << "," << static_cast<double>(span.y)
                  << "," << static_cast<double>(span.z) << ")"
                  << " mean_velocity=(" << static_cast<double>(mean_velocity.x) << ","
                  << static_cast<double>(mean_velocity.y) << ","
                  << static_cast<double>(mean_velocity.z) << ")";
        std::cout << "\n";
    }
    auto exit_code = 0;
    if (fail_on_invalid
        && (app.observed_invalid_state() || stats.escaped != 0u || stats.overflow != 0u
            || stats.nan_count != 0u || stats.invalid_events != 0u))
    {
        exit_code = 3;
    }
    if (max_center_delta.has_value() && center_delta_length > *max_center_delta)
    {
        std::cerr << "center delta " << static_cast<double>(center_delta_length)
                  << " exceeded limit " << static_cast<double>(*max_center_delta) << "\n";
        exit_code = exit_code == 0 ? 4 : exit_code;
    }
    if (max_speed.has_value() && max_speed_value > static_cast<double>(*max_speed))
    {
        std::cerr << "max speed " << max_speed_value << " exceeded limit "
                  << static_cast<double>(*max_speed) << "\n";
        exit_code = exit_code == 0 ? 5 : exit_code;
    }
    if (max_z_span.has_value() && span.z > *max_z_span)
    {
        std::cerr << "z span " << static_cast<double>(span.z) << " exceeded limit "
                  << static_cast<double>(*max_z_span) << "\n";
        exit_code = exit_code == 0 ? 6 : exit_code;
    }
    app.shutdown(runtime);
    return exit_code;
}
}  // namespace ds_vk_app::realtime_sph
