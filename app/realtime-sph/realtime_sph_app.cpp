#include "app/realtime-sph/realtime_sph_app.hpp"

#include "ds_vk/math.hpp"
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

struct GpuStats
{
    u32 escaped{};
    u32 overflow{};
    u32 active{};
    u32 nan_count{};
    u32 max_neighbors{};
    u32 max_speed_milli{};
    u32 reserved0{};
    u32 reserved1{};
};

static_assert(sizeof(GpuParticle) == 64zu);
static_assert(sizeof(GpuSimParams) == 128zu);
static_assert(sizeof(GpuStats) == 32zu);

struct FrameResources
{
    Buffer sim_params{};
    Buffer render_params{};
    Buffer stats{};
    VkDescriptorSet compute_descriptor{VK_NULL_HANDLE};
    VkDescriptorSet render_descriptor{VK_NULL_HANDLE};
};

struct SimulationSettings
{
    u32 target_particles{k_default_particles};
    f32 particle_radius{0.025f};
    f32 support_radius{0.10f};
    f32 fixed_dt{1.0f / 120.0f};
    u32 solver_iterations{4u};
    u32 max_substeps_per_frame{4u};
    f32 simulation_speed{1.0f};
    f32 speed_color_mix{0.55f};
    f32 render_radius_scale{1.0f};
    bool paused{};
};

struct Domain
{
    Vec3 min{-3.5f, -2.2f, 0.0f};
    Vec3 max{3.7f, 2.4f, 2.8f};
};

[[nodiscard]] auto ceil_div(u32 value, u32 divisor) noexcept -> u32
{
    return (value + divisor - 1u) / divisor;
}

[[nodiscard]] auto byte_size(usize count, usize element_size) -> VkDeviceSize
{
    return static_cast<VkDeviceSize>(count * element_size);
}

auto check_vk(VkResult result, std::string_view label) -> void
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::string{label} + " failed with VkResult "
                                 + std::to_string(static_cast<int>(result)));
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
        allocation_info.flags =
            VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
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
                const auto offset = spacing * Vec3{
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
    constexpr Vec3 block_min{-2.55f, -0.95f, 0.08f};
    constexpr u32 nx{28u};
    constexpr u32 ny{38u};
    const auto nz = ceil_div(settings.target_particles, nx * ny);
    auto particles = std::vector<GpuParticle>{};
    particles.reserve(settings.target_particles);

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
                particles.push_back(GpuParticle{
                    .position_radius = Vec4{position, settings.particle_radius},
                    .previous_density = Vec4{position, 0.0f},
                    .velocity_lambda = Vec4{0.0f},
                    .delta_neighbors = Vec4{0.0f},
                });
            }
        }
    }
    return particles;
}

class RealtimeSphApp
{
  public:
    explicit RealtimeSphApp(SimulationSettings settings) : settings_(settings)
    {
    }

    auto setup(Runtime& runtime) -> void
    {
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
            reset_requested_ = false;
        }

        accumulator_ += std::min(dt_seconds, 0.08f) * settings_.simulation_speed;
        auto substeps = 0u;
        while (!settings_.paused && accumulator_ >= settings_.fixed_dt
               && substeps < settings_.max_substeps_per_frame)
        {
            write_sim_params(frame, settings_.fixed_dt);
            record_simulation_step(frame);
            accumulator_ -= settings_.fixed_dt;
            ++substeps;
            ++sim_step_index_;
        }
        if (substeps == settings_.max_substeps_per_frame)
        {
            accumulator_ = std::min(accumulator_, settings_.fixed_dt);
        }

        configure_scene_draw(frame);
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

    auto draw_ui(FrameContext&) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{18.0f, 280.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{390.0f, 360.0f}, ImGuiCond_Once);
        if (ImGui::Begin("Realtime SPH"))
        {
            ImGui::Checkbox("Paused", &settings_.paused);
            if (ImGui::Button("Reset"))
            {
                reset_requested_ = true;
            }
            ImGui::SameLine();
            ImGui::Text("%u particles", particle_count_);
            ImGui::SliderFloat("Simulation speed", &settings_.simulation_speed, 0.0f, 2.0f, "%.2f");
            auto iterations = static_cast<int>(settings_.solver_iterations);
            if (ImGui::SliderInt("PBF iterations", &iterations, 1, 8))
            {
                settings_.solver_iterations = static_cast<u32>(std::max(1, iterations));
            }
            ImGui::SliderFloat("Render radius", &settings_.render_radius_scale, 0.55f, 1.6f, "%.2f");
            ImGui::SliderFloat("Speed coloring", &settings_.speed_color_mix, 0.0f, 1.0f, "%.2f");
            ImGui::Separator();
            ImGui::Text("GPU stats");
            ImGui::Text("step: %llu", static_cast<unsigned long long>(sim_step_index_));
            ImGui::Text("grid overflow: %u", last_stats_.overflow);
            ImGui::Text("escaped/teleported: %u", last_stats_.escaped);
            ImGui::Text("max neighbors: %u", last_stats_.max_neighbors);
            ImGui::Text("max speed: %.3f", static_cast<double>(last_stats_.max_speed_milli) / 1000.0);
            ImGui::Text("nan count: %u", last_stats_.nan_count);
        }
        ImGui::End();
    }

    auto shutdown(Runtime&) noexcept -> void
    {
        destroy();
    }

  private:
    auto ensure_resources(FrameContext& frame) -> void
    {
        if (device_ == VK_NULL_HANDLE)
        {
            device_ = frame.device;
            allocator_ = frame.allocator;
        }
        if (!resources_ready_)
        {
            create_static_resources(frame);
            create_frame_resources(frame.swapchain_image_count);
            create_descriptor_resources();
            create_compute_pipelines();
            reset_particles();
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

    auto create_static_resources(FrameContext& frame) -> void
    {
        grid_size_ = make_grid_size(domain_, settings_.support_radius);
        cell_count_ = flatten_grid_size(grid_size_);
        particle_capacity_ = settings_.target_particles;

        particle_buffer_ = create_buffer(
            frame.allocator,
            byte_size(particle_capacity_, sizeof(GpuParticle)),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true,
            "realtime sph particle buffer"
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
            const auto zero = GpuStats{};
            write_mapped(allocator_, frame.stats, zero);
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
        };
        VkDescriptorSetLayoutCreateInfo compute_layout_info{};
        compute_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        compute_layout_info.bindingCount = static_cast<u32>(compute_bindings.size());
        compute_layout_info.pBindings = compute_bindings.data();
        check_vk(
            vkCreateDescriptorSetLayout(device_, &compute_layout_info, nullptr, &compute_set_layout_),
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
                .descriptorCount = frame_count * 7u,
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

        auto compute_layouts = std::vector<VkDescriptorSetLayout>(frames_.size(), compute_set_layout_);
        auto render_layouts = std::vector<VkDescriptorSetLayout>(frames_.size(), render_set_layout_);
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
            VkDescriptorBufferInfo{.buffer = particle_buffer_.handle, .offset = 0, .range = particle_buffer_.size},
            VkDescriptorBufferInfo{.buffer = cell_counts_buffer_.handle, .offset = 0, .range = cell_counts_buffer_.size},
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
            VkDescriptorBufferInfo{.buffer = particle_buffer_.handle, .offset = 0, .range = particle_buffer_.size},
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
            device_,
            static_cast<u32>(compute_writes.size()),
            compute_writes.data(),
            0,
            nullptr
        );
        vkUpdateDescriptorSets(
            device_,
            static_cast<u32>(render_writes.size()),
            render_writes.data(),
            0,
            nullptr
        );
    }

    auto create_compute_pipelines() -> void
    {
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &compute_set_layout_;
        check_vk(
            vkCreatePipelineLayout(device_, &layout_info, nullptr, &compute_pipeline_layout_),
            "vkCreatePipelineLayout compute"
        );

        reset_pipeline_ = create_compute_pipeline("realtime_sph_reset.comp.spv");
        predict_pipeline_ = create_compute_pipeline("realtime_sph_predict.comp.spv");
        build_grid_pipeline_ = create_compute_pipeline("realtime_sph_build_grid.comp.spv");
        lambda_pipeline_ = create_compute_pipeline("realtime_sph_lambda.comp.spv");
        delta_pipeline_ = create_compute_pipeline("realtime_sph_delta.comp.spv");
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
            vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
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

        const auto vert = create_shader_module(device_, shader_path("realtime_sph_particles.vert.spv"));
        const auto frag = create_shader_module(device_, shader_path("realtime_sph_particles.frag.spv"));
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
            vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &render_pipeline_),
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
        particle_count_ = static_cast<u32>(particles.size());
        rest_density_ = lattice_rest_density(2.0f * settings_.particle_radius, settings_.support_radius);
        write_mapped_bytes(
            allocator_,
            particle_buffer_,
            particles.data(),
            byte_size(particles.size(), sizeof(GpuParticle))
        );
        last_stats_ = {};
        accumulator_ = 0.0f;
        sim_step_index_ = 0u;
    }

    auto read_stats(FrameContext& frame) -> void
    {
        if (frames_.empty())
        {
            return;
        }
        auto& stats_buffer = frames_[frame_slot_for(frame)].stats;
        check_vk(
            vmaInvalidateAllocation(allocator_, stats_buffer.allocation, 0, sizeof(GpuStats)),
            "vmaInvalidateAllocation stats"
        );
        std::memcpy(&last_stats_, stats_buffer.mapped, sizeof(GpuStats));
    }

    auto write_sim_params(FrameContext& frame, f32 dt) -> void
    {
        const auto params = GpuSimParams{
            .domain_min_dt = Vec4{domain_.min, dt},
            .domain_max_radius = Vec4{domain_.max, settings_.particle_radius},
            .gravity_rest_density = Vec4{0.0f, 0.0f, -9.81f, rest_density_},
            .kernel_poly6_spiky =
                Vec4{
                    poly6_constant(settings_.support_radius),
                    spiky_gradient_constant(settings_.support_radius),
                    settings_.support_radius,
                    1.0f,
                },
            .pbf_params =
                Vec4{
                    120.0f,
                    0.0010f,
                    0.30f * settings_.support_radius,
                    4.0f,
                },
            .viscosity_vorticity = Vec4{0.0f, 0.08f, 0.995f, 9.0f},
            .counts = glm::uvec4{particle_count_, cell_count_, k_max_particles_per_cell, settings_.solver_iterations},
            .grid_size = glm::uvec4{grid_size_, 0u},
        };
        write_mapped(allocator_, frames_[frame_slot_for(frame)].sim_params, params);
    }

    auto write_render_params(FrameContext& frame) -> void
    {
        const auto aspect =
            frame.extent.height == 0u ? 1.0f : static_cast<f32>(frame.extent.width) / static_cast<f32>(frame.extent.height);
        const auto params = GpuRenderParams{
            .view_projection = frame.camera.view_projection_matrix(aspect),
            .view = frame.camera.view_matrix(),
            .projection = frame.camera.projection_matrix(aspect),
            .camera_right = Vec4{frame.camera.right(), 0.0f},
            .camera_up = Vec4{frame.camera.up(), 0.0f},
            .camera_forward = Vec4{glm::normalize(frame.camera.pivot() - frame.camera.position()), 0.0f},
            .base_color = Vec4{0.05f, 0.58f, 0.85f, 1.0f},
            .light_direction = Vec4{-0.42f, -0.32f, -0.84f, 0.0f},
            .options =
                Vec4{
                    settings_.render_radius_scale,
                    5.0f,
                    settings_.speed_color_mix,
                    0.0f,
                },
        };
        write_mapped(allocator_, frames_[frame_slot_for(frame)].render_params, params);
    }

    auto record_simulation_step(FrameContext& frame) -> void
    {
        const auto frame_slot = frame_slot_for(frame);
        auto* cmd = frame.command_buffer;
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
        bind_and_dispatch(cmd, predict_pipeline_, particle_count_);
        compute_barrier(cmd);
        bind_and_dispatch(cmd, build_grid_pipeline_, particle_count_);
        compute_barrier(cmd);
        for (auto i = 0u; i < settings_.solver_iterations; ++i)
        {
            bind_and_dispatch(cmd, lambda_pipeline_, particle_count_);
            compute_barrier(cmd);
            bind_and_dispatch(cmd, delta_pipeline_, particle_count_);
            compute_barrier(cmd);
        }
        bind_and_dispatch(cmd, velocity_pipeline_, particle_count_);
        compute_to_render_barrier(cmd);
    }

    auto bind_and_dispatch(VkCommandBuffer cmd, VkPipeline pipeline, u32 item_count) const -> void
    {
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
        frame.draw.set_environment(EnvironmentConfig{
            .background_color = Color{0.17f, 0.19f, 0.20f, 1.0f},
            .background_top_color = Color{0.43f, 0.54f, 0.62f, 1.0f},
            .gradient_background = true,
        });
        frame.draw.set_ambient_light(Color{0.16f, 0.18f, 0.20f, 1.0f});
        frame.draw.directional_light(DirectionalLightConfig{
            .direction = Vec3{-0.42f, -0.32f, -0.84f},
            .color = Color{0.92f, 0.96f, 1.0f, 1.0f},
            .intensity = 1.65f,
            .shadow = {.enabled = false},
        });
        (void)viz::draw_aabb(
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

    [[nodiscard]] auto frame_slot_for(FrameContext& frame) const -> usize
    {
        return static_cast<usize>(frame.swapchain_image_index) % std::max<usize>(1zu, frames_.size());
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
            destroy_buffer(allocator_, frame.sim_params);
            destroy_buffer(allocator_, frame.render_params);
            destroy_buffer(allocator_, frame.stats);
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
              &lambda_pipeline_,
              &delta_pipeline_,
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
        destroy_buffer(allocator_, cell_particles_buffer_);
        destroy_buffer(allocator_, cell_counts_buffer_);
        destroy_buffer(allocator_, particle_buffer_);
        resources_ready_ = false;
    }

    SimulationSettings settings_{};
    Domain domain_{};
    VkDevice device_{VK_NULL_HANDLE};
    VmaAllocator allocator_{VK_NULL_HANDLE};
    Buffer particle_buffer_{};
    Buffer cell_counts_buffer_{};
    Buffer cell_particles_buffer_{};
    std::vector<FrameResources> frames_{};
    VkDescriptorSetLayout compute_set_layout_{VK_NULL_HANDLE};
    VkDescriptorSetLayout render_set_layout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    VkPipelineLayout compute_pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline reset_pipeline_{VK_NULL_HANDLE};
    VkPipeline predict_pipeline_{VK_NULL_HANDLE};
    VkPipeline build_grid_pipeline_{VK_NULL_HANDLE};
    VkPipeline lambda_pipeline_{VK_NULL_HANDLE};
    VkPipeline delta_pipeline_{VK_NULL_HANDLE};
    VkPipeline velocity_pipeline_{VK_NULL_HANDLE};
    VkPipelineLayout render_pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline render_pipeline_{VK_NULL_HANDLE};
    VkRenderPass render_pass_{VK_NULL_HANDLE};
    glm::uvec3 grid_size_{};
    u32 cell_count_{};
    u32 particle_capacity_{};
    u32 particle_count_{};
    f32 rest_density_{1.0f};
    f32 accumulator_{};
    u64 sim_step_index_{};
    GpuStats last_stats_{};
    bool resources_ready_{};
    bool reset_requested_{};
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

auto print_usage(const char* program) -> void
{
    std::cerr << "usage: " << program
              << " [--particles N] [--paused] [--smoke-frames N] [--screenshot PATH]"
                 " [--transparent-screenshot] [--hide-ui]\n";
}

}  // namespace

auto run_realtime_sph_app(int argc, char** argv) -> int
{
    auto settings = SimulationSettings{};
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
            settings.target_particles = std::clamp(parse_u32(argv[++i], k_default_particles), 1'000u, 50'000u);
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

    auto runtime = Runtime{runtime_config};
    auto app = RealtimeSphApp{settings};

    runtime.initialize();
    app.setup(runtime);
    while (auto* frame = runtime.begin_frame())
    {
        app.update(*frame, frame->dt_seconds);
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
    }
    app.shutdown(runtime);
    return 0;
}
}  // namespace ds_vk_app::realtime_sph
