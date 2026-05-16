#include "ds_vk/runtime.hpp"

#include "ds_vk/math.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <iostream>
#include <numbers>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ds_vk
{
namespace
{
constexpr auto k_swapchain_image_usage =
    VkImageUsageFlags{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
constexpr auto k_max_material_textures = u32{15};
constexpr auto k_max_lights = u32{16};
constexpr auto k_default_texture_index = 0u;

struct Buffer
{
    VkBuffer handle{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    void* mapped{};
    VkDeviceSize capacity{};
};

struct MeshResource
{
    Buffer vertices{};
    Buffer indices{};
    u32 vertex_count{};
    u32 index_count{};
};

struct TextureResource
{
    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};
    u32 width{};
    u32 height{};
    VkFormat format{VK_FORMAT_UNDEFINED};
};

struct ShadowMap
{
    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkSampler sampler{VK_NULL_HANDLE};
    VkRenderPass render_pass{VK_NULL_HANDLE};
    VkFramebuffer framebuffer{VK_NULL_HANDLE};
    u32 resolution{};
    VkFormat format{VK_FORMAT_UNDEFINED};
};

struct GpuMaterial
{
    Vec4 base_color{1.0f};
    Vec4 emissive_color{0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 pbr_params{0.0f, 0.55f, 1.0f, 0.0f};
    Vec4 texture_params{0.0f};
    Vec4 render_params{1.0f};
    Vec4 debug_color{1.0f, 0.0f, 1.0f, 0.85f};
    Vec4 debug_params{0.0f};
    Vec4 debug_params2{0.0f};
    Vec4 camera_position{0.0f, 0.0f, 1.0f, 1.0f};
    Vec4 camera_forward{0.0f, -1.0f, 0.0f, 0.0f};
};

struct GpuLight
{
    Vec4 position_range{};
    Vec4 direction_type{};
    Vec4 color_intensity{1.0f};
    Vec4 spot_shadow{};
};

struct GpuLighting
{
    Vec4 ambient_light_count{};
    Mat4 shadow_view_projection{1.0f};
    Vec4 shadow_params{};
    Vec4 environment_params{};
    std::array<GpuLight, k_max_lights> lights{};
};

struct DepthAttachment
{
    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
};

struct SwapchainCapture
{
    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkDeviceSize size{};
    u32 width{};
    u32 height{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    std::filesystem::path path;
    bool transparent_background{};
};

struct MeshPushConstants
{
    Mat4 view_projection{1.0f};
    Mat4 model{1.0f};
};

struct DebugPushConstants
{
    Mat4 view_projection{1.0f};
    Vec4 camera_position{0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 camera_right{k_axis_x, 0.0f};
};

struct EnvironmentPushConstants
{
    Mat4 inverse_view_projection{1.0f};
    Vec4 camera_position{0.0f, 0.0f, 0.0f, 1.0f};
    Vec4 params{};
};

// clang-format off
static_assert(sizeof(MeshPushConstants)              == 128zu);
static_assert(sizeof(GpuMaterial)                    == 160zu);
static_assert(offsetof(GpuMaterial, emissive_color)  == 16zu);
static_assert(offsetof(GpuMaterial, pbr_params)      == 32zu);
static_assert(offsetof(GpuMaterial, texture_params)  == 48zu);
static_assert(offsetof(GpuMaterial, render_params)   == 64zu);
static_assert(offsetof(GpuMaterial, debug_color)     == 80zu);
static_assert(offsetof(GpuMaterial, debug_params)    == 96zu);
static_assert(offsetof(GpuMaterial, debug_params2)   == 112zu);
static_assert(offsetof(GpuMaterial, camera_position) == 128zu);
static_assert(offsetof(GpuMaterial, camera_forward)  == 144zu);
static_assert(sizeof(GpuLight)                       == 64zu);
static_assert(sizeof(GpuLighting)                    == 112zu + static_cast<usize>(k_max_lights) * sizeof(GpuLight));
static_assert(sizeof(DebugPushConstants)             == 96zu);
static_assert(sizeof(EnvironmentPushConstants)       == 96zu);
// clang-format on
constexpr auto k_required_push_constant_bytes = std::max(
    {sizeof(MeshPushConstants), sizeof(DebugPushConstants), sizeof(EnvironmentPushConstants)}
);

[[nodiscard]] auto material_base_color_texture_index(const Material& material) noexcept -> u32
{
    if (material.textures.base_color.valid()
        and material.textures.base_color.id < k_max_material_textures)
    {
        return material.textures.base_color.id;
    }
    return k_default_texture_index;
}

auto to_gpu_material(
    const Material& material,
    const MeshRenderMask& mask,
    const MeshDebugConfig& debug,
    ObjectId object_id,
    f32 time,
    Vec3 camera_position,
    Vec3 camera_forward
) noexcept -> GpuMaterial
{
    auto debug_mode = debug.mode;
    if (debug.selected and debug_mode == MeshDebugMode::none)
    {
        debug_mode = MeshDebugMode::selected_pulse;
    }
    return GpuMaterial{
        .base_color = to_vec4(material.base_color),
        .emissive_color = to_vec4(material.emissive_color),
        .pbr_params =
            Vec4{
                material.metallic,
                material.roughness,
                material.ambient_occlusion,
                0.0f,
            },
        .texture_params =
            Vec4{
                static_cast<f32>(material_base_color_texture_index(material)),
                material.textures.base_color.valid() ? 1.0f : 0.0f,
                0.0f,
                0.0f,
            },
        .render_params =
            Vec4{
                mask.light_receiver ? 1.0f : 0.0f,
                mask.shadow_consumer ? 1.0f : 0.0f,
                mask.visible_to_camera ? 1.0f : 0.0f,
                mask.shadow_producer ? 1.0f : 0.0f,
            },
        .debug_color = to_vec4(debug.color),
        .debug_params =
            Vec4{
                static_cast<f32>(debug_mode),
                debug.scalar,
                debug.scalar_range.x,
                debug.scalar_range.y,
            },
        .debug_params2 =
            Vec4{
                time,
                object_id.valid() ? static_cast<f32>(object_id.value) : 0.0f,
                debug.selected ? 1.0f : 0.0f,
                0.0f,
            },
        .camera_position = Vec4{camera_position, 1.0f},
        .camera_forward = Vec4{camera_forward, 0.0f},
    };
}

[[nodiscard]] auto spot_angle_scale(const LightConfig& light) noexcept -> f32
{
    const auto inner = std::clamp(light.inner_cone_angle, 0.0f, std::numbers::pi_v<f32> * 0.49f);
    const auto outer =
        std::clamp(light.outer_cone_angle, inner + 0.001f, std::numbers::pi_v<f32> * 0.5f);
    return 1.0f / std::max(0.001f, std::cos(inner) - std::cos(outer));
}

[[nodiscard]] auto spot_angle_offset(const LightConfig& light) noexcept -> f32
{
    const auto inner = std::clamp(light.inner_cone_angle, 0.0f, std::numbers::pi_v<f32> * 0.49f);
    const auto outer =
        std::clamp(light.outer_cone_angle, inner + 0.001f, std::numbers::pi_v<f32> * 0.5f);
    return -std::cos(outer) * spot_angle_scale(light);
}

[[nodiscard]] auto to_gpu_light(const LightConfig& light, const bool casts_active_shadow) noexcept
    -> GpuLight
{
    const auto color = to_vec4(light.color);
    return GpuLight{
        .position_range = Vec4{light.position, std::max(0.0f, light.range)},
        .direction_type =
            Vec4{
                normalize_or(light.direction, -k_axis_z),
                static_cast<f32>(light.type),
            },
        .color_intensity =
            Vec4{
                color.r,
                color.g,
                color.b,
                std::max(0.0f, light.intensity),
            },
        .spot_shadow = Vec4{
            spot_angle_scale(light),
            spot_angle_offset(light),
            casts_active_shadow ? 1.0f : 0.0f,
            light.shadow.strength,
        },
    };
}

[[nodiscard]] auto shadow_supported_by_light(const LightConfig& light) noexcept -> bool
{
    if (!light.shadow.enabled)
    {
        return false;
    }

    switch (light.type)
    {
        case LightType::directional:
        case LightType::spot:
            return true;
        case LightType::radial:
            return false;
    }
    return false;
}

[[nodiscard]] auto shadow_light_index(std::span<const LightConfig> lights) noexcept -> u32
{
    for (auto i = 0u; i < std::min(static_cast<u32>(lights.size()), k_max_lights); ++i)
    {
        if (shadow_supported_by_light(lights[i]))
        {
            return i;
        }
    }
    return std::numeric_limits<u32>::max();
}

[[nodiscard]] auto light_view_matrix(Vec3 position, Vec3 direction) noexcept -> Mat4
{
    const auto forward = normalize_or(direction, -k_axis_z);
    const auto up_hint = std::abs(glm::dot(forward, k_axis_z)) > 0.92f ? k_axis_y : k_axis_z;
    return glm::lookAt(position, position + forward, up_hint);
}

[[nodiscard]] auto light_projection_matrix(const LightConfig& light, const Camera& camera) noexcept
    -> Mat4
{
    switch (light.type)
    {
        case LightType::spot:
            {
                const auto outer = std::clamp(
                    light.outer_cone_angle, glm::radians(1.0f), std::numbers::pi_v<f32> * 0.49f
                );
                const auto far_plane = light.shadow.far_plane > light.shadow.near_plane
                                           ? light.shadow.far_plane
                                           : std::max(light.shadow.near_plane + 0.1f, light.range);
                auto projection =
                    glm::perspective(2.0f * outer, 1.0f, light.shadow.near_plane, far_plane);
                projection[1][1] *= -1.0f;
                return projection;
            }
        case LightType::directional:
        case LightType::radial:
            {
                const auto extent = std::max(0.1f, light.shadow.ortho_extent);
                auto projection = glm::orthoRH_ZO(
                    -extent,
                    extent,
                    -extent,
                    extent,
                    light.shadow.near_plane,
                    light.shadow.far_plane
                );
                projection[1][1] *= -1.0f;
                (void) camera;
                return projection;
            }
    }
    return Mat4{1.0f};
}

[[nodiscard]] auto
light_view_projection_matrix(const LightConfig& light, const Camera& camera) noexcept -> Mat4
{
    switch (light.type)
    {
        case LightType::spot:
            return light_projection_matrix(light, camera)
                   * light_view_matrix(light.position, light.direction);
        case LightType::directional:
        case LightType::radial:
            {
                const auto direction = normalize_or(light.direction, -k_axis_z);
                const auto depth = std::max(light.shadow.far_plane - light.shadow.near_plane, 1.0f);
                const auto target = camera.pivot();
                const auto position = target - direction * (0.5f * depth);
                return light_projection_matrix(light, camera)
                       * light_view_matrix(position, direction);
            }
    }
    return Mat4{1.0f};
}

[[nodiscard]] auto build_gpu_lighting(
    std::span<const LightConfig> lights,
    Color ambient_light,
    const EnvironmentConfig& environment,
    const Camera& camera,
    u32 shadow_index,
    u32 shadow_resolution
) noexcept -> GpuLighting
{
    auto lighting = GpuLighting{
        .ambient_light_count =
            Vec4{
                ambient_light.r(),
                ambient_light.g(),
                ambient_light.b(),
                static_cast<f32>(std::min(static_cast<u32>(lights.size()), k_max_lights)),
            },
        .environment_params = Vec4{
            environment.texture.valid() ? environment.lighting_intensity : 0.0f,
            environment.texture.valid() and environment.visible_to_camera
                ? environment.background_intensity
                : 0.0f,
            environment.rotation_radians,
            environment.texture.valid() and environment.texture.id < k_max_material_textures
                ? static_cast<f32>(environment.texture.id)
                : -1.0f,
        },
    };
    if (shadow_index < lights.size())
    {
        const auto& shadow_light = lights[shadow_index];
        lighting.shadow_view_projection = light_view_projection_matrix(shadow_light, camera);
        lighting.shadow_params = Vec4{
            1.0f,
            shadow_light.shadow.bias,
            shadow_light.shadow.strength,
            static_cast<f32>(std::max(1u, shadow_resolution)),
        };
    }
    for (auto i = 0u; i < std::min(static_cast<u32>(lights.size()), k_max_lights); ++i)
    {
        lighting.lights[i] = to_gpu_light(lights[i], i == shadow_index);
    }
    return lighting;
}

auto check_vk_result(const VkResult result) -> void
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(
            std::format("Vulkan call failed with VkResult {}", static_cast<int>(result))
        );
    }
}

template <typename T>
auto data_byte_size(std::span<const T> values) noexcept -> VkDeviceSize
{
    return static_cast<VkDeviceSize>(values.size()) * static_cast<VkDeviceSize>(sizeof(T));
}

template <typename T>
auto data_byte_size(const std::vector<T>& values) noexcept -> VkDeviceSize
{
    return data_byte_size(std::span<const T>{values.data(), values.size()});
}

auto layer_available(const char* name) -> bool
{
    auto count = 0u;
    (void) vkEnumerateInstanceLayerProperties(&count, nullptr);
    auto layers = std::vector<VkLayerProperties>(count);
    (void) vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::ranges::any_of(
        layers,
        [&](const VkLayerProperties& layer) -> bool
        { return std::strcmp(layer.layerName, name) == 0; }
    );
}

auto extension_available(const std::vector<VkExtensionProperties>& properties, const char* name)
    -> bool
{
    return std::ranges::any_of(
        properties,
        [&](const VkExtensionProperties& extension) -> bool
        { return std::strcmp(extension.extensionName, name) == 0; }
    );
}

auto read_shader_words(const std::filesystem::path& path) -> std::vector<u32>
{
    auto input = std::ifstream{path, std::ios::binary | std::ios::ate};
    if (!input)
    {
        throw std::runtime_error(std::format("failed to open shader: {}", path.string()));
    }
    const auto end = input.tellg();
    if (end <= 0 or (static_cast<u64>(end) % sizeof(u32)) != 0u)
    {
        throw std::runtime_error(std::format("shader has invalid SPIR-V size: {}", path.string()));
    }
    auto words = std::vector<u32>(static_cast<usize>(end) / sizeof(u32));
    input.seekg(0, std::ios::beg);
    input.read(
        reinterpret_cast<char*>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(u32))
    );
    if (!input)
    {
        throw std::runtime_error(std::format("failed to read shader: {}", path.string()));
    }
    return words;
}

auto create_shader_module(VkDevice device, const std::filesystem::path& path) -> VkShaderModule
{
    const auto words = read_shader_words(path);
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = words.size() * sizeof(u32);
    info.pCode = words.data();

    auto module = VkShaderModule{VK_NULL_HANDLE};
    check_vk_result(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

auto find_depth_format(VkPhysicalDevice physical_device) -> VkFormat
{
    const auto candidates = std::array{
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM,
    };
    for (const auto format : candidates)
    {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            != 0u)
        {
            return format;
        }
    }
    throw std::runtime_error("no supported depth format found");
}

}  // namespace

auto DrawList::clear() -> void
{
    mesh_commands_.clear();
    debug_segments_.clear();
    lights_.clear();
    ambient_light_ = Color{0.035f, 0.040f, 0.050f, 1.0f};
    environment_ = {};
}

auto DrawList::set_ambient_light(Color color) -> void
{
    ambient_light_ = color;
}

auto DrawList::draw_mesh(const MeshDrawConfig& config) -> void
{
    if (!config.mesh.valid() or config.debug.hidden
        or (!config.mask.visible_to_camera and !config.mask.shadow_producer))
    {
        return;
    }
    mesh_commands_.push_back(
        MeshDrawCommand{
            .mesh = config.mesh,
            .object_id = config.object_id,
            .transform = config.transform,
            .material = config.material,
            .mask = config.mask,
            .debug = config.debug,
        }
    );
}

auto DrawList::draw_basic_mesh(const BasicMeshDrawConfig& config) -> void
{
    draw_mesh(
        MeshDrawConfig{
            .mesh = config.mesh,
            .object_id = config.object_id,
            .transform = config.transform,
            .material = Material{.base_color = config.color},
            .mask = config.mask,
            .debug = config.debug,
        }
    );
}

auto DrawList::debug_line(const DebugLineConfig& config) -> void
{
    debug_segments_.push_back(
        DebugSegment{
            .start = config.start,
            .width = config.width,
            .end = config.end,
            .arrow_tip = 0.0f,
            .color = config.color,
        }
    );
}

auto DrawList::debug_arrow(const DebugArrowConfig& config) -> void
{
    debug_segments_.push_back(
        DebugSegment{
            .start = config.origin,
            .width = config.width,
            .end = config.origin + config.vector,
            .arrow_tip = 1.0f,
            .color = config.color,
        }
    );
}

auto DrawList::debug_sphere(const DebugSphereConfig& config) -> void
{
    const auto safe_radius = std::max(0.0f, config.radius);
    if (safe_radius <= 0.0f)
    {
        return;
    }

    const auto safe_segments = std::max(8u, config.segments);
    const auto safe_segments_f = static_cast<f32>(safe_segments);
    for (auto i = 0u; i < safe_segments; ++i)
    {
        const auto pi2 = 2.0f * std::numbers::pi_v<f32>;
        const auto t0 = pi2 * static_cast<f32>(i) / safe_segments_f;
        const auto t1 = pi2 * static_cast<f32>(i + 1u) / safe_segments_f;
        const auto c0 = std::cos(t0) * safe_radius;
        const auto s0 = std::sin(t0) * safe_radius;
        const auto c1 = std::cos(t1) * safe_radius;
        const auto s1 = std::sin(t1) * safe_radius;
        debug_line(
            DebugLineConfig{
                .start = config.center + Vec3{c0, s0, 0.0f},
                .end = config.center + Vec3{c1, s1, 0.0f},
                .color = config.color,
                .width = config.width,
            }
        );
        debug_line(
            DebugLineConfig{
                .start = config.center + Vec3{c0, 0.0f, s0},
                .end = config.center + Vec3{c1, 0.0f, s1},
                .color = config.color,
                .width = config.width,
            }
        );
        debug_line(
            DebugLineConfig{
                .start = config.center + Vec3{0.0f, c0, s0},
                .end = config.center + Vec3{0.0f, c1, s1},
                .color = config.color,
                .width = config.width,
            }
        );
    }
}

auto DrawList::add_light(const LightConfig& config) -> void
{
    if (!config.enabled)
    {
        return;
    }
    lights_.push_back(config);
}

auto DrawList::directional_light(const DirectionalLightConfig& config) -> void
{
    add_light(
        LightConfig{
            .type = LightType::directional,
            .direction = config.direction,
            .color = config.color,
            .intensity = config.intensity,
            .shadow = config.shadow,
            .enabled = config.enabled,
        }
    );
}

auto DrawList::radial_light(const RadialLightConfig& config) -> void
{
    add_light(
        LightConfig{
            .type = LightType::radial,
            .position = config.position,
            .color = config.color,
            .intensity = config.intensity,
            .range = config.range,
            .enabled = config.enabled,
        }
    );
}

auto DrawList::spot_light(const SpotLightConfig& config) -> void
{
    add_light(
        LightConfig{
            .type = LightType::spot,
            .position = config.position,
            .direction = config.direction,
            .color = config.color,
            .intensity = config.intensity,
            .range = config.range,
            .inner_cone_angle = config.inner_cone_angle,
            .outer_cone_angle = config.outer_cone_angle,
            .shadow = config.shadow,
            .enabled = config.enabled,
        }
    );
}

auto DrawList::set_environment(const EnvironmentConfig& config) -> void
{
    environment_ = config;
}

auto DrawList::mesh_commands() const noexcept -> std::span<const MeshDrawCommand>
{
    return std::span<const MeshDrawCommand>{mesh_commands_.data(), mesh_commands_.size()};
}

auto DrawList::debug_segments() const noexcept -> std::span<const DebugSegment>
{
    return std::span<const DebugSegment>{debug_segments_.data(), debug_segments_.size()};
}

auto DrawList::lights() const noexcept -> std::span<const LightConfig>
{
    return std::span<const LightConfig>{lights_.data(), lights_.size()};
}

auto DrawList::ambient_light() const noexcept -> Color
{
    return ambient_light_;
}

auto DrawList::environment() const noexcept -> const EnvironmentConfig&
{
    return environment_;
}

struct Runtime::Impl
{
    explicit Impl(RuntimeConfig config_in) : config(std::move(config_in))
    {
    }

    RuntimeConfig config{};
    Camera camera{};
    RuntimeStats stats{};
    f32 elapsed_seconds{};
    DescriptorIndexingSupport descriptor_indexing{};
    SDL_Window* window{};
    VkAllocationCallbacks* allocation_callbacks{};
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    u32 queue_family{std::numeric_limits<u32>::max()};
    VkQueue queue{VK_NULL_HANDLE};
    VmaAllocator vma_allocator{VK_NULL_HANDLE};
    VkDescriptorPool imgui_descriptor_pool{VK_NULL_HANDLE};
    VkPipelineCache pipeline_cache{VK_NULL_HANDLE};
    ImGui_ImplVulkanH_Window window_data{};
    u32 min_image_count{2u};
    u32 vulkan_api_version{VK_API_VERSION_1_2};
    bool swapchain_rebuild{};
    VkFormat depth_format{VK_FORMAT_UNDEFINED};
    std::vector<DepthAttachment> depth_attachments;
    VkDescriptorSetLayout mesh_descriptor_set_layout{VK_NULL_HANDLE};
    VkDescriptorPool mesh_descriptor_pool{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> mesh_descriptor_sets;
    VkPipelineLayout mesh_pipeline_layout{VK_NULL_HANDLE};
    VkPipeline environment_pipeline{VK_NULL_HANDLE};
    VkPipeline mesh_pipeline{VK_NULL_HANDLE};
    VkPipelineLayout shadow_pipeline_layout{VK_NULL_HANDLE};
    VkPipeline shadow_pipeline{VK_NULL_HANDLE};
    VkPipelineLayout debug_pipeline_layout{VK_NULL_HANDLE};
    VkPipeline debug_pipeline{VK_NULL_HANDLE};
    ShadowMap shadow_map{};
    std::vector<MeshResource> meshes;
    std::vector<TextureResource> textures;
    std::vector<Buffer> debug_segment_buffers;
    std::vector<Buffer> mesh_material_buffers;
    std::vector<Buffer> mesh_lighting_buffers;
    std::vector<GpuMaterial> mesh_material_upload;
    GpuLighting mesh_lighting_upload{};
    DrawList draw_list{};
    InputState input{};
    std::filesystem::path pending_screenshot;
    bool pending_screenshot_transparent{};
    bool imgui_ready{};
    bool sdl_ready{};

    auto run(const detail::RuntimeCallbacks& callbacks, Runtime& runtime) -> int;
    auto initialize() -> void;
    auto shutdown() noexcept -> void;
    auto setup_sdl() -> void;
    auto setup_vulkan(std::vector<const char*> instance_extensions) -> void;
    auto setup_vulkan_window(VkSurfaceKHR surface, int width, int height) -> void;
    auto setup_imgui() -> void;
    auto create_pipelines() -> void;
    auto destroy_pipelines() noexcept -> void;
    auto create_shadow_map() -> void;
    auto destroy_shadow_map() noexcept -> void;
    auto destroy_depth_attachments() noexcept -> void;
    auto install_depth_rendering() -> void;
    [[nodiscard]] auto create_depth_attachment(u32 width, u32 height) -> DepthAttachment;
    auto create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        bool mapped,
        VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO
    ) -> Buffer;
    auto destroy_buffer(Buffer& buffer) noexcept -> void;
    auto begin_immediate_commands() -> VkCommandBuffer;
    auto end_immediate_commands(VkCommandBuffer command_buffer) -> void;
    auto create_texture_resource(
        const void* pixels, u32 width, u32 height, VkFormat format, VkDeviceSize bytes_per_pixel
    ) -> TextureResource;
    auto create_default_texture() -> void;
    auto destroy_texture(TextureResource& texture) noexcept -> void;
    auto load_texture(const std::filesystem::path& path, const TextureLoadConfig& load_config)
        -> TextureHandle;
    auto
    load_hdr_texture(const std::filesystem::path& path, const HdrTextureLoadConfig& load_config)
        -> TextureHandle;
    auto ensure_debug_buffer(u32 frame_index, VkDeviceSize size) -> Buffer&;
    auto ensure_mesh_material_buffer(u32 frame_index, VkDeviceSize size) -> Buffer&;
    auto ensure_mesh_lighting_buffer(u32 frame_index) -> Buffer&;
    auto update_mesh_material_descriptor(u32 frame_index, const Buffer& buffer) -> void;
    auto update_mesh_lighting_descriptor(u32 frame_index, const Buffer& buffer) -> void;
    auto update_mesh_texture_descriptors() -> void;
    auto update_mesh_shadow_descriptors() -> void;
    auto create_mesh_resource(const MeshData& mesh) -> MeshResource;
    auto upload_mesh(const MeshData& mesh) -> MeshHandle;
    auto replace_mesh(MeshHandle handle, const MeshData& mesh) -> MeshHandle;
    auto render_frame(
        VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index, ImDrawData* draw_data
    ) -> void;
    auto draw_shadow_map(VkCommandBuffer command_buffer) -> void;
    auto draw_environment(VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index)
        -> void;
    auto draw_meshes(VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index) -> void;
    auto draw_debug(VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index) -> void;
    auto draw_runtime_ui() -> void;
    auto handle_event(const SDL_Event& event, bool& done, bool& orbiting, bool& panning) -> void;
    [[nodiscard]] auto framebuffer_mouse_position(f32 window_x, f32 window_y) const -> Vec2;
    [[nodiscard]] auto current_modifiers() const noexcept -> KeyboardModifiers;
    auto reset_input_frame() -> void;
    auto rebuild_swapchain_if_needed() -> void;
    auto create_capture_buffer(SwapchainCapture& capture) -> void;
    auto destroy_capture_buffer(SwapchainCapture& capture) noexcept -> void;
    auto record_capture_commands(
        VkCommandBuffer command_buffer,
        const ImGui_ImplVulkanH_Frame* frame,
        const SwapchainCapture& capture
    ) -> void;
    auto write_capture_png(const SwapchainCapture& capture) -> void;
    auto present_frame() -> void;
};

auto Runtime::Impl::create_buffer(
    const VkDeviceSize size,
    const VkBufferUsageFlags usage,
    const bool mapped,
    const VmaMemoryUsage memory_usage
) -> Buffer
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = std::max<VkDeviceSize>(size, 1u);
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = memory_usage;
    if (mapped)
    {
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    else
    {
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }

    VmaAllocationInfo allocation_result{};
    auto buffer = Buffer{.capacity = buffer_info.size};
    check_vk_result(vmaCreateBuffer(
        vma_allocator,
        &buffer_info,
        &allocation_info,
        &buffer.handle,
        &buffer.allocation,
        &allocation_result
    ));
    buffer.mapped = mapped ? allocation_result.pMappedData : nullptr;
    return buffer;
}

auto Runtime::Impl::destroy_buffer(Buffer& buffer) noexcept -> void
{
    if (buffer.handle != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(vma_allocator, buffer.handle, buffer.allocation);
    }
    buffer = {};
}

auto Runtime::Impl::begin_immediate_commands() -> VkCommandBuffer
{
    if (window_data.Frames.Size <= 0 or window_data.Frames[0].CommandPool == VK_NULL_HANDLE)
    {
        throw std::runtime_error("immediate Vulkan upload requires an initialized command pool");
    }

    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = window_data.Frames[0].CommandPool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;

    auto command_buffer = VkCommandBuffer{VK_NULL_HANDLE};
    check_vk_result(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk_result(vkBeginCommandBuffer(command_buffer, &begin_info));
    return command_buffer;
}

auto Runtime::Impl::end_immediate_commands(const VkCommandBuffer command_buffer) -> void
{
    check_vk_result(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    check_vk_result(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
    check_vk_result(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, window_data.Frames[0].CommandPool, 1, &command_buffer);
}

auto Runtime::Impl::create_texture_resource(
    const void* pixels,
    const u32 width,
    const u32 height,
    const VkFormat format,
    const VkDeviceSize bytes_per_pixel
) -> TextureResource
{
    if (width == 0u or height == 0u or !pixels or bytes_per_pixel == 0u)
    {
        throw std::runtime_error("cannot create texture from empty image data");
    }

    const auto pixel_bytes =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * bytes_per_pixel;
    auto staging = create_buffer(
        pixel_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, VMA_MEMORY_USAGE_AUTO_PREFER_HOST
    );
    auto texture = TextureResource{.width = width, .height = height, .format = format};
    try
    {
        std::memcpy(staging.mapped, pixels, static_cast<usize>(pixel_bytes));

        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = VkExtent3D{.width = width, .height = height, .depth = 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        check_vk_result(vmaCreateImage(
            vma_allocator,
            &image_info,
            &allocation_info,
            &texture.image,
            &texture.allocation,
            nullptr
        ));

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        check_vk_result(vkCreateImageView(device, &view_info, allocation_callbacks, &texture.view));

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = 0.0f;
        sampler_info.maxAnisotropy = 1.0f;
        check_vk_result(
            vkCreateSampler(device, &sampler_info, allocation_callbacks, &texture.sampler)
        );

        const auto command_buffer = begin_immediate_commands();
        VkImageMemoryBarrier upload_barrier{};
        upload_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        upload_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        upload_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        upload_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        upload_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        upload_barrier.image = texture.image;
        upload_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        upload_barrier.subresourceRange.baseMipLevel = 0;
        upload_barrier.subresourceRange.levelCount = 1;
        upload_barrier.subresourceRange.baseArrayLayer = 0;
        upload_barrier.subresourceRange.layerCount = 1;
        upload_barrier.srcAccessMask = 0;
        upload_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &upload_barrier
        );

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset = 0;
        copy_region.bufferRowLength = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageExtent = VkExtent3D{.width = width, .height = height, .depth = 1};
        vkCmdCopyBufferToImage(
            command_buffer,
            staging.handle,
            texture.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy_region
        );

        VkImageMemoryBarrier shader_read_barrier = upload_barrier;
        shader_read_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        shader_read_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        shader_read_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        shader_read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &shader_read_barrier
        );
        end_immediate_commands(command_buffer);
    }
    catch (...)
    {
        destroy_texture(texture);
        destroy_buffer(staging);
        throw;
    }
    destroy_buffer(staging);
    return texture;
}

auto Runtime::Impl::create_default_texture() -> void
{
    if (!textures.empty())
    {
        return;
    }
    constexpr auto white = std::array<u8, 4>{255u, 255u, 255u, 255u};
    textures.push_back(create_texture_resource(white.data(), 1u, 1u, VK_FORMAT_R8G8B8A8_UNORM, 4u));
}

auto Runtime::Impl::destroy_texture(TextureResource& texture) noexcept -> void
{
    if (texture.sampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, texture.sampler, allocation_callbacks);
    }
    if (texture.view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, texture.view, allocation_callbacks);
    }
    if (texture.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(vma_allocator, texture.image, texture.allocation);
    }
    texture = {};
}

auto Runtime::Impl::load_texture(
    const std::filesystem::path& path, const TextureLoadConfig& load_config
) -> TextureHandle
{
    if (textures.size() >= k_max_material_textures)
    {
        throw std::runtime_error(
            std::format("ds_vk material texture table is full (max {})", k_max_material_textures)
        );
    }

    auto width = int{};
    auto height = int{};
    auto channels = int{};
    auto* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels)
    {
        throw std::runtime_error(
            std::format("failed to load texture {}: {}", path.string(), stbi_failure_reason())
        );
    }

    auto texture = TextureResource{};
    try
    {
        const auto format = load_config.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        texture = create_texture_resource(
            pixels, static_cast<u32>(width), static_cast<u32>(height), format, 4u
        );
    }
    catch (...)
    {
        stbi_image_free(pixels);
        throw;
    }
    stbi_image_free(pixels);

    const auto index = static_cast<u32>(textures.size());
    try
    {
        textures.push_back(texture);
    }
    catch (...)
    {
        destroy_texture(texture);
        throw;
    }
    update_mesh_texture_descriptors();
    return TextureHandle{.id = index};
}

auto Runtime::Impl::load_hdr_texture(
    const std::filesystem::path& path, const HdrTextureLoadConfig& load_config
) -> TextureHandle
{
    if (textures.size() >= k_max_material_textures)
    {
        throw std::runtime_error(
            std::format("ds_vk texture table is full (max {})", k_max_material_textures)
        );
    }

    auto width = int{};
    auto height = int{};
    auto channels = int{};
    auto* pixels = stbi_loadf(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels)
    {
        throw std::runtime_error(
            std::format("failed to load HDR texture {}: {}", path.string(), stbi_failure_reason())
        );
    }
    if (width <= 0 or height <= 0)
    {
        stbi_image_free(pixels);
        throw std::runtime_error(std::format("HDR texture has invalid size: {}", path.string()));
    }
    const auto pixel_count = static_cast<usize>(width) * static_cast<usize>(height) * 4zu;
    for (auto i = 0zu; i < pixel_count; ++i)
    {
        pixels[i] *= load_config.exposure;
    }

    auto texture = TextureResource{};
    try
    {
        texture = create_texture_resource(
            pixels,
            static_cast<u32>(width),
            static_cast<u32>(height),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            4u * sizeof(f32)
        );
    }
    catch (...)
    {
        stbi_image_free(pixels);
        throw;
    }
    stbi_image_free(pixels);

    const auto index = static_cast<u32>(textures.size());
    try
    {
        textures.push_back(texture);
    }
    catch (...)
    {
        destroy_texture(texture);
        throw;
    }
    update_mesh_texture_descriptors();
    return TextureHandle{.id = index};
}

auto Runtime::Impl::ensure_debug_buffer(u32 frame_index, VkDeviceSize size) -> Buffer&
{
    if (debug_segment_buffers.size() < window_data.ImageCount)
    {
        debug_segment_buffers.resize(window_data.ImageCount);
    }
    auto& buffer = debug_segment_buffers.at(frame_index);
    if (buffer.capacity >= size and buffer.handle != VK_NULL_HANDLE)
    {
        return buffer;
    }

    check_vk_result(vkDeviceWaitIdle(device));
    destroy_buffer(buffer);
    buffer = create_buffer(
        size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, VMA_MEMORY_USAGE_AUTO_PREFER_HOST
    );
    return buffer;
}

auto Runtime::Impl::ensure_mesh_material_buffer(u32 frame_index, VkDeviceSize size) -> Buffer&
{
    if (mesh_material_buffers.size() < window_data.ImageCount)
    {
        mesh_material_buffers.resize(window_data.ImageCount);
    }
    auto& buffer = mesh_material_buffers.at(frame_index);
    if (buffer.capacity >= size and buffer.handle != VK_NULL_HANDLE)
    {
        return buffer;
    }

    check_vk_result(vkDeviceWaitIdle(device));
    destroy_buffer(buffer);
    buffer = create_buffer(
        size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, VMA_MEMORY_USAGE_AUTO_PREFER_HOST
    );
    update_mesh_material_descriptor(frame_index, buffer);
    return buffer;
}

auto Runtime::Impl::ensure_mesh_lighting_buffer(u32 frame_index) -> Buffer&
{
    if (mesh_lighting_buffers.size() < window_data.ImageCount)
    {
        mesh_lighting_buffers.resize(window_data.ImageCount);
    }
    auto& buffer = mesh_lighting_buffers.at(frame_index);
    if (buffer.capacity >= sizeof(GpuLighting) and buffer.handle != VK_NULL_HANDLE)
    {
        return buffer;
    }

    check_vk_result(vkDeviceWaitIdle(device));
    destroy_buffer(buffer);
    buffer = create_buffer(
        sizeof(GpuLighting),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        true,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST
    );
    update_mesh_lighting_descriptor(frame_index, buffer);
    return buffer;
}

auto Runtime::Impl::update_mesh_material_descriptor(u32 frame_index, const Buffer& buffer) -> void
{
    if (mesh_descriptor_sets.empty())
    {
        throw std::runtime_error("mesh material descriptor sets are not initialized");
    }

    const auto buffer_info = VkDescriptorBufferInfo{
        .buffer = buffer.handle,
        .offset = 0,
        .range = buffer.capacity,
    };
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = mesh_descriptor_sets.at(frame_index);
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

auto Runtime::Impl::update_mesh_lighting_descriptor(u32 frame_index, const Buffer& buffer) -> void
{
    if (mesh_descriptor_sets.empty())
    {
        throw std::runtime_error("mesh lighting descriptor sets are not initialized");
    }

    const auto buffer_info = VkDescriptorBufferInfo{
        .buffer = buffer.handle,
        .offset = 0,
        .range = buffer.capacity,
    };
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = mesh_descriptor_sets.at(frame_index);
    write.dstBinding = 2;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

auto Runtime::Impl::update_mesh_texture_descriptors() -> void
{
    if (mesh_descriptor_sets.empty())
    {
        return;
    }
    if (textures.empty())
    {
        throw std::runtime_error("mesh texture descriptors require the default texture");
    }

    auto image_infos = std::array<VkDescriptorImageInfo, k_max_material_textures>{};
    const auto& default_texture = textures.at(k_default_texture_index);
    const auto default_info = VkDescriptorImageInfo{
        .sampler = default_texture.sampler,
        .imageView = default_texture.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    image_infos.fill(default_info);
    for (auto i = 0u; i < std::min(static_cast<u32>(textures.size()), k_max_material_textures); ++i)
    {
        image_infos[i] = VkDescriptorImageInfo{
            .sampler = textures[i].sampler,
            .imageView = textures[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
    }

    for (const auto descriptor_set : mesh_descriptor_sets)
    {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = 1;
        write.dstArrayElement = 0;
        write.descriptorCount = k_max_material_textures;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = image_infos.data();
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

auto Runtime::Impl::update_mesh_shadow_descriptors() -> void
{
    if (mesh_descriptor_sets.empty() or shadow_map.view == VK_NULL_HANDLE)
    {
        return;
    }

    const auto image_info = VkDescriptorImageInfo{
        .sampler = shadow_map.sampler,
        .imageView = shadow_map.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    for (const auto descriptor_set : mesh_descriptor_sets)
    {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = 3;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image_info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

auto Runtime::Impl::create_mesh_resource(const MeshData& mesh) -> MeshResource
{
    if (vma_allocator == VK_NULL_HANDLE)
    {
        throw std::runtime_error("mesh upload requires an initialized ds_vk::Runtime");
    }
    if (mesh.vertices.empty() or mesh.indices.empty() or !has_valid_indices(mesh))
    {
        throw std::runtime_error("cannot upload empty mesh or mesh with invalid indices");
    }

    const auto vertex_bytes = data_byte_size(mesh.vertices);
    const auto index_bytes = data_byte_size(mesh.indices);
    auto resource = MeshResource{};
    try
    {
        resource.vertex_count = static_cast<u32>(mesh.vertices.size());
        resource.index_count = static_cast<u32>(mesh.indices.size());
        resource.vertices = create_buffer(vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false);
        resource.indices = create_buffer(index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, false);
        check_vk_result(vmaCopyMemoryToAllocation(
            vma_allocator, mesh.vertices.data(), resource.vertices.allocation, 0, vertex_bytes
        ));
        check_vk_result(vmaCopyMemoryToAllocation(
            vma_allocator, mesh.indices.data(), resource.indices.allocation, 0, index_bytes
        ));
    }
    catch (...)
    {
        destroy_buffer(resource.vertices);
        destroy_buffer(resource.indices);
        throw;
    }
    return resource;
}

auto Runtime::Impl::upload_mesh(const MeshData& mesh) -> MeshHandle
{
    auto resource = create_mesh_resource(mesh);
    const auto index = static_cast<u32>(meshes.size());
    try
    {
        meshes.push_back(resource);
    }
    catch (...)
    {
        destroy_buffer(resource.vertices);
        destroy_buffer(resource.indices);
        throw;
    }
    return MeshHandle{.id = index};
}

auto Runtime::Impl::replace_mesh(MeshHandle handle, const MeshData& mesh) -> MeshHandle
{
    if (!handle.valid() or handle.id >= meshes.size())
    {
        return upload_mesh(mesh);
    }

    auto replacement = create_mesh_resource(mesh);
    try
    {
        check_vk_result(vkDeviceWaitIdle(device));
    }
    catch (...)
    {
        destroy_buffer(replacement.vertices);
        destroy_buffer(replacement.indices);
        throw;
    }
    auto old = meshes[handle.id];
    meshes[handle.id] = replacement;
    destroy_buffer(old.vertices);
    destroy_buffer(old.indices);
    return handle;
}

auto Runtime::Impl::create_shadow_map() -> void
{
    destroy_shadow_map();

    const auto resolution = std::max(256u, config.shadow_map_resolution);
    shadow_map.resolution = resolution;
    shadow_map.format =
        depth_format == VK_FORMAT_UNDEFINED ? find_depth_format(physical_device) : depth_format;

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = shadow_map.format;
    image_info.extent = VkExtent3D{.width = resolution, .height = resolution, .depth = 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check_vk_result(vmaCreateImage(
        vma_allocator,
        &image_info,
        &allocation_info,
        &shadow_map.image,
        &shadow_map.allocation,
        nullptr
    ));

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = shadow_map.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = shadow_map.format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    check_vk_result(vkCreateImageView(device, &view_info, allocation_callbacks, &shadow_map.view));

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;
    sampler_info.maxAnisotropy = 1.0f;
    check_vk_result(
        vkCreateSampler(device, &sampler_info, allocation_callbacks, &shadow_map.sampler)
    );

    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = shadow_map.format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depth_reference{};
    depth_reference.attachment = 0;
    depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depth_reference;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &depth_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    check_vk_result(
        vkCreateRenderPass(device, &render_pass_info, allocation_callbacks, &shadow_map.render_pass)
    );

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = shadow_map.render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &shadow_map.view;
    framebuffer_info.width = resolution;
    framebuffer_info.height = resolution;
    framebuffer_info.layers = 1;
    check_vk_result(vkCreateFramebuffer(
        device, &framebuffer_info, allocation_callbacks, &shadow_map.framebuffer
    ));

    const auto command_buffer = begin_immediate_commands();
    VkImageMemoryBarrier shader_read_barrier{};
    shader_read_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    shader_read_barrier.srcAccessMask = 0;
    shader_read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    shader_read_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    shader_read_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    shader_read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    shader_read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    shader_read_barrier.image = shadow_map.image;
    shader_read_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    shader_read_barrier.subresourceRange.baseMipLevel = 0;
    shader_read_barrier.subresourceRange.levelCount = 1;
    shader_read_barrier.subresourceRange.baseArrayLayer = 0;
    shader_read_barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &shader_read_barrier
    );
    end_immediate_commands(command_buffer);
}

auto Runtime::Impl::destroy_shadow_map() noexcept -> void
{
    if (shadow_map.framebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, shadow_map.framebuffer, allocation_callbacks);
    }
    if (shadow_map.render_pass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, shadow_map.render_pass, allocation_callbacks);
    }
    if (shadow_map.sampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, shadow_map.sampler, allocation_callbacks);
    }
    if (shadow_map.view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, shadow_map.view, allocation_callbacks);
    }
    if (shadow_map.image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(vma_allocator, shadow_map.image, shadow_map.allocation);
    }
    shadow_map = {};
}

auto Runtime::Impl::create_depth_attachment(u32 width, u32 height) -> DepthAttachment
{
    auto depth = DepthAttachment{};
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = depth_format;
    image_info.extent = VkExtent3D{.width = width, .height = height, .depth = 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    check_vk_result(vmaCreateImage(
        vma_allocator, &image_info, &allocation_info, &depth.image, &depth.allocation, nullptr
    ));

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = depth.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = depth_format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    check_vk_result(vkCreateImageView(device, &view_info, allocation_callbacks, &depth.view));
    return depth;
}

auto Runtime::Impl::destroy_depth_attachments() noexcept -> void
{
    for (auto& depth : depth_attachments)
    {
        if (depth.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, depth.view, allocation_callbacks);
        }
        if (depth.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(vma_allocator, depth.image, depth.allocation);
        }
    }
    depth_attachments.clear();
}

auto Runtime::Impl::install_depth_rendering() -> void
{
    destroy_depth_attachments();
    for (int i = 0; i < window_data.Frames.Size; ++i)
    {
        auto& frame = window_data.Frames[i];
        if (frame.Framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(device, frame.Framebuffer, allocation_callbacks);
            frame.Framebuffer = VK_NULL_HANDLE;
        }
    }
    if (window_data.RenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, window_data.RenderPass, allocation_callbacks);
        window_data.RenderPass = VK_NULL_HANDLE;
    }
    if (window_data.Width <= 0 or window_data.Height <= 0 or window_data.ImageCount == 0u)
    {
        return;
    }

    depth_format = find_depth_format(physical_device);
    depth_attachments.reserve(window_data.ImageCount);
    for (u32 i = 0; i < window_data.ImageCount; ++i)
    {
        depth_attachments.push_back(create_depth_attachment(
            static_cast<u32>(window_data.Width), static_cast<u32>(window_data.Height)
        ));
    }

    auto color_attachment = window_data.AttachmentDesc;
    if (color_attachment.format == VK_FORMAT_UNDEFINED)
    {
        color_attachment.format = window_data.SurfaceFormat.format;
    }
    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = depth_format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    const auto attachments = std::array{color_attachment, depth_attachment};
    VkAttachmentReference color_reference{};
    color_reference.attachment = 0;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depth_reference{};
    depth_reference.attachment = 1;
    depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pDepthStencilAttachment = &depth_reference;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<u32>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;
    check_vk_result(
        vkCreateRenderPass(device, &render_pass_info, allocation_callbacks, &window_data.RenderPass)
    );

    for (u32 i = 0; i < window_data.ImageCount; ++i)
    {
        auto& frame = window_data.Frames[static_cast<int>(i)];
        const auto framebuffer_attachments =
            std::array{frame.BackbufferView, depth_attachments[i].view};
        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = window_data.RenderPass;
        framebuffer_info.attachmentCount = static_cast<u32>(framebuffer_attachments.size());
        framebuffer_info.pAttachments = framebuffer_attachments.data();
        framebuffer_info.width = static_cast<u32>(window_data.Width);
        framebuffer_info.height = static_cast<u32>(window_data.Height);
        framebuffer_info.layers = 1;
        check_vk_result(
            vkCreateFramebuffer(device, &framebuffer_info, allocation_callbacks, &frame.Framebuffer)
        );
    }
}

auto Runtime::Impl::setup_sdl() -> void
{
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD))
    {
        throw std::runtime_error(SDL_GetError());
    }
    sdl_ready = true;

    const auto display_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    const auto flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN
                       | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    window = SDL_CreateWindow(
        config.window_title.c_str(),
        static_cast<int>(static_cast<f32>(config.initial_width) * display_scale),
        static_cast<int>(static_cast<f32>(config.initial_height) * display_scale),
        flags
    );
    if (!window)
    {
        throw std::runtime_error(SDL_GetError());
    }
}

auto Runtime::Impl::setup_vulkan(std::vector<const char*> instance_extensions) -> void
{
    auto instance_extension_count = 0u;
    check_vk_result(
        vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr)
    );
    auto instance_properties = std::vector<VkExtensionProperties>(instance_extension_count);
    check_vk_result(vkEnumerateInstanceExtensionProperties(
        nullptr, &instance_extension_count, instance_properties.data()
    ));

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = config.window_title.c_str();
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "ds_vk";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    if (extension_available(
            instance_properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
        ))
    {
        instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (extension_available(instance_properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
    {
        instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif

    const auto* validation_layer = "VK_LAYER_KHRONOS_validation";
    if (config.enable_validation and layer_available(validation_layer))
    {
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = &validation_layer;
    }
    create_info.enabledExtensionCount = static_cast<u32>(instance_extensions.size());
    create_info.ppEnabledExtensionNames = instance_extensions.data();
    check_vk_result(vkCreateInstance(&create_info, allocation_callbacks, &instance));

    physical_device = ImGui_ImplVulkanH_SelectPhysicalDevice(instance);
    if (physical_device == VK_NULL_HANDLE)
    {
        throw std::runtime_error("no Vulkan physical device selected");
    }
    queue_family = ImGui_ImplVulkanH_SelectQueueFamilyIndex(physical_device);
    if (queue_family == std::numeric_limits<u32>::max())
    {
        throw std::runtime_error("no Vulkan graphics queue family selected");
    }
    VkPhysicalDeviceProperties physical_device_properties{};
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
    vulkan_api_version = std::min(physical_device_properties.apiVersion, VK_API_VERSION_1_2);
    if (physical_device_properties.limits.maxPushConstantsSize < k_required_push_constant_bytes)
    {
        throw std::runtime_error("physical device maxPushConstantsSize is too small for ds_vk");
    }
    if (physical_device_properties.limits.maxPerStageDescriptorSamplers
        < k_max_material_textures + 1u)
    {
        throw std::runtime_error(
            "physical device maxPerStageDescriptorSamplers is too small for ds_vk"
        );
    }

    auto device_extension_count = 0u;
    check_vk_result(vkEnumerateDeviceExtensionProperties(
        physical_device, nullptr, &device_extension_count, nullptr
    ));
    auto device_properties = std::vector<VkExtensionProperties>(device_extension_count);
    check_vk_result(vkEnumerateDeviceExtensionProperties(
        physical_device, nullptr, &device_extension_count, device_properties.data()
    ));

    auto device_extensions = std::vector<const char*>{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (extension_available(device_properties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
    {
        device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#else
    constexpr auto* k_portability_subset_extension = "VK_KHR_portability_subset";
    if (extension_available(device_properties, k_portability_subset_extension))
    {
        device_extensions.push_back(k_portability_subset_extension);
    }
#endif
#ifdef VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME
    constexpr auto* k_descriptor_indexing_extension = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME;
#else
    constexpr auto* k_descriptor_indexing_extension = "VK_EXT_descriptor_indexing";
#endif
    const auto has_descriptor_indexing_extension =
        extension_available(device_properties, k_descriptor_indexing_extension);
    if (has_descriptor_indexing_extension)
    {
        device_extensions.push_back(k_descriptor_indexing_extension);
    }

    const auto supports_vulkan_1_2 =
        VK_VERSION_MAJOR(physical_device_properties.apiVersion) > 1u
        or (VK_VERSION_MAJOR(physical_device_properties.apiVersion) == 1u
            and VK_VERSION_MINOR(physical_device_properties.apiVersion) >= 2u);
    const auto can_enable_descriptor_indexing =
        supports_vulkan_1_2 or has_descriptor_indexing_extension;

    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_features{};
    descriptor_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    VkPhysicalDeviceFeatures2 available_features{};
    available_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    available_features.pNext = can_enable_descriptor_indexing ? &descriptor_features : nullptr;
    vkGetPhysicalDeviceFeatures2(physical_device, &available_features);

    VkPhysicalDeviceDescriptorIndexingFeatures enabled_descriptor_features{};
    enabled_descriptor_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    enabled_descriptor_features.runtimeDescriptorArray = descriptor_features.runtimeDescriptorArray;
    enabled_descriptor_features.descriptorBindingPartiallyBound =
        descriptor_features.descriptorBindingPartiallyBound;
    enabled_descriptor_features.shaderSampledImageArrayNonUniformIndexing =
        descriptor_features.shaderSampledImageArrayNonUniformIndexing;
    enabled_descriptor_features.shaderStorageBufferArrayNonUniformIndexing =
        descriptor_features.shaderStorageBufferArrayNonUniformIndexing;
    enabled_descriptor_features.descriptorBindingSampledImageUpdateAfterBind =
        descriptor_features.descriptorBindingSampledImageUpdateAfterBind;
    enabled_descriptor_features.descriptorBindingStorageBufferUpdateAfterBind =
        descriptor_features.descriptorBindingStorageBufferUpdateAfterBind;

    descriptor_indexing = DescriptorIndexingSupport{
        .descriptor_indexing = descriptor_features.runtimeDescriptorArray == VK_TRUE
                               or descriptor_features.descriptorBindingPartiallyBound == VK_TRUE,
        .sampled_image_array_dynamic_indexing =
            available_features.features.shaderSampledImageArrayDynamicIndexing == VK_TRUE,
        .runtime_descriptor_array = descriptor_features.runtimeDescriptorArray == VK_TRUE,
        .descriptor_binding_partially_bound =
            descriptor_features.descriptorBindingPartiallyBound == VK_TRUE,
        .sampled_image_non_uniform_indexing =
            descriptor_features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE,
        .storage_buffer_non_uniform_indexing =
            descriptor_features.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE,
        .sampled_image_update_after_bind =
            descriptor_features.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE,
        .storage_buffer_update_after_bind =
            descriptor_features.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE,
    };

    VkPhysicalDeviceFeatures2 enabled_features{};
    enabled_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabled_features.features.shaderSampledImageArrayDynamicIndexing =
        available_features.features.shaderSampledImageArrayDynamicIndexing;
    enabled_features.pNext =
        can_enable_descriptor_indexing ? &enabled_descriptor_features : nullptr;

    const auto queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &enabled_features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<u32>(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.data();
    check_vk_result(vkCreateDevice(physical_device, &device_info, allocation_callbacks, &device));
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.instance = instance;
    allocator_info.physicalDevice = physical_device;
    allocator_info.device = device;
    allocator_info.vulkanApiVersion = vulkan_api_version;
    check_vk_result(vmaCreateAllocator(&allocator_info, &vma_allocator));

    const auto pool_sizes = std::array{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE,
        },
    };
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    for (const auto& pool_size : pool_sizes)
    {
        pool_info.maxSets += pool_size.descriptorCount;
    }
    pool_info.poolSizeCount = static_cast<u32>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    check_vk_result(
        vkCreateDescriptorPool(device, &pool_info, allocation_callbacks, &imgui_descriptor_pool)
    );

    VkPipelineCacheCreateInfo pipeline_cache_info{};
    pipeline_cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    check_vk_result(
        vkCreatePipelineCache(device, &pipeline_cache_info, allocation_callbacks, &pipeline_cache)
    );
}

auto Runtime::Impl::setup_vulkan_window(
    const VkSurfaceKHR surface, const int width, const int height
) -> void
{
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family, surface, &supported);
    if (supported != VK_TRUE)
    {
        throw std::runtime_error("selected Vulkan queue has no WSI support");
    }

    const auto formats = std::array{
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8_UNORM,
        VK_FORMAT_R8G8B8_UNORM,
    };
    window_data.Surface = surface;
    window_data.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        physical_device,
        window_data.Surface,
        formats.data(),
        static_cast<int>(formats.size()),
        VK_COLORSPACE_SRGB_NONLINEAR_KHR
    );
    auto present_modes = std::array{VK_PRESENT_MODE_FIFO_KHR};
    window_data.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        physical_device,
        window_data.Surface,
        present_modes.data(),
        static_cast<int>(present_modes.size())
    );
    window_data.ClearValue.color.float32[0] = config.clear_color.r();
    window_data.ClearValue.color.float32[1] = config.clear_color.g();
    window_data.ClearValue.color.float32[2] = config.clear_color.b();
    window_data.ClearValue.color.float32[3] =
        config.transparent_screenshot ? 0.0f : config.clear_color.a();
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        instance,
        physical_device,
        device,
        &window_data,
        queue_family,
        allocation_callbacks,
        width,
        height,
        min_image_count,
        k_swapchain_image_usage
    );
    install_depth_rendering();
}

auto Runtime::Impl::setup_imgui() -> void
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion = vulkan_api_version;
    init_info.Instance = instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.QueueFamily = queue_family;
    init_info.Queue = queue;
    init_info.PipelineCache = pipeline_cache;
    init_info.DescriptorPool = imgui_descriptor_pool;
    init_info.MinImageCount = min_image_count;
    init_info.ImageCount = window_data.ImageCount;
    init_info.Allocator = allocation_callbacks;
    init_info.PipelineInfoMain.RenderPass = window_data.RenderPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = check_vk_result;
    if (!ImGui_ImplVulkan_Init(&init_info))
    {
        throw std::runtime_error("failed to initialize ImGui Vulkan backend");
    }
    imgui_ready = true;
}

auto Runtime::Impl::create_pipelines() -> void
{
    destroy_pipelines();

    const auto shader_dir =
        config.shader_dir.empty() ? std::filesystem::path{DS_VK_SHADER_DIR} : config.shader_dir;

    const auto mesh_vert = create_shader_module(device, shader_dir / "mesh.vert.spv");
    const auto mesh_frag = create_shader_module(device, shader_dir / "mesh.frag.spv");
    const auto environment_vert = create_shader_module(device, shader_dir / "environment.vert.spv");
    const auto environment_frag = create_shader_module(device, shader_dir / "environment.frag.spv");
    const auto shadow_vert = create_shader_module(device, shader_dir / "shadow.vert.spv");
    const auto debug_vert = create_shader_module(device, shader_dir / "debug_line.vert.spv");
    const auto debug_frag = create_shader_module(device, shader_dir / "debug_line.frag.spv");

    const auto destroy_shader_modules = [&]() -> void
    {
        vkDestroyShaderModule(device, mesh_vert, allocation_callbacks);
        vkDestroyShaderModule(device, mesh_frag, allocation_callbacks);
        vkDestroyShaderModule(device, environment_vert, allocation_callbacks);
        vkDestroyShaderModule(device, environment_frag, allocation_callbacks);
        vkDestroyShaderModule(device, shadow_vert, allocation_callbacks);
        vkDestroyShaderModule(device, debug_vert, allocation_callbacks);
        vkDestroyShaderModule(device, debug_frag, allocation_callbacks);
    };

    VkPushConstantRange mesh_push_range{};
    mesh_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    mesh_push_range.offset = 0;
    mesh_push_range.size = sizeof(MeshPushConstants);

    VkDescriptorSetLayoutBinding material_binding{};
    material_binding.binding = 0;
    material_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    material_binding.descriptorCount = 1;
    material_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding texture_binding{};
    texture_binding.binding = 1;
    texture_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texture_binding.descriptorCount = k_max_material_textures;
    texture_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding lighting_binding{};
    lighting_binding.binding = 2;
    lighting_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lighting_binding.descriptorCount = 1;
    lighting_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding shadow_binding{};
    shadow_binding.binding = 3;
    shadow_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadow_binding.descriptorCount = 1;
    shadow_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    const auto mesh_descriptor_bindings = std::array{
        material_binding,
        texture_binding,
        lighting_binding,
        shadow_binding,
    };
    VkDescriptorSetLayoutCreateInfo mesh_descriptor_layout_info{};
    mesh_descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    mesh_descriptor_layout_info.bindingCount = static_cast<u32>(mesh_descriptor_bindings.size());
    mesh_descriptor_layout_info.pBindings = mesh_descriptor_bindings.data();
    check_vk_result(vkCreateDescriptorSetLayout(
        device, &mesh_descriptor_layout_info, allocation_callbacks, &mesh_descriptor_set_layout
    ));

    const auto descriptor_pool_sizes = std::array{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = std::max(1u, window_data.ImageCount) * 2u,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount =
                std::max(1u, window_data.ImageCount) * (k_max_material_textures + 1u),
        },
    };
    VkDescriptorPoolCreateInfo mesh_descriptor_pool_info{};
    mesh_descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    mesh_descriptor_pool_info.maxSets = std::max(1u, window_data.ImageCount);
    mesh_descriptor_pool_info.poolSizeCount = static_cast<u32>(descriptor_pool_sizes.size());
    mesh_descriptor_pool_info.pPoolSizes = descriptor_pool_sizes.data();
    check_vk_result(vkCreateDescriptorPool(
        device, &mesh_descriptor_pool_info, allocation_callbacks, &mesh_descriptor_pool
    ));

    mesh_descriptor_sets.resize(window_data.ImageCount);
    const auto descriptor_layouts =
        std::vector<VkDescriptorSetLayout>(window_data.ImageCount, mesh_descriptor_set_layout);
    VkDescriptorSetAllocateInfo descriptor_allocate_info{};
    descriptor_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate_info.descriptorPool = mesh_descriptor_pool;
    descriptor_allocate_info.descriptorSetCount = static_cast<u32>(descriptor_layouts.size());
    descriptor_allocate_info.pSetLayouts = descriptor_layouts.data();
    check_vk_result(
        vkAllocateDescriptorSets(device, &descriptor_allocate_info, mesh_descriptor_sets.data())
    );
    for (auto i = 0u;
         i < std::min(static_cast<u32>(mesh_material_buffers.size()), window_data.ImageCount);
         ++i)
    {
        if (mesh_material_buffers[i].handle != VK_NULL_HANDLE)
        {
            update_mesh_material_descriptor(i, mesh_material_buffers[i]);
        }
        if (i < mesh_lighting_buffers.size() and mesh_lighting_buffers[i].handle != VK_NULL_HANDLE)
        {
            update_mesh_lighting_descriptor(i, mesh_lighting_buffers[i]);
        }
    }
    update_mesh_texture_descriptors();
    update_mesh_shadow_descriptors();

    VkPipelineLayoutCreateInfo mesh_layout_info{};
    mesh_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    mesh_layout_info.setLayoutCount = 1;
    mesh_layout_info.pSetLayouts = &mesh_descriptor_set_layout;
    mesh_layout_info.pushConstantRangeCount = 1;
    mesh_layout_info.pPushConstantRanges = &mesh_push_range;
    check_vk_result(vkCreatePipelineLayout(
        device, &mesh_layout_info, allocation_callbacks, &mesh_pipeline_layout
    ));

    VkPipelineLayoutCreateInfo shadow_layout_info{};
    shadow_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    shadow_layout_info.pushConstantRangeCount = 1;
    shadow_layout_info.pPushConstantRanges = &mesh_push_range;
    check_vk_result(vkCreatePipelineLayout(
        device, &shadow_layout_info, allocation_callbacks, &shadow_pipeline_layout
    ));

    VkPushConstantRange debug_push_range{};
    debug_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    debug_push_range.offset = 0;
    debug_push_range.size = sizeof(DebugPushConstants);
    VkPipelineLayoutCreateInfo debug_layout_info{};
    debug_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    debug_layout_info.pushConstantRangeCount = 1;
    debug_layout_info.pPushConstantRanges = &debug_push_range;
    check_vk_result(vkCreatePipelineLayout(
        device, &debug_layout_info, allocation_callbacks, &debug_pipeline_layout
    ));

    const auto make_stage = [](const VkShaderModule module,
                               const VkShaderStageFlagBits stage) -> VkPipelineShaderStageCreateInfo
    {
        VkPipelineShaderStageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = module;
        info.pName = "main";
        return info;
    };

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

    VkPipelineDepthStencilStateCreateInfo depth_state{};
    depth_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_state.depthTestEnable = VK_TRUE;
    depth_state.depthWriteEnable = VK_TRUE;
    depth_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                      | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend_state{};
    blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_state.attachmentCount = 1;
    blend_state.pAttachments = &color_attachment;

    const auto dynamic_states = std::array{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    const auto mesh_binding = VkVertexInputBindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const auto mesh_attributes = std::array{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
        VkVertexInputAttributeDescription{
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, texcoord),
        },
    };
    VkPipelineVertexInputStateCreateInfo mesh_vertex_input{};
    mesh_vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    mesh_vertex_input.vertexBindingDescriptionCount = 1;
    mesh_vertex_input.pVertexBindingDescriptions = &mesh_binding;
    mesh_vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(mesh_attributes.size());
    mesh_vertex_input.pVertexAttributeDescriptions = mesh_attributes.data();

    const auto mesh_stages = std::array{
        make_stage(mesh_vert, VK_SHADER_STAGE_VERTEX_BIT),
        make_stage(mesh_frag, VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    VkGraphicsPipelineCreateInfo mesh_pipeline_info{};
    mesh_pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    mesh_pipeline_info.stageCount = static_cast<u32>(mesh_stages.size());
    mesh_pipeline_info.pStages = mesh_stages.data();
    mesh_pipeline_info.pVertexInputState = &mesh_vertex_input;
    mesh_pipeline_info.pInputAssemblyState = &input_assembly;
    mesh_pipeline_info.pViewportState = &viewport_state;
    mesh_pipeline_info.pRasterizationState = &rasterization;
    mesh_pipeline_info.pMultisampleState = &multisample;
    mesh_pipeline_info.pDepthStencilState = &depth_state;
    mesh_pipeline_info.pColorBlendState = &blend_state;
    mesh_pipeline_info.pDynamicState = &dynamic_state;
    mesh_pipeline_info.layout = mesh_pipeline_layout;
    mesh_pipeline_info.renderPass = window_data.RenderPass;
    mesh_pipeline_info.subpass = 0;
    check_vk_result(vkCreateGraphicsPipelines(
        device, pipeline_cache, 1, &mesh_pipeline_info, allocation_callbacks, &mesh_pipeline
    ));

    VkPipelineVertexInputStateCreateInfo environment_vertex_input{};
    environment_vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto environment_depth_state = depth_state;
    environment_depth_state.depthTestEnable = VK_TRUE;
    environment_depth_state.depthWriteEnable = VK_FALSE;
    environment_depth_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    const auto environment_stages = std::array{
        make_stage(environment_vert, VK_SHADER_STAGE_VERTEX_BIT),
        make_stage(environment_frag, VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    VkGraphicsPipelineCreateInfo environment_pipeline_info = mesh_pipeline_info;
    environment_pipeline_info.stageCount = static_cast<u32>(environment_stages.size());
    environment_pipeline_info.pStages = environment_stages.data();
    environment_pipeline_info.pVertexInputState = &environment_vertex_input;
    environment_pipeline_info.pDepthStencilState = &environment_depth_state;
    check_vk_result(vkCreateGraphicsPipelines(
        device,
        pipeline_cache,
        1,
        &environment_pipeline_info,
        allocation_callbacks,
        &environment_pipeline
    ));

    const auto shadow_attributes = std::array{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
    };
    VkPipelineVertexInputStateCreateInfo shadow_vertex_input{};
    shadow_vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    shadow_vertex_input.vertexBindingDescriptionCount = 1;
    shadow_vertex_input.pVertexBindingDescriptions = &mesh_binding;
    shadow_vertex_input.vertexAttributeDescriptionCount =
        static_cast<u32>(shadow_attributes.size());
    shadow_vertex_input.pVertexAttributeDescriptions = shadow_attributes.data();

    auto shadow_rasterization = rasterization;
    shadow_rasterization.depthBiasEnable = VK_TRUE;
    shadow_rasterization.depthBiasConstantFactor = 1.25f;
    shadow_rasterization.depthBiasSlopeFactor = 1.75f;
    auto shadow_depth_state = depth_state;
    shadow_depth_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendStateCreateInfo shadow_blend_state{};
    shadow_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    const auto shadow_stages = std::array{make_stage(shadow_vert, VK_SHADER_STAGE_VERTEX_BIT)};
    VkGraphicsPipelineCreateInfo shadow_pipeline_info = mesh_pipeline_info;
    shadow_pipeline_info.stageCount = static_cast<u32>(shadow_stages.size());
    shadow_pipeline_info.pStages = shadow_stages.data();
    shadow_pipeline_info.pVertexInputState = &shadow_vertex_input;
    shadow_pipeline_info.pRasterizationState = &shadow_rasterization;
    shadow_pipeline_info.pDepthStencilState = &shadow_depth_state;
    shadow_pipeline_info.pColorBlendState = &shadow_blend_state;
    shadow_pipeline_info.layout = shadow_pipeline_layout;
    shadow_pipeline_info.renderPass = shadow_map.render_pass;
    check_vk_result(vkCreateGraphicsPipelines(
        device, pipeline_cache, 1, &shadow_pipeline_info, allocation_callbacks, &shadow_pipeline
    ));

    const auto debug_binding = VkVertexInputBindingDescription{
        .binding = 0,
        .stride = sizeof(DebugSegment),
        .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
    };
    const auto debug_attributes = std::array{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(DebugSegment, start),
        },
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32_SFLOAT,
            .offset = offsetof(DebugSegment, width),
        },
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(DebugSegment, end),
        },
        VkVertexInputAttributeDescription{
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32_SFLOAT,
            .offset = offsetof(DebugSegment, arrow_tip),
        },
        VkVertexInputAttributeDescription{
            .location = 4,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(DebugSegment, color),
        },
    };
    VkPipelineVertexInputStateCreateInfo debug_vertex_input{};
    debug_vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    debug_vertex_input.vertexBindingDescriptionCount = 1;
    debug_vertex_input.pVertexBindingDescriptions = &debug_binding;
    debug_vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(debug_attributes.size());
    debug_vertex_input.pVertexAttributeDescriptions = debug_attributes.data();

    auto debug_depth_state = depth_state;
    debug_depth_state.depthWriteEnable = VK_FALSE;
    auto debug_color_attachment = color_attachment;
    debug_color_attachment.blendEnable = VK_TRUE;
    debug_color_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    debug_color_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    debug_color_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    debug_color_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    debug_color_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    debug_color_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    auto debug_blend_state = blend_state;
    debug_blend_state.pAttachments = &debug_color_attachment;

    const auto debug_stages = std::array{
        make_stage(debug_vert, VK_SHADER_STAGE_VERTEX_BIT),
        make_stage(debug_frag, VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    VkGraphicsPipelineCreateInfo debug_pipeline_info = mesh_pipeline_info;
    debug_pipeline_info.stageCount = static_cast<u32>(debug_stages.size());
    debug_pipeline_info.pStages = debug_stages.data();
    debug_pipeline_info.pVertexInputState = &debug_vertex_input;
    debug_pipeline_info.pDepthStencilState = &debug_depth_state;
    debug_pipeline_info.pColorBlendState = &debug_blend_state;
    debug_pipeline_info.layout = debug_pipeline_layout;
    check_vk_result(vkCreateGraphicsPipelines(
        device, pipeline_cache, 1, &debug_pipeline_info, allocation_callbacks, &debug_pipeline
    ));

    destroy_shader_modules();
}

auto Runtime::Impl::destroy_pipelines() noexcept -> void
{
    if (debug_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, debug_pipeline, allocation_callbacks);
        debug_pipeline = VK_NULL_HANDLE;
    }
    if (debug_pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, debug_pipeline_layout, allocation_callbacks);
        debug_pipeline_layout = VK_NULL_HANDLE;
    }
    if (shadow_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, shadow_pipeline, allocation_callbacks);
        shadow_pipeline = VK_NULL_HANDLE;
    }
    if (shadow_pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, shadow_pipeline_layout, allocation_callbacks);
        shadow_pipeline_layout = VK_NULL_HANDLE;
    }
    if (mesh_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, mesh_pipeline, allocation_callbacks);
        mesh_pipeline = VK_NULL_HANDLE;
    }
    if (environment_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, environment_pipeline, allocation_callbacks);
        environment_pipeline = VK_NULL_HANDLE;
    }
    if (mesh_pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, mesh_pipeline_layout, allocation_callbacks);
        mesh_pipeline_layout = VK_NULL_HANDLE;
    }
    mesh_descriptor_sets.clear();
    if (mesh_descriptor_pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, mesh_descriptor_pool, allocation_callbacks);
        mesh_descriptor_pool = VK_NULL_HANDLE;
    }
    if (mesh_descriptor_set_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, mesh_descriptor_set_layout, allocation_callbacks);
        mesh_descriptor_set_layout = VK_NULL_HANDLE;
    }
}

auto Runtime::Impl::draw_shadow_map(const VkCommandBuffer command_buffer) -> void
{
    const auto& mesh_commands = draw_list.mesh_commands();
    const auto& lights = draw_list.lights();
    const auto shadow_index = shadow_light_index(lights);
    if (mesh_commands.empty() or shadow_index >= lights.size() or shadow_pipeline == VK_NULL_HANDLE
        or shadow_map.framebuffer == VK_NULL_HANDLE)
    {
        return;
    }

    const auto shadow_view_projection = light_view_projection_matrix(lights[shadow_index], camera);
    const auto clear =
        VkClearValue{.depthStencil = VkClearDepthStencilValue{.depth = 1.0f, .stencil = 0}};
    VkRenderPassBeginInfo render_pass_begin{};
    render_pass_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin.renderPass = shadow_map.render_pass;
    render_pass_begin.framebuffer = shadow_map.framebuffer;
    render_pass_begin.renderArea = VkRect2D{
        .offset = VkOffset2D{.x = 0, .y = 0},
        .extent = VkExtent2D{.width = shadow_map.resolution, .height = shadow_map.resolution},
    };
    render_pass_begin.clearValueCount = 1;
    render_pass_begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    const auto viewport = VkViewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<f32>(shadow_map.resolution),
        .height = static_cast<f32>(shadow_map.resolution),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    const auto scissor = VkRect2D{
        .offset = VkOffset2D{.x = 0, .y = 0},
        .extent = VkExtent2D{.width = shadow_map.resolution, .height = shadow_map.resolution},
    };
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

    for (const auto& command : mesh_commands)
    {
        if (!command.mask.shadow_producer or !command.mesh.valid()
            or command.mesh.id >= meshes.size())
        {
            continue;
        }
        const auto& mesh = meshes[command.mesh.id];
        const auto offsets = std::array<VkDeviceSize, 1>{0};
        const auto vertex_buffers = std::array{mesh.vertices.handle};
        vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers.data(), offsets.data());
        vkCmdBindIndexBuffer(command_buffer, mesh.indices.handle, 0, VK_INDEX_TYPE_UINT32);
        const auto push = MeshPushConstants{
            .view_projection = shadow_view_projection,
            .model = command.transform.matrix(),
        };
        vkCmdPushConstants(
            command_buffer,
            shadow_pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(push),
            &push
        );
        vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
    }
    vkCmdEndRenderPass(command_buffer);

    VkImageMemoryBarrier read_barrier{};
    read_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    read_barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    read_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    read_barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    read_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    read_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    read_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    read_barrier.image = shadow_map.image;
    read_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    read_barrier.subresourceRange.levelCount = 1;
    read_barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &read_barrier
    );
}

auto Runtime::Impl::draw_environment(
    VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index
) -> void
{
    const auto& environment = draw_list.environment();
    if (!environment.texture.valid() or !environment.visible_to_camera
        or environment.texture.id >= k_max_material_textures
        or environment.background_intensity <= 0.0f or environment_pipeline == VK_NULL_HANDLE)
    {
        return;
    }
    const auto aspect = static_cast<f32>(std::max(1u, extent.width))
                        / static_cast<f32>(std::max(1u, extent.height));
    const auto push = EnvironmentPushConstants{
        .inverse_view_projection = glm::inverse(camera.view_projection_matrix(aspect)),
        .camera_position = Vec4{camera.position(), 1.0f},
        .params = Vec4{
            environment.background_intensity,
            environment.rotation_radians,
            static_cast<f32>(environment.texture.id),
            0.0f,
        },
    };
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, environment_pipeline);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        mesh_pipeline_layout,
        0,
        1,
        &mesh_descriptor_sets.at(frame_index),
        0,
        nullptr
    );
    vkCmdPushConstants(
        command_buffer, mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push
    );
    vkCmdDraw(command_buffer, 3u, 1u, 0u, 0u);
}

auto Runtime::Impl::draw_meshes(VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index)
    -> void
{
    if (draw_list.mesh_commands().empty() or mesh_pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    const auto& mesh_commands = draw_list.mesh_commands();
    const auto camera_position = camera.position();
    const auto camera_forward = normalize_or(camera.pivot() - camera_position, -k_axis_y);
    const auto& lights = draw_list.lights();
    const auto active_shadow_index = shadow_light_index(lights);
    mesh_lighting_upload = build_gpu_lighting(
        lights,
        draw_list.ambient_light(),
        draw_list.environment(),
        camera,
        active_shadow_index,
        shadow_map.resolution
    );
    auto& lighting_buffer = ensure_mesh_lighting_buffer(frame_index);
    std::memcpy(lighting_buffer.mapped, &mesh_lighting_upload, sizeof(mesh_lighting_upload));

    mesh_material_upload.clear();
    mesh_material_upload.reserve(mesh_commands.size());
    for (const auto& command : mesh_commands)
    {
        mesh_material_upload.push_back(to_gpu_material(
            command.material,
            command.mask,
            command.debug,
            command.object_id,
            elapsed_seconds,
            camera_position,
            camera_forward
        ));
    }
    const auto material_byte_count = data_byte_size(mesh_material_upload);
    auto& material_buffer = ensure_mesh_material_buffer(frame_index, material_byte_count);
    std::memcpy(
        material_buffer.mapped, mesh_material_upload.data(), static_cast<usize>(material_byte_count)
    );

    const auto aspect = static_cast<f32>(std::max(1u, extent.width))
                        / static_cast<f32>(std::max(1u, extent.height));
    const auto view_projection = camera.view_projection_matrix(aspect);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        mesh_pipeline_layout,
        0,
        1,
        &mesh_descriptor_sets.at(frame_index),
        0,
        nullptr
    );
    for (auto command_index = 0zu; command_index < mesh_commands.size(); ++command_index)
    {
        const auto& command = mesh_commands[command_index];
        if (!command.mask.visible_to_camera or !command.mesh.valid()
            or command.mesh.id >= meshes.size())
        {
            continue;
        }
        const auto& mesh = meshes[command.mesh.id];
        const auto offsets = std::array<VkDeviceSize, 1>{0};
        const auto vertex_buffers = std::array{mesh.vertices.handle};
        vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers.data(), offsets.data());
        vkCmdBindIndexBuffer(command_buffer, mesh.indices.handle, 0, VK_INDEX_TYPE_UINT32);

        const auto model = command.transform.matrix();
        const auto push = MeshPushConstants{
            .view_projection = view_projection,
            .model = model,
        };
        vkCmdPushConstants(
            command_buffer, mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push
        );
        vkCmdDrawIndexed(
            command_buffer, mesh.index_count, 1, 0, 0, static_cast<u32>(command_index)
        );
    }
}

auto Runtime::Impl::draw_debug(VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index)
    -> void
{
    if (draw_list.debug_segments().empty() or debug_pipeline == VK_NULL_HANDLE)
    {
        return;
    }

    const auto byte_count = data_byte_size(draw_list.debug_segments());
    auto& buffer = ensure_debug_buffer(frame_index, byte_count);
    std::memcpy(buffer.mapped, draw_list.debug_segments().data(), static_cast<usize>(byte_count));

    const auto aspect = static_cast<f32>(std::max(1u, extent.width))
                        / static_cast<f32>(std::max(1u, extent.height));
    const auto push = DebugPushConstants{
        .view_projection = camera.view_projection_matrix(aspect),
        .camera_position = Vec4{camera.position(), 1.0f},
        .camera_right = Vec4{camera.right(), 0.0f},
    };

    const auto vertex_buffers = std::array{buffer.handle};
    const auto offsets = std::array<VkDeviceSize, 1>{0};
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debug_pipeline);
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers.data(), offsets.data());
    vkCmdPushConstants(
        command_buffer, debug_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push
    );
    vkCmdDraw(command_buffer, 18u, static_cast<u32>(draw_list.debug_segments().size()), 0, 0);
}

auto Runtime::Impl::render_frame(
    VkCommandBuffer command_buffer, VkExtent2D extent, u32 frame_index, ImDrawData* draw_data
) -> void
{
    const auto viewport = VkViewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<f32>(extent.width),
        .height = static_cast<f32>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    const auto scissor = VkRect2D{.offset = VkOffset2D{.x = 0, .y = 0}, .extent = extent};
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    draw_meshes(command_buffer, extent, frame_index);
    draw_environment(command_buffer, extent, frame_index);
    draw_debug(command_buffer, extent, frame_index);

    if (draw_data)
    {
        ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
    }
}

auto Runtime::Impl::draw_runtime_ui() -> void
{
    ImGui::SetNextWindowPos(ImVec2{20.0f, 20.0f}, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2{500.0f, 260.0f}, ImGuiCond_Once);
    if (!ImGui::Begin("ds_vk Camera"))
    {
        ImGui::End();
        return;
    }
    ImGui::PushItemWidth(300.0f);

    auto fov_degrees = glm::degrees(camera.fov_y());
    if (ImGui::SliderFloat("FOV", &fov_degrees, 10.0f, 120.0f, "%.1f deg"))
    {
        camera.fov_y() = glm::radians(std::clamp(fov_degrees, 10.0f, 120.0f));
    }
    if (ImGui::SliderFloat("Orbit sensitivity", &camera.orbit_sensitivity(), 0.10f, 4.0f, "%.2f"))
    {
        camera.orbit_sensitivity() = std::clamp(camera.orbit_sensitivity(), 0.10f, 4.0f);
    }
    if (ImGui::SliderFloat("Pivot sensitivity", &camera.pivot_sensitivity(), 0.10f, 4.0f, "%.2f"))
    {
        camera.pivot_sensitivity() = std::clamp(camera.pivot_sensitivity(), 0.10f, 4.0f);
    }
    if (ImGui::SliderFloat("Zoom sensitivity", &camera.zoom_sensitivity(), 0.10f, 4.0f, "%.2f"))
    {
        camera.zoom_sensitivity() = std::clamp(camera.zoom_sensitivity(), 0.10f, 4.0f);
    }

    const auto projection_labels = std::array{"Perspective", "Orthographic"};
    auto projection_index = camera.projection_mode() == ProjectionMode::orthographic ? 1 : 0;
    if (ImGui::Combo("Projection", &projection_index, projection_labels.data(), 2))
    {
        camera.projection_mode() =
            projection_index == 1 ? ProjectionMode::orthographic : ProjectionMode::perspective;
    }

    ImGui::DragFloat3("Pivot", &camera.pivot().x, 0.01f);
    ImGui::Text(
        "Frame %.2f ms | draw %u | debug %u | lights %u",
        static_cast<double>(stats.last_frame_ms),
        stats.mesh_draws,
        stats.debug_segments,
        stats.lights
    );
    ImGui::Text(
        "Descriptor indexing: %s",
        descriptor_indexing.descriptor_indexing ? "available" : "not available"
    );
    ImGui::PopItemWidth();
    ImGui::End();
}

auto Runtime::Impl::handle_event(const SDL_Event& event, bool& done, bool& orbiting, bool& panning)
    -> void
{
    auto& io = ImGui::GetIO();
    ImGui_ImplSDL3_ProcessEvent(&event);

    if (event.type == SDL_EVENT_QUIT)
    {
        done = true;
    }
    if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED
        and event.window.windowID == SDL_GetWindowID(window))
    {
        done = true;
    }
    if (event.type == SDL_EVENT_KEY_DOWN and event.key.key == SDLK_ESCAPE)
    {
        done = true;
    }
    if (event.type == SDL_EVENT_KEY_DOWN and event.key.key == SDLK_SPACE and !event.key.repeat
        and !io.WantCaptureKeyboard)
    {
        input.space_pressed = true;
    }
    if (event.type == SDL_EVENT_WINDOW_RESIZED or event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
    {
        swapchain_rebuild = true;
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN and !io.WantCaptureMouse)
    {
        input.mouse_px = framebuffer_mouse_position(event.button.x, event.button.y);
        if (event.button.button == SDL_BUTTON_RIGHT)
        {
            orbiting = true;
        }
        else if (event.button.button == SDL_BUTTON_MIDDLE)
        {
            panning = true;
        }
        else if (event.button.button == SDL_BUTTON_LEFT)
        {
            input.left_click = MouseClick{
                .occurred = true,
                .position_px = input.mouse_px,
                .click_count = static_cast<u8>(event.button.clicks),
                .modifiers = current_modifiers(),
            };
        }
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP)
    {
        input.mouse_px = framebuffer_mouse_position(event.button.x, event.button.y);
        if (event.button.button == SDL_BUTTON_RIGHT)
        {
            orbiting = false;
        }
        else if (event.button.button == SDL_BUTTON_MIDDLE)
        {
            panning = false;
        }
    }
    if (event.type == SDL_EVENT_MOUSE_MOTION and !io.WantCaptureMouse)
    {
        input.mouse_px = framebuffer_mouse_position(event.motion.x, event.motion.y);
        auto framebuffer_width = 1;
        auto framebuffer_height = 1;
        SDL_GetWindowSizeInPixels(window, &framebuffer_width, &framebuffer_height);
        if (orbiting)
        {
            const auto sensitivity = std::clamp(camera.orbit_sensitivity(), 0.10f, 4.0f);
            camera.yaw() -= event.motion.xrel * 0.006f * sensitivity;
            camera.pitch() = std::clamp(
                camera.pitch() + event.motion.yrel * 0.006f * sensitivity,
                glm::radians(-82.0f),
                glm::radians(82.0f)
            );
        }
        else if (panning)
        {
            const auto sensitivity = std::clamp(camera.pivot_sensitivity(), 0.10f, 4.0f);
            camera.pivot() += camera.pan_offset_world(
                event.motion.xrel * sensitivity,
                event.motion.yrel * sensitivity,
                static_cast<f32>(std::max(1, framebuffer_height))
            );
        }
    }
    if (event.type == SDL_EVENT_MOUSE_WHEEL and !io.WantCaptureMouse)
    {
        const auto sensitivity = std::clamp(camera.zoom_sensitivity(), 0.10f, 4.0f);
        camera.distance() *= std::exp(-event.wheel.y * 0.12f * sensitivity);
        camera.distance() = std::clamp(camera.distance(), 0.12f, 200.0f);
    }
}

auto Runtime::Impl::framebuffer_mouse_position(f32 window_x, f32 window_y) const -> Vec2
{
    auto window_width = 1;
    auto window_height = 1;
    auto framebuffer_width = 1;
    auto framebuffer_height = 1;
    SDL_GetWindowSize(window, &window_width, &window_height);
    SDL_GetWindowSizeInPixels(window, &framebuffer_width, &framebuffer_height);
    const auto scale_x =
        static_cast<f32>(framebuffer_width) / static_cast<f32>(std::max(1, window_width));
    const auto scale_y =
        static_cast<f32>(framebuffer_height) / static_cast<f32>(std::max(1, window_height));
    return Vec2{window_x * scale_x, window_y * scale_y};
}

auto Runtime::Impl::current_modifiers() const noexcept -> KeyboardModifiers
{
    const auto mods = SDL_GetModState();
    return KeyboardModifiers{
        .shift = (mods & SDL_KMOD_SHIFT) != 0,
        .control = (mods & SDL_KMOD_CTRL) != 0,
        .alt = (mods & SDL_KMOD_ALT) != 0,
        .super = (mods & SDL_KMOD_GUI) != 0,
    };
}

auto Runtime::Impl::reset_input_frame() -> void
{
    auto mouse_x = 0.0f;
    auto mouse_y = 0.0f;
    (void) SDL_GetMouseState(&mouse_x, &mouse_y);
    input.left_click = {};
    input.space_pressed = false;
    input.mouse_px = framebuffer_mouse_position(mouse_x, mouse_y);
    input.mouse_captured_by_ui = ImGui::GetIO().WantCaptureMouse;
}

auto Runtime::Impl::rebuild_swapchain_if_needed() -> void
{
    if (!swapchain_rebuild)
    {
        return;
    }

    auto width = 0;
    auto height = 0;
    SDL_GetWindowSizeInPixels(window, &width, &height);
    if (width <= 0 or height <= 0)
    {
        return;
    }

    check_vk_result(vkDeviceWaitIdle(device));
    ImGui_ImplVulkan_SetMinImageCount(min_image_count);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        instance,
        physical_device,
        device,
        &window_data,
        queue_family,
        allocation_callbacks,
        width,
        height,
        min_image_count,
        k_swapchain_image_usage
    );
    install_depth_rendering();
    swapchain_rebuild = false;
}

auto Runtime::Impl::create_capture_buffer(SwapchainCapture& capture) -> void
{
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = capture.size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    check_vk_result(vmaCreateBuffer(
        vma_allocator, &buffer_info, &allocation_info, &capture.buffer, &capture.allocation, nullptr
    ));
}

auto Runtime::Impl::destroy_capture_buffer(SwapchainCapture& capture) noexcept -> void
{
    if (capture.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(vma_allocator, capture.buffer, capture.allocation);
        capture.buffer = VK_NULL_HANDLE;
        capture.allocation = VK_NULL_HANDLE;
    }
}

auto Runtime::Impl::record_capture_commands(
    const VkCommandBuffer command_buffer,
    const ImGui_ImplVulkanH_Frame* frame,
    const SwapchainCapture& capture
) -> void
{
    VkImageMemoryBarrier to_transfer{};
    to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_transfer.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_transfer.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.image = frame->Backbuffer;
    to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_transfer.subresourceRange.levelCount = 1;
    to_transfer.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &to_transfer
    );

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = VkExtent3D{.width = capture.width, .height = capture.height, .depth = 1};
    vkCmdCopyImageToBuffer(
        command_buffer,
        frame->Backbuffer,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        capture.buffer,
        1,
        &copy
    );

    auto to_present = to_transfer;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &to_present
    );
}

auto Runtime::Impl::write_capture_png(const SwapchainCapture& capture) -> void
{
    if (capture.format != VK_FORMAT_B8G8R8A8_UNORM and capture.format != VK_FORMAT_B8G8R8A8_SRGB
        and capture.format != VK_FORMAT_R8G8B8A8_UNORM
        and capture.format != VK_FORMAT_R8G8B8A8_SRGB)
    {
        throw std::runtime_error("screenshot capture only supports 8-bit RGBA/BGRA formats");
    }

    if (!capture.path.parent_path().empty())
    {
        std::filesystem::create_directories(capture.path.parent_path());
    }
    const auto bgra =
        capture.format == VK_FORMAT_B8G8R8A8_UNORM or capture.format == VK_FORMAT_B8G8R8A8_SRGB;
    auto rgba = std::vector<u8>(
        static_cast<usize>(capture.width) * static_cast<usize>(capture.height) * 4zu
    );

    void* mapped = nullptr;
    check_vk_result(vmaMapMemory(vma_allocator, capture.allocation, &mapped));
    try
    {
        check_vk_result(
            vmaInvalidateAllocation(vma_allocator, capture.allocation, 0, capture.size)
        );
        const auto* pixels = static_cast<const u8*>(mapped);
        for (auto y = 0u; y < capture.height; ++y)
        {
            for (auto x = 0u; x < capture.width; ++x)
            {
                const auto i = (static_cast<usize>(y) * capture.width + x) * 4zu;
                rgba[i + 0zu] = bgra ? pixels[i + 2zu] : pixels[i + 0zu];
                rgba[i + 1zu] = pixels[i + 1zu];
                rgba[i + 2zu] = bgra ? pixels[i + 0zu] : pixels[i + 2zu];
                rgba[i + 3zu] = capture.transparent_background ? pixels[i + 3zu] : 255u;
            }
        }
    }
    catch (...)
    {
        vmaUnmapMemory(vma_allocator, capture.allocation);
        throw;
    }
    vmaUnmapMemory(vma_allocator, capture.allocation);

    const auto result = stbi_write_png(
        capture.path.string().c_str(),
        static_cast<int>(capture.width),
        static_cast<int>(capture.height),
        4,
        rgba.data(),
        static_cast<int>(capture.width * 4u)
    );
    if (result == 0)
    {
        throw std::runtime_error("failed to write screenshot PNG");
    }
}

auto Runtime::Impl::present_frame() -> void
{
    if (swapchain_rebuild)
    {
        return;
    }
    const auto semaphore_index = static_cast<int>(window_data.SemaphoreIndex);
    const auto render_complete =
        window_data.FrameSemaphores[semaphore_index].RenderCompleteSemaphore;
    VkPresentInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete;
    info.swapchainCount = 1;
    info.pSwapchains = &window_data.Swapchain;
    info.pImageIndices = &window_data.FrameIndex;
    const auto result = vkQueuePresentKHR(queue, &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR or result == VK_SUBOPTIMAL_KHR)
    {
        swapchain_rebuild = true;
    }
    else
    {
        check_vk_result(result);
    }
    window_data.SemaphoreIndex = (window_data.SemaphoreIndex + 1u) % window_data.SemaphoreCount;
}

auto Runtime::Impl::initialize() -> void
{
    setup_sdl();

    auto sdl_extension_count = 0u;
    const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
    if (!sdl_extensions)
    {
        throw std::runtime_error("SDL_Vulkan_GetInstanceExtensions failed");
    }
    auto instance_extensions = std::vector<const char*>{};
    instance_extensions.reserve(sdl_extension_count + 4zu);
    for (auto i = 0u; i < sdl_extension_count; ++i)
    {
        instance_extensions.push_back(sdl_extensions[i]);
    }
    setup_vulkan(std::move(instance_extensions));

    auto surface = VkSurfaceKHR{VK_NULL_HANDLE};
    if (!SDL_Vulkan_CreateSurface(window, instance, allocation_callbacks, &surface))
    {
        throw std::runtime_error("SDL_Vulkan_CreateSurface failed");
    }

    auto width = 0;
    auto height = 0;
    SDL_GetWindowSizeInPixels(window, &width, &height);
    setup_vulkan_window(surface, width, height);
    create_shadow_map();
    create_default_texture();
    setup_imgui();
    create_pipelines();
    SDL_ShowWindow(window);
}

auto Runtime::Impl::shutdown() noexcept -> void
{
    if (device != VK_NULL_HANDLE)
    {
        (void) vkDeviceWaitIdle(device);
    }

    for (auto& buffer : debug_segment_buffers)
    {
        destroy_buffer(buffer);
    }
    debug_segment_buffers.clear();

    for (auto& buffer : mesh_material_buffers)
    {
        destroy_buffer(buffer);
    }
    mesh_material_buffers.clear();
    mesh_material_upload.clear();

    for (auto& buffer : mesh_lighting_buffers)
    {
        destroy_buffer(buffer);
    }
    mesh_lighting_buffers.clear();
    mesh_lighting_upload = {};

    for (auto& mesh : meshes)
    {
        destroy_buffer(mesh.vertices);
        destroy_buffer(mesh.indices);
    }
    meshes.clear();

    destroy_pipelines();
    destroy_shadow_map();

    for (auto& texture : textures)
    {
        destroy_texture(texture);
    }
    textures.clear();

    if (imgui_ready)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        imgui_ready = false;
    }

    if (window_data.Surface != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkanH_DestroyWindow(instance, device, &window_data, allocation_callbacks);
        destroy_depth_attachments();
        vkDestroySurfaceKHR(instance, window_data.Surface, allocation_callbacks);
        window_data.Surface = VK_NULL_HANDLE;
    }

    if (pipeline_cache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device, pipeline_cache, allocation_callbacks);
        pipeline_cache = VK_NULL_HANDLE;
    }
    if (imgui_descriptor_pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, imgui_descriptor_pool, allocation_callbacks);
        imgui_descriptor_pool = VK_NULL_HANDLE;
    }
    if (vma_allocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(vma_allocator);
        vma_allocator = VK_NULL_HANDLE;
    }
    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, allocation_callbacks);
        device = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, allocation_callbacks);
        instance = VK_NULL_HANDLE;
    }
    if (window)
    {
        SDL_DestroyWindow(window);
        window = nullptr;
    }
    if (sdl_ready)
    {
        SDL_Quit();
        sdl_ready = false;
    }
}

auto Runtime::Impl::run(const detail::RuntimeCallbacks& callbacks, Runtime& runtime) -> int
{
    initialize();
    callbacks.setup(callbacks.user, runtime);

    auto done = false;
    auto orbiting = false;
    auto panning = false;
    auto frame_counter = 0u;
    auto previous = std::chrono::steady_clock::now();
    pending_screenshot = config.screenshot_path;
    pending_screenshot_transparent = config.transparent_screenshot;

    while (!done)
    {
        const auto frame_begin_cpu = std::chrono::steady_clock::now();
        const auto now = frame_begin_cpu;
        const auto dt_seconds = std::chrono::duration<f32>(now - previous).count();
        previous = now;
        elapsed_seconds += dt_seconds;
        reset_input_frame();

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            handle_event(event, done, orbiting, panning);
        }
        rebuild_swapchain_if_needed();
        if (done)
        {
            break;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        const auto semaphore_index = static_cast<int>(window_data.SemaphoreIndex);
        const auto image_acquired =
            window_data.FrameSemaphores[semaphore_index].ImageAcquiredSemaphore;
        const auto render_complete =
            window_data.FrameSemaphores[semaphore_index].RenderCompleteSemaphore;
        const auto acquire_result = vkAcquireNextImageKHR(
            device,
            window_data.Swapchain,
            UINT64_MAX,
            image_acquired,
            VK_NULL_HANDLE,
            &window_data.FrameIndex
        );
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            swapchain_rebuild = true;
            ImGui::EndFrame();
            continue;
        }
        if (acquire_result == VK_SUBOPTIMAL_KHR)
        {
            swapchain_rebuild = true;
        }
        else
        {
            check_vk_result(acquire_result);
        }

        const auto* frame = &window_data.Frames[static_cast<int>(window_data.FrameIndex)];
        check_vk_result(vkWaitForFences(device, 1, &frame->Fence, VK_TRUE, UINT64_MAX));
        check_vk_result(vkResetFences(device, 1, &frame->Fence));
        check_vk_result(vkResetCommandPool(device, frame->CommandPool, 0));

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk_result(vkBeginCommandBuffer(frame->CommandBuffer, &begin_info));

        draw_list.clear();
        const auto extent = VkExtent2D{
            .width = static_cast<u32>(std::max(0, window_data.Width)),
            .height = static_cast<u32>(std::max(0, window_data.Height)),
        };
        auto frame_context = FrameContext{
            .instance = instance,
            .physical_device = physical_device,
            .device = device,
            .graphics_queue = queue,
            .graphics_queue_family = queue_family,
            .command_buffer = frame->CommandBuffer,
            .allocator = vma_allocator,
            .extent = extent,
            .frame_index = frame_counter,
            .dt_seconds = dt_seconds,
            .camera = camera,
            .draw = draw_list,
            .input = input,
            .descriptor_indexing = descriptor_indexing,
            .stats = stats,
        };

        const auto update_begin = std::chrono::steady_clock::now();
        callbacks.update(callbacks.user, frame_context, dt_seconds);
        const auto update_end = std::chrono::steady_clock::now();

        if (!config.hide_ui)
        {
            draw_runtime_ui();
            callbacks.draw_ui(callbacks.user, frame_context);
        }
        ImGui::Render();
        const auto ui_end = std::chrono::steady_clock::now();

        auto capture = SwapchainCapture{};
        if (!pending_screenshot.empty() and frame_counter >= 4u)
        {
            capture.width = extent.width;
            capture.height = extent.height;
            capture.format = window_data.SurfaceFormat.format;
            capture.size = static_cast<VkDeviceSize>(capture.width)
                           * static_cast<VkDeviceSize>(capture.height) * 4u;
            capture.path = pending_screenshot;
            capture.transparent_background = pending_screenshot_transparent;
            create_capture_buffer(capture);
        }

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = window_data.RenderPass;
        render_pass_info.framebuffer = frame->Framebuffer;
        render_pass_info.renderArea.extent = extent;
        auto clear_values = std::array{
            window_data.ClearValue,
            VkClearValue{.depthStencil = {.depth = 1.0f, .stencil = 0}},
        };
        render_pass_info.clearValueCount = static_cast<u32>(clear_values.size());
        render_pass_info.pClearValues = clear_values.data();
        const auto render_begin = std::chrono::steady_clock::now();
        draw_shadow_map(frame->CommandBuffer);
        vkCmdBeginRenderPass(frame->CommandBuffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
        render_frame(
            frame->CommandBuffer,
            extent,
            window_data.FrameIndex,
            config.hide_ui ? nullptr : ImGui::GetDrawData()
        );
        vkCmdEndRenderPass(frame->CommandBuffer);
        if (capture.buffer != VK_NULL_HANDLE)
        {
            record_capture_commands(frame->CommandBuffer, frame, capture);
        }

        const auto wait_stage = VkPipelineStageFlags{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_acquired;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &frame->CommandBuffer;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_complete;

        check_vk_result(vkEndCommandBuffer(frame->CommandBuffer));
        check_vk_result(vkQueueSubmit(queue, 1, &submit_info, frame->Fence));
        if (capture.buffer != VK_NULL_HANDLE)
        {
            check_vk_result(vkWaitForFences(device, 1, &frame->Fence, VK_TRUE, UINT64_MAX));
            write_capture_png(capture);
            destroy_capture_buffer(capture);
            std::cout << "[screenshot] wrote " << pending_screenshot << '\n' << std::flush;
            pending_screenshot.clear();
            pending_screenshot_transparent = false;
        }
        const auto render_end = std::chrono::steady_clock::now();

        present_frame();
        const auto frame_end_cpu = std::chrono::steady_clock::now();

        stats = RuntimeStats{
            .last_frame_ms =
                std::chrono::duration<f32, std::milli>(frame_end_cpu - frame_begin_cpu).count(),
            .last_update_ms =
                std::chrono::duration<f32, std::milli>(update_end - update_begin).count(),
            .last_ui_ms = std::chrono::duration<f32, std::milli>(ui_end - update_end).count(),
            .last_render_ms =
                std::chrono::duration<f32, std::milli>(render_end - render_begin).count(),
            .mesh_draws = static_cast<u32>(draw_list.mesh_commands().size()),
            .debug_segments = static_cast<u32>(draw_list.debug_segments().size()),
            .lights = static_cast<u32>(draw_list.lights().size()),
        };

        ++frame_counter;
        if (config.smoke_frames > 0u and frame_counter >= config.smoke_frames
            and pending_screenshot.empty())
        {
            done = true;
        }
    }

    callbacks.shutdown(callbacks.user, runtime);
    return 0;
}

Runtime::Runtime(RuntimeConfig config) : impl_(std::make_unique<Impl>(std::move(config)))
{
}

Runtime::~Runtime()
{
    if (impl_)
    {
        impl_->shutdown();
    }
}

Runtime::Runtime(Runtime&&) noexcept = default;

auto Runtime::operator=(Runtime&& other) noexcept -> Runtime&
{
    if (this != &other)
    {
        if (impl_)
        {
            impl_->shutdown();
        }
        impl_ = std::move(other.impl_);
    }
    return *this;
}

auto Runtime::run_callbacks(const detail::RuntimeCallbacks& callbacks) -> int
{
    return impl_->run(callbacks, *this);
}

auto Runtime::upload_mesh(const MeshData& mesh) -> MeshHandle
{
    return impl_->upload_mesh(mesh);
}

auto Runtime::replace_mesh(MeshHandle handle, const MeshData& mesh) -> MeshHandle
{
    return impl_->replace_mesh(handle, mesh);
}

auto Runtime::load_texture(const std::filesystem::path& path, const TextureLoadConfig& config)
    -> TextureHandle
{
    return impl_->load_texture(path, config);
}

auto Runtime::load_hdr_texture(
    const std::filesystem::path& path, const HdrTextureLoadConfig& config
) -> TextureHandle
{
    return impl_->load_hdr_texture(path, config);
}

auto Runtime::request_screenshot(std::filesystem::path path, const bool transparent) -> void
{
    impl_->pending_screenshot = std::move(path);
    impl_->pending_screenshot_transparent = transparent;
}

auto Runtime::camera(const CameraConfig& config) noexcept -> Camera&
{
    return impl_->camera.configure(config);
}

auto Runtime::camera() noexcept -> Camera&
{
    return impl_->camera;
}

auto Runtime::camera() const noexcept -> const Camera&
{
    return impl_->camera;
}

auto Runtime::stats() const noexcept -> const RuntimeStats&
{
    return impl_->stats;
}

auto Runtime::descriptor_indexing_support() const noexcept -> const DescriptorIndexingSupport&
{
    return impl_->descriptor_indexing;
}
}  // namespace ds_vk
