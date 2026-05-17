#pragma once

#include "ds_vk/camera.hpp"
#include "ds_vk/debug_draw.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/types.hpp"

#include <concepts>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace ds_vk
{

struct DescriptorIndexingSupport
{
    bool descriptor_indexing{};
    bool sampled_image_array_dynamic_indexing{};
    bool runtime_descriptor_array{};
    bool descriptor_binding_partially_bound{};
    bool sampled_image_non_uniform_indexing{};
    bool storage_buffer_non_uniform_indexing{};
    bool sampled_image_update_after_bind{};
    bool storage_buffer_update_after_bind{};
};

struct MaterialTextures
{
    TextureHandle base_color{};
};

struct Material
{
    Color base_color{Color::white};
    Color emissive_color{Color::black};
    f32 metallic{0.0f};
    f32 roughness{0.55f};
    f32 ambient_occlusion{1.0f};
    MaterialTextures textures{};
};

enum class MeshDebugMode : u8
{
    none = 0,
    color_override = 1,
    selected_pulse = 2,
    scalar_heatmap = 3,
    normal = 4,
    object_id = 5,
    camera_depth = 6,
    triangle_selected_pulse = 7,
    world_z_ramp = 8,
    facet_color = 9,
    angle_shaded = 10,
};

struct MeshDebugConfig
{
    MeshDebugMode mode{MeshDebugMode::none};
    Color color{1.0f, 0.0f, 1.0f, 0.85f};
    f32 scalar{};
    Vec2 scalar_range{0.0f, 1.0f};
    bool selected{};
    bool hidden{};
};

struct MeshRenderMask
{
    bool visible_to_camera{true};
    bool shadow_producer{true};
    bool shadow_consumer{true};
    bool light_receiver{true};
};

struct MeshDrawConfig
{
    MeshHandle mesh{};
    ObjectId object_id{};
    Transform transform{};
    Material material{};
    MeshRenderMask mask{};
    MeshDebugConfig debug{};
};

struct BasicMeshDrawConfig
{
    MeshHandle mesh{};
    ObjectId object_id{};
    Transform transform{};
    Color color{Color::white};
    MeshRenderMask mask{};
    MeshDebugConfig debug{};
};

struct MeshDrawCommand
{
    MeshHandle mesh{};
    ObjectId object_id{};
    Transform transform{};
    Material material{};
    MeshRenderMask mask{};
    MeshDebugConfig debug{};
};

enum class LightType : u8
{
    directional = 0,
    radial = 1,
    spot = 2,
};

struct LightShadowConfig
{
    bool enabled{};
    f32 bias{0.0025f};
    f32 strength{0.70f};
    f32 near_plane{0.05f};
    f32 far_plane{28.0f};
    f32 ortho_extent{7.0f};
};

struct LightConfig
{
    LightType type{LightType::directional};
    Vec3 position{0.0f, 0.0f, 2.0f};
    Vec3 direction{-0.45f, -0.35f, -0.82f};
    Color color{Color::white};
    f32 intensity{1.0f};
    f32 range{6.0f};
    f32 inner_cone_angle{glm::radians(12.0f)};
    f32 outer_cone_angle{glm::radians(24.0f)};
    LightShadowConfig shadow{};
    bool enabled{true};
};

struct EnvironmentConfig
{
    TextureHandle texture{};
    f32 lighting_intensity{0.0f};
    f32 background_intensity{0.0f};
    f32 rotation_radians{};
    Color background_color{0.14f, 0.16f, 0.18f, 0.0f};
    Color background_top_color{0.36f, 0.43f, 0.48f, 0.0f};
    bool gradient_background{};
    bool visible_to_camera{true};
};

struct DirectionalLightConfig
{
    Vec3 direction{-0.45f, -0.35f, -0.82f};
    Color color{Color::white};
    f32 intensity{1.0f};
    LightShadowConfig shadow{};
    bool enabled{true};
};

struct RadialLightConfig
{
    Vec3 position{0.0f, 0.0f, 2.0f};
    Color color{Color::white};
    f32 intensity{8.0f};
    f32 range{5.0f};
    bool enabled{true};
};

struct SpotLightConfig
{
    Vec3 position{0.0f, 0.0f, 3.0f};
    Vec3 direction{-k_axis_z};
    Color color{Color::white};
    f32 intensity{18.0f};
    f32 range{7.0f};
    f32 inner_cone_angle{glm::radians(12.0f)};
    f32 outer_cone_angle{glm::radians(24.0f)};
    LightShadowConfig shadow{};
    bool enabled{true};
};

class DrawList
{
  public:
    auto clear() -> void;
    auto set_ambient_light(Color color) -> void;
    // clang-format off
    auto draw_mesh        (const MeshDrawConfig&        ) -> void;
    auto draw_basic_mesh  (const BasicMeshDrawConfig&   ) -> void;
    auto debug_line       (const DebugLineConfig&       ) -> void;
    auto debug_arrow      (const DebugArrowConfig&      ) -> void;
    auto debug_sphere     (const DebugSphereConfig&     ) -> void;
    auto add_light        (const LightConfig&           ) -> void;
    auto directional_light(const DirectionalLightConfig&) -> void;
    auto radial_light     (const RadialLightConfig&     ) -> void;
    auto spot_light       (const SpotLightConfig&       ) -> void;
    auto set_environment  (const EnvironmentConfig&     ) -> void;

    [[nodiscard]] auto mesh_commands()         const noexcept -> std::span<const MeshDrawCommand>;
    [[nodiscard]] auto debug_segments()        const noexcept -> std::span<const DebugSegment>;
    [[nodiscard]] auto debug_on_top_segments() const noexcept -> std::span<const DebugSegment>;
    [[nodiscard]] auto lights()                const noexcept -> std::span<const LightConfig>;
    [[nodiscard]] auto ambient_light()         const noexcept -> Color;
    [[nodiscard]] auto environment()           const noexcept -> const EnvironmentConfig&;
    // clang-format on

  private:
    std::vector<MeshDrawCommand> mesh_commands_{};
    std::vector<DebugSegment> debug_segments_{};
    std::vector<DebugSegment> debug_on_top_segments_{};
    std::vector<LightConfig> lights_{};
    Color ambient_light_{0.035f, 0.040f, 0.050f, 1.0f};
    EnvironmentConfig environment_{};
};

struct RuntimeStats
{
    f32 last_frame_ms{};
    f32 last_update_ms{};
    f32 last_ui_ms{};
    f32 last_render_ms{};
    u32 mesh_draws{};
    u32 mesh_batches{};
    u32 debug_segments{};
    u32 lights{};
};

struct RuntimeConfig
{
    std::string window_title{"ds_vk app"};
    u32 initial_width{1280};
    u32 initial_height{800};
    std::filesystem::path shader_dir{};
    std::filesystem::path screenshot_path{};
    u32 smoke_frames{};
    bool hide_ui{};
    bool transparent_screenshot{};
    bool enable_validation{true};
    Color clear_color{0.035f, 0.045f, 0.055f, 1.0f};
    u32 shadow_map_resolution{2048};
};

struct TextureLoadConfig
{
    bool srgb{true};
};

struct HdrTextureLoadConfig
{
    f32 exposure{1.0f};
};

struct MeshReserveConfig
{
    MeshHandle mesh{};
    usize vertex_capacity{};
    usize index_capacity{};
    MeshVertexFormat vertex_format{MeshVertexFormat::standard};
};

struct MeshUpdateConfig
{
    bool validate_indices{};
};

struct KeyboardModifiers
{
    bool shift{};
    bool control{};
    bool alt{};
    bool super{};
};

struct MouseClick
{
    bool occurred{};
    Vec2 position_px{};
    u8 click_count{};
    KeyboardModifiers modifiers{};
};

struct InputState
{
    Vec2 mouse_px{};
    bool mouse_captured_by_ui{};
    bool space_pressed{};
    bool key_g_pressed{};
    bool key_r_pressed{};
    bool key_s_pressed{};
    bool key_x_pressed{};
    bool key_y_pressed{};
    bool key_z_pressed{};
    bool key_c_pressed{};
    bool key_enter_pressed{};
    MouseClick left_click{};
};

struct FrameContext
{
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue graphics_queue{VK_NULL_HANDLE};
    u32 graphics_queue_family{};
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
    VkRenderPass main_render_pass{VK_NULL_HANDLE};
    VmaAllocator allocator{VK_NULL_HANDLE};
    VkExtent2D extent{};
    u32 frame_index{};
    u32 swapchain_image_index{};
    u32 swapchain_image_count{};
    f32 dt_seconds{};
    Camera& camera;
    DrawList& draw;
    const InputState& input;
    const DescriptorIndexingSupport& descriptor_indexing;
    const RuntimeStats& stats;
};

class Runtime;

namespace detail
{
template <typename App>
concept has_setup = requires(App& app, Runtime& runtime) {
    { app.setup(runtime) } -> std::same_as<void>;
};

template <typename App>
concept has_update = requires(App& app, FrameContext& frame, f32 dt_seconds) {
    { app.update(frame, dt_seconds) } -> std::same_as<void>;
};

template <typename App>
concept has_draw_ui = requires(App& app, FrameContext& frame) {
    { app.draw_ui(frame) } -> std::same_as<void>;
};

template <typename App>
concept has_shutdown = requires(App& app, Runtime& runtime) {
    { app.shutdown(runtime) } -> std::same_as<void>;
};

template <typename App>
concept has_runtime_hook =
    has_setup<App> or has_update<App> or has_draw_ui<App> or has_shutdown<App>;
}  // namespace detail

class Runtime
{
  public:
    explicit Runtime(RuntimeConfig = {});
    ~Runtime();

    Runtime(const Runtime&) = delete;
    auto operator=(const Runtime&) -> Runtime& = delete;
    Runtime(Runtime&&) noexcept;
    auto operator=(Runtime&&) noexcept -> Runtime&;

    // clang-format off
    auto initialize()                -> void;
    auto shutdown() noexcept         -> void;

    [[nodiscard]] auto begin_frame() -> FrameContext*;
    [[nodiscard]] auto frame()       -> FrameContext&;
    [[nodiscard]] auto frame() const -> const FrameContext&;

    auto draw_runtime_ui()    -> void;
    auto render_shadow_pass() -> void;
    auto begin_main_pass()    -> void;
    auto render_draw_list()   -> void;
    auto render_imgui()       -> void;
    auto end_main_pass()      -> void;
    auto end_frame()          -> void;

    [[nodiscard]] auto ui_visible() const noexcept -> bool;

    [[nodiscard]] auto upload_mesh(const MeshData&)                        -> MeshHandle;
    [[nodiscard]] auto upload_mesh(const PositionNormalMeshData&)          -> MeshHandle;
    [[nodiscard]] auto upload_mesh(const QuantizedPositionNormalMeshData&) -> MeshHandle;
    [[nodiscard]] auto reserve_mesh_capacity(const MeshReserveConfig&)     -> MeshHandle;

    // Reuses existing buffers when capacity permits. Callers must avoid updating a handle
    // still used by in-flight command buffers.
    [[nodiscard]] auto update_mesh(MeshHandle, const MeshData&, const MeshUpdateConfig& = {})                        -> MeshHandle;
    [[nodiscard]] auto update_mesh(MeshHandle, const PositionNormalMeshData&, const MeshUpdateConfig& = {})          -> MeshHandle;
    [[nodiscard]] auto update_mesh(MeshHandle, const QuantizedPositionNormalMeshData&, const MeshUpdateConfig& = {}) -> MeshHandle;

    [[nodiscard]] auto replace_mesh(MeshHandle, const MeshData&) -> MeshHandle;
    [[nodiscard]] auto load_texture(const std::filesystem::path&, const TextureLoadConfig& = {}) -> TextureHandle;
    [[nodiscard]] auto load_hdr_texture(const std::filesystem::path&, const HdrTextureLoadConfig& = {}) -> TextureHandle;
    [[nodiscard]] auto upload_texture_rgba(std::span<const ColorU8>, u32 width, u32 height, const TextureLoadConfig& = {}) -> TextureHandle;
    [[nodiscard]] auto imgui_texture_id(TextureHandle) -> uptr;
    auto request_screenshot(std::filesystem::path path, bool transparent = false) -> void;

    auto camera(const CameraConfig&)                       noexcept -> Camera&;
    [[nodiscard]] auto camera()                            noexcept -> Camera&;
    [[nodiscard]] auto camera()                      const noexcept -> const Camera&;
    [[nodiscard]] auto stats()                       const noexcept -> const RuntimeStats&;
    [[nodiscard]] auto descriptor_indexing_support() const noexcept -> const DescriptorIndexingSupport&;
    // clang-format on

    template <typename App>
    [[nodiscard]]
    auto run_prototype(App& app) -> int
    {
        static_assert(
            detail::has_runtime_hook<App>,
            "ds_vk prototype apps must provide at least one of setup(Runtime&), "
            "update(FrameContext&, f32), draw_ui(FrameContext&), or shutdown(Runtime&)."
        );

        initialize();
        if constexpr (detail::has_setup<App>)
        {
            app.setup(*this);
        }

        while (auto* current_frame = begin_frame())
        {
            if constexpr (detail::has_update<App>)
            {
                app.update(*current_frame, current_frame->dt_seconds);
            }

            if (ui_visible())
            {
                draw_runtime_ui();
                if constexpr (detail::has_draw_ui<App>)
                {
                    app.draw_ui(*current_frame);
                }
            }

            render_shadow_pass();
            begin_main_pass();
            render_draw_list();
            render_imgui();
            end_main_pass();
            end_frame();
        }

        if constexpr (detail::has_shutdown<App>)
        {
            app.shutdown(*this);
        }
        return 0;
    }

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace ds_vk
