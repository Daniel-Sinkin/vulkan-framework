#pragma once

#include "ds_vk/camera.hpp"
#include "ds_vk/mesh.hpp"

#include <concepts>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace ds_vk
{
struct MeshHandle
{
    u32 index{std::numeric_limits<u32>::max()};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return index != std::numeric_limits<u32>::max();
    }
};

struct TextureHandle
{
    u32 index{std::numeric_limits<u32>::max()};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return index != std::numeric_limits<u32>::max();
    }
};

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

struct MeshDrawConfig
{
    MeshHandle mesh{};
    ObjectId object_id{};
    Transform transform{};
    Material material{};
    MeshDebugConfig debug{};
};

struct BasicMeshDrawConfig
{
    MeshHandle mesh{};
    ObjectId object_id{};
    Transform transform{};
    Color color{Color::white};
    MeshDebugConfig debug{};
};

struct MeshDrawCommand
{
    MeshHandle mesh{};
    ObjectId object_id{};
    Transform transform{};
    Material material{};
    MeshDebugConfig debug{};
};

struct DebugLineConfig
{
    Vec3 start{};
    Vec3 end{};
    Color color{Color::white};
    f32 width{0.012f};
};

struct DebugArrowConfig
{
    Vec3 origin{};
    Vec3 vector{};
    Color color{Color::white};
    f32 width{0.016f};
};

struct DebugSphereConfig
{
    Vec3 center{};
    f32 radius{1.0f};
    Color color{Color::white};
    u32 segments{32u};
    f32 width{0.010f};
};

struct DebugSegment
{
    Vec3 start{};
    f32 width{0.012f};
    Vec3 end{};
    f32 arrow_tip{};
    Color color{Color::white};
};

class DrawList
{
  public:
    auto clear() -> void;
    auto draw_mesh(const MeshDrawConfig& config) -> void;
    auto draw_basic_mesh(const BasicMeshDrawConfig& config) -> void;
    auto draw_basic_mesh(
        MeshHandle mesh, const Transform& transform = Transform{}, Color color = Color::white
    ) -> void;
    auto debug_line(const DebugLineConfig& config) -> void;
    auto debug_line(Vec3 start, Vec3 end, Color color, f32 width = 0.012f) -> void;
    auto debug_arrow(const DebugArrowConfig& config) -> void;
    auto debug_arrow(Vec3 origin, Vec3 vector, Color color, f32 width = 0.016f) -> void;
    auto debug_sphere(const DebugSphereConfig& config) -> void;
    auto debug_sphere(Vec3 center, f32 radius, Color color, u32 segments = 32u, f32 width = 0.010f)
        -> void;

    [[nodiscard]] auto mesh_commands() const noexcept -> const std::vector<MeshDrawCommand>&;
    [[nodiscard]] auto debug_segments() const noexcept -> const std::vector<DebugSegment>&;

  private:
    std::vector<MeshDrawCommand> mesh_commands_{};
    std::vector<DebugSegment> debug_segments_{};
};

struct RuntimeStats
{
    f32 last_frame_ms{};
    f32 last_update_ms{};
    f32 last_ui_ms{};
    f32 last_render_ms{};
    u32 mesh_draws{};
    u32 debug_segments{};
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
};

struct TextureLoadConfig
{
    bool srgb{true};
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
    VmaAllocator allocator{VK_NULL_HANDLE};
    VkExtent2D extent{};
    u32 frame_index{};
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

struct RuntimeCallbacks
{
    using SetupFn = auto (*)(void*, Runtime&) -> void;
    using UpdateFn = auto (*)(void*, FrameContext&, f32) -> void;
    using DrawUiFn = auto (*)(void*, FrameContext&) -> void;
    using ShutdownFn = auto (*)(void*, Runtime&) -> void;

    void* user{};
    SetupFn setup{};
    UpdateFn update{};
    DrawUiFn draw_ui{};
    ShutdownFn shutdown{};
};

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
    has_setup<App> || has_update<App> || has_draw_ui<App> || has_shutdown<App>;
}  // namespace detail

class Runtime
{
  public:
    explicit Runtime(RuntimeConfig config = {});
    ~Runtime();

    Runtime(const Runtime&) = delete;
    auto operator=(const Runtime&) -> Runtime& = delete;
    Runtime(Runtime&&) noexcept;
    auto operator=(Runtime&&) noexcept -> Runtime&;

    template <typename App>
    auto run(App& app) -> int
    {
        static_assert(
            detail::has_runtime_hook<App>,
            "ds_vk apps must provide at least one of setup(Runtime&), "
            "update(FrameContext&, f32), draw_ui(FrameContext&), or shutdown(Runtime&)."
        );

        const auto callbacks = detail::RuntimeCallbacks{
            .user = &app,
            .setup = [](void* user, Runtime& runtime) -> void
            {
                if constexpr (detail::has_setup<App>)
                {
                    static_cast<App*>(user)->setup(runtime);
                }
            },
            .update = [](void* user, FrameContext& frame, const f32 dt_seconds) -> void
            {
                if constexpr (detail::has_update<App>)
                {
                    static_cast<App*>(user)->update(frame, dt_seconds);
                }
            },
            .draw_ui = [](void* user, FrameContext& frame) -> void
            {
                if constexpr (detail::has_draw_ui<App>)
                {
                    static_cast<App*>(user)->draw_ui(frame);
                }
            },
            .shutdown = [](void* user, Runtime& runtime) -> void
            {
                if constexpr (detail::has_shutdown<App>)
                {
                    static_cast<App*>(user)->shutdown(runtime);
                }
            },
        };
        return run_callbacks(callbacks);
    }

    auto upload_mesh(const MeshData& mesh) -> MeshHandle;
    auto replace_mesh(MeshHandle handle, const MeshData& mesh) -> MeshHandle;
    auto load_texture(const std::filesystem::path& path, const TextureLoadConfig& config = {})
        -> TextureHandle;
    auto request_screenshot(std::filesystem::path path, bool transparent = false) -> void;

    auto camera(const CameraConfig& config) noexcept -> Camera&;
    [[nodiscard]] auto camera() noexcept -> Camera&;
    [[nodiscard]] auto camera() const noexcept -> const Camera&;
    [[nodiscard]] auto stats() const noexcept -> const RuntimeStats&;
    [[nodiscard]] auto descriptor_indexing_support() const noexcept
        -> const DescriptorIndexingSupport&;

  private:
    auto run_callbacks(const detail::RuntimeCallbacks& callbacks) -> int;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace ds_vk
