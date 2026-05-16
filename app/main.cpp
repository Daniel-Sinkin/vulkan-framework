#include "ds_vk/plugins/picker.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace ds_vk;

#ifndef DS_VK_ASSET_DIR
#    define DS_VK_ASSET_DIR "assets"
#endif

constexpr auto k_selection_color = Color{1.0f, 0.0f, 0.85f, 0.86f};

[[nodiscard]] auto asset_path(const std::filesystem::path& relative) -> std::filesystem::path
{
    return std::filesystem::path{DS_VK_ASSET_DIR} / relative;
}

class BasicViewerApp final
{
  public:
    auto setup(ds_vk::Runtime& runtime) -> void
    {
        runtime_ = &runtime;

        constexpr Color floor_color{0.55f, 0.58f, 0.53f};
        constexpr Color cube_color{0.55f, 0.58f, 0.53f};

        floor_mesh_ = runtime.upload_mesh(make_quad(8.0f, floor_color));
        cube_mesh_ = runtime.upload_mesh(make_cube(1.0f, cube_color));
        floor_texture_ =
            runtime.load_texture(asset_path("textures/polyhaven/concrete_floor_diff_1k.jpg"));
        cube_texture_ =
            runtime.load_texture(asset_path("textures/polyhaven/wood_table_001_diff_1k.png"));
        materials_.floor.textures.base_color = floor_texture_;
        materials_.cube.textures.base_color = cube_texture_;
        rebuild_sphere();
        rebuild_vector_field();

        runtime.camera({
            .pivot = 0.7f * k_axis_z,
            .distance = 5.4f,
            .yaw = glm::radians(42.0f),
            .pitch = glm::radians(25.0f),
        });
    }

    auto update(ds_vk::FrameContext& frame, const ds_vk::f32) -> void
    {
        picker_.clear();
        register_pick_targets();
        handle_selection_click(frame);

        frame.draw.draw_mesh({
            .mesh = floor_mesh_,
            .object_id = object_ids_.floor,
            .material = materials_.floor,
            .debug = object_debug_config(object_ids_.floor),
        });
        frame.draw.draw_mesh({
            .mesh = sphere_mesh_,
            .object_id = object_ids_.sphere,
            .transform =
                Transform{
                    .translation = sphere_position_,
                    .scale = Vec3{sphere_radius_},
                },
            .material = materials_.sphere,
            .debug = object_debug_config(object_ids_.sphere),
        });
        if (show_sphere_wire_)
        {
            frame.draw.debug_sphere({
                .center = sphere_position_,
                .radius = sphere_radius_,
                .color = Color{0.55f, 0.84f, 1.0f, 0.9f},
                .segments = std::clamp(sphere_slices_, 12u, 64u),
                .width = 0.006f,
            });
        }
        frame.draw.draw_mesh({
            .mesh = cube_mesh_,
            .object_id = object_ids_.cube,
            .transform = cube_transform(),
            .material = materials_.cube,
            .debug = object_debug_config(object_ids_.cube, hide_cube_),
        });

        if (show_debug_axes_)
        {
            const auto& cfg = debug_axis_cfg_;
            frame.draw.debug_arrow({
                .origin = cfg.pos,
                .vector = cfg.arrow_length * k_axis_x,
                .color = cfg.color_x,
            });
            frame.draw.debug_arrow({
                .origin = cfg.pos,
                .vector = cfg.arrow_length * k_axis_y,
                .color = cfg.color_y,
            });
            frame.draw.debug_arrow({
                .origin = cfg.pos,
                .vector = cfg.arrow_length * k_axis_z,
                .color = cfg.color_z,
            });
        }
        if (show_vector_field_)
        {
            viz::draw_vector_field(
                frame.draw,
                viz::VectorFieldConfig{
                    .positions = std::span<const Vec3>{vector_field_positions_},
                    .vectors = std::span<const Vec3>{vector_field_vectors_},
                    .scale = vector_field_cfg_.scale,
                    .width = vector_field_cfg_.width,
                    .color_by_magnitude = true,
                    .color_ramp = vector_field_ramp_,
                }
            );
        }
        if (show_floor_grid_)
        {
            const auto& cfg = floor_grid_cfg_;
            const auto dx = cfg.half_extent * k_axis_x;
            const auto dy = cfg.half_extent * k_axis_y;

            const auto draw_line = [&](int i) -> void
            {
                const auto color = (i == 0) ? cfg.color_major : cfg.color_minor;
                const auto p = static_cast<f32>(i);
                const Vec3 x_base{p, 0.0f, cfg.height};
                const Vec3 y_base{0.0f, p, cfg.height};

                frame.draw.debug_line({
                    .start = x_base - dy,
                    .end = x_base + dy,
                    .color = color,
                    .width = cfg.line_width,
                });
                frame.draw.debug_line({
                    .start = y_base - dx,
                    .end = y_base + dx,
                    .color = color,
                    .width = cfg.line_width,
                });
            };
            for (auto i = -cfg.count_per_side; i <= cfg.count_per_side; ++i)
            {
                draw_line(i);
            }
        }
    }

    auto draw_ui(ds_vk::FrameContext& frame) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{20.0f, 290.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{500.0f, 360.0f}, ImGuiCond_Once);
        if (ImGui::Begin("Basic Viewer"))
        {
            ImGui::PushItemWidth(300.0f);

            ImGui::Checkbox("Debug axes", &show_debug_axes_);
            ImGui::Checkbox("Floor grid", &show_floor_grid_);
            ImGui::Checkbox("Vector field", &show_vector_field_);
            ImGui::Checkbox("Sphere wire", &show_sphere_wire_);
            ImGui::Checkbox("Normal debug", &show_normal_debug_);
            ImGui::Checkbox("Hide cube", &hide_cube_);
            ImGui::Separator();
            ImGui::DragFloat3("Sphere position", &sphere_position_.x, 0.025f);
            ImGui::SliderFloat("Sphere radius", &sphere_radius_, 0.1f, 2.0f, "%.2f");
            ImGui::ColorEdit4(
                "Sphere base color",
                materials_.sphere.base_color.data(),
                ImGuiColorEditFlags_NoInputs
            );
            ImGui::ColorEdit3("Sphere emissive", materials_.sphere.emissive_color.data());
            ImGui::SliderFloat("Sphere metallic", &materials_.sphere.metallic, 0.0f, 1.0f);
            ImGui::SliderFloat("Sphere roughness", &materials_.sphere.roughness, 0.04f, 1.0f);
            ImGui::SliderFloat("Sphere AO", &materials_.sphere.ambient_occlusion, 0.0f, 1.0f);
            auto slices = static_cast<int>(sphere_slices_);
            auto stacks = static_cast<int>(sphere_stacks_);
            auto changed = false;
            changed |= ImGui::SliderInt("Sphere slices", &slices, 8, 96);
            changed |= ImGui::SliderInt("Sphere stacks", &stacks, 4, 64);
            sphere_slices_ = static_cast<ds_vk::u32>(std::clamp(slices, 8, 96));
            sphere_stacks_ = static_cast<ds_vk::u32>(std::clamp(stacks, 4, 64));
            if (changed || ImGui::Button("Rebuild sphere mesh"))
            {
                rebuild_sphere();
            }
            ImGui::Separator();
            ImGui::Text(
                "Framebuffer: %ux%u",
                static_cast<unsigned>(frame.extent.width),
                static_cast<unsigned>(frame.extent.height)
            );
            ImGui::Text("Vulkan command buffer: %p", static_cast<void*>(frame.command_buffer));
            ImGui::PopItemWidth();
        }
        ImGui::End();
        draw_selection_window();
    }

    auto set_normal_debug(const bool enabled) noexcept -> void
    {
        show_normal_debug_ = enabled;
    }

  private:
    [[nodiscard]] auto is_selected(const ObjectId object_id) const noexcept -> bool
    {
        return selected_object_id_.valid() && selected_object_id_.value == object_id.value;
    }

    [[nodiscard]] auto
    object_debug_config(const ObjectId object_id, const bool hidden = false) const noexcept
        -> MeshDebugConfig
    {
        auto debug = MeshDebugConfig{.hidden = hidden};
        if (show_normal_debug_ && !hidden)
        {
            debug.mode = MeshDebugMode::normal;
            return debug;
        }
        if (is_selected(object_id))
        {
            debug.mode = MeshDebugMode::selected_pulse;
            debug.color = k_selection_color;
            debug.selected = true;
        }
        return debug;
    }

    [[nodiscard]] auto cube_transform() const noexcept -> Transform
    {
        return Transform{
            .translation = {-1.6f, 1.1f, 0.5f},
            .rotation = glm::angleAxis(glm::radians(24.0f), k_axis_z),
            .scale = {0.5f, 0.5f, 0.5f},
        };
    }

    auto register_pick_targets() -> void
    {
        picker_.add_sphere({
            .object_id = object_ids_.sphere,
            .center = sphere_position_,
            .radius = sphere_radius_,
        });

        if (!hide_cube_)
        {
            const auto cube = cube_transform();
            picker_.add_obb({
                .object_id = object_ids_.cube,
                .center = cube.translation,
                .half_extent = 0.5f * glm::abs(cube.scale),
                .rotation = cube.rotation,
            });
        }
    }

    auto handle_selection_click(const FrameContext& frame) -> void
    {
        if (!frame.input.left_click.occurred)
        {
            return;
        }

        const auto hit = picker_.click({
            .camera = frame.camera,
            .mouse_px = frame.input.left_click.position_px,
            .viewport_px = Vec2{
                static_cast<f32>(frame.extent.width),
                static_cast<f32>(frame.extent.height),
            },
        });
        selected_object_id_ = hit.has_value() ? hit->object_id : ObjectId{};
        selection_window_open_ = hit.has_value();
    }

    [[nodiscard]] auto selected_label() const noexcept -> const char*
    {
        if (is_selected(object_ids_.sphere))
        {
            return "Sphere";
        }
        if (is_selected(object_ids_.cube))
        {
            return "Cube";
        }
        return "None";
    }

    auto draw_selection_window() -> void
    {
        if (!selection_window_open_ || !selected_object_id_.valid())
        {
            return;
        }

        ImGui::SetNextWindowPos(ImVec2{540.0f, 290.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{320.0f, 180.0f}, ImGuiCond_Once);
        auto open = selection_window_open_;
        if (ImGui::Begin("Selection", &open))
        {
            ImGui::Text("Object: %s", selected_label());
            ImGui::Text("Object id: %u", selected_object_id_.value);
            if (is_selected(object_ids_.sphere))
            {
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    static_cast<double>(sphere_position_.x),
                    static_cast<double>(sphere_position_.y),
                    static_cast<double>(sphere_position_.z)
                );
                ImGui::Text("Radius: %.3f", static_cast<double>(sphere_radius_));
                ImGui::Text("Mesh: %u slices, %u stacks", sphere_slices_, sphere_stacks_);
                ImGui::Text(
                    "Material: metallic %.2f roughness %.2f",
                    static_cast<double>(materials_.sphere.metallic),
                    static_cast<double>(materials_.sphere.roughness)
                );
            }
            else if (is_selected(object_ids_.cube))
            {
                const auto cube = cube_transform();
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    static_cast<double>(cube.translation.x),
                    static_cast<double>(cube.translation.y),
                    static_cast<double>(cube.translation.z)
                );
                ImGui::Text("Collider: OBB");
                ImGui::Text(
                    "Material: metallic %.2f roughness %.2f",
                    static_cast<double>(materials_.cube.metallic),
                    static_cast<double>(materials_.cube.roughness)
                );
            }
            if (ImGui::Button("Clear selection"))
            {
                open = false;
            }
        }
        ImGui::End();
        selection_window_open_ = open;
        if (!selection_window_open_)
        {
            selected_object_id_ = {};
        }
    }

    auto rebuild_vector_field() -> void
    {
        vector_field_positions_.clear();
        vector_field_vectors_.clear();
        for (auto y = -4; y <= 4; ++y)
        {
            for (auto x = -4; x <= 4; ++x)
            {
                const auto xf = static_cast<f32>(x) * 0.58f;
                const auto yf = static_cast<f32>(y) * 0.58f;
                const auto radius_squared = xf * xf + yf * yf;
                const auto lift = 0.30f + 0.16f * std::sin(1.4f * xf) * std::cos(1.2f * yf);
                const auto tangent = Vec3{-yf, xf, 0.38f * std::cos(0.7f * radius_squared)};
                const auto direction =
                    glm::dot(tangent, tangent) > 1.0e-8f ? glm::normalize(tangent) : k_axis_z;
                const auto strength = 0.35f + 0.58f * std::exp(-0.16f * radius_squared);
                vector_field_positions_.emplace_back(xf, yf, lift);
                vector_field_vectors_.push_back(strength * direction);
            }
        }
    }

    auto rebuild_sphere() -> void
    {
        if (runtime_ == nullptr) return;
        const auto sphere =
            make_uv_sphere(1.0f, sphere_slices_, sphere_stacks_, ds_vk::Color::white);
        sphere_mesh_ = sphere_mesh_.valid() ? runtime_->replace_mesh(sphere_mesh_, sphere)
                                            : runtime_->upload_mesh(sphere);
    }

    Runtime* runtime_{};
    Picker picker_{};
    MeshHandle floor_mesh_{};
    MeshHandle sphere_mesh_{};
    MeshHandle cube_mesh_{};
    TextureHandle floor_texture_{};
    TextureHandle cube_texture_{};
    ObjectId selected_object_id_{};
    Vec3 sphere_position_{0.0f, 0.0f, 1.0f};
    f32 sphere_radius_{0.75f};
    u32 sphere_slices_{40u};
    u32 sphere_stacks_{20u};
    bool show_debug_axes_{true};
    bool show_floor_grid_{true};
    bool show_vector_field_{true};
    bool show_sphere_wire_{true};
    bool show_normal_debug_{};
    bool hide_cube_{};
    bool selection_window_open_{};

    struct ObjectIdsConfig
    {
        ObjectId floor{1u};
        ObjectId sphere{2u};
        ObjectId cube{3u};
    };
    ObjectIdsConfig object_ids_{};

    struct MaterialsConfig
    {
        Material floor{
            .base_color = {0.48f, 0.52f, 0.48f, 1.0f},
            .metallic = 0.0f,
            .roughness = 0.88f,
            .ambient_occlusion = 1.0f,
        };
        Material sphere{
            .base_color = {0.18f, 0.52f, 0.95f, 1.0f},
            .metallic = 0.0f,
            .roughness = 0.34f,
            .ambient_occlusion = 1.0f,
        };
        Material cube{
            .base_color = {0.9f, 0.46f, 0.2f, 1.0f},
            .metallic = 0.0f,
            .roughness = 0.62f,
            .ambient_occlusion = 1.0f,
        };
    };
    MaterialsConfig materials_{};

    struct DebugAxisConfig
    {
        f32 arrow_length{1.4f};
        Vec3 pos{0.0f, 0.0f, 0.04f};
        Color color_x{0.96f, 0.18f, 0.14f, 1.0f};
        Color color_y{0.20f, 0.78f, 0.24f, 1.0f};
        Color color_z{0.20f, 0.42f, 1.00f, 1.0f};
    };
    DebugAxisConfig debug_axis_cfg_{};

    struct VectorFieldDisplayConfig
    {
        f32 scale{0.36f};
        f32 width{0.010f};
    };
    VectorFieldDisplayConfig vector_field_cfg_{};
    viz::ColorRamp vector_field_ramp_{
        viz::ColorRampConfig{
            .preset = viz::ColorPreset::turbo,
            .range = {.min = 0.25f, .max = 1.00f},
        },
    };
    std::vector<Vec3> vector_field_positions_;
    std::vector<Vec3> vector_field_vectors_;

    struct FloorGridConfig
    {
        Color color_major{0.18f, 0.22f, 0.25f, 1.0f};
        Color color_minor{0.18f, 0.22f, 0.25f, 1.0f};
        int count_per_side{4};
        f32 half_extent{4.0f};
        f32 height{0.012f};
        f32 line_width{0.006f};
    };
    FloorGridConfig floor_grid_cfg_{};
};

auto parse_u32(const char* text, const ds_vk::u32 fallback) -> ds_vk::u32
{
    char* end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text)
    {
        return fallback;
    }
    return static_cast<ds_vk::u32>(value);
}

auto print_usage(const char* executable) -> void
{
    std::cout << "usage: " << executable
              << " [--smoke-frames N] [--screenshot PATH] [--hide-ui]"
                 " [--transparent-screenshot] [--normal-debug]\n";
}
}  // namespace

auto main(const int argc, char** argv) -> int
{
    try
    {
        auto config = ds_vk::RuntimeConfig{.window_title = "ds_vk Basic Viewer"};
        auto start_normal_debug = false;
        for (auto i = 1; i < argc; ++i)
        {
            const auto arg = std::string_view{argv[i]};
            if (arg == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--smoke-frames" && i + 1 < argc)
            {
                config.smoke_frames = parse_u32(argv[++i], 0u);
            }
            else if (arg == "--screenshot" && i + 1 < argc)
            {
                config.screenshot_path = argv[++i];
            }
            else if (arg == "--hide-ui")
            {
                config.hide_ui = true;
            }
            else if (arg == "--transparent-screenshot")
            {
                config.transparent_screenshot = true;
            }
            else if (arg == "--normal-debug")
            {
                start_normal_debug = true;
            }
            else
            {
                std::cerr << "unknown or incomplete argument: " << arg << '\n';
                print_usage(argv[0]);
                return 2;
            }
        }

        auto app = BasicViewerApp{};
        app.set_normal_debug(start_normal_debug);
        auto runtime = ds_vk::Runtime{std::move(config)};
        return runtime.run(app);
    }
    catch (const std::exception& error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
