#include "ds_vk/math.hpp"
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
        configure_lighting(frame.draw);
        const auto show_debug_overlays = !show_camera_depth_debug_;

        frame.draw.draw_mesh({
            .mesh = floor_mesh_,
            .object_id = object_ids_.floor,
            .material = materials_.floor,
            .mask = {.shadow_producer = false},
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
        if (show_debug_overlays && show_sphere_wire_)
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
        frame.draw.draw_mesh({
            .mesh = cube_mesh_,
            .object_id = object_ids_.column,
            .transform = column_transform(),
            .material = materials_.column,
            .debug = object_debug_config(object_ids_.column),
        });
        frame.draw.draw_mesh({
            .mesh = sphere_mesh_,
            .object_id = object_ids_.small_sphere,
            .transform =
                Transform{
                    .translation = small_sphere_position_,
                    .scale = Vec3{0.36f},
                },
            .material = materials_.small_sphere,
            .debug = object_debug_config(object_ids_.small_sphere),
        });

        if (show_debug_overlays && show_debug_axes_)
        {
            const auto& cfg = debug_axis_cfg_;
            frame.draw.debug_arrow({
                .origin = cfg.origin,
                .vector = cfg.arrow_length * k_axis_x,
                .color = cfg.x_color,
            });
            frame.draw.debug_arrow({
                .origin = cfg.origin,
                .vector = cfg.arrow_length * k_axis_y,
                .color = cfg.y_color,
            });
            frame.draw.debug_arrow({
                .origin = cfg.origin,
                .vector = cfg.arrow_length * k_axis_z,
                .color = cfg.z_color,
            });
        }
        if (show_debug_overlays && show_vector_field_)
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
        if (show_debug_overlays && show_floor_grid_)
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
        if (show_debug_overlays && show_light_gizmos_)
        {
            draw_light_gizmos(frame.draw);
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
            if (ImGui::Checkbox("Normal debug", &show_normal_debug_) && show_normal_debug_)
            {
                show_camera_depth_debug_ = false;
            }
            if (ImGui::Checkbox("Camera depth debug", &show_camera_depth_debug_)
                && show_camera_depth_debug_)
            {
                show_normal_debug_ = false;
            }
            ImGui::DragFloatRange2(
                "Depth range",
                &camera_depth_debug_cfg_.near_distance,
                &camera_depth_debug_cfg_.far_distance,
                0.05f,
                0.0f,
                30.0f,
                "%.2f",
                "%.2f"
            );
            ImGui::Checkbox("Hide cube", &hide_cube_);
            ImGui::Checkbox("Light gizmos", &show_light_gizmos_);
            ImGui::Separator();
            {  // Light
                ImGui::ColorEdit3("Ambient", lights_.ambient.data());
                ImGui::Checkbox("Directional light", &lights_.sun.enabled);
                ImGui::DragFloat3("Sun direction", &lights_.sun.direction.x, 0.02f);
                ImGui::SliderFloat("Sun intensity", &lights_.sun.intensity, 0.0f, 6.0f, "%.2f");
                ImGui::SliderFloat("Shadow bias", &lights_.sun.shadow.bias, 0.0001f, 0.02f, "%.4f");
                ImGui::SliderFloat(
                    "Shadow strength", &lights_.sun.shadow.strength, 0.0f, 1.0f, "%.2f"
                );
                ImGui::Checkbox("Radial light", &lights_.radial.enabled);
                ImGui::DragFloat3("Radial position", &lights_.radial.position.x, 0.03f);
                ImGui::SliderFloat(
                    "Radial intensity", &lights_.radial.intensity, 0.0f, 30.0f, "%.2f"
                );
                ImGui::SliderFloat("Radial range", &lights_.radial.range, 0.5f, 10.0f, "%.2f");
                ImGui::Checkbox("Spot light", &lights_.spot.enabled);
                ImGui::DragFloat3("Spot position", &lights_.spot.position.x, 0.03f);
                ImGui::DragFloat3("Spot direction", &lights_.spot.direction.x, 0.02f);
                ImGui::SliderFloat("Spot intensity", &lights_.spot.intensity, 0.0f, 60.0f, "%.2f");
                ImGui::SliderFloat("Spot range", &lights_.spot.range, 0.5f, 12.0f, "%.2f");
            }
            ImGui::Separator();
            {  // Sphere
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
        if (enabled)
        {
            show_camera_depth_debug_ = false;
        }
    }

    auto set_camera_depth_debug(const bool enabled) noexcept -> void
    {
        show_camera_depth_debug_ = enabled;
        if (enabled)
        {
            show_normal_debug_ = false;
        }
    }

  private:
    [[nodiscard]] auto is_selected(ObjectId object_id) const noexcept -> bool
    {
        return selected_object_id_.valid() && selected_object_id_.value == object_id.value;
    }

    [[nodiscard]] auto object_debug_config(ObjectId object_id, bool hidden = false) const noexcept
        -> MeshDebugConfig
    {
        auto debug = MeshDebugConfig{.hidden = hidden};
        if (show_camera_depth_debug_ && !hidden)
        {
            const auto& cfg = camera_depth_debug_cfg_;
            debug.mode = MeshDebugMode::camera_depth;
            debug.scalar_range = {
                cfg.near_distance, std::max(cfg.near_distance + 0.001f, cfg.far_distance)
            };
            return debug;
        }
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

    [[nodiscard]] auto column_transform() const noexcept -> Transform
    {
        return Transform{
            .translation = {1.35f, 1.25f, 0.68f},
            .rotation = glm::angleAxis(glm::radians(-12.0f), k_axis_z),
            .scale = {0.36f, 0.36f, 1.36f},
        };
    }

    auto configure_lighting(DrawList& draw) const -> void
    {
        draw.set_ambient_light(lights_.ambient);
        draw.directional_light(lights_.sun);
        draw.radial_light(lights_.radial);
        draw.spot_light(lights_.spot);
    }

    auto draw_light_gizmos(DrawList& draw) const -> void
    {
        if (lights_.sun.enabled)
        {
            const auto sun_dir = normalize_or(lights_.sun.direction, -k_axis_z);
            draw.debug_arrow({
                .origin = {-2.8f, -2.8f, 2.1f},
                .vector = 0.9f * sun_dir,
                .color = lights_.sun.color,
                .width = 0.018f,
            });
        }
        if (lights_.radial.enabled)
        {
            draw.debug_sphere({
                .center = lights_.radial.position,
                .radius = 0.08f,
                .color = lights_.radial.color,
                .segments = 12u,
                .width = 0.012f,
            });
        }
        if (lights_.spot.enabled)
        {
            draw.debug_arrow({
                .origin = lights_.spot.position,
                .vector = 0.65f * normalize_or(lights_.spot.direction, -k_axis_z),
                .color = lights_.spot.color,
                .width = 0.018f,
            });
        }
    }

    auto register_pick_targets() -> void
    {
        static_cast<void>(picker_.add_sphere({
            .object_id = object_ids_.sphere,
            .center = sphere_position_,
            .radius = sphere_radius_,
        }));

        if (!hide_cube_)
        {
            const auto cube = cube_transform();
            static_cast<void>(picker_.add_obb({
                .object_id = object_ids_.cube,
                .center = cube.translation,
                .half_extent = 0.5f * glm::abs(cube.scale),
                .rotation = cube.rotation,
            }));
        }
        const auto column = column_transform();
        static_cast<void>(picker_.add_obb({
            .object_id = object_ids_.column,
            .center = column.translation,
            .half_extent = 0.5f * glm::abs(column.scale),
            .rotation = column.rotation,
        }));
        static_cast<void>(picker_.add_sphere({
            .object_id = object_ids_.small_sphere,
            .center = small_sphere_position_,
            .radius = 0.36f,
        }));
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
        if (is_selected(object_ids_.column))
        {
            return "Column";
        }
        if (is_selected(object_ids_.small_sphere))
        {
            return "Small sphere";
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
                    static_cast<f64>(sphere_position_.x),
                    static_cast<f64>(sphere_position_.y),
                    static_cast<f64>(sphere_position_.z)
                );
                ImGui::Text("Radius: %.3f", static_cast<f64>(sphere_radius_));
                ImGui::Text("Mesh: %u slices, %u stacks", sphere_slices_, sphere_stacks_);
                ImGui::Text(
                    "Material: metallic %.2f roughness %.2f",
                    static_cast<f64>(materials_.sphere.metallic),
                    static_cast<f64>(materials_.sphere.roughness)
                );
            }
            else if (is_selected(object_ids_.cube))
            {
                const auto cube = cube_transform();
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    static_cast<f64>(cube.translation.x),
                    static_cast<f64>(cube.translation.y),
                    static_cast<f64>(cube.translation.z)
                );
                ImGui::Text("Collider: OBB");
                ImGui::Text(
                    "Material: metallic %.2f roughness %.2f",
                    static_cast<f64>(materials_.cube.metallic),
                    static_cast<f64>(materials_.cube.roughness)
                );
            }
            else if (is_selected(object_ids_.column))
            {
                const auto column = column_transform();
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    static_cast<f64>(column.translation.x),
                    static_cast<f64>(column.translation.y),
                    static_cast<f64>(column.translation.z)
                );
                ImGui::Text("Mask: camera + shadow producer + receiver");
                ImGui::Text(
                    "Material: metallic %.2f roughness %.2f",
                    static_cast<f64>(materials_.column.metallic),
                    static_cast<f64>(materials_.column.roughness)
                );
            }
            else if (is_selected(object_ids_.small_sphere))
            {
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    static_cast<f64>(small_sphere_position_.x),
                    static_cast<f64>(small_sphere_position_.y),
                    static_cast<f64>(small_sphere_position_.z)
                );
                ImGui::Text("Mask: camera + shadow producer + receiver");
                ImGui::Text(
                    "Material: metallic %.2f roughness %.2f",
                    static_cast<f64>(materials_.small_sphere.metallic),
                    static_cast<f64>(materials_.small_sphere.roughness)
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
                const auto direction = normalize_or(tangent, k_axis_z);
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
    Vec3 small_sphere_position_{1.15f, -1.45f, 0.42f};
    f32 sphere_radius_{0.75f};
    u32 sphere_slices_{40u};
    u32 sphere_stacks_{20u};
    bool show_debug_axes_{true};
    bool show_floor_grid_{true};
    bool show_vector_field_{true};
    bool show_sphere_wire_{true};
    bool show_normal_debug_{};
    bool show_camera_depth_debug_{};
    bool show_light_gizmos_{true};
    bool hide_cube_{};
    bool selection_window_open_{};

    struct ObjectIdsConfig
    {
        ObjectId floor{1u};
        ObjectId sphere{2u};
        ObjectId cube{3u};
        ObjectId column{4u};
        ObjectId small_sphere{5u};
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
        Material column{
            .base_color = {0.72f, 0.68f, 0.58f, 1.0f},
            .metallic = 0.0f,
            .roughness = 0.44f,
            .ambient_occlusion = 1.0f,
        };
        Material small_sphere{
            .base_color = {0.95f, 0.82f, 0.34f, 1.0f},
            .metallic = 0.55f,
            .roughness = 0.26f,
            .ambient_occlusion = 1.0f,
        };
    };
    MaterialsConfig materials_{};

    struct LightsConfig
    {
        Color ambient{0.030f, 0.036f, 0.046f, 1.0f};
        DirectionalLightConfig sun{
            .direction = {-0.42f, -0.34f, -0.84f},
            .color = {1.0f, 0.94f, 0.84f, 1.0f},
            .intensity = 2.45f,
            .shadow = {
                .enabled = true,
                .bias = 0.0040f,
                .strength = 0.78f,
                .near_plane = 0.05f,
                .far_plane = 24.0f,
                .ortho_extent = 5.2f,
            },
        };
        RadialLightConfig radial{
            .position = {-2.2f, -1.3f, 1.55f},
            .color = {0.24f, 0.78f, 1.0f, 1.0f},
            .intensity = 18.0f,
            .range = 5.0f,
        };
        SpotLightConfig spot{
            .position = {2.4f, -2.2f, 2.65f},
            .direction = {-0.62f, 0.48f, -0.62f},
            .color = {1.0f, 0.44f, 0.22f, 1.0f},
            .intensity = 32.0f,
            .range = 7.0f,
            .inner_cone_angle = glm::radians(10.0f),
            .outer_cone_angle = glm::radians(24.0f),
        };
    };
    LightsConfig lights_{};

    struct CameraDepthDebugConfig
    {
        f32 near_distance{2.0f};
        f32 far_distance{8.5f};
    };
    CameraDepthDebugConfig camera_depth_debug_cfg_{};

    struct DebugAxisConfig
    {
        f32 arrow_length{1.4f};
        Vec3 origin{0.0f, 0.0f, 0.04f};
        Color x_color{0.96f, 0.18f, 0.14f, 1.0f};
        Color y_color{0.20f, 0.78f, 0.24f, 1.0f};
        Color z_color{0.20f, 0.42f, 1.00f, 1.0f};
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
                 " [--transparent-screenshot] [--normal-debug] [--depth-debug]\n";
}
}  // namespace

auto main(const int argc, char** argv) -> int
{
    try
    {
        auto config = ds_vk::RuntimeConfig{.window_title = "ds_vk Basic Viewer"};
        auto start_normal_debug = false;
        auto start_depth_debug = false;
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
            else if (arg == "--depth-debug")
            {
                start_depth_debug = true;
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
        app.set_camera_depth_debug(start_depth_debug);
        auto runtime = ds_vk::Runtime{std::move(config)};
        return runtime.run(app);
    }
    catch (const std::exception& error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
