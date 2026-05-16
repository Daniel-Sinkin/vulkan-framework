#include "app/pba/physics.hpp"
#include "ds_vk/assets.hpp"
#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace ds_vk;

#ifndef DS_VK_ASSET_DIR
#    define DS_VK_ASSET_DIR "assets"
#endif

struct PbaDebugConfig
{
    bool speed_coloring{true};
    bool velocity_arrows{true};
    f32 velocity_arrow_scale{0.08f};
    usize max_velocity_arrows{256zu};
};

[[nodiscard]] auto asset_path(const std::filesystem::path& relative) -> std::filesystem::path
{
    return std::filesystem::path{DS_VK_ASSET_DIR} / relative;
}

auto print_usage(const char* program) -> void
{
    std::cerr << "usage: " << program
              << " [--smoke-frames N] [--screenshot PATH] [--hide-ui]"
                 " [--transparent-screenshot]\n";
}

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

class PbaPyramidApp
{
  public:
    auto setup(Runtime& runtime) -> void
    {
        cube_mesh_ = runtime.upload_mesh(make_cube(1.0f, Color::white));
        floor_mesh_ = runtime.upload_mesh(make_quad(12.0f, Color::white));
        const auto test_gltf = asset_path("test/triangle.gltf");
        if (std::filesystem::exists(test_gltf))
        {
            gltf_marker_mesh_ = runtime.upload_mesh(load_gltf_mesh(
                test_gltf,
                GltfMeshLoadConfig{
                    .transform = {.scale = Vec3{0.35f}},
                    .color = Color{0.95f, 0.72f, 0.28f, 1.0f},
                }
            ));
        }
        reset_scene();
        runtime.camera({
            .pivot = 1.2f * k_axis_z,
            .distance = 9.3f,
            .yaw = glm::radians(43.0f),
            .pitch = glm::radians(28.0f),
        });
    }

    auto update(FrameContext& frame, f32 dt_seconds) -> void
    {
        if (frame.input.space_pressed)
        {
            paused_ = !paused_;
        }
        if (!paused_)
        {
            accumulator_ += std::min(dt_seconds, 0.08f) * simulation_speed_;
            while (accumulator_ >= ds_vk_app::pba::k_fixed_dt)
            {
                simulation_.step(ds_vk_app::pba::k_fixed_dt);
                accumulator_ -= ds_vk_app::pba::k_fixed_dt;
            }
        }

        configure_lighting(frame.draw);
        frame.draw.draw_mesh({
            .mesh = floor_mesh_,
            .material =
                Material{
                    .base_color = Color{0.18f, 0.19f, 0.17f, 1.0f},
                    .roughness = 0.88f,
                },
            .mask = {.shadow_producer = false},
        });
        if (gltf_marker_mesh_.valid())
        {
            frame.draw.draw_mesh({
                .mesh = gltf_marker_mesh_,
                .transform =
                    Transform{
                        .translation = {-4.2f, -4.2f, 0.05f},
                        .rotation = glm::angleAxis(glm::radians(35.0f), k_axis_z),
                    },
                .material =
                    Material{
                        .base_color = Color{0.95f, 0.72f, 0.28f, 1.0f},
                        .roughness = 0.38f,
                    },
                .mask = {.shadow_producer = false},
            });
        }

        velocity_positions_.clear();
        velocity_vectors_.clear();
        speed_values_.clear();
        const auto bodies = simulation_.bodies();
        velocity_positions_.reserve(bodies.size());
        velocity_vectors_.reserve(bodies.size());
        speed_values_.reserve(bodies.size());
        for (const auto& body : bodies)
        {
            const auto speed = glm::length(body.velocity);
            speed_values_.push_back(speed);
        }
        const auto speed_range = viz::range_from_values(std::span<const f32>{speed_values_});
        speed_ramp_.configure({
            .preset = viz::ColorPreset::turbo,
            .range = {.min = 0.0f, .max = std::max(1.0f, speed_range.max)},
        });

        for (auto i = 0zu; i < bodies.size(); ++i)
        {
            const auto& body = bodies[i];
            const auto color =
                debug_.speed_coloring ? speed_ramp_.sample(speed_values_[i]) : body.color;
            frame.draw.draw_mesh({
                .mesh = cube_mesh_,
                .object_id = body.object_id,
                .transform =
                    Transform{
                        .translation = body.position,
                        .rotation = body.orientation,
                        .scale = 2.0f * body.half_extent,
                    },
                .material =
                    Material{
                        .base_color = color,
                        .metallic = 0.0f,
                        .roughness = 0.62f,
                    },
                .debug = MeshDebugConfig{
                    .selected = body.object_id.value == selected_id_.value,
                },
            });
            if (show_bounds_)
            {
                (void) viz::draw_aabb(
                    frame.draw,
                    viz::AabbMarkerConfig{
                        .aabb = ds_vk_app::pba::body_aabb(body),
                        .color = Color{0.34f, 0.42f, 0.48f, 0.45f},
                        .width = 0.004f,
                    }
                );
            }
            if (debug_.velocity_arrows and body.inv_mass > 0.0f)
            {
                velocity_positions_.push_back(body.position);
                velocity_vectors_.push_back(body.velocity);
            }
        }

        if (debug_.velocity_arrows)
        {
            (void) viz::draw_vector_field(
                frame.draw,
                viz::VectorFieldConfig{
                    .positions = std::span<const Vec3>{velocity_positions_},
                    .vectors = std::span<const Vec3>{velocity_vectors_},
                    .scale = debug_.velocity_arrow_scale,
                    .width = 0.010f,
                    .color_by_magnitude = true,
                    .color_ramp = speed_ramp_,
                    .max_vectors = debug_.max_velocity_arrows,
                }
            );
        }
    }

    auto draw_ui(FrameContext&) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{18.0f, 280.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{440.0f, 330.0f}, ImGuiCond_Once);
        if (ImGui::Begin("PBA Pyramid"))
        {
            ImGui::Text("Space toggles simulation pause");
            ImGui::Checkbox("Paused", &paused_);
            ImGui::SliderFloat("Simulation speed", &simulation_speed_, 0.0f, 4.0f, "%.2f");
            if (ImGui::Button("Reset pyramid"))
            {
                reset_scene();
            }
            ImGui::Separator();
            ImGui::Checkbox("Speed coloring", &debug_.speed_coloring);
            ImGui::Checkbox("Velocity arrows", &debug_.velocity_arrows);
            ImGui::SliderFloat("Arrow scale", &debug_.velocity_arrow_scale, 0.0f, 0.25f, "%.3f");
            auto max_arrows = static_cast<int>(debug_.max_velocity_arrows);
            if (ImGui::SliderInt("Max arrows", &max_arrows, 0, 512))
            {
                debug_.max_velocity_arrows = static_cast<usize>(std::max(0, max_arrows));
            }
            ImGui::Checkbox("AABB bounds", &show_bounds_);
            ImGui::Separator();
            auto& physics = simulation_.physics();
            ImGui::SliderFloat("Restitution", &physics.restitution, 0.0f, 0.8f, "%.2f");
            ImGui::SliderFloat("Friction", &physics.friction, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Damping", &physics.linear_damping, 0.0f, 0.5f, "%.3f");
            auto iterations = static_cast<int>(physics.solver_iterations);
            if (ImGui::SliderInt("Solver iterations", &iterations, 1, 12))
            {
                physics.solver_iterations = static_cast<u32>(std::max(1, iterations));
            }
            ImGui::Text("Bodies: %zu", simulation_.bodies().size());
            ImGui::Text("Steps: %zu", simulation_.step_count());
        }
        ImGui::End();
    }

  private:
    auto reset_scene() -> void
    {
        simulation_.reset();
        accumulator_ = 0.0f;
        if (!simulation_.bodies().empty())
        {
            selected_id_ = simulation_.bodies().back().object_id;
        }
    }

    auto configure_lighting(DrawList& draw) const -> void
    {
        draw.set_ambient_light(Color{0.045f, 0.050f, 0.060f, 1.0f});
        draw.directional_light({
            .direction = normalize_or(Vec3{-0.42f, -0.48f, -1.0f}, -k_axis_z),
            .color = Color{1.0f, 0.96f, 0.86f, 1.0f},
            .intensity = 2.4f,
            .shadow = {.enabled = true, .ortho_extent = 7.0f},
        });
        draw.radial_light({
            .position = {2.8f, -3.4f, 2.7f},
            .color = Color{0.50f, 0.76f, 1.0f, 1.0f},
            .intensity = 8.0f,
            .range = 6.0f,
        });
    }

    MeshHandle cube_mesh_{};
    MeshHandle floor_mesh_{};
    MeshHandle gltf_marker_mesh_{};
    ds_vk_app::pba::PyramidSimulation simulation_{};
    std::vector<Vec3> velocity_positions_{};
    std::vector<Vec3> velocity_vectors_{};
    std::vector<f32> speed_values_{};
    viz::ColorRamp speed_ramp_{};
    PbaDebugConfig debug_{};
    ObjectId selected_id_{};
    f32 accumulator_{};
    f32 simulation_speed_{1.0f};
    bool paused_{false};
    bool show_bounds_{false};
};
}  // namespace

auto main(const int argc, char** argv) -> int
{
    try
    {
        auto config = RuntimeConfig{
            .window_title = "ds_vk PBA pyramid",
            .initial_width = 1280u,
            .initial_height = 820u,
            .clear_color = Color{0.024f, 0.028f, 0.034f, 1.0f},
        };
        for (auto i = 1; i < argc; ++i)
        {
            const auto arg = std::string_view{argv[i]};
            if (arg == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--smoke-frames" and i + 1 < argc)
            {
                config.smoke_frames = parse_u32(argv[++i], 0u);
            }
            else if (arg == "--screenshot" and i + 1 < argc)
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
            else
            {
                print_usage(argv[0]);
                return 2;
            }
        }

        auto runtime = Runtime{std::move(config)};
        auto app = PbaPyramidApp{};
        return runtime.run(app);
    }
    catch (const std::exception& error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
