#include "ds_vk/math.hpp"
#include "ds_vk/plugins/picker.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace
{
using namespace ds_vk;

enum class VectorfieldExample : u8
{
    divergence_free_twist = 0,
    expanding_twist = 1,
    linear_saddle = 2,
    vortex_shear = 3,
    lorenz_attractor = 4,
    count = 5,
};

constexpr auto k_example_count = static_cast<usize>(VectorfieldExample::count);
constexpr ObjectId k_vector_target_id{.value = 700u};
constexpr auto k_invalid_selected_vector = k_invalid_index;

[[nodiscard]] auto example_name(VectorfieldExample example) noexcept -> const char*
{
    switch (example)
    {
        case VectorfieldExample::divergence_free_twist:
            return "divergence-free twist";
        case VectorfieldExample::expanding_twist:
            return "expanding twist";
        case VectorfieldExample::linear_saddle:
            return "linear saddle";
        case VectorfieldExample::vortex_shear:
            return "vortex shear";
        case VectorfieldExample::lorenz_attractor:
            return "lorenz attractor";
        case VectorfieldExample::count:
            break;
    }
    return "unknown";
}

[[nodiscard]] auto example_names() noexcept -> std::array<const char*, k_example_count>
{
    return {
        example_name(VectorfieldExample::divergence_free_twist),
        example_name(VectorfieldExample::expanding_twist),
        example_name(VectorfieldExample::linear_saddle),
        example_name(VectorfieldExample::vortex_shear),
        example_name(VectorfieldExample::lorenz_attractor),
    };
}

[[nodiscard]] auto divergence_free_twist_field(Vec3 p, f32) noexcept -> Vec3
{
    constexpr auto pulse = 0.72f;
    return {
        -pulse * p.y + 0.18f * std::sin(2.0f * p.z),
        pulse * p.x + 0.12f * std::cos(1.5f * p.z),
        0.22f * std::sin(1.4f * p.x) * std::cos(1.1f * p.y),
    };
}

[[nodiscard]] auto expanding_twist_field(Vec3 p, f32) noexcept -> Vec3
{
    constexpr auto spin = 0.82f;
    return {0.34f * p.x - spin * p.y, spin * p.x + 0.34f * p.y, 0.26f * p.z};
}

[[nodiscard]] auto linear_saddle_field(Vec3 p, f32) noexcept -> Vec3
{
    return {0.38f * p.x, -0.54f * p.y, -0.18f * p.z};
}

[[nodiscard]] auto vortex_shear_field(Vec3 p, f32) noexcept -> Vec3
{
    const auto r2 = p.x * p.x + p.y * p.y;
    const auto swirl = std::exp(-0.44f * r2);
    return {
        -0.74f * swirl * p.y + 0.22f,
        0.74f * swirl * p.x + 0.16f * std::sin(1.6f * p.z),
        0.14f * std::sin(1.15f * p.x) - 0.10f * p.z,
    };
}

[[nodiscard]] auto lorenz_rhs(Vec3 p) noexcept -> Vec3
{
    constexpr auto sigma = 10.0f;
    constexpr auto rho = 28.0f;
    constexpr auto beta = 8.0f / 3.0f;
    return {sigma * (p.y - p.x), p.x * (rho - p.z) - p.y, p.x * p.y - beta * p.z};
}

[[nodiscard]] auto lorenz_visual_field(Vec3 p, f32) noexcept -> Vec3
{
    constexpr auto visual_scale = 0.035f;
    constexpr Vec3 visual_offset{0.0f, 0.0f, -0.6f};
    constexpr auto trajectory_speed_scale = 0.18f;
    const auto raw = (p - visual_offset) / visual_scale;
    return visual_scale * trajectory_speed_scale * lorenz_rhs(raw);
}

[[nodiscard]] auto field_for_example(VectorfieldExample example, Vec3 p, f32 time_seconds) noexcept
    -> Vec3
{
    switch (example)
    {
        case VectorfieldExample::divergence_free_twist:
            return divergence_free_twist_field(p, time_seconds);
        case VectorfieldExample::expanding_twist:
            return expanding_twist_field(p, time_seconds);
        case VectorfieldExample::linear_saddle:
            return linear_saddle_field(p, time_seconds);
        case VectorfieldExample::vortex_shear:
            return vortex_shear_field(p, time_seconds);
        case VectorfieldExample::lorenz_attractor:
            return lorenz_visual_field(p, time_seconds);
        case VectorfieldExample::count:
            break;
    }
    return {};
}

[[nodiscard]] auto rk4_step(Vec3 p, f32 time_seconds, f32 dt, VectorfieldExample example) noexcept
    -> Vec3
{
    const auto k1 = field_for_example(example, p, time_seconds);
    const auto k2 = field_for_example(example, p + 0.5f * dt * k1, time_seconds + 0.5f * dt);
    const auto k3 = field_for_example(example, p + 0.5f * dt * k2, time_seconds + 0.5f * dt);
    const auto k4 = field_for_example(example, p + dt * k3, time_seconds + dt);
    return p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}

[[nodiscard]] auto
advect_from_time(Vec3 p, f32 start_time, f32 duration_seconds, VectorfieldExample example) noexcept
    -> Vec3
{
    constexpr auto k_steps_per_second = 72.0f;
    const auto duration = std::abs(duration_seconds);
    if (duration <= 1.0e-7f)
    {
        return p;
    }

    const auto steps = static_cast<usize>(std::max(1.0f, std::ceil(duration * k_steps_per_second)));
    const auto dt = duration_seconds / static_cast<f32>(steps);
    auto time = start_time;
    for (auto i = 0zu; i < steps; ++i)
    {
        p = rk4_step(p, time, dt, example);
        time += dt;
    }
    return p;
}

[[nodiscard]] auto default_trace_seeds(VectorfieldExample example) -> std::vector<Vec3>
{
    switch (example)
    {
        case VectorfieldExample::linear_saddle:
            return {
                {-0.88f, -0.64f, -0.28f},
                {-0.28f, -0.78f, 0.34f},
                {0.24f, 0.64f, -0.22f},
                {0.86f, 0.58f, 0.26f},
            };
        case VectorfieldExample::vortex_shear:
            return {
                {-0.94f, -0.52f, -0.32f},
                {-0.86f, 0.20f, 0.34f},
                {-0.36f, -0.72f, 0.24f},
                {0.46f, 0.74f, 0.20f},
            };
        case VectorfieldExample::lorenz_attractor:
            {
                constexpr auto seed_z = -0.60f + 0.035f;
                return {
                    {0.0f, 0.035f, seed_z},
                    {0.0f, 0.0364f, seed_z},
                    {0.0014f, 0.035f, seed_z},
                };
            }
        case VectorfieldExample::divergence_free_twist:
        case VectorfieldExample::expanding_twist:
        case VectorfieldExample::count:
            return {
                {-0.72f, -0.18f, -0.10f},
                {-0.46f, 0.40f, 0.18f},
                {0.18f, -0.62f, 0.06f},
            };
    }
    return {};
}

[[nodiscard]] auto seed_color(usize index, usize count, viz::ColorPreset preset) noexcept -> Color
{
    const auto t = count <= 1zu ? 0.65f : static_cast<f32>(index) / static_cast<f32>(count - 1zu);
    return viz::sample_color(preset, t);
}

struct VectorfieldConfig
{
    VectorfieldExample example{VectorfieldExample::divergence_free_twist};
    viz::ColorPreset color_preset{viz::ColorPreset::turbo};
    Vec3 sample_extent{1.15f, 1.15f, 0.85f};
    f32 sample_spacing{0.42f};
    f32 vector_scale{0.18f};
    f32 vector_width{0.008f};
    f32 trail_duration{4.0f};
    u32 trail_steps{120u};
    f32 playback_speed{1.0f};
    f32 time_seconds{};
    f32 max_time_seconds{8.0f};
    bool paused{false};
    bool show_vectors{true};
    bool show_trails{true};
    bool show_bounds{true};
    bool show_seed_markers{true};
};

class VectorfieldApp final
{
  public:
    auto setup(Runtime& runtime) -> void
    {
        floor_mesh_ = runtime.upload_mesh(make_quad(7.0f, Color::white));
        seed_mesh_ = runtime.upload_mesh(make_uv_sphere({
            .radius = 1.0f,
            .slices = 16u,
            .stacks = 8u,
            .color = Color::white,
        }));
        trace_seeds_ = default_trace_seeds(config_.example);
        runtime.camera({
            .pivot = 0.22f * k_axis_z,
            .distance = 5.0f,
            .yaw = glm::radians(43.0f),
            .pitch = glm::radians(27.0f),
        });
    }

    auto update(FrameContext& frame, f32 dt_seconds) -> void
    {
        advance_time(dt_seconds);
        rebuild_vector_samples();
        picker_.clear();
        register_vector_targets();
        handle_click(frame);

        frame.draw.set_ambient_light(Color{0.045f, 0.050f, 0.062f, 1.0f});
        frame.draw.directional_light({
            .direction = {-0.42f, -0.28f, -0.86f},
            .color = {1.0f, 0.94f, 0.84f, 1.0f},
            .intensity = 2.2f,
        });
        frame.draw.radial_light({
            .position = {-2.1f, -1.4f, 1.6f},
            .color = {0.24f, 0.78f, 1.0f, 1.0f},
            .intensity = 11.0f,
            .range = 5.5f,
        });

        draw_floor(frame.draw);
        if (config_.show_bounds)
        {
            draw_bounds(frame.draw);
        }
        if (config_.show_vectors)
        {
            draw_vectors(frame.draw);
        }
        if (config_.show_trails)
        {
            draw_trails(frame);
        }
        draw_selected_vector(frame);
    }

    auto draw_ui(FrameContext& frame) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{20.0f, 280.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{450.0f, 470.0f}, ImGuiCond_Once);
        if (ImGui::Begin("Vectorfield"))
        {
            ImGui::PushItemWidth(270.0f);
            draw_example_combo();
            draw_color_combo();
            ImGui::Checkbox("Play", &play_checkbox_value_);
            config_.paused = !play_checkbox_value_;
            ImGui::SameLine();
            if (ImGui::Button("Reset time"))
            {
                config_.time_seconds = 0.0f;
                selected_vector_index_ = k_invalid_selected_vector;
            }
            ImGui::SliderFloat(
                "Time", &config_.time_seconds, 0.0f, config_.max_time_seconds, "%.2f"
            );
            ImGui::SliderFloat("Playback speed", &config_.playback_speed, 0.0f, 3.0f, "%.2f");
            ImGui::Separator();
            ImGui::Checkbox("Vectors", &config_.show_vectors);
            ImGui::Checkbox("Trails", &config_.show_trails);
            ImGui::Checkbox("Bounds", &config_.show_bounds);
            ImGui::Checkbox("Seed markers", &config_.show_seed_markers);
            ImGui::SliderFloat("Sample spacing", &config_.sample_spacing, 0.22f, 0.70f, "%.2f");
            ImGui::SliderFloat("Vector scale", &config_.vector_scale, 0.04f, 0.42f, "%.2f");
            ImGui::SliderFloat("Trail duration", &config_.trail_duration, 0.25f, 8.0f, "%.2f");
            auto trail_steps = static_cast<int>(config_.trail_steps);
            if (ImGui::SliderInt("Trail steps", &trail_steps, 16, 240))
            {
                config_.trail_steps = static_cast<u32>(std::clamp(trail_steps, 16, 240));
            }
            ImGui::Separator();
            ImGui::Text("Samples: %u", static_cast<unsigned>(vector_positions_.size()));
            ImGui::Text("Trace seeds: %u", static_cast<unsigned>(trace_seeds_.size()));
            ImGui::Text(
                "Frame: %.2f ms | render %.2f ms",
                static_cast<f64>(frame.stats.last_frame_ms),
                static_cast<f64>(frame.stats.last_render_ms)
            );
            if (selected_vector_index_ != k_invalid_selected_vector)
            {
                ImGui::Text("Selected vector: %u", static_cast<unsigned>(selected_vector_index_));
            }
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }

  private:
    auto advance_time(f32 dt_seconds) noexcept -> void
    {
        play_checkbox_value_ = !config_.paused;
        if (config_.paused)
        {
            return;
        }
        config_.time_seconds += dt_seconds * config_.playback_speed;
        if (config_.time_seconds > config_.max_time_seconds)
        {
            config_.time_seconds = std::fmod(config_.time_seconds, config_.max_time_seconds);
            selected_vector_index_ = k_invalid_selected_vector;
        }
    }

    auto rebuild_vector_samples() -> void
    {
        vector_positions_.clear();
        vector_values_.clear();
        vector_magnitudes_.clear();

        const auto extent = glm::max(glm::abs(config_.sample_extent), Vec3{0.05f});
        const auto spacing = std::max(0.05f, config_.sample_spacing);
        const auto axis_count = [spacing](f32 axis_extent) noexcept -> u32
        { return static_cast<u32>(std::floor((2.0f * axis_extent) / spacing)) + 1u; };
        const glm::uvec3 counts{
            axis_count(extent.x),
            axis_count(extent.y),
            axis_count(extent.z),
        };
        const auto position_at = [](f32 axis_extent, u32 count, u32 index) noexcept -> f32
        {
            if (count <= 1u)
            {
                return 0.0f;
            }
            return -axis_extent
                   + 2.0f * axis_extent * static_cast<f32>(index) / static_cast<f32>(count - 1u);
        };

        const auto sample_count = static_cast<usize>(counts.x) * static_cast<usize>(counts.y)
                                  * static_cast<usize>(counts.z);
        vector_positions_.reserve(sample_count);
        vector_values_.reserve(sample_count);
        vector_magnitudes_.reserve(sample_count);
        for (auto z = 0u; z < counts.z; ++z)
        {
            for (auto y = 0u; y < counts.y; ++y)
            {
                for (auto x = 0u; x < counts.x; ++x)
                {
                    const Vec3 p{
                        position_at(extent.x, counts.x, x),
                        position_at(extent.y, counts.y, y),
                        position_at(extent.z, counts.z, z),
                    };
                    const auto v = field_for_example(config_.example, p, config_.time_seconds);
                    vector_positions_.push_back(p);
                    vector_values_.push_back(v);
                    vector_magnitudes_.push_back(glm::length(v));
                }
            }
        }
        vector_ramp_.configure({
            .preset = config_.color_preset,
            .range = viz::range_from_values(
                std::span<const f32>{vector_magnitudes_.data(), vector_magnitudes_.size()}
            ),
        });
    }

    auto register_vector_targets() -> void
    {
        const auto count = std::min(vector_positions_.size(), vector_values_.size());
        for (auto i = 0zu; i < count; ++i)
        {
            if (i > static_cast<usize>(std::numeric_limits<u32>::max()))
            {
                break;
            }
            const auto start = vector_positions_[i];
            const auto end = start + config_.vector_scale * vector_values_[i];
            (void) picker_.add_screen_segment({
                .object_id = k_vector_target_id,
                .sub_index = static_cast<u32>(i),
                .segment = {.start = start, .end = end},
                .radius_px = 10.0f,
            });
        }
    }

    auto handle_click(const FrameContext& frame) -> void
    {
        if (!frame.input.left_click.occurred)
        {
            return;
        }
        const auto hit = picker_.click({
            .camera = frame.camera,
            .mouse_px = frame.input.left_click.position_px,
            .viewport_px = {
                static_cast<f32>(frame.extent.width),
                static_cast<f32>(frame.extent.height),
            },
        });
        if (hit.has_value() and hit->object_id.value == k_vector_target_id.value)
        {
            selected_vector_index_ = static_cast<usize>(hit->sub_index);
            return;
        }
        selected_vector_index_ = k_invalid_selected_vector;
    }

    auto draw_floor(DrawList& draw) const -> void
    {
        draw.draw_mesh({
            .mesh = floor_mesh_,
            .transform = {.translation = -0.72f * k_axis_z},
            .material =
                Material{
                    .base_color = {0.18f, 0.20f, 0.18f, 1.0f},
                    .roughness = 0.86f,
                },
            .mask = {.shadow_producer = false},
        });

        constexpr auto line_count = 6;
        constexpr auto extent = 3.5f;
        constexpr auto z = -0.705f;
        for (auto i = -line_count; i <= line_count; ++i)
        {
            const auto p = static_cast<f32>(i) * extent / static_cast<f32>(line_count);
            const auto color =
                i == 0 ? Color{0.30f, 0.36f, 0.38f, 1.0f} : Color{0.20f, 0.25f, 0.27f, 1.0f};
            draw.debug_line({
                .start = {-extent, p, z},
                .end = {extent, p, z},
                .color = color,
                .width = 0.0045f,
            });
            draw.debug_line({
                .start = {p, -extent, z},
                .end = {p, extent, z},
                .color = color,
                .width = 0.0045f,
            });
        }
    }

    auto draw_bounds(DrawList& draw) const -> void
    {
        const auto e = config_.sample_extent;
        const std::array corners{
            Vec3{-e.x, -e.y, -e.z},
            Vec3{e.x, -e.y, -e.z},
            Vec3{e.x, e.y, -e.z},
            Vec3{-e.x, e.y, -e.z},
            Vec3{-e.x, -e.y, e.z},
            Vec3{e.x, -e.y, e.z},
            Vec3{e.x, e.y, e.z},
            Vec3{-e.x, e.y, e.z},
        };
        constexpr std::array edges{
            std::array{0zu, 1zu},
            std::array{1zu, 2zu},
            std::array{2zu, 3zu},
            std::array{3zu, 0zu},
            std::array{4zu, 5zu},
            std::array{5zu, 6zu},
            std::array{6zu, 7zu},
            std::array{7zu, 4zu},
            std::array{0zu, 4zu},
            std::array{1zu, 5zu},
            std::array{2zu, 6zu},
            std::array{3zu, 7zu},
        };
        for (const auto edge : edges)
        {
            draw.debug_line({
                .start = corners[edge[0]],
                .end = corners[edge[1]],
                .color = {0.42f, 0.70f, 0.88f, 0.45f},
                .width = 0.004f,
            });
        }
    }

    auto draw_vectors(DrawList& draw) const -> void
    {
        (void) viz::draw_vector_field(
            draw,
            viz::VectorFieldConfig{
                .positions =
                    std::span<const Vec3>{vector_positions_.data(), vector_positions_.size()},
                .vectors = std::span<const Vec3>{vector_values_.data(), vector_values_.size()},
                .scale = config_.vector_scale,
                .width = config_.vector_width,
                .color_by_magnitude = true,
                .color_ramp = vector_ramp_,
            }
        );
    }

    [[nodiscard]] auto trail_points(Vec3 seed) const -> std::vector<Vec3>
    {
        std::vector<Vec3> points{};
        const auto step_count = std::max(2u, config_.trail_steps);
        points.reserve(static_cast<usize>(step_count));
        const auto end_time = std::max(0.0f, config_.time_seconds);
        const auto start_time = std::max(0.0f, end_time - std::max(0.0f, config_.trail_duration));
        auto p = advect_from_time(seed, 0.0f, start_time, config_.example);
        points.push_back(p);
        const auto dt = (end_time - start_time) / static_cast<f32>(step_count - 1u);
        auto time = start_time;
        for (auto i = 1u; i < step_count; ++i)
        {
            p = advect_from_time(p, time, dt, config_.example);
            time += dt;
            points.push_back(p);
        }
        return points;
    }

    auto draw_trails(FrameContext& frame) const -> void
    {
        for (auto i = 0zu; i < trace_seeds_.size(); ++i)
        {
            const auto color = seed_color(i, trace_seeds_.size(), config_.color_preset);
            const auto points = trail_points(trace_seeds_[i]);
            (void) viz::draw_trail(
                frame.draw,
                viz::TrailConfig{
                    .points = std::span<const Vec3>{points.data(), points.size()},
                    .color = color,
                    .width = 0.007f,
                    .tail_alpha = 0.16f,
                    .head_alpha = 0.96f,
                }
            );
            if (config_.show_seed_markers and !points.empty())
            {
                frame.draw.draw_mesh({
                    .mesh = seed_mesh_,
                    .transform =
                        Transform{
                            .translation = points.back(),
                            .scale = Vec3{0.035f},
                        },
                    .material =
                        Material{
                            .base_color = with_alpha(color, 0.95f),
                            .emissive_color = with_alpha(color, 1.0f),
                            .roughness = 0.38f,
                        },
                    .mask = {.shadow_producer = false},
                });
                viz::draw_cross_marker(
                    frame.draw,
                    viz::CrossMarkerConfig{
                        .camera = frame.camera,
                        .center = points.back(),
                        .radius = 0.045f,
                        .color = with_alpha(color, 0.96f),
                        .width = 0.009f,
                    }
                );
            }
        }
    }

    auto draw_selected_vector(FrameContext& frame) const -> void
    {
        if (selected_vector_index_ == k_invalid_selected_vector
            or selected_vector_index_ >= vector_positions_.size()
            or selected_vector_index_ >= vector_values_.size())
        {
            return;
        }
        const auto start = vector_positions_[selected_vector_index_];
        const auto vector = config_.vector_scale * vector_values_[selected_vector_index_];
        frame.draw.debug_arrow({
            .origin = start,
            .vector = vector,
            .color = {1.0f, 0.95f, 0.30f, 1.0f},
            .width = 0.022f,
        });
        viz::draw_cross_marker(
            frame.draw,
            viz::CrossMarkerConfig{
                .camera = frame.camera,
                .center = start + vector,
                .radius = 0.055f,
                .color = {1.0f, 1.0f, 1.0f, 0.96f},
                .width = 0.010f,
            }
        );
    }

    auto draw_example_combo() -> void
    {
        const auto names = example_names();
        auto current = static_cast<usize>(config_.example);
        if (ImGui::BeginCombo("Example", example_name(config_.example)))
        {
            for (auto i = 0zu; i < names.size(); ++i)
            {
                const auto selected = i == current;
                if (ImGui::Selectable(names[i], selected))
                {
                    current = i;
                    config_.example = static_cast<VectorfieldExample>(static_cast<u8>(i));
                    trace_seeds_ = default_trace_seeds(config_.example);
                    selected_vector_index_ = k_invalid_selected_vector;
                    config_.time_seconds = 0.0f;
                }
                if (selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    auto draw_color_combo() -> void
    {
        constexpr std::array presets{
            viz::ColorPreset::turbo,
            viz::ColorPreset::viridis,
            viz::ColorPreset::magma,
            viz::ColorPreset::blue_red,
            viz::ColorPreset::grayscale,
        };
        const auto name = [](viz::ColorPreset preset) noexcept -> const char*
        {
            switch (preset)
            {
                case viz::ColorPreset::grayscale:
                    return "grayscale";
                case viz::ColorPreset::blue_red:
                    return "blue-red";
                case viz::ColorPreset::viridis:
                    return "viridis";
                case viz::ColorPreset::magma:
                    return "magma";
                case viz::ColorPreset::turbo:
                    return "turbo";
            }
            return "unknown";
        };
        if (ImGui::BeginCombo("Color map", name(config_.color_preset)))
        {
            for (const auto preset : presets)
            {
                const auto selected = preset == config_.color_preset;
                if (ImGui::Selectable(name(preset), selected))
                {
                    config_.color_preset = preset;
                }
                if (selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    Picker picker_{};
    MeshHandle floor_mesh_{};
    MeshHandle seed_mesh_{};
    VectorfieldConfig config_{};
    bool play_checkbox_value_{true};
    usize selected_vector_index_{k_invalid_selected_vector};
    std::vector<Vec3> trace_seeds_{};
    std::vector<Vec3> vector_positions_{};
    std::vector<Vec3> vector_values_{};
    std::vector<f32> vector_magnitudes_{};
    viz::ColorRamp vector_ramp_{};
};

auto parse_u32(const char* text, u32 fallback) -> u32
{
    char* end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text)
    {
        return fallback;
    }
    return static_cast<u32>(value);
}

auto print_usage(const char* executable) -> void
{
    std::cout << "usage: " << executable
              << " [--smoke-frames N] [--screenshot PATH] [--hide-ui]"
                 " [--transparent-screenshot]\n";
}
}  // namespace

auto main(int argc, char** argv) -> int
{
    try
    {
        RuntimeConfig config{.window_title = "ds_vk Vectorfield"};
        for (auto i = 1; i < argc; ++i)
        {
            const std::string_view arg{argv[i]};
            if (arg == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--smoke-frames" and i + 1 < argc)
            {
                config.smoke_frames = parse_u32(argv[++i], 0u);
                continue;
            }
            if (arg == "--screenshot" and i + 1 < argc)
            {
                config.screenshot_path = std::filesystem::path{argv[++i]};
                continue;
            }
            if (arg == "--hide-ui")
            {
                config.hide_ui = true;
                continue;
            }
            if (arg == "--transparent-screenshot")
            {
                config.transparent_screenshot = true;
                continue;
            }
            std::cerr << "unknown argument: " << arg << '\n';
            print_usage(argv[0]);
            return 2;
        }

        Runtime runtime{config};
        VectorfieldApp app{};
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
            runtime.render_imgui();
            runtime.end_main_pass();
            runtime.end_frame();
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "fatal: " << e.what() << '\n';
        return 1;
    }
}
