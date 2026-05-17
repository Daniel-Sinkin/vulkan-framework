#pragma once

#include "ds_vk/camera.hpp"
#include "ds_vk/debug_draw.hpp"
#include "ds_vk/geometry.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace ds_vk::viz
{
enum class ColorPreset : u8
{
    grayscale = 0,
    blue_red = 1,
    viridis = 2,
    magma = 3,
    turbo = 4,
};

struct ScalarRange
{
    f32 min{0.0f};
    f32 max{1.0f};
};

struct ColorRampConfig
{
    ColorPreset preset{ColorPreset::turbo};
    ScalarRange range{};
    bool clamp{true};
    Color nan_color{1.0f, 0.0f, 1.0f, 1.0f};
};

class ColorRamp
{
  public:
    explicit ColorRamp(ColorRampConfig = {}) noexcept;

    // clang-format off
    auto configure(const ColorRampConfig&) noexcept               -> ColorRamp&;

    [[nodiscard]] auto sample(f32 value) const noexcept           -> Color;
    [[nodiscard]] auto normalized_value(f32 value) const noexcept -> f32;
    [[nodiscard]] auto config() const noexcept                    -> const ColorRampConfig&;
    // clang-format on

  private:
    ColorRampConfig config_{};
};

// clang-format off
[[nodiscard]] auto sample_color(ColorPreset preset, f32 normalized_value) noexcept -> Color;
[[nodiscard]] auto range_from_values(std::span<const f32> values) noexcept         -> ScalarRange;
// clang-format on

struct VectorFieldConfig
{
    std::span<const Vec3> positions{};
    std::span<const Vec3> vectors{};
    f32 scale{1.0f};
    Color color{0.12f, 0.92f, 0.58f, 1.0f};
    f32 width{0.010f};
    bool color_by_magnitude{};
    ColorRamp color_ramp{};
    f32 min_vector_length{1.0e-5f};
    usize max_vectors{std::numeric_limits<usize>::max()};
    bool draw_on_top{};
};

struct CrossMarkerConfig
{
    const Camera& camera;
    Vec3 center{};
    f32 radius{0.08f};
    Color color{1.0f, 0.0f, 0.85f, 1.0f};
    f32 width{0.012f};
};

struct TrailConfig
{
    std::span<const Vec3> points{};
    Color color{1.0f, 0.72f, 0.20f, 1.0f};
    f32 width{0.007f};
    bool fade_alpha{true};
    f32 tail_alpha{0.22f};
    f32 head_alpha{1.0f};
};

struct AabbMarkerConfig
{
    Aabb aabb{};
    Color color{0.42f, 0.70f, 0.88f, 0.55f};
    f32 width{0.006f};
    bool draw_on_top{};
};

template <typename DrawSink>
auto draw_vector_field(DrawSink& draw, const VectorFieldConfig& config) -> usize
{
    const auto count =
        std::min({config.positions.size(), config.vectors.size(), config.max_vectors});
    auto drawn = 0zu;
    const auto min_length_squared = config.min_vector_length * config.min_vector_length;
    for (auto i = 0zu; i < count; ++i)
    {
        const auto vector = config.vectors[i];
        const auto length_squared = glm::dot(vector, vector);
        if (length_squared > min_length_squared)
        {
            const auto color = config.color_by_magnitude
                                   ? config.color_ramp.sample(std::sqrt(length_squared))
                                   : config.color;
            draw.debug_arrow(
                DebugArrowConfig{
                    .origin = config.positions[i],
                    .vector = vector * config.scale,
                    .color = color,
                    .width = config.width,
                    .draw_on_top = config.draw_on_top,
                }
            );
            ++drawn;
        }
    }
    return drawn;
}

template <typename DrawSink>
auto draw_cross_marker(DrawSink& draw, const CrossMarkerConfig& config) -> usize
{
    const auto radius = std::max(0.0f, config.radius);
    if (radius <= 0.0f)
    {
        return 0zu;
    }

    const auto right = config.camera.right();
    const auto up = config.camera.up();
    draw.debug_line(
        DebugLineConfig{
            .start = config.center - radius * right - radius * up,
            .end = config.center + radius * right + radius * up,
            .color = config.color,
            .width = config.width,
        }
    );
    draw.debug_line(
        DebugLineConfig{
            .start = config.center - radius * right + radius * up,
            .end = config.center + radius * right - radius * up,
            .color = config.color,
            .width = config.width,
        }
    );
    return 2zu;
}

template <typename DrawSink>
auto draw_trail(DrawSink& draw, const TrailConfig& config) -> usize
{
    if (config.points.size() < 2zu)
    {
        return 0zu;
    }

    const auto segment_count = config.points.size() - 1zu;
    for (auto i = 0zu; i < segment_count; ++i)
    {
        auto color = config.color;
        if (config.fade_alpha)
        {
            const auto denom = std::max(1zu, segment_count - 1zu);
            const auto t = static_cast<f32>(i) / static_cast<f32>(denom);
            color = with_alpha(
                color, std::lerp(config.tail_alpha, config.head_alpha, std::clamp(t, 0.0f, 1.0f))
            );
        }
        draw.debug_line(
            DebugLineConfig{
                .start = config.points[i],
                .end = config.points[i + 1zu],
                .color = color,
                .width = config.width,
            }
        );
    }
    return segment_count;
}

template <typename DrawSink>
auto draw_aabb(DrawSink& draw, const AabbMarkerConfig& config) -> usize
{
    const auto min = glm::min(config.aabb.min, config.aabb.max);
    const auto max = glm::max(config.aabb.min, config.aabb.max);
    const Vec3 p000{min.x, min.y, min.z};
    const Vec3 p100{max.x, min.y, min.z};
    const Vec3 p010{min.x, max.y, min.z};
    const Vec3 p110{max.x, max.y, min.z};
    const Vec3 p001{min.x, min.y, max.z};
    const Vec3 p101{max.x, min.y, max.z};
    const Vec3 p011{min.x, max.y, max.z};
    const Vec3 p111{max.x, max.y, max.z};
    const auto line = [&](Vec3 start, Vec3 end) -> void
    {
        draw.debug_line(
            DebugLineConfig{
                .start = start,
                .end = end,
                .color = config.color,
                .width = config.width,
                .draw_on_top = config.draw_on_top,
            }
        );
    };
    line(p000, p100);
    line(p100, p110);
    line(p110, p010);
    line(p010, p000);
    line(p001, p101);
    line(p101, p111);
    line(p111, p011);
    line(p011, p001);
    line(p000, p001);
    line(p100, p101);
    line(p010, p011);
    line(p110, p111);
    return 12zu;
}
}  // namespace ds_vk::viz
