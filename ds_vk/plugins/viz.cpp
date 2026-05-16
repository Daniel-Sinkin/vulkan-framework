#include "ds_vk/plugins/viz.hpp"

#include <array>
#include <cmath>

namespace ds_vk::viz
{
namespace
{
struct ColorStop
{
    f32 t{};
    Color color;
};

[[nodiscard]] auto saturate(f32 value) noexcept -> f32
{
    return std::clamp(value, 0.0f, 1.0f);
}

[[nodiscard]] auto saturate_color(Color color) noexcept -> Color
{
    return Color{
        saturate(color.r()), saturate(color.g()), saturate(color.b()), saturate(color.a())
    };
}

[[nodiscard]] auto sample_stops(std::span<const ColorStop> stops, f32 t_in) noexcept -> Color
{
    if (stops.empty())
    {
        return Color::white;
    }

    const auto t = saturate(t_in);
    if (t <= stops.front().t)
    {
        return saturate_color(stops.front().color);
    }
    if (t >= stops.back().t)
    {
        return saturate_color(stops.back().color);
    }

    for (auto i = 1zu; i < stops.size(); ++i)
    {
        const auto& right = stops[i];
        if (t <= right.t)
        {
            const auto& left = stops[i - 1zu];
            const auto denom = std::max(1.0e-6f, right.t - left.t);
            const auto local_t = (t - left.t) / denom;
            return saturate_color(mix_color(left.color, right.color, local_t));
        }
    }
    return saturate_color(stops.back().color);
}

[[nodiscard]] auto grayscale(f32 t) noexcept -> Color
{
    const auto v = saturate(t);
    return Color{v, v, v, 1.0f};
}
}  // namespace

ColorRamp::ColorRamp(ColorRampConfig config) noexcept : config_(config)
{
}

auto ColorRamp::configure(const ColorRampConfig& config) noexcept -> ColorRamp&
{
    config_ = config;
    return *this;
}

auto ColorRamp::sample(f32 value) const noexcept -> Color
{
    if (!std::isfinite(value))
    {
        return config_.nan_color;
    }
    return sample_color(config_.preset, normalized_value(value));
}

auto ColorRamp::normalized_value(f32 value) const noexcept -> f32
{
    const auto denom = config_.range.max - config_.range.min;
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    if (std::abs(denom) <= 1.0e-8f)
    {
        return 0.5f;
    }

    const auto normalized = (value - config_.range.min) / denom;
    return config_.clamp ? saturate(normalized) : normalized;
}

auto ColorRamp::config() const noexcept -> const ColorRampConfig&
{
    return config_;
}

auto sample_color(ColorPreset preset, f32 normalized_value) noexcept -> Color
{
    switch (preset)
    {
        case ColorPreset::grayscale:
            return grayscale(normalized_value);
        case ColorPreset::blue_red:
            {
                static constexpr std::array stops{
                    ColorStop{.t = 0.00f, .color = {0.10f, 0.18f, 0.84f}},
                    ColorStop{.t = 0.25f, .color = {0.12f, 0.68f, 0.92f}},
                    ColorStop{.t = 0.50f, .color = {0.96f, 0.96f, 0.90f}},
                    ColorStop{.t = 0.75f, .color = {0.98f, 0.70f, 0.16f}},
                    ColorStop{.t = 1.00f, .color = {0.82f, 0.08f, 0.12f}},
                };
                return sample_stops(stops, normalized_value);
            }
        case ColorPreset::viridis:
            {
                static constexpr std::array stops{
                    ColorStop{.t = 0.00f, .color = {0.267f, 0.005f, 0.329f}},
                    ColorStop{.t = 0.15f, .color = {0.283f, 0.141f, 0.458f}},
                    ColorStop{.t = 0.30f, .color = {0.254f, 0.265f, 0.530f}},
                    ColorStop{.t = 0.45f, .color = {0.207f, 0.372f, 0.553f}},
                    ColorStop{.t = 0.60f, .color = {0.128f, 0.567f, 0.551f}},
                    ColorStop{.t = 0.75f, .color = {0.267f, 0.749f, 0.441f}},
                    ColorStop{.t = 1.00f, .color = {0.993f, 0.906f, 0.144f}},
                };
                return sample_stops(stops, normalized_value);
            }
        case ColorPreset::magma:
            {
                static constexpr std::array stops{
                    ColorStop{.t = 0.00f, .color = {0.001f, 0.000f, 0.014f}},
                    ColorStop{.t = 0.17f, .color = {0.111f, 0.064f, 0.262f}},
                    ColorStop{.t = 0.34f, .color = {0.316f, 0.071f, 0.485f}},
                    ColorStop{.t = 0.50f, .color = {0.594f, 0.176f, 0.501f}},
                    ColorStop{.t = 0.67f, .color = {0.857f, 0.360f, 0.410f}},
                    ColorStop{.t = 0.84f, .color = {0.988f, 0.647f, 0.211f}},
                    ColorStop{.t = 1.00f, .color = {0.987f, 0.991f, 0.749f}},
                };
                return sample_stops(stops, normalized_value);
            }
        case ColorPreset::turbo:
            {
                static constexpr std::array stops{
                    ColorStop{.t = 0.00f, .color = {0.190f, 0.072f, 0.232f}},
                    ColorStop{.t = 0.14f, .color = {0.145f, 0.365f, 0.901f}},
                    ColorStop{.t = 0.28f, .color = {0.023f, 0.627f, 0.886f}},
                    ColorStop{.t = 0.42f, .color = {0.187f, 0.808f, 0.559f}},
                    ColorStop{.t = 0.56f, .color = {0.627f, 0.897f, 0.196f}},
                    ColorStop{.t = 0.70f, .color = {0.956f, 0.706f, 0.095f}},
                    ColorStop{.t = 0.84f, .color = {0.892f, 0.314f, 0.087f}},
                    ColorStop{.t = 1.00f, .color = {0.480f, 0.016f, 0.010f}},
                };
                return sample_stops(stops, normalized_value);
            }
    }
    return grayscale(normalized_value);
}

auto range_from_values(std::span<const f32> values) noexcept -> ScalarRange
{
    auto found = false;
    auto min_value = 0.0f;
    auto max_value = 0.0f;
    for (const auto value : values)
    {
        if (!std::isfinite(value))
        {
            continue;
        }
        if (!found)
        {
            min_value = value;
            max_value = value;
            found = true;
            continue;
        }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    if (!found)
    {
        return ScalarRange{};
    }
    if (std::abs(max_value - min_value) <= 1.0e-8f)
    {
        return ScalarRange{.min = min_value - 0.5f, .max = max_value + 0.5f};
    }
    return ScalarRange{.min = min_value, .max = max_value};
}
}  // namespace ds_vk::viz
