#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>

namespace ds_vk
{
using usize = std::size_t;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using f32 = float;
using f64 = double;

static_assert(sizeof(f32) == 4);
static_assert(sizeof(f64) == 8);

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Quat = glm::quat;
using Mat4 = glm::mat4;

struct Color
{
    std::array<f32, 4> channels{1.0f, 1.0f, 1.0f, 1.0f};

    constexpr Color() noexcept = default;
    constexpr Color(f32 r, f32 g, f32 b, f32 a = 1.0f) noexcept : channels{r, g, b, a}
    {
    }

    constexpr auto data() noexcept -> f32*
    {
        return channels.data();
    }

    constexpr auto data() const noexcept -> const f32*
    {
        return channels.data();
    }

    constexpr auto r() noexcept -> f32&
    {
        return channels[0];
    }

    constexpr auto g() noexcept -> f32&
    {
        return channels[1];
    }

    constexpr auto b() noexcept -> f32&
    {
        return channels[2];
    }

    constexpr auto a() noexcept -> f32&
    {
        return channels[3];
    }

    constexpr auto r() const noexcept -> f32
    {
        return channels[0];
    }

    constexpr auto g() const noexcept -> f32
    {
        return channels[1];
    }

    constexpr auto b() const noexcept -> f32
    {
        return channels[2];
    }

    constexpr auto a() const noexcept -> f32
    {
        return channels[3];
    }

    static const Color black;
    static const Color white;
    static const Color red;
    static const Color green;
    static const Color blue;
    static const Color yellow;
    static const Color cyan;
    static const Color magenta;
    static const Color orange;
    static const Color gray;
};

inline constexpr Color Color::black{0.0f, 0.0f, 0.0f, 1.0f};
inline constexpr Color Color::white{1.0f, 1.0f, 1.0f, 1.0f};
inline constexpr Color Color::red{1.0f, 0.0f, 0.0f, 1.0f};
inline constexpr Color Color::green{0.0f, 1.0f, 0.0f, 1.0f};
inline constexpr Color Color::blue{0.0f, 0.0f, 1.0f, 1.0f};
inline constexpr Color Color::yellow{1.0f, 1.0f, 0.0f, 1.0f};
inline constexpr Color Color::cyan{0.0f, 1.0f, 1.0f, 1.0f};
inline constexpr Color Color::magenta{1.0f, 0.0f, 1.0f, 1.0f};
inline constexpr Color Color::orange{1.0f, 0.5f, 0.0f, 1.0f};
inline constexpr Color Color::gray{0.5f, 0.5f, 0.5f, 1.0f};

struct ColorU8
{
    std::array<u8, 4> channels{255u, 255u, 255u, 255u};

    constexpr ColorU8() noexcept = default;
    constexpr ColorU8(u8 r, u8 g, u8 b, u8 a = 255u) noexcept : channels{r, g, b, a}
    {
    }

    constexpr auto data() noexcept -> u8*
    {
        return channels.data();
    }

    constexpr auto data() const noexcept -> const u8*
    {
        return channels.data();
    }

    constexpr auto r() noexcept -> u8&
    {
        return channels[0];
    }

    constexpr auto g() noexcept -> u8&
    {
        return channels[1];
    }

    constexpr auto b() noexcept -> u8&
    {
        return channels[2];
    }

    constexpr auto a() noexcept -> u8&
    {
        return channels[3];
    }

    constexpr auto r() const noexcept -> u8
    {
        return channels[0];
    }

    constexpr auto g() const noexcept -> u8
    {
        return channels[1];
    }

    constexpr auto b() const noexcept -> u8
    {
        return channels[2];
    }

    constexpr auto a() const noexcept -> u8
    {
        return channels[3];
    }
};

static_assert(sizeof(Color) == 4u * sizeof(f32));
static_assert(sizeof(ColorU8) == 4u * sizeof(u8));

[[nodiscard]] constexpr auto to_vec4(const Color color) noexcept -> Vec4
{
    return Vec4{color.r(), color.g(), color.b(), color.a()};
}

[[nodiscard]] constexpr auto to_color(const Vec4 value) noexcept -> Color
{
    return Color{value.r, value.g, value.b, value.a};
}

[[nodiscard]] constexpr auto to_color(const ColorU8 color) noexcept -> Color
{
    constexpr auto inv_255 = 1.0f / 255.0f;
    return Color{
        static_cast<f32>(color.r()) * inv_255,
        static_cast<f32>(color.g()) * inv_255,
        static_cast<f32>(color.b()) * inv_255,
        static_cast<f32>(color.a()) * inv_255,
    };
}

[[nodiscard]] constexpr auto color_channel_to_u8(const f32 value) noexcept -> u8
{
    const auto scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f;
    return static_cast<u8>(scaled);
}

[[nodiscard]] constexpr auto to_color_u8(const Color color) noexcept -> ColorU8
{
    return ColorU8{
        color_channel_to_u8(color.r()),
        color_channel_to_u8(color.g()),
        color_channel_to_u8(color.b()),
        color_channel_to_u8(color.a()),
    };
}

[[nodiscard]] constexpr auto with_alpha(const Color color, const f32 alpha) noexcept -> Color
{
    return Color{color.r(), color.g(), color.b(), alpha};
}

[[nodiscard]] constexpr auto mix_color(const Color a, const Color b, const f32 t) noexcept -> Color
{
    const auto clamped_t = std::clamp(t, 0.0f, 1.0f);
    const auto inv_t = 1.0f - clamped_t;
    return Color{
        a.r() * inv_t + b.r() * clamped_t,
        a.g() * inv_t + b.g() * clamped_t,
        a.b() * inv_t + b.b() * clamped_t,
        a.a() * inv_t + b.a() * clamped_t,
    };
}

inline constexpr auto k_invalid_index = std::numeric_limits<usize>::max();
inline constexpr auto k_invalid_object_id = std::numeric_limits<u32>::max();

struct ObjectId
{
    u32 value{k_invalid_object_id};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return value != k_invalid_object_id;
    }
};

inline constexpr auto k_axis_x = Vec3{1.0f, 0.0f, 0.0f};
inline constexpr auto k_axis_y = Vec3{0.0f, 1.0f, 0.0f};
inline constexpr auto k_axis_z = Vec3{0.0f, 0.0f, 1.0f};
}  // namespace ds_vk
