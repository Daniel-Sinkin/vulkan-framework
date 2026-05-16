#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>

#if defined(__has_include)
#    if __has_include(<stdfloat>)
#        include <stdfloat>
#    endif
#endif

namespace ds_vk
{
using usize = std::size_t;
using isize = std::ptrdiff_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using uptr = std::uintptr_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using iptr = std::intptr_t;

#if defined(__STDCPP_FLOAT32_T__) and defined(__STDCPP_FLOAT64_T__)
using f32 = std::float32_t;
using f64 = std::float64_t;
#else
using f32 = float;
using f64 = double;
#endif
static_assert(sizeof(f32) == 4zu);
static_assert(sizeof(f64) == 8zu);

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

    // clang-format off
    [[nodiscard]] constexpr auto data()       noexcept -> f32*       { return channels.data(); }
    [[nodiscard]] constexpr auto data() const noexcept -> const f32* { return channels.data(); }

    [[nodiscard]] constexpr auto r()          noexcept -> f32&       { return channels[0]; }
    [[nodiscard]] constexpr auto g()          noexcept -> f32&       { return channels[1]; }
    [[nodiscard]] constexpr auto b()          noexcept -> f32&       { return channels[2]; }
    [[nodiscard]] constexpr auto a()          noexcept -> f32&       { return channels[3]; }

    [[nodiscard]] constexpr auto r()    const noexcept -> f32        { return channels[0]; }
    [[nodiscard]] constexpr auto g()    const noexcept -> f32        { return channels[1]; }
    [[nodiscard]] constexpr auto b()    const noexcept -> f32        { return channels[2]; }
    [[nodiscard]] constexpr auto a()    const noexcept -> f32        { return channels[3]; }
    // clang-format on

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

// clang-format off
inline constexpr Color Color::black  {0.0f, 0.0f, 0.0f, 1.0f};
inline constexpr Color Color::white  {1.0f, 1.0f, 1.0f, 1.0f};
inline constexpr Color Color::red    {1.0f, 0.0f, 0.0f, 1.0f};
inline constexpr Color Color::green  {0.0f, 1.0f, 0.0f, 1.0f};
inline constexpr Color Color::blue   {0.0f, 0.0f, 1.0f, 1.0f};
inline constexpr Color Color::yellow {1.0f, 1.0f, 0.0f, 1.0f};
inline constexpr Color Color::cyan   {0.0f, 1.0f, 1.0f, 1.0f};
inline constexpr Color Color::magenta{1.0f, 0.0f, 1.0f, 1.0f};
inline constexpr Color Color::orange {1.0f, 0.5f, 0.0f, 1.0f};
inline constexpr Color Color::gray   {0.5f, 0.5f, 0.5f, 1.0f};
// clang-format on

struct ColorU8
{
    std::array<u8, 4> channels{255u, 255u, 255u, 255u};

    constexpr ColorU8() noexcept = default;
    constexpr ColorU8(u8 r, u8 g, u8 b, u8 a = 255u) noexcept : channels{r, g, b, a}
    {
    }

    // clang-format off
    [[nodiscard]] constexpr auto data()       noexcept -> u8*       { return channels.data(); }
    [[nodiscard]] constexpr auto data() const noexcept -> const u8* { return channels.data(); }

    [[nodiscard]] constexpr auto r()          noexcept -> u8&       { return channels[0]; }
    [[nodiscard]] constexpr auto g()          noexcept -> u8&       { return channels[1]; }
    [[nodiscard]] constexpr auto b()          noexcept -> u8&       { return channels[2]; }
    [[nodiscard]] constexpr auto a()          noexcept -> u8&       { return channels[3]; }

    [[nodiscard]] constexpr auto r()    const noexcept -> u8        { return channels[0]; }
    [[nodiscard]] constexpr auto g()    const noexcept -> u8        { return channels[1]; }
    [[nodiscard]] constexpr auto b()    const noexcept -> u8        { return channels[2]; }
    [[nodiscard]] constexpr auto a()    const noexcept -> u8        { return channels[3]; }
    // clang-format on
};

static_assert(sizeof(Color) == 4zu * sizeof(f32));
static_assert(sizeof(ColorU8) == 4zu * sizeof(u8));

[[nodiscard]] constexpr auto to_vec4(Color color) noexcept -> Vec4
{
    return Vec4{color.r(), color.g(), color.b(), color.a()};
}

[[nodiscard]] constexpr auto to_color(Vec4 value) noexcept -> Color
{
    return Color{value.r, value.g, value.b, value.a};
}

[[nodiscard]] constexpr auto to_color(ColorU8 color) noexcept -> Color
{
    constexpr auto inv_255 = 1.0f / 255.0f;
    return Color{
        static_cast<f32>(color.r()) * inv_255,
        static_cast<f32>(color.g()) * inv_255,
        static_cast<f32>(color.b()) * inv_255,
        static_cast<f32>(color.a()) * inv_255,
    };
}

[[nodiscard]] constexpr auto color_channel_to_u8(f32 value) noexcept -> u8
{
    const auto scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f;
    return static_cast<u8>(scaled);
}

[[nodiscard]] constexpr auto to_color_u8(Color color) noexcept -> ColorU8
{
    return ColorU8{
        color_channel_to_u8(color.r()),
        color_channel_to_u8(color.g()),
        color_channel_to_u8(color.b()),
        color_channel_to_u8(color.a()),
    };
}

[[nodiscard]] constexpr auto with_alpha(Color color, f32 alpha) noexcept -> Color
{
    return Color{color.r(), color.g(), color.b(), alpha};
}

[[nodiscard]] constexpr auto mix_color(Color a, Color b, f32 t) noexcept -> Color
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
inline constexpr auto k_invalid_id = std::numeric_limits<u32>::max();

struct MeshHandle
{
    u32 id{k_invalid_id};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return id != k_invalid_id;
    }
};
struct TextureHandle
{
    u32 id{k_invalid_id};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return id != k_invalid_id;
    }
};
struct ObjectId
{
    u32 value{k_invalid_id};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return value != k_invalid_id;
    }
};

inline constexpr Vec3 k_axis_x{1.0f, 0.0f, 0.0f};
inline constexpr Vec3 k_axis_y{0.0f, 1.0f, 0.0f};
inline constexpr Vec3 k_axis_z{0.0f, 0.0f, 1.0f};
inline constexpr Quat k_quat_identity{1.0f, 0.0f, 0.0f, 0.0f};
}  // namespace ds_vk
