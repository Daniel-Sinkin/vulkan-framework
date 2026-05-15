#pragma once

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

inline constexpr auto k_invalid_index = std::numeric_limits<usize>::max();
inline constexpr auto k_axis_x = Vec3{1.0f, 0.0f, 0.0f};
inline constexpr auto k_axis_y = Vec3{0.0f, 1.0f, 0.0f};
inline constexpr auto k_axis_z = Vec3{0.0f, 0.0f, 1.0f};
}  // namespace ds_vk
