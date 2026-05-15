#include "ds_vk/camera.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace ds_vk
{
namespace
{
auto safe_normalize(const Vec3 value, const Vec3 fallback) noexcept -> Vec3
{
    const auto length_squared = glm::dot(value, value);
    if (length_squared <= 1.0e-12f)
    {
        return glm::normalize(fallback);
    }
    return value * glm::inversesqrt(length_squared);
}
}  // namespace

auto Camera::position() const noexcept -> Vec3
{
    const auto cos_pitch = std::cos(pitch);
    const auto offset = Vec3{
        distance * cos_pitch * std::cos(yaw),
        distance * cos_pitch * std::sin(yaw),
        distance * std::sin(pitch),
    };
    return pivot + offset;
}

auto Camera::view_matrix() const noexcept -> Mat4
{
    return glm::lookAt(position(), pivot, k_axis_z);
}

auto Camera::projection_matrix(const f32 aspect) const noexcept -> Mat4
{
    const auto clamped_aspect = std::max(0.01f, aspect);
    if (projection_mode == ProjectionMode::orthographic)
    {
        const auto half_height = 0.5f * view_height();
        const auto half_width = half_height * clamped_aspect;
        auto proj =
            glm::orthoRH_ZO(-half_width, half_width, -half_height, half_height, z_near, z_far);
        proj[1][1] *= -1.0f;
        return proj;
    }

    auto proj = glm::perspective(fov_y, clamped_aspect, z_near, z_far);
    proj[1][1] *= -1.0f;
    return proj;
}

auto Camera::view_projection_matrix(const f32 aspect) const noexcept -> Mat4
{
    return projection_matrix(aspect) * view_matrix();
}

auto Camera::right() const noexcept -> Vec3
{
    const auto forward = safe_normalize(pivot - position(), -k_axis_y);
    return safe_normalize(glm::cross(forward, k_axis_z), k_axis_x);
}

auto Camera::up() const noexcept -> Vec3
{
    const auto forward = safe_normalize(pivot - position(), -k_axis_y);
    return safe_normalize(glm::cross(right(), forward), k_axis_z);
}

auto Camera::view_height() const noexcept -> f32
{
    return 2.0f * distance * std::tan(0.5f * fov_y);
}

auto Camera::units_per_pixel_y(const f32 viewport_height_px) const noexcept -> f32
{
    return view_height() / std::max(1.0f, viewport_height_px);
}

auto Camera::pan_offset_world(
    const f32 dx_px, const f32 dy_px, const f32 viewport_height_px
) const noexcept -> Vec3
{
    const auto units = units_per_pixel_y(viewport_height_px);
    return (-dx_px * units) * right() + (dy_px * units) * up();
}
}  // namespace ds_vk
