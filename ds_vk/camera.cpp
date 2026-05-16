#include "ds_vk/camera.hpp"

#include "ds_vk/math.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace ds_vk
{
auto Camera::configure(const CameraConfig& config) noexcept -> Camera&
{
    pivot_ = config.pivot;
    distance_ = config.distance;
    yaw_ = config.yaw;
    pitch_ = config.pitch;
    fov_y_ = config.fov_y;
    orbit_sensitivity_ = config.orbit_sensitivity;
    pivot_sensitivity_ = config.pivot_sensitivity;
    zoom_sensitivity_ = config.zoom_sensitivity;
    z_near_ = config.z_near;
    z_far_ = config.z_far;
    projection_mode_ = config.projection_mode;
    return *this;
}

auto Camera::pivot() noexcept -> Vec3&
{
    return pivot_;
}

auto Camera::pivot() const noexcept -> const Vec3&
{
    return pivot_;
}

auto Camera::distance() noexcept -> f32&
{
    return distance_;
}

auto Camera::distance() const noexcept -> f32
{
    return distance_;
}

auto Camera::yaw() noexcept -> f32&
{
    return yaw_;
}

auto Camera::yaw() const noexcept -> f32
{
    return yaw_;
}

auto Camera::pitch() noexcept -> f32&
{
    return pitch_;
}

auto Camera::pitch() const noexcept -> f32
{
    return pitch_;
}

auto Camera::fov_y() noexcept -> f32&
{
    return fov_y_;
}

auto Camera::fov_y() const noexcept -> f32
{
    return fov_y_;
}

auto Camera::orbit_sensitivity() noexcept -> f32&
{
    return orbit_sensitivity_;
}

auto Camera::orbit_sensitivity() const noexcept -> f32
{
    return orbit_sensitivity_;
}

auto Camera::pivot_sensitivity() noexcept -> f32&
{
    return pivot_sensitivity_;
}

auto Camera::pivot_sensitivity() const noexcept -> f32
{
    return pivot_sensitivity_;
}

auto Camera::zoom_sensitivity() noexcept -> f32&
{
    return zoom_sensitivity_;
}

auto Camera::zoom_sensitivity() const noexcept -> f32
{
    return zoom_sensitivity_;
}

auto Camera::z_near() noexcept -> f32&
{
    return z_near_;
}

auto Camera::z_near() const noexcept -> f32
{
    return z_near_;
}

auto Camera::z_far() noexcept -> f32&
{
    return z_far_;
}

auto Camera::z_far() const noexcept -> f32
{
    return z_far_;
}

auto Camera::projection_mode() noexcept -> ProjectionMode&
{
    return projection_mode_;
}

auto Camera::projection_mode() const noexcept -> ProjectionMode
{
    return projection_mode_;
}

auto Camera::position() const noexcept -> Vec3
{
    const auto cos_pitch = std::cos(pitch_);
    const auto offset = Vec3{
        distance_ * cos_pitch * std::cos(yaw_),
        distance_ * cos_pitch * std::sin(yaw_),
        distance_ * std::sin(pitch_),
    };
    return pivot_ + offset;
}

auto Camera::view_matrix() const noexcept -> Mat4
{
    return glm::lookAt(position(), pivot_, k_axis_z);
}

auto Camera::projection_matrix(f32 aspect) const noexcept -> Mat4
{
    const auto clamped_aspect = std::max(0.01f, aspect);
    if (projection_mode_ == ProjectionMode::orthographic)
    {
        const auto half_height = 0.5f * view_height();
        const auto half_width = half_height * clamped_aspect;
        auto proj =
            glm::orthoRH_ZO(-half_width, half_width, -half_height, half_height, z_near_, z_far_);
        proj[1][1] *= -1.0f;
        return proj;
    }

    auto proj = glm::perspective(fov_y_, clamped_aspect, z_near_, z_far_);
    proj[1][1] *= -1.0f;
    return proj;
}

auto Camera::view_projection_matrix(f32 aspect) const noexcept -> Mat4
{
    return projection_matrix(aspect) * view_matrix();
}

auto Camera::right() const noexcept -> Vec3
{
    const auto forward = normalize_or(pivot_ - position(), -k_axis_y);
    return normalize_or(glm::cross(forward, k_axis_z), k_axis_x);
}

auto Camera::up() const noexcept -> Vec3
{
    const auto forward = normalize_or(pivot_ - position(), -k_axis_y);
    return normalize_or(glm::cross(right(), forward), k_axis_z);
}

auto Camera::view_height() const noexcept -> f32
{
    return 2.0f * distance_ * std::tan(0.5f * fov_y_);
}

auto Camera::units_per_pixel_y(f32 viewport_height_px) const noexcept -> f32
{
    return view_height() / std::max(1.0f, viewport_height_px);
}

auto Camera::pan_offset_world(f32 dx_px, f32 dy_px, f32 viewport_height_px) const noexcept -> Vec3
{
    const auto units = units_per_pixel_y(viewport_height_px);
    return (-dx_px * units) * right() + (dy_px * units) * up();
}
}  // namespace ds_vk
