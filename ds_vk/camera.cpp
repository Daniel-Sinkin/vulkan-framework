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
    allow_pivot_move_ = config.allow_pivot_move;
    clamp_position_z_min_ = config.clamp_position_z_min;
    min_position_z_ = config.min_position_z;
    projection_mode_ = config.projection_mode;
    apply_constraints();
    return *this;
}

// clang-format off
auto Camera::pivot() const noexcept                  -> const Vec3&     { return pivot_; }
auto Camera::distance() const noexcept               -> f32             { return distance_; }
auto Camera::yaw() const noexcept                    -> f32             { return yaw_; }
auto Camera::pitch() const noexcept                  -> f32             { return pitch_; }
auto Camera::fov_y() const noexcept                  -> f32             { return fov_y_; }
auto Camera::orbit_sensitivity() const noexcept      -> f32             { return orbit_sensitivity_; }
auto Camera::pivot_sensitivity() const noexcept      -> f32             { return pivot_sensitivity_; }
auto Camera::zoom_sensitivity() const noexcept       -> f32             { return zoom_sensitivity_; }
auto Camera::z_near() const noexcept                 -> f32             { return z_near_; }
auto Camera::z_far() const noexcept                  -> f32             { return z_far_; }
auto Camera::allow_pivot_move() const noexcept       -> bool            { return allow_pivot_move_; }
auto Camera::clamp_position_z_min() const noexcept   -> bool            { return clamp_position_z_min_; }
auto Camera::min_position_z() const noexcept         -> f32             { return min_position_z_; }
auto Camera::projection_mode() const noexcept        -> ProjectionMode  { return projection_mode_; }
// clang-format on

auto Camera::set_pivot(const Vec3 pivot) noexcept -> Camera&
{
    pivot_ = pivot;
    apply_constraints();
    return *this;
}

auto Camera::translate_pivot(const Vec3 offset) noexcept -> Camera&
{
    return set_pivot(pivot_ + offset);
}

auto Camera::set_distance(const f32 distance) noexcept -> Camera&
{
    distance_ = distance;
    apply_constraints();
    return *this;
}

auto Camera::set_yaw(const f32 yaw) noexcept -> Camera&
{
    yaw_ = yaw;
    return *this;
}

auto Camera::set_pitch(const f32 pitch) noexcept -> Camera&
{
    pitch_ = pitch;
    apply_constraints();
    return *this;
}

auto Camera::set_fov_y(const f32 fov_y) noexcept -> Camera&
{
    fov_y_ = fov_y;
    return *this;
}

auto Camera::set_orbit_sensitivity(const f32 sensitivity) noexcept -> Camera&
{
    orbit_sensitivity_ = sensitivity;
    return *this;
}

auto Camera::set_pivot_sensitivity(const f32 sensitivity) noexcept -> Camera&
{
    pivot_sensitivity_ = sensitivity;
    return *this;
}

auto Camera::set_zoom_sensitivity(const f32 sensitivity) noexcept -> Camera&
{
    zoom_sensitivity_ = sensitivity;
    return *this;
}

auto Camera::set_z_near(const f32 z_near) noexcept -> Camera&
{
    z_near_ = z_near;
    return *this;
}

auto Camera::set_z_far(const f32 z_far) noexcept -> Camera&
{
    z_far_ = z_far;
    return *this;
}

auto Camera::set_allow_pivot_move(const bool allow) noexcept -> Camera&
{
    allow_pivot_move_ = allow;
    return *this;
}

auto Camera::set_clamp_position_z_min(const bool clamp) noexcept -> Camera&
{
    clamp_position_z_min_ = clamp;
    apply_constraints();
    return *this;
}

auto Camera::set_min_position_z(const f32 min_z) noexcept -> Camera&
{
    min_position_z_ = min_z;
    apply_constraints();
    return *this;
}

auto Camera::set_projection_mode(const ProjectionMode projection_mode) noexcept -> Camera&
{
    projection_mode_ = projection_mode;
    return *this;
}

auto Camera::apply_constraints() noexcept -> void
{
    if (!clamp_position_z_min_)
    {
        return;
    }
    const auto min_offset_z = min_position_z_ - pivot_.z;
    if (min_offset_z <= -distance_)
    {
        return;
    }
    const auto min_pitch =
        std::asin(std::clamp(min_offset_z / std::max(distance_, 0.001f), -1.0f, 1.0f));
    pitch_ = std::max(pitch_, min_pitch);
}

auto Camera::position() const noexcept -> Vec3
{
    const auto cos_pitch = std::cos(pitch_);
    const Vec3 offset{
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
    switch (projection_mode_)
    {
        case ProjectionMode::orthographic:
            {
                const auto half_height = 0.5f * view_height();
                const auto half_width = half_height * clamped_aspect;
                auto proj = glm::orthoRH_ZO(
                    -half_width, half_width, -half_height, half_height, z_near_, z_far_
                );
                proj[1][1] *= -1.0f;
                return proj;
            }
        case ProjectionMode::perspective:
            {
                auto proj = glm::perspective(fov_y_, clamped_aspect, z_near_, z_far_);
                proj[1][1] *= -1.0f;
                return proj;
            }
    }
    return Mat4{1.0f};
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
