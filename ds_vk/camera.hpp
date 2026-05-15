#pragma once

#include "ds_vk/types.hpp"

namespace ds_vk
{
enum class ProjectionMode : u32
{
    perspective = 0,
    orthographic = 1,
};

struct Camera
{
    Vec3 pivot{0.0f, 0.0f, 0.5f};
    f32 distance{5.0f};
    f32 yaw{glm::radians(45.0f)};
    f32 pitch{glm::radians(24.0f)};
    f32 fov_y{glm::radians(55.0f)};
    f32 orbit_sensitivity{1.0f};
    f32 pivot_sensitivity{1.0f};
    f32 zoom_sensitivity{1.0f};
    f32 z_near{0.02f};
    f32 z_far{200.0f};
    ProjectionMode projection_mode{ProjectionMode::perspective};

    [[nodiscard]] auto position() const noexcept -> Vec3;
    [[nodiscard]] auto view_matrix() const noexcept -> Mat4;
    [[nodiscard]] auto projection_matrix(f32 aspect) const noexcept -> Mat4;
    [[nodiscard]] auto view_projection_matrix(f32 aspect) const noexcept -> Mat4;
    [[nodiscard]] auto right() const noexcept -> Vec3;
    [[nodiscard]] auto up() const noexcept -> Vec3;
    [[nodiscard]] auto view_height() const noexcept -> f32;
    [[nodiscard]] auto units_per_pixel_y(f32 viewport_height_px) const noexcept -> f32;
    [[nodiscard]] auto pan_offset_world(f32 dx_px, f32 dy_px, f32 viewport_height_px) const noexcept
        -> Vec3;
};
}  // namespace ds_vk
