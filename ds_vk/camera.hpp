#pragma once

#include "ds_vk/types.hpp"

namespace ds_vk
{
enum class ProjectionMode : u8
{
    perspective = 0,
    orthographic = 1,
};

struct CameraConfig
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
    bool allow_pivot_move{true};
    bool clamp_position_z_min{};
    f32 min_position_z{};
    ProjectionMode projection_mode{ProjectionMode::perspective};
};

class Camera
{
  public:
    // clang-format off
    auto configure(const CameraConfig&) noexcept -> Camera&;

    [[nodiscard]] auto pivot() const noexcept             -> const Vec3&;
    [[nodiscard]] auto distance() const noexcept          -> f32;
    [[nodiscard]] auto yaw() const noexcept               -> f32;
    [[nodiscard]] auto pitch() const noexcept             -> f32;
    [[nodiscard]] auto fov_y() const noexcept             -> f32;
    [[nodiscard]] auto orbit_sensitivity() const noexcept -> f32;
    [[nodiscard]] auto pivot_sensitivity() const noexcept -> f32;
    [[nodiscard]] auto zoom_sensitivity() const noexcept  -> f32;
    [[nodiscard]] auto z_near() const noexcept            -> f32;
    [[nodiscard]] auto z_far() const noexcept             -> f32;
    [[nodiscard]] auto allow_pivot_move() const noexcept  -> bool;
    [[nodiscard]] auto clamp_position_z_min() const noexcept -> bool;
    [[nodiscard]] auto min_position_z() const noexcept    -> f32;
    [[nodiscard]] auto projection_mode() const noexcept   -> ProjectionMode;

    auto set_pivot(Vec3) noexcept                         -> Camera&;
    auto translate_pivot(Vec3) noexcept                   -> Camera&;
    auto set_distance(f32) noexcept                       -> Camera&;
    auto set_yaw(f32) noexcept                            -> Camera&;
    auto set_pitch(f32) noexcept                          -> Camera&;
    auto set_fov_y(f32) noexcept                          -> Camera&;
    auto set_orbit_sensitivity(f32) noexcept              -> Camera&;
    auto set_pivot_sensitivity(f32) noexcept              -> Camera&;
    auto set_zoom_sensitivity(f32) noexcept               -> Camera&;
    auto set_z_near(f32) noexcept                         -> Camera&;
    auto set_z_far(f32) noexcept                          -> Camera&;
    auto set_allow_pivot_move(bool) noexcept              -> Camera&;
    auto set_clamp_position_z_min(bool) noexcept          -> Camera&;
    auto set_min_position_z(f32) noexcept                 -> Camera&;
    auto set_projection_mode(ProjectionMode) noexcept     -> Camera&;
    auto apply_constraints() noexcept -> void;

    [[nodiscard]] auto position() const noexcept                                                     -> Vec3;
    [[nodiscard]] auto view_matrix() const noexcept                                                  -> Mat4;
    [[nodiscard]] auto projection_matrix(f32 aspect) const noexcept                                  -> Mat4;
    [[nodiscard]] auto view_projection_matrix(f32 aspect) const noexcept                             -> Mat4;
    [[nodiscard]] auto right() const noexcept                                                        -> Vec3;
    [[nodiscard]] auto up() const noexcept                                                           -> Vec3;
    [[nodiscard]] auto view_height() const noexcept                                                  -> f32;
    [[nodiscard]] auto units_per_pixel_y(f32 viewport_height_px) const noexcept                      -> f32;
    [[nodiscard]] auto pan_offset_world(f32 dx_px, f32 dy_px, f32 viewport_height_px) const noexcept -> Vec3;
    // clang-format on

  private:
    Vec3 pivot_{0.0f, 0.0f, 0.5f};
    f32 distance_{5.0f};
    f32 yaw_{glm::radians(45.0f)};
    f32 pitch_{glm::radians(24.0f)};
    f32 fov_y_{glm::radians(55.0f)};
    f32 orbit_sensitivity_{1.0f};
    f32 pivot_sensitivity_{1.0f};
    f32 zoom_sensitivity_{1.0f};
    f32 z_near_{0.02f};
    f32 z_far_{200.0f};
    bool allow_pivot_move_{true};
    bool clamp_position_z_min_{};
    f32 min_position_z_{};
    ProjectionMode projection_mode_{ProjectionMode::perspective};
};
}  // namespace ds_vk
