#pragma once

#include "ds_vk/geometry.hpp"
#include "ds_vk/types.hpp"

#include <span>
#include <string>
#include <vector>

namespace ds_vk_app::pba
{
inline constexpr auto k_body_id_base = ds_vk::u32{2000u};
inline constexpr auto k_fixed_dt = 1.0f / 120.0f;

struct Body
{
    ds_vk::ObjectId object_id{};
    ds_vk::Vec3 half_extent{0.5f};
    ds_vk::Vec3 position{};
    ds_vk::Vec3 previous_position{};
    ds_vk::Vec3 velocity{};
    ds_vk::Quat orientation{ds_vk::k_quat_identity};
    ds_vk::f32 inv_mass{1.0f};
    ds_vk::Color color{0.58f, 0.61f, 0.58f, 1.0f};
    std::string name{};
};

struct PhysicsConfig
{
    ds_vk::Vec3 gravity{-9.81f * ds_vk::k_axis_z};
    ds_vk::f32 restitution{0.05f};
    ds_vk::f32 friction{0.78f};
    ds_vk::f32 linear_damping{0.04f};
    ds_vk::u32 solver_iterations{5u};
};

struct PyramidSceneConfig
{
    int base{5};
    ds_vk::f32 step{1.05f};
    bool add_projectile{true};
};

[[nodiscard]] auto body_aabb(const Body& body) noexcept -> ds_vk::Aabb;

class PyramidSimulation
{
  public:
    auto reset(const PyramidSceneConfig& cfg = {}) -> void;
    auto step(ds_vk::f32 dt) -> void;

    // clang-format off
    [[nodiscard]] auto bodies() noexcept              -> std::span<Body>;
    [[nodiscard]] auto bodies() const noexcept        -> std::span<const Body>;
    [[nodiscard]] auto physics() noexcept             -> PhysicsConfig&;
    [[nodiscard]] auto physics() const noexcept       -> const PhysicsConfig&;
    [[nodiscard]] auto step_count() const noexcept    -> ds_vk::usize;
    // clang-format on

  private:
    std::vector<Body> bodies_{};
    PhysicsConfig physics_{};
    ds_vk::usize step_count_{};
};
}  // namespace ds_vk_app::pba
