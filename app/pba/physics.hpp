#pragma once

#include "ds_vk/geometry.hpp"
#include "ds_vk/types.hpp"

#include <span>
#include <string>
#include <variant>
#include <vector>

namespace ds_vk_app::pba
{
inline constexpr ds_vk::u32 k_body_id_base{2000u};
inline constexpr auto k_fixed_dt = 1.0f / 120.0f;

struct Body
{
    ds_vk::ObjectId object_id{};
    ds_vk::Vec3 half_extent{0.5f};
    ds_vk::Vec3 position{};
    ds_vk::Vec3 previous_position{};
    ds_vk::Vec3 velocity{};
    ds_vk::Vec3 angular_velocity{};
    ds_vk::Vec3 force_accum{};
    ds_vk::Vec3 torque_accum{};
    ds_vk::Quat orientation{ds_vk::k_quat_identity};
    ds_vk::f32 inv_mass{1.0f};
    ds_vk::Color color{0.58f, 0.61f, 0.58f, 1.0f};
    bool grabbed{};
    std::string name{};
};

struct GravityForce
{
    ds_vk::Vec3 accel{-9.81f * ds_vk::k_axis_z};
};

struct AttractorForce
{
    ds_vk::Vec3 target{};
    ds_vk::f32 magnitude{10.0f};
    ds_vk::f32 min_radius{0.25f};
};

struct RepulsionForce
{
    ds_vk::Vec3 target{};
    ds_vk::f32 accel_max{12.0f};
    ds_vk::f32 range{4.0f};
    ds_vk::f32 min_radius{0.35f};
};

struct NBodyForce
{
    ds_vk::f32 g{1.0f};
    ds_vk::f32 softening{1.0e-3f};
};

using SimpleForce = std::variant<GravityForce, AttractorForce, RepulsionForce>;
using ComplexForce = std::variant<NBodyForce>;

struct PhysicsConfig
{
    ds_vk::f32 restitution{0.05f};
    ds_vk::f32 friction{0.78f};
    ds_vk::f32 linear_damping{0.04f};
    ds_vk::f32 angular_damping{0.08f};
    ds_vk::u32 solver_iterations{5u};
    std::vector<SimpleForce> simple_forces{SimpleForce{GravityForce{}}};
    std::vector<ComplexForce> complex_forces{};
};

struct PhysicsStepStats
{
    ds_vk::usize broadphase_candidates{};
    ds_vk::usize body_pair_count{};
};

struct PyramidSceneConfig
{
    ds_vk::usize base{5zu};
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
    [[nodiscard]] auto find_body(ds_vk::ObjectId id) noexcept       -> Body*;
    [[nodiscard]] auto find_body(ds_vk::ObjectId id) const noexcept -> const Body*;
    [[nodiscard]] auto physics() noexcept             -> PhysicsConfig&;
    [[nodiscard]] auto physics() const noexcept       -> const PhysicsConfig&;
    [[nodiscard]] auto step_count() const noexcept    -> ds_vk::usize;
    [[nodiscard]] auto last_step_stats() const noexcept -> const PhysicsStepStats&;
    // clang-format on

  private:
    struct CollisionPair
    {
        ds_vk::usize a{};
        ds_vk::usize b{};
    };

    auto rebuild_collision_pairs() -> void;

    std::vector<Body> bodies_{};
    std::vector<ds_vk::Aabb> body_bounds_{};
    std::vector<ds_vk::usize> broadphase_order_{};
    std::vector<ds_vk::usize> broadphase_active_{};
    std::vector<CollisionPair> collision_pairs_{};
    PhysicsConfig physics_{};
    PhysicsStepStats last_step_stats_{};
    ds_vk::usize step_count_{};
};
}  // namespace ds_vk_app::pba
