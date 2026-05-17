#pragma once

#include "ds_vk/types.hpp"

#include <optional>

namespace ds_vk
{
class Camera;

struct Ray
{
    Vec3 origin{};
    Vec3 direction{-k_axis_z};
};

struct Sphere
{
    Vec3 center{};
    f32 radius{1.0f};
};

struct Aabb
{
    Vec3 min{};
    Vec3 max{};
};

struct Obb
{
    Vec3 center{};
    Vec3 half_extent{0.5f};
    Quat rotation{k_quat_identity};
};

struct Capsule
{
    Vec3 a{};
    Vec3 b{k_axis_z};
    f32 radius{0.1f};
};

struct Segment
{
    Vec3 start{};
    Vec3 end{};
};

struct RayHit
{
    f32 distance{};
    Vec3 position{};
    Vec3 normal{k_axis_z};
};

// clang-format off
[[nodiscard]] auto make_camera_ray(const Camera&, Vec2 cursor_px, Vec2 viewport_px) noexcept -> Ray;

[[nodiscard]] auto intersect_sphere(const Ray&, const Sphere&) noexcept    -> std::optional<f32>;
[[nodiscard]] auto intersect_aabb(const Ray&, const Aabb&) noexcept        -> std::optional<f32>;
[[nodiscard]] auto intersect_obb(const Ray&, const Obb&) noexcept          -> std::optional<f32>;
[[nodiscard]] auto intersect_capsule(const Ray&, const Capsule&) noexcept  -> std::optional<f32>;

[[nodiscard]] auto hit_sphere(const Ray&, const Sphere&) noexcept          -> std::optional<RayHit>;
[[nodiscard]] auto hit_aabb(const Ray&, const Aabb&) noexcept              -> std::optional<RayHit>;
[[nodiscard]] auto hit_obb(const Ray&, const Obb&) noexcept                -> std::optional<RayHit>;
[[nodiscard]] auto hit_capsule(const Ray&, const Capsule&) noexcept        -> std::optional<RayHit>;
// clang-format on

}  // namespace ds_vk
