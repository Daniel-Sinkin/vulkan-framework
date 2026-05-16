#pragma once

#include "ds_vk/camera.hpp"
#include "ds_vk/types.hpp"

#include <optional>
#include <span>

namespace ds_vk
{
struct PickRay
{
    Vec3 origin{};
    Vec3 direction{0.0f, 0.0f, -1.0f};
};

struct PickSphere
{
    Vec3 center{};
    f32 radius{1.0f};
};

struct PickAabb
{
    Vec3 min{};
    Vec3 max{};
};

enum class PickShapeType : u32
{
    sphere = 0,
    aabb = 1,
};

struct PickCandidate
{
    ObjectId object_id{};
    PickShapeType type{PickShapeType::sphere};
    PickSphere sphere{};
    PickAabb aabb{};
};

struct PickHit
{
    ObjectId object_id{};
    f32 distance{};
    Vec3 position{};
};

[[nodiscard]] auto make_pick_ray(const Camera& camera, Vec2 cursor_px, Vec2 viewport_px) noexcept
    -> PickRay;
[[nodiscard]] auto intersect_sphere(const PickRay& ray, const PickSphere& sphere) noexcept
    -> std::optional<f32>;
[[nodiscard]] auto intersect_aabb(const PickRay& ray, const PickAabb& aabb) noexcept
    -> std::optional<f32>;
[[nodiscard]] auto pick_nearest(const PickRay& ray, std::span<const PickCandidate> candidates)
    -> std::optional<PickHit>;
}  // namespace ds_vk
