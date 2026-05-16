#include "ds_vk/selection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ds_vk
{
namespace
{
auto safe_direction(const Vec3 value) noexcept -> Vec3
{
    const auto length_squared = glm::dot(value, value);
    if (length_squared <= 1.0e-12f)
    {
        return Vec3{0.0f, 0.0f, -1.0f};
    }
    return value * glm::inversesqrt(length_squared);
}
}  // namespace

auto make_pick_ray(const Camera& camera, const Vec2 cursor_px, const Vec2 viewport_px) noexcept
    -> PickRay
{
    const auto viewport = glm::max(viewport_px, Vec2{1.0f});
    const auto ndc_x = 2.0f * cursor_px.x / viewport.x - 1.0f;
    const auto ndc_y = 2.0f * cursor_px.y / viewport.y - 1.0f;
    const auto aspect = viewport.x / viewport.y;
    const auto inverse_view_projection = glm::inverse(camera.view_projection_matrix(aspect));
    const auto near_clip = Vec4{ndc_x, ndc_y, 0.0f, 1.0f};
    const auto far_clip = Vec4{ndc_x, ndc_y, 1.0f, 1.0f};
    auto near_world = inverse_view_projection * near_clip;
    auto far_world = inverse_view_projection * far_clip;
    near_world /= near_world.w;
    far_world /= far_world.w;
    const auto origin = Vec3{near_world};
    return PickRay{.origin = origin, .direction = safe_direction(Vec3{far_world - near_world})};
}

auto intersect_sphere(const PickRay& ray, const PickSphere& sphere) noexcept -> std::optional<f32>
{
    const auto oc = ray.origin - sphere.center;
    const auto radius = std::max(0.0f, sphere.radius);
    const auto a = glm::dot(ray.direction, ray.direction);
    const auto half_b = glm::dot(oc, ray.direction);
    const auto c = glm::dot(oc, oc) - radius * radius;
    const auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f)
    {
        return std::nullopt;
    }

    const auto root = std::sqrt(discriminant);
    const auto near_t = (-half_b - root) / a;
    if (near_t >= 0.0f)
    {
        return near_t;
    }
    const auto far_t = (-half_b + root) / a;
    if (far_t >= 0.0f)
    {
        return far_t;
    }
    return std::nullopt;
}

auto intersect_aabb(const PickRay& ray, const PickAabb& aabb) noexcept -> std::optional<f32>
{
    const auto box_min = glm::min(aabb.min, aabb.max);
    const auto box_max = glm::max(aabb.min, aabb.max);
    auto t_min = 0.0f;
    auto t_max = std::numeric_limits<f32>::max();
    auto hit = true;
    for (auto axis = 0; axis < 3 && hit; ++axis)
    {
        const auto origin = ray.origin[axis];
        const auto direction = ray.direction[axis];
        if (std::abs(direction) <= 1.0e-8f)
        {
            if (origin < box_min[axis] || origin > box_max[axis])
            {
                hit = false;
            }
        }
        else
        {
            const auto inv_direction = 1.0f / direction;
            auto t0 = (box_min[axis] - origin) * inv_direction;
            auto t1 = (box_max[axis] - origin) * inv_direction;
            if (t0 > t1)
            {
                std::swap(t0, t1);
            }
            t_min = std::max(t_min, t0);
            t_max = std::min(t_max, t1);
            if (t_max < t_min)
            {
                hit = false;
            }
        }
    }
    if (!hit)
    {
        return std::nullopt;
    }
    return t_min;
}

auto pick_nearest(const PickRay& ray, std::span<const PickCandidate> candidates)
    -> std::optional<PickHit>
{
    auto best = std::optional<PickHit>{};
    for (const auto& candidate : candidates)
    {
        if (candidate.object_id.valid())
        {
            const auto distance = candidate.type == PickShapeType::sphere
                                      ? intersect_sphere(ray, candidate.sphere)
                                      : intersect_aabb(ray, candidate.aabb);
            if (distance.has_value() && (!best.has_value() || *distance < best->distance))
            {
                best = PickHit{
                    .object_id = candidate.object_id,
                    .distance = *distance,
                    .position = ray.origin + *distance * ray.direction,
                };
            }
        }
    }
    return best;
}
}  // namespace ds_vk
