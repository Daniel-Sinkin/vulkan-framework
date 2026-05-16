#include "ds_vk/geometry.hpp"

#include "ds_vk/camera.hpp"
#include "ds_vk/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ds_vk
{
namespace
{
struct FaceDistance
{
    f32 distance{};
    Vec3 normal{};
};

[[nodiscard]] auto aabb_normal_at(Vec3 position, Vec3 box_min, Vec3 box_max) noexcept -> Vec3
{
    const std::array face_distances{
        FaceDistance{.distance = std::abs(position.x - box_min.x), .normal = -k_axis_x},
        FaceDistance{.distance = std::abs(position.x - box_max.x), .normal = k_axis_x},
        FaceDistance{.distance = std::abs(position.y - box_min.y), .normal = -k_axis_y},
        FaceDistance{.distance = std::abs(position.y - box_max.y), .normal = k_axis_y},
        FaceDistance{.distance = std::abs(position.z - box_min.z), .normal = -k_axis_z},
        FaceDistance{.distance = std::abs(position.z - box_max.z), .normal = k_axis_z},
    };
    const auto best = std::ranges::min_element(
        face_distances,
        [](const FaceDistance& lhs, const FaceDistance& rhs) noexcept -> bool
        { return lhs.distance < rhs.distance; }
    );
    return best == face_distances.end() ? k_axis_z : best->normal;
}
}  // namespace

auto make_camera_ray(const Camera& camera, Vec2 cursor_px, Vec2 viewport_px) noexcept -> Ray
{
    const auto viewport = glm::max(viewport_px, Vec2{1.0f});
    const auto ndc_x = 2.0f * cursor_px.x / viewport.x - 1.0f;
    const auto ndc_y = 2.0f * cursor_px.y / viewport.y - 1.0f;
    const auto aspect = viewport.x / viewport.y;
    const auto inverse_view_projection = glm::inverse(camera.view_projection_matrix(aspect));
    const Vec4 near_clip{ndc_x, ndc_y, 0.0f, 1.0f};
    const Vec4 far_clip{ndc_x, ndc_y, 1.0f, 1.0f};
    auto near_world = inverse_view_projection * near_clip;
    auto far_world = inverse_view_projection * far_clip;
    near_world /= near_world.w;
    far_world /= far_world.w;
    const Vec3 origin{near_world};
    return Ray{
        .origin = origin, .direction = normalize_or(Vec3{far_world - near_world}, -k_axis_z)
    };
}

auto intersect_sphere(const Ray& ray, const Sphere& sphere) noexcept -> std::optional<f32>
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

auto intersect_aabb(const Ray& ray, const Aabb& aabb) noexcept -> std::optional<f32>
{
    const auto box_min = glm::min(aabb.min, aabb.max);
    const auto box_max = glm::max(aabb.min, aabb.max);
    auto t_min = 0.0f;
    auto t_max = std::numeric_limits<f32>::max();
    auto hit = true;
    for (Vec3::length_type axis{0}; axis < Vec3::length_type{3} and hit; ++axis)
    {
        const auto origin = ray.origin[axis];
        const auto direction = ray.direction[axis];
        if (std::abs(direction) <= 1.0e-8f)
        {
            if (origin < box_min[axis] or origin > box_max[axis])
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

auto intersect_obb(const Ray& ray, const Obb& obb) noexcept -> std::optional<f32>
{
    const auto hit = hit_obb(ray, obb);
    if (!hit.has_value())
    {
        return std::nullopt;
    }
    return hit->distance;
}

auto intersect_capsule(const Ray& ray, const Capsule& capsule) noexcept -> std::optional<f32>
{
    const auto hit = hit_capsule(ray, capsule);
    if (!hit.has_value())
    {
        return std::nullopt;
    }
    return hit->distance;
}

auto hit_sphere(const Ray& ray, const Sphere& sphere) noexcept -> std::optional<RayHit>
{
    const auto distance = intersect_sphere(ray, sphere);
    if (!distance.has_value())
    {
        return std::nullopt;
    }
    const auto position = ray.origin + *distance * ray.direction;
    return RayHit{
        .distance = *distance,
        .position = position,
        .normal = normalize_or(position - sphere.center, ray.direction),
    };
}

auto hit_aabb(const Ray& ray, const Aabb& aabb) noexcept -> std::optional<RayHit>
{
    const auto distance = intersect_aabb(ray, aabb);
    if (!distance.has_value())
    {
        return std::nullopt;
    }
    const auto box_min = glm::min(aabb.min, aabb.max);
    const auto box_max = glm::max(aabb.min, aabb.max);
    const auto position = ray.origin + *distance * ray.direction;
    return RayHit{
        .distance = *distance,
        .position = position,
        .normal = aabb_normal_at(position, box_min, box_max),
    };
}

auto hit_obb(const Ray& ray, const Obb& obb) noexcept -> std::optional<RayHit>
{
    const auto inverse_rotation = glm::inverse(obb.rotation);
    const Ray local_ray{
        .origin = inverse_rotation * (ray.origin - obb.center),
        .direction = normalize_or(inverse_rotation * ray.direction, ray.direction),
    };
    const auto local_hit = hit_aabb(
        local_ray, Aabb{.min = -glm::abs(obb.half_extent), .max = glm::abs(obb.half_extent)}
    );
    if (!local_hit.has_value())
    {
        return std::nullopt;
    }
    const auto world_position = obb.center + obb.rotation * local_hit->position;
    return RayHit{
        .distance = glm::dot(world_position - ray.origin, ray.direction),
        .position = world_position,
        .normal = normalize_or(obb.rotation * local_hit->normal, ray.direction),
    };
}

auto hit_capsule(const Ray& ray, const Capsule& capsule) noexcept -> std::optional<RayHit>
{
    const auto radius = std::max(0.0f, capsule.radius);
    const auto segment = capsule.b - capsule.a;
    const auto segment_length_squared = glm::dot(segment, segment);
    if (segment_length_squared <= 1.0e-12f)
    {
        return hit_sphere(ray, Sphere{.center = capsule.a, .radius = radius});
    }

    const auto ray_direction = normalize_or(ray.direction, k_axis_x);
    const auto ray_to_a = ray.origin - capsule.a;
    const auto ray_segment_dot = glm::dot(ray_direction, segment);
    const auto ray_to_a_dot_ray = glm::dot(ray_to_a, ray_direction);
    const auto ray_to_a_dot_segment = glm::dot(ray_to_a, segment);
    const auto denom = segment_length_squared - ray_segment_dot * ray_segment_dot;

    auto ray_t = 0.0f;
    if (std::abs(denom) > 1.0e-8f)
    {
        ray_t = (ray_segment_dot * ray_to_a_dot_segment - segment_length_squared * ray_to_a_dot_ray)
                / denom;
    }
    ray_t = std::max(0.0f, ray_t);

    auto segment_t = (ray_segment_dot * ray_t + ray_to_a_dot_segment) / segment_length_squared;
    segment_t = std::clamp(segment_t, 0.0f, 1.0f);
    ray_t = std::max(0.0f, ray_segment_dot * segment_t - ray_to_a_dot_ray);

    const auto axis_position = capsule.a + segment_t * segment;
    const auto closest_ray_position = ray.origin + ray_t * ray_direction;
    const auto offset = closest_ray_position - axis_position;
    const auto distance_squared = glm::dot(offset, offset);
    if (distance_squared > radius * radius)
    {
        return std::nullopt;
    }

    const auto entry_offset = std::sqrt(std::max(0.0f, radius * radius - distance_squared));
    const auto entry_t = std::max(0.0f, ray_t - entry_offset);
    const auto hit_position = ray.origin + entry_t * ray_direction;
    const auto hit_segment_t = std::clamp(
        glm::dot(hit_position - capsule.a, segment) / segment_length_squared, 0.0f, 1.0f
    );
    const auto hit_axis_position = capsule.a + hit_segment_t * segment;
    return RayHit{
        .distance = entry_t,
        .position = hit_position,
        .normal = normalize_or(hit_position - hit_axis_position, -ray_direction),
    };
}

}  // namespace ds_vk
