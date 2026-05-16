#include "ds_vk/plugins/picker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ds_vk
{
namespace
{
struct RayHit
{
    f32 distance{};
    Vec3 position{};
    Vec3 normal{0.0f, 0.0f, 1.0f};
};

struct ScreenPoint
{
    Vec2 position{};
    f32 depth{};
};

[[nodiscard]] auto normalize_or(const Vec3 value, const Vec3 fallback) noexcept -> Vec3
{
    const auto length_squared = glm::dot(value, value);
    if (length_squared <= 1.0e-12f)
    {
        return glm::normalize(fallback);
    }
    return value * glm::inversesqrt(length_squared);
}

[[nodiscard]] auto layer_matches(const u32 layer, const u32 layer_mask) noexcept -> bool
{
    return (layer & layer_mask) != 0u;
}

template <typename Config>
[[nodiscard]] auto common_from(const Config& config) noexcept -> PickerTargetCommon
{
    return PickerTargetCommon{
        .object_id = config.object_id,
        .layer = config.layer,
        .sub_index = config.sub_index,
        .user_bits = config.user_bits,
        .enabled = config.enabled,
    };
}

[[nodiscard]] auto aabb_normal_at(const Vec3 position, const Vec3 min, const Vec3 max) noexcept
    -> Vec3
{
    const auto distances = std::array{
        std::pair{std::abs(position.x - min.x), Vec3{-1.0f, 0.0f, 0.0f}},
        std::pair{std::abs(position.x - max.x), Vec3{1.0f, 0.0f, 0.0f}},
        std::pair{std::abs(position.y - min.y), Vec3{0.0f, -1.0f, 0.0f}},
        std::pair{std::abs(position.y - max.y), Vec3{0.0f, 1.0f, 0.0f}},
        std::pair{std::abs(position.z - min.z), Vec3{0.0f, 0.0f, -1.0f}},
        std::pair{std::abs(position.z - max.z), Vec3{0.0f, 0.0f, 1.0f}},
    };
    const auto best = std::ranges::min_element(
        distances,
        [](const auto& lhs, const auto& rhs) noexcept -> bool { return lhs.first < rhs.first; }
    );
    return best == distances.end() ? Vec3{0.0f, 0.0f, 1.0f} : best->second;
}

[[nodiscard]] auto sphere_hit(const PickRay& ray, const Vec3 center, const f32 radius) noexcept
    -> std::optional<RayHit>
{
    const auto distance = intersect_sphere(ray, PickSphere{.center = center, .radius = radius});
    if (!distance.has_value())
    {
        return std::nullopt;
    }
    const auto position = ray.origin + *distance * ray.direction;
    return RayHit{
        .distance = *distance,
        .position = position,
        .normal = normalize_or(position - center, ray.direction),
    };
}

[[nodiscard]] auto aabb_hit(const PickRay& ray, const Vec3 min, const Vec3 max) noexcept
    -> std::optional<RayHit>
{
    const auto distance = intersect_aabb(ray, PickAabb{.min = min, .max = max});
    if (!distance.has_value())
    {
        return std::nullopt;
    }
    const auto box_min = glm::min(min, max);
    const auto box_max = glm::max(min, max);
    const auto position = ray.origin + *distance * ray.direction;
    return RayHit{
        .distance = *distance,
        .position = position,
        .normal = aabb_normal_at(position, box_min, box_max),
    };
}

[[nodiscard]] auto obb_hit(const PickRay& ray, const PickerObbConfig& obb) noexcept
    -> std::optional<RayHit>
{
    const auto inverse_rotation = glm::inverse(obb.rotation);
    const auto local_ray = PickRay{
        .origin = inverse_rotation * (ray.origin - obb.center),
        .direction = normalize_or(inverse_rotation * ray.direction, ray.direction),
    };
    const auto local_min = -glm::abs(obb.half_extent);
    const auto local_max = glm::abs(obb.half_extent);
    const auto local_hit = aabb_hit(local_ray, local_min, local_max);
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

[[nodiscard]] auto capsule_hit(const PickRay& ray, const PickerCapsuleConfig& capsule) noexcept
    -> std::optional<RayHit>
{
    const auto radius = std::max(0.0f, capsule.radius);
    const auto segment = capsule.b - capsule.a;
    const auto segment_length_squared = glm::dot(segment, segment);
    if (segment_length_squared <= 1.0e-12f)
    {
        return sphere_hit(ray, capsule.a, radius);
    }

    const auto ray_direction = normalize_or(ray.direction, {1.0f, 0.0f, 0.0f});
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
    ray_t = std::max(0.0f, (ray_segment_dot * segment_t - ray_to_a_dot_ray));

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

[[nodiscard]] auto
project_to_screen(const Camera& camera, const Vec3 position, const Vec2 viewport_px) noexcept
    -> std::optional<ScreenPoint>
{
    const auto viewport = glm::max(viewport_px, Vec2{1.0f});
    const auto clip = camera.view_projection_matrix(viewport.x / viewport.y) * Vec4{position, 1.0f};
    if (clip.w <= 0.0f)
    {
        return std::nullopt;
    }
    const auto ndc = Vec3{clip} / clip.w;
    if (ndc.z < 0.0f || ndc.z > 1.0f)
    {
        return std::nullopt;
    }
    return ScreenPoint{
        .position =
            Vec2{
                (ndc.x * 0.5f + 0.5f) * viewport.x,
                (ndc.y * 0.5f + 0.5f) * viewport.y,
            },
        .depth = ndc.z,
    };
}

[[nodiscard]] auto screen_segment_hit(
    const PickRay& ray,
    const PickerScreenSegmentConfig& segment,
    const Camera& camera,
    const Vec2 mouse_px,
    const Vec2 viewport_px
) noexcept -> std::optional<RayHit>
{
    const auto start = project_to_screen(camera, segment.start, viewport_px);
    const auto end = project_to_screen(camera, segment.end, viewport_px);
    if (!start.has_value() || !end.has_value())
    {
        return std::nullopt;
    }

    const auto screen_segment = end->position - start->position;
    const auto length_squared = glm::dot(screen_segment, screen_segment);
    if (length_squared <= 1.0e-8f)
    {
        return std::nullopt;
    }
    const auto t = std::clamp(
        glm::dot(mouse_px - start->position, screen_segment) / length_squared, 0.0f, 1.0f
    );
    const auto closest = start->position + t * screen_segment;
    if (glm::length(mouse_px - closest) > std::max(0.0f, segment.radius_px))
    {
        return std::nullopt;
    }

    const auto world_position = segment.start + t * (segment.end - segment.start);
    return RayHit{
        .distance = std::max(0.0f, glm::dot(world_position - ray.origin, ray.direction)),
        .position = world_position,
        .normal = normalize_or(glm::cross(segment.end - segment.start, ray.direction), k_axis_z),
    };
}
}  // namespace

auto Picker::clear() -> void
{
    targets_.clear();
    next_target_id_ = 0u;
}

auto Picker::add_target(Target target) -> PickTargetId
{
    target.target_id = PickTargetId{.value = next_target_id_++};
    const auto id = target.target_id;
    targets_.push_back(target);
    return id;
}

auto Picker::add_sphere(const PickerSphereConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::sphere,
            .common = common_from(config),
            .sphere = config,
        }
    );
}

auto Picker::add_aabb(const PickerAabbConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::aabb,
            .common = common_from(config),
            .aabb = config,
        }
    );
}

auto Picker::add_obb(const PickerObbConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::obb,
            .common = common_from(config),
            .obb = config,
        }
    );
}

auto Picker::add_capsule(const PickerCapsuleConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::capsule,
            .common = common_from(config),
            .capsule = config,
        }
    );
}

auto Picker::add_screen_segment(const PickerScreenSegmentConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::screen_segment,
            .common = common_from(config),
            .screen_segment = config,
        }
    );
}

auto Picker::raycast(const PickerRaycastConfig& config) const -> std::optional<PickerHit>
{
    auto best = std::optional<PickerHit>{};
    for (const auto& target : targets_)
    {
        const auto target_pickable = target.common.enabled && target.common.object_id.valid()
                                     && layer_matches(target.common.layer, config.layer_mask)
                                     && target.shape != PickerShapeType::screen_segment;
        if (target_pickable)
        {
            auto hit = std::optional<RayHit>{};
            if (target.shape == PickerShapeType::sphere)
            {
                hit = sphere_hit(config.ray, target.sphere.center, target.sphere.radius);
            }
            else if (target.shape == PickerShapeType::aabb)
            {
                hit = aabb_hit(config.ray, target.aabb.min, target.aabb.max);
            }
            else if (target.shape == PickerShapeType::obb)
            {
                hit = obb_hit(config.ray, target.obb);
            }
            else if (target.shape == PickerShapeType::capsule)
            {
                hit = capsule_hit(config.ray, target.capsule);
            }

            if (hit.has_value() && (!best.has_value() || hit->distance < best->distance))
            {
                best = PickerHit{
                    .object_id = target.common.object_id,
                    .target_id = target.target_id,
                    .shape = target.shape,
                    .distance = hit->distance,
                    .world_position = hit->position,
                    .world_normal = hit->normal,
                    .layer = target.common.layer,
                    .sub_index = target.common.sub_index,
                    .user_bits = target.common.user_bits,
                };
            }
        }
    }
    return best;
}

auto Picker::raycast(const PickRay& ray) const -> std::optional<PickerHit>
{
    return raycast(PickerRaycastConfig{.ray = ray});
}

auto Picker::click(const PickerClickConfig& config) const -> std::optional<PickerHit>
{
    const auto ray = make_pick_ray(config.camera, config.mouse_px, config.viewport_px);
    auto best = raycast(PickerRaycastConfig{.ray = ray, .layer_mask = config.layer_mask});
    for (const auto& target : targets_)
    {
        const auto target_pickable = target.common.enabled && target.common.object_id.valid()
                                     && layer_matches(target.common.layer, config.layer_mask)
                                     && target.shape == PickerShapeType::screen_segment;
        if (target_pickable)
        {
            const auto hit = screen_segment_hit(
                ray, target.screen_segment, config.camera, config.mouse_px, config.viewport_px
            );
            if (hit.has_value() && (!best.has_value() || hit->distance < best->distance))
            {
                best = PickerHit{
                    .object_id = target.common.object_id,
                    .target_id = target.target_id,
                    .shape = target.shape,
                    .distance = hit->distance,
                    .world_position = hit->position,
                    .world_normal = hit->normal,
                    .layer = target.common.layer,
                    .sub_index = target.common.sub_index,
                    .user_bits = target.common.user_bits,
                };
            }
        }
    }
    return best;
}

auto Picker::target_count() const noexcept -> usize
{
    return targets_.size();
}
}  // namespace ds_vk
