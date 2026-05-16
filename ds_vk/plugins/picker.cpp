#include "ds_vk/plugins/picker.hpp"

#include "ds_vk/camera.hpp"
#include "ds_vk/math.hpp"

#include <algorithm>
#include <cmath>

namespace ds_vk
{
namespace
{
struct ScreenPoint
{
    Vec2 position{};
    f32 depth{};
};

[[nodiscard]] auto layer_matches(Layer layer, LayerMask layer_mask) noexcept -> bool
{
    return (layer & layer_mask) != LayerMask{};
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

[[nodiscard]] auto project_to_screen(const Camera& camera, Vec3 position, Vec2 viewport_px) noexcept
    -> std::optional<ScreenPoint>
{
    const auto viewport = glm::max(viewport_px, Vec2{1.0f});
    const auto clip = camera.view_projection_matrix(viewport.x / viewport.y) * Vec4{position, 1.0f};
    if (clip.w <= 0.0f)
    {
        return std::nullopt;
    }
    const Vec3 ndc = Vec3{clip} / clip.w;
    if (ndc.z < 0.0f or ndc.z > 1.0f)
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
    const Ray& ray,
    const Segment& segment,
    f32 radius_px,
    const Camera& camera,
    Vec2 mouse_px,
    Vec2 viewport_px
) noexcept -> std::optional<RayHit>
{
    const auto start = project_to_screen(camera, segment.start, viewport_px);
    const auto end = project_to_screen(camera, segment.end, viewport_px);
    if (!start.has_value() or !end.has_value())
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
    if (glm::length(mouse_px - closest) > std::max(0.0f, radius_px))
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
            .sphere = config.sphere,
        }
    );
}

auto Picker::add_aabb(const PickerAabbConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::aabb,
            .common = common_from(config),
            .aabb = config.aabb,
        }
    );
}

auto Picker::add_obb(const PickerObbConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::obb,
            .common = common_from(config),
            .obb = config.obb,
        }
    );
}

auto Picker::add_capsule(const PickerCapsuleConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::capsule,
            .common = common_from(config),
            .capsule = config.capsule,
        }
    );
}

auto Picker::add_screen_segment(const PickerScreenSegmentConfig& config) -> PickTargetId
{
    return add_target(
        Target{
            .shape = PickerShapeType::screen_segment,
            .common = common_from(config),
            .screen_segment = config.segment,
            .screen_segment_radius_px = config.radius_px,
        }
    );
}

auto Picker::raycast(const PickerRaycastConfig& config) const -> std::optional<PickerHit>
{
    std::optional<PickerHit> best{};
    for (const auto& target : targets_)
    {
        const auto target_pickable = target.common.enabled and target.common.object_id.valid()
                                     and layer_matches(target.common.layer, config.layer_mask)
                                     and target.shape != PickerShapeType::screen_segment;
        if (!target_pickable)
        {
            continue;
        }

        std::optional<RayHit> hit{};
        switch (target.shape)
        {
            case PickerShapeType::sphere:
                hit = hit_sphere(config.ray, target.sphere);
                break;
            case PickerShapeType::aabb:
                hit = hit_aabb(config.ray, target.aabb);
                break;
            case PickerShapeType::obb:
                hit = hit_obb(config.ray, target.obb);
                break;
            case PickerShapeType::capsule:
                hit = hit_capsule(config.ray, target.capsule);
                break;
            case PickerShapeType::screen_segment:
                break;
        }

        if (hit.has_value() and (!best.has_value() or hit->distance < best->distance))
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
    return best;
}

auto Picker::raycast(const Ray& ray) const -> std::optional<PickerHit>
{
    return raycast(PickerRaycastConfig{.ray = ray});
}

auto Picker::click(const PickerClickConfig& config) const -> std::optional<PickerHit>
{
    const auto ray = make_camera_ray(config.camera, config.mouse_px, config.viewport_px);
    auto best = raycast(PickerRaycastConfig{.ray = ray, .layer_mask = config.layer_mask});
    for (const auto& target : targets_)
    {
        const auto target_pickable = target.common.enabled and target.common.object_id.valid()
                                     and layer_matches(target.common.layer, config.layer_mask)
                                     and target.shape == PickerShapeType::screen_segment;
        if (!target_pickable)
        {
            continue;
        }

        const auto hit = screen_segment_hit(
            ray,
            target.screen_segment,
            target.screen_segment_radius_px,
            config.camera,
            config.mouse_px,
            config.viewport_px
        );
        if (hit.has_value() and (!best.has_value() or hit->distance < best->distance))
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
    return best;
}

auto Picker::target_count() const noexcept -> usize
{
    return targets_.size();
}
}  // namespace ds_vk
