#pragma once

#include "ds_vk/geometry.hpp"
#include "ds_vk/types.hpp"

#include <limits>
#include <optional>
#include <vector>

namespace ds_vk
{
using Layer = u32;
using LayerMask = u32;

inline constexpr Layer k_pick_layer_default{1u << 0u};
inline constexpr auto k_pick_layer_all = std::numeric_limits<LayerMask>::max();

struct PickTargetId
{
    u32 value{k_invalid_id};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return value != k_invalid_id;
    }
};

enum class PickerShapeType : u8
{
    sphere = 0,
    aabb = 1,
    obb = 2,
    capsule = 3,
    screen_segment = 4,
};

struct PickerTargetCommon
{
    ObjectId object_id{};
    Layer layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
};

struct PickerSphereConfig
{
    ObjectId object_id{};
    Layer layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Sphere sphere{};
};

struct PickerAabbConfig
{
    ObjectId object_id{};
    Layer layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Aabb aabb{};
};

struct PickerObbConfig
{
    ObjectId object_id{};
    Layer layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Obb obb{};
};

struct PickerCapsuleConfig
{
    ObjectId object_id{};
    Layer layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Capsule capsule{};
};

struct PickerScreenSegmentConfig
{
    ObjectId object_id{};
    Layer layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Segment segment{};
    f32 radius_px{6.0f};
};

struct PickerRaycastConfig
{
    Ray ray{};
    LayerMask layer_mask{k_pick_layer_all};
};

struct PickerClickConfig
{
    const Camera& camera;
    Vec2 mouse_px{};
    Vec2 viewport_px{};
    LayerMask layer_mask{k_pick_layer_all};
};

struct PickerHit
{
    ObjectId object_id{};
    PickTargetId target_id{};
    PickerShapeType shape{PickerShapeType::sphere};
    f32 distance{};
    Vec3 world_position{};
    Vec3 world_normal{k_axis_z};
    Layer layer{};
    u32 sub_index{};
    u64 user_bits{};
};

class Picker
{
  public:
    // clang-format off
    auto clear()                                                                     -> void;

    [[nodiscard]] auto add_sphere(const PickerSphereConfig&)                         -> PickTargetId;
    [[nodiscard]] auto add_aabb(const PickerAabbConfig&)                             -> PickTargetId;
    [[nodiscard]] auto add_obb(const PickerObbConfig&)                               -> PickTargetId;
    [[nodiscard]] auto add_capsule(const PickerCapsuleConfig&)                       -> PickTargetId;
    [[nodiscard]] auto add_screen_segment(const PickerScreenSegmentConfig&)          -> PickTargetId;

    [[nodiscard]] auto raycast(const PickerRaycastConfig&) const                     -> std::optional<PickerHit>;
    [[nodiscard]] auto raycast(const Ray&) const                                     -> std::optional<PickerHit>;
    [[nodiscard]] auto click(const PickerClickConfig&) const                         -> std::optional<PickerHit>;

    [[nodiscard]] auto target_count() const noexcept                                 -> usize;
    // clang-format on

  private:
    struct Target
    {
        PickTargetId target_id{};
        PickerShapeType shape{PickerShapeType::sphere};
        PickerTargetCommon common{};
        Sphere sphere{};
        Aabb aabb{};
        Obb obb{};
        Capsule capsule{};
        Segment screen_segment{};
        f32 screen_segment_radius_px{};
    };

    [[nodiscard]] auto add_target(Target target) -> PickTargetId;

    std::vector<Target> targets_{};
    u32 next_target_id_{};
};
}  // namespace ds_vk
