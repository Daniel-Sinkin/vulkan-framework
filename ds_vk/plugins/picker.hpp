#pragma once

#include "ds_vk/selection.hpp"

#include <limits>
#include <optional>
#include <vector>

namespace ds_vk
{
inline constexpr auto k_pick_layer_default = u32{1u << 0u};
inline constexpr auto k_pick_layer_all = std::numeric_limits<u32>::max();
inline constexpr auto k_invalid_pick_target_id = std::numeric_limits<u32>::max();

struct PickTargetId
{
    u32 value{k_invalid_pick_target_id};

    [[nodiscard]] auto valid() const noexcept -> bool
    {
        return value != k_invalid_pick_target_id;
    }
};

enum class PickerShapeType : u32
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
    u32 layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
};

struct PickerSphereConfig
{
    ObjectId object_id{};
    u32 layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Vec3 center{};
    f32 radius{1.0f};
};

struct PickerAabbConfig
{
    ObjectId object_id{};
    u32 layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Vec3 min{};
    Vec3 max{};
};

struct PickerObbConfig
{
    ObjectId object_id{};
    u32 layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Vec3 center{};
    Vec3 half_extent{0.5f};
    Quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
};

struct PickerCapsuleConfig
{
    ObjectId object_id{};
    u32 layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Vec3 a{};
    Vec3 b{0.0f, 0.0f, 1.0f};
    f32 radius{0.1f};
};

struct PickerScreenSegmentConfig
{
    ObjectId object_id{};
    u32 layer{k_pick_layer_default};
    u32 sub_index{};
    u64 user_bits{};
    bool enabled{true};
    Vec3 start{};
    Vec3 end{};
    f32 radius_px{6.0f};
};

struct PickerRaycastConfig
{
    PickRay ray{};
    u32 layer_mask{k_pick_layer_all};
};

struct PickerClickConfig
{
    const Camera& camera;
    Vec2 mouse_px{};
    Vec2 viewport_px{};
    u32 layer_mask{k_pick_layer_all};
};

struct PickerHit
{
    ObjectId object_id{};
    PickTargetId target_id{};
    PickerShapeType shape{PickerShapeType::sphere};
    f32 distance{};
    Vec3 world_position{};
    Vec3 world_normal{0.0f, 0.0f, 1.0f};
    u32 layer{};
    u32 sub_index{};
    u64 user_bits{};
};

class Picker
{
  public:
    auto clear() -> void;

    auto add_sphere(const PickerSphereConfig& config) -> PickTargetId;
    auto add_aabb(const PickerAabbConfig& config) -> PickTargetId;
    auto add_obb(const PickerObbConfig& config) -> PickTargetId;
    auto add_capsule(const PickerCapsuleConfig& config) -> PickTargetId;
    auto add_screen_segment(const PickerScreenSegmentConfig& config) -> PickTargetId;

    [[nodiscard]] auto raycast(const PickerRaycastConfig& config) const -> std::optional<PickerHit>;
    [[nodiscard]] auto raycast(const PickRay& ray) const -> std::optional<PickerHit>;
    [[nodiscard]] auto click(const PickerClickConfig& config) const -> std::optional<PickerHit>;

    [[nodiscard]] auto target_count() const noexcept -> usize;

  private:
    struct Target
    {
        PickTargetId target_id{};
        PickerShapeType shape{PickerShapeType::sphere};
        PickerTargetCommon common{};
        PickerSphereConfig sphere{};
        PickerAabbConfig aabb{};
        PickerObbConfig obb{};
        PickerCapsuleConfig capsule{};
        PickerScreenSegmentConfig screen_segment{};
    };

    auto add_target(Target target) -> PickTargetId;

    std::vector<Target> targets_{};
    u32 next_target_id_{};
};
}  // namespace ds_vk
