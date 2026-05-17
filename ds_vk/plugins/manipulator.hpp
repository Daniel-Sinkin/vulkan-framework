#pragma once

#include "ds_vk/camera.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/types.hpp"

#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace ds_vk
{
enum class ManipulatorMode : u8
{
    none = 0,
    translate = 1,
    rotate = 2,
    scale = 3,
};

enum class ManipulatorAxis : u8
{
    none = 0,
    x = 1,
    y = 2,
    z = 3,
};

struct ManipulatorInput
{
    const Camera& camera;
    Vec2 mouse_px{};
    Vec2 viewport_px{1.0f};
    bool mouse_captured_by_ui{};
    bool translate_pressed{};
    bool rotate_pressed{};
    bool scale_pressed{};
    bool x_pressed{};
    bool y_pressed{};
    bool z_pressed{};
    bool confirm_pressed{};
    bool cancel_pressed{};
};

struct ManipulatorCallbacks
{
    std::function<std::optional<Transform>(ObjectId)> get_transform{};
    std::function<void(ObjectId, const Transform&)> set_transform{};
};

struct ManipulatorConfig
{
    ManipulatorInput input;
    std::span<const ObjectId> selected_ids{};
    ManipulatorCallbacks callbacks{};
    f32 translate_sensitivity{1.0f};
    f32 rotate_sensitivity{1.0f};
    f32 scale_sensitivity{1.0f};
};

class Manipulator
{
  public:
    auto update(const ManipulatorConfig&) -> void;
    auto cancel() -> void;
    auto confirm() noexcept -> void;

    // clang-format off
    [[nodiscard]] auto active() const noexcept     -> bool;
    [[nodiscard]] auto mode() const noexcept       -> ManipulatorMode;
    [[nodiscard]] auto axis() const noexcept       -> ManipulatorAxis;
    [[nodiscard]] auto active_ids() const noexcept -> std::span<const ObjectId>;
    // clang-format on

  private:
    struct ActiveTarget
    {
        ObjectId id{};
        Transform start_transform{};
    };

    auto begin(const ManipulatorConfig&, ManipulatorMode mode) -> void;
    auto apply(const ManipulatorConfig&) -> void;
    auto set_axis_from_input(const ManipulatorInput&) noexcept -> void;

    bool active_{};
    ManipulatorMode mode_{ManipulatorMode::none};
    ManipulatorAxis axis_{ManipulatorAxis::none};
    Vec2 start_mouse_px_{};
    ManipulatorCallbacks callbacks_{};
    std::vector<ActiveTarget> targets_{};
    std::vector<ObjectId> active_ids_{};
};
}  // namespace ds_vk
