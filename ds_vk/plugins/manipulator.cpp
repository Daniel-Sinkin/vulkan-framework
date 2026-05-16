#include "ds_vk/plugins/manipulator.hpp"

#include "ds_vk/math.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

namespace ds_vk
{
namespace
{
[[nodiscard]] auto axis_vector(ManipulatorAxis axis, const Camera& camera) noexcept -> Vec3
{
    switch (axis)
    {
        case ManipulatorAxis::none:
            return normalize_or(camera.up(), k_axis_z);
        case ManipulatorAxis::x:
            return k_axis_x;
        case ManipulatorAxis::y:
            return k_axis_y;
        case ManipulatorAxis::z:
            return k_axis_z;
    }
    return k_axis_z;
}

[[nodiscard]] auto constrained_delta(Vec3 delta, ManipulatorAxis axis) noexcept -> Vec3
{
    switch (axis)
    {
        case ManipulatorAxis::none:
            return delta;
        case ManipulatorAxis::x:
            return Vec3{delta.x, 0.0f, 0.0f};
        case ManipulatorAxis::y:
            return Vec3{0.0f, delta.y, 0.0f};
        case ManipulatorAxis::z:
            return Vec3{0.0f, 0.0f, delta.z};
    }
    return delta;
}

[[nodiscard]] auto
scaled_transform(const Transform& transform, ManipulatorAxis axis, f32 scale_factor) noexcept
    -> Transform
{
    auto result = transform;
    const Vec3 min_scale{0.05f};
    switch (axis)
    {
        case ManipulatorAxis::none:
            result.scale *= scale_factor;
            break;
        case ManipulatorAxis::x:
            result.scale.x *= scale_factor;
            break;
        case ManipulatorAxis::y:
            result.scale.y *= scale_factor;
            break;
        case ManipulatorAxis::z:
            result.scale.z *= scale_factor;
            break;
    }
    result.scale = glm::max(glm::abs(result.scale), min_scale);
    return result;
}
}  // namespace

auto Manipulator::update(const ManipulatorConfig& cfg) -> void
{
    if (!active_)
    {
        if (cfg.input.mouse_captured_by_ui)
        {
            return;
        }
        if (cfg.input.translate_pressed)
        {
            begin(cfg, ManipulatorMode::translate);
        }
        else if (cfg.input.rotate_pressed)
        {
            begin(cfg, ManipulatorMode::rotate);
        }
        else if (cfg.input.scale_pressed)
        {
            begin(cfg, ManipulatorMode::scale);
        }
        return;
    }

    if (cfg.input.cancel_pressed)
    {
        cancel();
        return;
    }

    callbacks_ = cfg.callbacks;
    set_axis_from_input(cfg.input);
    apply(cfg);
    if (cfg.input.confirm_pressed)
    {
        confirm();
    }
}

auto Manipulator::begin(const ManipulatorConfig& cfg, ManipulatorMode mode) -> void
{
    if (!cfg.callbacks.get_transform or !cfg.callbacks.set_transform or cfg.selected_ids.empty())
    {
        return;
    }

    callbacks_ = cfg.callbacks;
    targets_.clear();
    active_ids_.clear();
    targets_.reserve(cfg.selected_ids.size());
    active_ids_.reserve(cfg.selected_ids.size());

    for (const auto id : cfg.selected_ids)
    {
        if (!id.valid())
        {
            continue;
        }
        const auto transform = callbacks_.get_transform(id);
        if (!transform.has_value())
        {
            continue;
        }
        targets_.push_back(ActiveTarget{.id = id, .start_transform = *transform});
        active_ids_.push_back(id);
    }
    if (targets_.empty())
    {
        return;
    }

    active_ = true;
    mode_ = mode;
    axis_ = ManipulatorAxis::none;
    start_mouse_px_ = cfg.input.mouse_px;
    apply(cfg);
}

auto Manipulator::apply(const ManipulatorConfig& cfg) -> void
{
    if (!callbacks_.set_transform)
    {
        return;
    }

    const auto mouse_delta = cfg.input.mouse_px - start_mouse_px_;
    const auto viewport_y = std::max(1.0f, cfg.input.viewport_px.y);
    const auto units_per_px =
        cfg.input.camera.units_per_pixel_y(viewport_y) * std::max(0.0f, cfg.translate_sensitivity);
    const auto world_delta = constrained_delta(
        mouse_delta.x * units_per_px * cfg.input.camera.right()
            - mouse_delta.y * units_per_px * cfg.input.camera.up(),
        axis_
    );
    const auto scale_factor =
        std::exp(mouse_delta.x * 0.006f * std::max(0.0f, cfg.scale_sensitivity));
    const auto angle = mouse_delta.x * 0.010f * cfg.rotate_sensitivity;
    const auto rotation_axis = axis_vector(axis_, cfg.input.camera);
    const auto rotation_delta = glm::angleAxis(angle, rotation_axis);

    for (const auto& target : targets_)
    {
        auto transform = target.start_transform;
        switch (mode_)
        {
            case ManipulatorMode::none:
                break;
            case ManipulatorMode::translate:
                transform.translation += world_delta;
                break;
            case ManipulatorMode::rotate:
                transform.rotation = glm::normalize(rotation_delta * transform.rotation);
                break;
            case ManipulatorMode::scale:
                transform = scaled_transform(transform, axis_, scale_factor);
                break;
        }
        callbacks_.set_transform(target.id, transform);
    }
}

auto Manipulator::cancel() -> void
{
    if (!active_)
    {
        return;
    }
    if (callbacks_.set_transform)
    {
        for (const auto& target : targets_)
        {
            callbacks_.set_transform(target.id, target.start_transform);
        }
    }
    active_ = false;
    mode_ = ManipulatorMode::none;
    axis_ = ManipulatorAxis::none;
    targets_.clear();
    active_ids_.clear();
    callbacks_ = {};
}

auto Manipulator::confirm() noexcept -> void
{
    if (!active_)
    {
        return;
    }
    active_ = false;
    mode_ = ManipulatorMode::none;
    axis_ = ManipulatorAxis::none;
    targets_.clear();
    active_ids_.clear();
    callbacks_ = {};
}

auto Manipulator::set_axis_from_input(const ManipulatorInput& input) noexcept -> void
{
    if (input.x_pressed)
    {
        axis_ = ManipulatorAxis::x;
    }
    else if (input.y_pressed)
    {
        axis_ = ManipulatorAxis::y;
    }
    else if (input.z_pressed)
    {
        axis_ = ManipulatorAxis::z;
    }
}

// clang-format off
auto Manipulator::active() const noexcept     -> bool             { return active_; }
auto Manipulator::mode() const noexcept       -> ManipulatorMode  { return mode_; }
auto Manipulator::axis() const noexcept       -> ManipulatorAxis  { return axis_; }
auto Manipulator::active_ids() const noexcept -> std::span<const ObjectId>
{
    return std::span<const ObjectId>{active_ids_.data(), active_ids_.size()};
}
// clang-format on
}  // namespace ds_vk
