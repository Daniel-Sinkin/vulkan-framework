#include "app/pba/physics.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace ds_vk_app::pba
{
namespace
{
[[nodiscard]] auto overlap_amount(const ds_vk::Aabb& a, const ds_vk::Aabb& b) noexcept
    -> ds_vk::Vec3
{
    const auto a_min = glm::min(a.min, a.max);
    const auto a_max = glm::max(a.min, a.max);
    const auto b_min = glm::min(b.min, b.max);
    const auto b_max = glm::max(b.min, b.max);
    return {
        std::min(a_max.x, b_max.x) - std::max(a_min.x, b_min.x),
        std::min(a_max.y, b_max.y) - std::max(a_min.y, b_min.y),
        std::min(a_max.z, b_max.z) - std::max(a_min.z, b_min.z),
    };
}

[[nodiscard]] auto collision_normal_from_overlap(const Body& a, const Body& b, ds_vk::Vec3 overlap)
    -> ds_vk::Vec3
{
    auto normal = ds_vk::k_axis_x;
    auto min_overlap = overlap.x;
    if (overlap.y < min_overlap)
    {
        min_overlap = overlap.y;
        normal = ds_vk::k_axis_y;
    }
    if (overlap.z < min_overlap)
    {
        normal = ds_vk::k_axis_z;
    }

    const auto center_delta = b.position - a.position;
    if (glm::dot(center_delta, normal) < 0.0f)
    {
        normal = -normal;
    }
    return normal;
}

auto apply_impulse(Body& a, Body& b, ds_vk::Vec3 normal, ds_vk::f32 restitution) noexcept -> void
{
    const auto inv_mass_sum = a.inv_mass + b.inv_mass;
    if (inv_mass_sum <= 0.0f)
    {
        return;
    }

    const auto relative_velocity = b.velocity - a.velocity;
    const auto velocity_along_normal = glm::dot(relative_velocity, normal);
    if (velocity_along_normal > 0.0f)
    {
        return;
    }

    const auto impulse_magnitude = -(1.0f + restitution) * velocity_along_normal / inv_mass_sum;
    const auto impulse = impulse_magnitude * normal;
    a.velocity -= impulse * a.inv_mass;
    b.velocity += impulse * b.inv_mass;
}

auto resolve_body_pair(Body& a, Body& b, const PhysicsConfig& cfg) noexcept -> void
{
    if (a.inv_mass <= 0.0f and b.inv_mass <= 0.0f)
    {
        return;
    }
    const auto overlap = overlap_amount(body_aabb(a), body_aabb(b));
    if (overlap.x <= 0.0f or overlap.y <= 0.0f or overlap.z <= 0.0f)
    {
        return;
    }

    const auto normal = collision_normal_from_overlap(a, b, overlap);
    const auto penetration = glm::dot(overlap, glm::abs(normal));
    const auto inv_mass_sum = a.inv_mass + b.inv_mass;
    if (inv_mass_sum > 0.0f)
    {
        constexpr auto percent = 0.82f;
        const auto correction = percent * penetration * normal / inv_mass_sum;
        a.position -= correction * a.inv_mass;
        b.position += correction * b.inv_mass;
    }
    apply_impulse(a, b, normal, cfg.restitution);

    const auto relative_velocity = b.velocity - a.velocity;
    const auto tangent_velocity = relative_velocity - glm::dot(relative_velocity, normal) * normal;
    const auto tangent_len2 = glm::dot(tangent_velocity, tangent_velocity);
    if (tangent_len2 > 1.0e-8f)
    {
        const auto friction_direction = tangent_velocity * glm::inversesqrt(tangent_len2);
        const auto friction = cfg.friction * 0.04f;
        a.velocity += friction * friction_direction * a.inv_mass;
        b.velocity -= friction * friction_direction * b.inv_mass;
    }
}

auto integrate_body(Body& body, const PhysicsConfig& cfg, ds_vk::f32 dt) noexcept -> void
{
    body.previous_position = body.position;
    if (body.inv_mass <= 0.0f)
    {
        return;
    }
    body.velocity += cfg.gravity * dt;
    body.velocity *= std::max(0.0f, 1.0f - cfg.linear_damping * dt);
    body.position += body.velocity * dt;
}

auto resolve_ground(Body& body, const PhysicsConfig& cfg) noexcept -> void
{
    if (body.inv_mass <= 0.0f)
    {
        return;
    }
    const auto min_z = body.position.z - body.half_extent.z;
    if (min_z >= 0.0f)
    {
        return;
    }
    body.position.z -= min_z;
    if (body.velocity.z < 0.0f)
    {
        body.velocity.z = -body.velocity.z * cfg.restitution;
    }
    body.velocity.x *= std::clamp(1.0f - cfg.friction * 0.08f, 0.0f, 1.0f);
    body.velocity.y *= std::clamp(1.0f - cfg.friction * 0.08f, 0.0f, 1.0f);
}
}  // namespace

auto body_aabb(const Body& body) noexcept -> ds_vk::Aabb
{
    return {
        .min = body.position - body.half_extent,
        .max = body.position + body.half_extent,
    };
}

auto PyramidSimulation::reset(const PyramidSceneConfig& cfg) -> void
{
    bodies_.clear();
    step_count_ = 0zu;

    constexpr auto half = ds_vk::Vec3{0.5f, 0.5f, 0.5f};
    for (auto layer = 0; layer < cfg.base; ++layer)
    {
        const auto n = cfg.base - layer;
        const auto half_span = 0.5f * static_cast<ds_vk::f32>(n - 1) * cfg.step;
        for (auto x = 0; x < n; ++x)
        {
            for (auto y = 0; y < n; ++y)
            {
                const ds_vk::Vec3 position{
                    static_cast<ds_vk::f32>(x) * cfg.step - half_span,
                    static_cast<ds_vk::f32>(y) * cfg.step - half_span,
                    0.5f + static_cast<ds_vk::f32>(layer) * cfg.step,
                };
                const auto id = static_cast<ds_vk::u32>(bodies_.size());
                bodies_.push_back(Body{
                    .object_id = {.value = k_body_id_base + id},
                    .half_extent = half,
                    .position = position,
                    .previous_position = position,
                    .velocity = 0.08f
                                * ds_vk::Vec3{
                                    std::sin(static_cast<ds_vk::f32>(id) * 0.53f),
                                    std::cos(static_cast<ds_vk::f32>(id) * 0.37f),
                                    0.0f,
                                },
                    .color = ds_vk::Color{0.56f, 0.59f, 0.56f, 1.0f},
                    .name = std::format("Cube {}", id),
                });
            }
        }
    }

    if (!cfg.add_projectile)
    {
        return;
    }

    const auto projectile_id = static_cast<ds_vk::u32>(bodies_.size());
    bodies_.push_back(
        Body{
            .object_id = {.value = k_body_id_base + projectile_id},
            .half_extent = ds_vk::Vec3{0.42f},
            .position = {-2.8f, -2.9f, 4.6f},
            .previous_position = {-2.8f, -2.9f, 4.6f},
            .velocity = {2.6f, 2.1f, 0.7f},
            .color = ds_vk::Color{0.95f, 0.32f, 0.18f, 1.0f},
            .name = "Projectile cube",
        }
    );
}

auto PyramidSimulation::step(ds_vk::f32 dt) -> void
{
    for (auto& body : bodies_)
    {
        integrate_body(body, physics_, dt);
        resolve_ground(body, physics_);
    }

    for (auto iteration = 0u; iteration < physics_.solver_iterations; ++iteration)
    {
        for (auto i = 0zu; i < bodies_.size(); ++i)
        {
            for (auto j = i + 1zu; j < bodies_.size(); ++j)
            {
                resolve_body_pair(bodies_[i], bodies_[j], physics_);
            }
        }
        for (auto& body : bodies_)
        {
            resolve_ground(body, physics_);
        }
    }
    ++step_count_;
}

auto PyramidSimulation::bodies() noexcept -> std::span<Body>
{
    return std::span<Body>{bodies_.data(), bodies_.size()};
}

auto PyramidSimulation::bodies() const noexcept -> std::span<const Body>
{
    return std::span<const Body>{bodies_.data(), bodies_.size()};
}

auto PyramidSimulation::physics() noexcept -> PhysicsConfig&
{
    return physics_;
}

auto PyramidSimulation::physics() const noexcept -> const PhysicsConfig&
{
    return physics_;
}

auto PyramidSimulation::step_count() const noexcept -> ds_vk::usize
{
    return step_count_;
}
}  // namespace ds_vk_app::pba
