#include "app/pba/physics.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <variant>

namespace ds_vk_app::pba
{
namespace
{
template <class... Ts>
struct Overloaded : Ts...
{
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

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

[[nodiscard]] auto inverse_inertia_world(const Body& body) noexcept -> glm::mat3
{
    if (body.inv_mass <= 0.0f or body.grabbed)
    {
        return glm::mat3{0.0f};
    }

    const auto mass = 1.0f / body.inv_mass;
    const auto size = 2.0f * glm::max(glm::abs(body.half_extent), ds_vk::Vec3{1.0e-4f});
    const ds_vk::Vec3 inertia{
        mass * (size.y * size.y + size.z * size.z) / 12.0f,
        mass * (size.x * size.x + size.z * size.z) / 12.0f,
        mass * (size.x * size.x + size.y * size.y) / 12.0f,
    };
    glm::mat3 local_inverse{0.0f};
    local_inverse[0][0] = 1.0f / inertia.x;
    local_inverse[1][1] = 1.0f / inertia.y;
    local_inverse[2][2] = 1.0f / inertia.z;

    const auto rotation = glm::mat3_cast(glm::normalize(body.orientation));
    return rotation * local_inverse * glm::transpose(rotation);
}

[[nodiscard]] auto velocity_at(const Body& body, ds_vk::Vec3 world_point) noexcept -> ds_vk::Vec3
{
    return body.velocity + glm::cross(body.angular_velocity, world_point - body.position);
}

[[nodiscard]] auto
impulse_denominator(const Body& body, ds_vk::Vec3 world_point, ds_vk::Vec3 direction) noexcept
    -> ds_vk::f32
{
    if (body.inv_mass <= 0.0f or body.grabbed)
    {
        return 0.0f;
    }
    const auto offset = world_point - body.position;
    const auto angular =
        glm::cross(inverse_inertia_world(body) * glm::cross(offset, direction), offset);
    return body.inv_mass + glm::dot(direction, angular);
}

auto apply_impulse_at(Body& body, ds_vk::Vec3 world_point, ds_vk::Vec3 impulse) noexcept -> void
{
    if (body.inv_mass <= 0.0f or body.grabbed)
    {
        return;
    }
    body.velocity += impulse * body.inv_mass;
    body.angular_velocity +=
        inverse_inertia_world(body) * glm::cross(world_point - body.position, impulse);
}

[[nodiscard]] auto overlap_contact_point(const ds_vk::Aabb& a, const ds_vk::Aabb& b) noexcept
    -> ds_vk::Vec3
{
    const auto a_min = glm::min(a.min, a.max);
    const auto a_max = glm::max(a.min, a.max);
    const auto b_min = glm::min(b.min, b.max);
    const auto b_max = glm::max(b.min, b.max);
    return 0.5f * (glm::max(a_min, b_min) + glm::min(a_max, b_max));
}

auto apply_contact_impulse(
    Body& a, Body& b, ds_vk::Vec3 contact_point, ds_vk::Vec3 normal, const PhysicsConfig& cfg
) noexcept -> void
{
    if (a.grabbed or b.grabbed)
    {
        return;
    }
    const auto normal_denominator = impulse_denominator(a, contact_point, normal)
                                    + impulse_denominator(b, contact_point, normal);
    if (normal_denominator <= 1.0e-8f)
    {
        return;
    }

    const auto relative_velocity = velocity_at(b, contact_point) - velocity_at(a, contact_point);
    const auto velocity_along_normal = glm::dot(relative_velocity, normal);
    if (velocity_along_normal > 0.0f)
    {
        return;
    }

    const auto normal_impulse_magnitude =
        -(1.0f + cfg.restitution) * velocity_along_normal / normal_denominator;
    const auto normal_impulse = normal_impulse_magnitude * normal;
    apply_impulse_at(a, contact_point, -normal_impulse);
    apply_impulse_at(b, contact_point, normal_impulse);

    const auto relative_velocity_after =
        velocity_at(b, contact_point) - velocity_at(a, contact_point);
    const auto tangent_velocity =
        relative_velocity_after - glm::dot(relative_velocity_after, normal) * normal;
    const auto tangent_len2 = glm::dot(tangent_velocity, tangent_velocity);
    if (tangent_len2 <= 1.0e-8f)
    {
        return;
    }

    const auto tangent = tangent_velocity * glm::inversesqrt(tangent_len2);
    const auto tangent_denominator = impulse_denominator(a, contact_point, tangent)
                                     + impulse_denominator(b, contact_point, tangent);
    if (tangent_denominator <= 1.0e-8f)
    {
        return;
    }

    const auto unclamped_tangent_impulse =
        -glm::dot(relative_velocity_after, tangent) / tangent_denominator;
    const auto max_friction_impulse = cfg.friction * normal_impulse_magnitude;
    const auto tangent_impulse_magnitude =
        std::clamp(unclamped_tangent_impulse, -max_friction_impulse, max_friction_impulse);
    const auto tangent_impulse = tangent_impulse_magnitude * tangent;
    apply_impulse_at(a, contact_point, -tangent_impulse);
    apply_impulse_at(b, contact_point, tangent_impulse);
}

auto resolve_body_pair(Body& a, Body& b, const PhysicsConfig& cfg) noexcept -> void
{
    if (a.grabbed or b.grabbed)
    {
        return;
    }
    if (a.inv_mass <= 0.0f and b.inv_mass <= 0.0f)
    {
        return;
    }
    const auto a_bounds = body_aabb(a);
    const auto b_bounds = body_aabb(b);
    const auto overlap = overlap_amount(a_bounds, b_bounds);
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
    apply_contact_impulse(a, b, overlap_contact_point(a_bounds, b_bounds), normal, cfg);
}

auto integrate_body(Body& body, const PhysicsConfig& cfg, ds_vk::f32 dt) noexcept -> void
{
    body.previous_position = body.position;
    if (body.inv_mass <= 0.0f or body.grabbed)
    {
        if (body.grabbed)
        {
            body.velocity = {};
            body.angular_velocity = {};
        }
        return;
    }
    body.velocity += body.force_accum * body.inv_mass * dt;
    body.velocity *= std::max(0.0f, 1.0f - cfg.linear_damping * dt);
    body.position += body.velocity * dt;

    body.angular_velocity += inverse_inertia_world(body) * body.torque_accum * dt;
    body.angular_velocity *= std::max(0.0f, 1.0f - cfg.angular_damping * dt);
    if (glm::dot(body.angular_velocity, body.angular_velocity) > 1.0e-10f)
    {
        const ds_vk::Quat spin{
            0.0f,
            body.angular_velocity.x,
            body.angular_velocity.y,
            body.angular_velocity.z,
        };
        body.orientation =
            glm::normalize(body.orientation + (spin * body.orientation) * (0.5f * dt));
    }
}

auto resolve_ground(Body& body, const PhysicsConfig& cfg) noexcept -> void
{
    if (body.inv_mass <= 0.0f or body.grabbed)
    {
        return;
    }
    const auto bounds = body_aabb(body);
    const auto min_z = std::min(bounds.min.z, bounds.max.z);
    if (min_z >= 0.0f)
    {
        return;
    }
    body.position.z -= min_z;

    const ds_vk::Vec3 contact_point{body.position.x, body.position.y, 0.0f};
    const auto contact_velocity = velocity_at(body, contact_point);
    const auto normal_velocity = glm::dot(contact_velocity, ds_vk::k_axis_z);
    if (normal_velocity < 0.0f)
    {
        const auto normal_denominator = impulse_denominator(body, contact_point, ds_vk::k_axis_z);
        if (normal_denominator > 1.0e-8f)
        {
            const auto normal_impulse_magnitude =
                -(1.0f + cfg.restitution) * normal_velocity / normal_denominator;
            apply_impulse_at(body, contact_point, normal_impulse_magnitude * ds_vk::k_axis_z);

            const auto velocity_after = velocity_at(body, contact_point);
            const auto tangent_velocity =
                velocity_after - glm::dot(velocity_after, ds_vk::k_axis_z) * ds_vk::k_axis_z;
            const auto tangent_len2 = glm::dot(tangent_velocity, tangent_velocity);
            if (tangent_len2 > 1.0e-8f)
            {
                const auto tangent = tangent_velocity * glm::inversesqrt(tangent_len2);
                const auto tangent_denominator = impulse_denominator(body, contact_point, tangent);
                if (tangent_denominator > 1.0e-8f)
                {
                    const auto unclamped_tangent_impulse =
                        -glm::dot(velocity_after, tangent) / tangent_denominator;
                    const auto max_friction_impulse = cfg.friction * normal_impulse_magnitude;
                    const auto tangent_impulse_magnitude = std::clamp(
                        unclamped_tangent_impulse, -max_friction_impulse, max_friction_impulse
                    );
                    apply_impulse_at(body, contact_point, tangent_impulse_magnitude * tangent);
                }
            }
        }
    }
}

auto apply_simple_force(Body& body, const SimpleForce& force) -> void
{
    if (body.inv_mass <= 0.0f or body.grabbed)
    {
        return;
    }

    std::visit(
        Overloaded{
            [&](const GravityForce& gravity) noexcept
            { body.force_accum += gravity.accel / body.inv_mass; },
            [&](const AttractorForce& attractor) noexcept
            {
                const auto min_radius = std::max(attractor.min_radius, 1.0e-6f);
                const auto min_radius_squared = min_radius * min_radius;
                const auto offset = attractor.target - body.position;
                const auto distance_squared = glm::dot(offset, offset);
                if (distance_squared <= min_radius_squared)
                {
                    return;
                }
                const auto direction = offset * glm::inversesqrt(distance_squared);
                body.force_accum += (attractor.magnitude * direction) / body.inv_mass;
            },
            [&](const RepulsionForce& repulsion) noexcept
            {
                const auto min_radius = std::max(repulsion.min_radius, 1.0e-6f);
                const auto range = std::max(repulsion.range, min_radius);
                const auto offset = body.position - repulsion.target;
                const auto distance_squared = glm::dot(offset, offset);
                if (distance_squared <= min_radius * min_radius)
                {
                    return;
                }
                const auto distance = std::sqrt(distance_squared);
                if (distance >= range)
                {
                    return;
                }
                const auto t = (range - distance) / (range - min_radius);
                const auto acceleration = std::clamp(t, 0.0f, 1.0f) * repulsion.accel_max;
                body.force_accum += (acceleration * offset / distance) / body.inv_mass;
            },
        },
        force
    );
}

auto apply_complex_force(std::span<Body> bodies, const ComplexForce& force) -> void
{
    std::visit(
        Overloaded{
            [&](const NBodyForce& nbody) noexcept
            {
                const auto eps2 = nbody.softening * nbody.softening;
                for (auto i = 0zu; i < bodies.size(); ++i)
                {
                    auto& a = bodies[i];
                    if (a.inv_mass <= 0.0f or a.grabbed)
                    {
                        continue;
                    }
                    const auto mass_a = 1.0f / a.inv_mass;
                    for (auto j = i + 1zu; j < bodies.size(); ++j)
                    {
                        auto& b = bodies[j];
                        if (b.inv_mass <= 0.0f or b.grabbed)
                        {
                            continue;
                        }
                        const auto mass_b = 1.0f / b.inv_mass;
                        const auto offset = b.position - a.position;
                        const auto distance_squared = glm::dot(offset, offset) + eps2;
                        const auto inv_distance = glm::inversesqrt(distance_squared);
                        const auto inv_distance3 = inv_distance * inv_distance * inv_distance;
                        const auto force_ab = (nbody.g * mass_a * mass_b) * offset * inv_distance3;
                        a.force_accum += force_ab;
                        b.force_accum -= force_ab;
                    }
                }
            },
        },
        force
    );
}

auto clear_accumulators(std::span<Body> bodies) noexcept -> void
{
    for (auto& body : bodies)
    {
        body.force_accum = {};
        body.torque_accum = {};
    }
}

auto apply_forces(std::span<Body> bodies, const PhysicsConfig& cfg) -> void
{
    clear_accumulators(bodies);
    for (auto& body : bodies)
    {
        for (const auto& force : cfg.simple_forces)
        {
            apply_simple_force(body, force);
        }
    }
    for (const auto& force : cfg.complex_forces)
    {
        apply_complex_force(bodies, force);
    }
}
}  // namespace

auto body_aabb(const Body& body) noexcept -> ds_vk::Aabb
{
    const auto axes = glm::mat3_cast(glm::normalize(body.orientation));
    const ds_vk::Vec3 world_half_extent{
        std::abs(axes[0].x) * body.half_extent.x + std::abs(axes[1].x) * body.half_extent.y
            + std::abs(axes[2].x) * body.half_extent.z,
        std::abs(axes[0].y) * body.half_extent.x + std::abs(axes[1].y) * body.half_extent.y
            + std::abs(axes[2].y) * body.half_extent.z,
        std::abs(axes[0].z) * body.half_extent.x + std::abs(axes[1].z) * body.half_extent.y
            + std::abs(axes[2].z) * body.half_extent.z,
    };
    return {
        .min = body.position - world_half_extent,
        .max = body.position + world_half_extent,
    };
}

auto PyramidSimulation::reset(const PyramidSceneConfig& cfg) -> void
{
    bodies_.clear();
    body_bounds_.clear();
    broadphase_order_.clear();
    broadphase_active_.clear();
    collision_pairs_.clear();
    last_step_stats_ = {};
    step_count_ = 0zu;

    constexpr ds_vk::Vec3 half{0.5f, 0.5f, 0.5f};
    for (auto layer = 0zu; layer < cfg.base; ++layer)
    {
        const auto n = cfg.base - layer;
        const auto half_span = 0.5f * static_cast<ds_vk::f32>(n - 1) * cfg.step;
        for (auto x = 0zu; x < n; ++x)
        {
            for (auto y = 0zu; y < n; ++y)
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
    apply_forces(std::span<Body>{bodies_}, physics_);
    for (auto& body : bodies_)
    {
        integrate_body(body, physics_, dt);
        resolve_ground(body, physics_);
    }

    for (auto iteration = 0u; iteration < physics_.solver_iterations; ++iteration)
    {
        rebuild_collision_pairs();
        for (const auto& pair : collision_pairs_)
        {
            resolve_body_pair(bodies_[pair.a], bodies_[pair.b], physics_);
        }
        for (auto& body : bodies_)
        {
            resolve_ground(body, physics_);
        }
    }
    clear_accumulators(std::span<Body>{bodies_});
    ++step_count_;
}

auto PyramidSimulation::rebuild_collision_pairs() -> void
{
    body_bounds_.resize(bodies_.size());
    broadphase_order_.resize(bodies_.size());
    collision_pairs_.clear();
    broadphase_active_.clear();

    for (auto i = 0zu; i < bodies_.size(); ++i)
    {
        body_bounds_[i] = body_aabb(bodies_[i]);
        broadphase_order_[i] = i;
    }

    std::ranges::sort(
        broadphase_order_,
        [&](const ds_vk::usize lhs, const ds_vk::usize rhs) noexcept
        {
            const auto lhs_min = std::min(body_bounds_[lhs].min.x, body_bounds_[lhs].max.x);
            const auto rhs_min = std::min(body_bounds_[rhs].min.x, body_bounds_[rhs].max.x);
            if (lhs_min < rhs_min)
            {
                return true;
            }
            if (lhs_min > rhs_min)
            {
                return false;
            }
            return lhs < rhs;
        }
    );

    for (const auto idx : broadphase_order_)
    {
        const auto idx_min = glm::min(body_bounds_[idx].min, body_bounds_[idx].max);
        const auto idx_max = glm::max(body_bounds_[idx].min, body_bounds_[idx].max);

        auto write = 0zu;
        for (const auto active_idx : broadphase_active_)
        {
            const auto active_max_x =
                std::max(body_bounds_[active_idx].min.x, body_bounds_[active_idx].max.x);
            if (active_max_x >= idx_min.x)
            {
                broadphase_active_[write++] = active_idx;
            }
        }
        broadphase_active_.resize(write);

        for (const auto active_idx : broadphase_active_)
        {
            if (bodies_[idx].grabbed or bodies_[active_idx].grabbed)
            {
                continue;
            }
            if (bodies_[idx].inv_mass <= 0.0f and bodies_[active_idx].inv_mass <= 0.0f)
            {
                continue;
            }

            const auto active_min =
                glm::min(body_bounds_[active_idx].min, body_bounds_[active_idx].max);
            const auto active_max =
                glm::max(body_bounds_[active_idx].min, body_bounds_[active_idx].max);
            const auto overlap_y = idx_min.y <= active_max.y and idx_max.y >= active_min.y;
            const auto overlap_z = idx_min.z <= active_max.z and idx_max.z >= active_min.z;
            if (overlap_y and overlap_z)
            {
                collision_pairs_.push_back(
                    CollisionPair{
                        .a = std::min(idx, active_idx),
                        .b = std::max(idx, active_idx),
                    }
                );
            }
        }

        broadphase_active_.push_back(idx);
    }

    const auto body_count = bodies_.size();
    const auto body_pair_count = body_count < 2zu ? 0zu : body_count * (body_count - 1zu) / 2zu;
    last_step_stats_ = PhysicsStepStats{
        .broadphase_candidates = collision_pairs_.size(),
        .body_pair_count = body_pair_count,
    };
}

auto PyramidSimulation::bodies() noexcept -> std::span<Body>
{
    return std::span<Body>{bodies_.data(), bodies_.size()};
}

auto PyramidSimulation::bodies() const noexcept -> std::span<const Body>
{
    return std::span<const Body>{bodies_.data(), bodies_.size()};
}

auto PyramidSimulation::find_body(ds_vk::ObjectId id) noexcept -> Body*
{
    const auto iter = std::ranges::find_if(
        bodies_, [id](const Body& body) noexcept { return body.object_id.value == id.value; }
    );
    return iter == bodies_.end() ? nullptr : &*iter;
}

auto PyramidSimulation::find_body(ds_vk::ObjectId id) const noexcept -> const Body*
{
    const auto iter = std::ranges::find_if(
        bodies_, [id](const Body& body) noexcept { return body.object_id.value == id.value; }
    );
    return iter == bodies_.end() ? nullptr : &*iter;
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

auto PyramidSimulation::last_step_stats() const noexcept -> const PhysicsStepStats&
{
    return last_step_stats_;
}
}  // namespace ds_vk_app::pba
