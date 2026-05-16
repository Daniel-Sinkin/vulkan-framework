#include "app/pba/physics.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <glm/geometric.hpp>
#include <iostream>
#include <string_view>

namespace
{
auto g_failures = 0;

auto check(bool condition, const std::string_view message) -> void
{
    if (!condition)
    {
        ++g_failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

auto finite_vec3(const ds_vk::Vec3 value) -> bool
{
    return std::isfinite(value.x) and std::isfinite(value.y) and std::isfinite(value.z);
}

auto finite_quat(const ds_vk::Quat value) -> bool
{
    return std::isfinite(value.w) and std::isfinite(value.x) and std::isfinite(value.y)
           and std::isfinite(value.z);
}

auto quat_vector_len(const ds_vk::Quat value) -> ds_vk::f32
{
    return glm::length(ds_vk::Vec3{value.x, value.y, value.z});
}

auto test_pyramid_scene() -> void
{
    ds_vk_app::pba::PyramidSimulation sim{};
    sim.reset();
    check(sim.bodies().size() == 56zu, "pyramid reset creates 55 stack bodies plus projectile");
    check(sim.step_count() == 0zu, "pyramid reset clears step count");
    check(sim.bodies().back().name == "Projectile cube", "pyramid reset names projectile");
    check(sim.bodies().back().velocity.x > 0.0f, "projectile starts moving");

    const auto initial_z = sim.bodies().back().position.z;
    for (auto i = 0zu; i < 16zu; ++i)
    {
        sim.step(ds_vk_app::pba::k_fixed_dt);
    }
    check(sim.step_count() == 16zu, "pyramid sim counts fixed steps");
    check(sim.bodies().back().position.z != initial_z, "projectile position changes over time");
    for (const auto& body : sim.bodies())
    {
        check(finite_vec3(body.position), "pyramid sim keeps positions finite");
        check(finite_vec3(body.velocity), "pyramid sim keeps velocities finite");
        check(finite_vec3(body.angular_velocity), "pyramid sim keeps angular velocities finite");
        check(finite_quat(body.orientation), "pyramid sim keeps orientations finite");
        const auto bounds = ds_vk_app::pba::body_aabb(body);
        check(
            std::min(bounds.min.z, bounds.max.z) >= -1.0e-4f,
            "pyramid sim keeps bodies above ground"
        );
    }
    check(
        sim.last_step_stats().broadphase_candidates < sim.last_step_stats().body_pair_count,
        "pyramid sim broadphase prunes all-pairs body checks"
    );
}

auto test_projectile_optional() -> void
{
    ds_vk_app::pba::PyramidSimulation sim{};
    sim.reset({.base = 3zu, .add_projectile = false});
    check(sim.bodies().size() == 14zu, "pyramid reset can omit projectile");
}

auto test_forces_and_grabbed_bodies() -> void
{
    ds_vk_app::pba::PyramidSimulation sim{};
    sim.reset({.base = 1zu, .add_projectile = false});
    auto& body = sim.bodies().front();
    body.velocity = {};

    sim.physics().simple_forces.clear();
    const auto start_position = body.position;
    sim.step(ds_vk_app::pba::k_fixed_dt);
    check(glm::length(body.position - start_position) == 0.0f, "body without forces stays still");

    sim.physics().simple_forces = {
        ds_vk_app::pba::SimpleForce{
            ds_vk_app::pba::GravityForce{.accel = 4.0f * ds_vk::k_axis_x},
        },
    };
    sim.step(ds_vk_app::pba::k_fixed_dt);
    check(body.velocity.x > 0.0f, "gravity force path accelerates bodies");

    body.grabbed = true;
    body.velocity = {};
    const auto grabbed_position = body.position;
    sim.step(ds_vk_app::pba::k_fixed_dt);
    check(
        glm::length(body.position - grabbed_position) == 0.0f,
        "grabbed body is excluded from force integration"
    );
    check(glm::length(body.velocity) == 0.0f, "grabbed body velocity is cleared");
    check(sim.find_body(body.object_id) == &body, "simulation can find body by object id");
}

auto test_off_center_collision_generates_spin() -> void
{
    ds_vk_app::pba::PyramidSimulation sim{};
    sim.reset({.base = 1zu, .add_projectile = true});
    sim.physics().simple_forces.clear();
    sim.physics().solver_iterations = 1u;

    auto bodies = sim.bodies();
    auto& a = bodies[0];
    auto& b = bodies[1];
    a.position = {0.0f, 0.0f, 0.5f};
    a.previous_position = a.position;
    a.velocity = {};
    a.angular_velocity = {};
    b.position = {0.82f, 0.32f, 0.5f};
    b.previous_position = b.position;
    b.velocity = {-2.0f, 0.0f, 0.0f};
    b.angular_velocity = {};

    sim.step(ds_vk_app::pba::k_fixed_dt);

    check(
        glm::length(a.angular_velocity) > 1.0e-4f or glm::length(b.angular_velocity) > 1.0e-4f,
        "off-center box collision generates angular velocity"
    );

    sim.step(ds_vk_app::pba::k_fixed_dt);
    check(
        quat_vector_len(a.orientation) > 1.0e-5f or quat_vector_len(b.orientation) > 1.0e-5f,
        "angular velocity advances box orientation"
    );
}
}  // namespace

auto main() -> int
{
    try
    {
        test_pyramid_scene();
        test_projectile_optional();
        test_forces_and_grabbed_bodies();
        test_off_center_collision_generates_spin();
    }
    catch (const std::exception& error)
    {
        std::cerr << "[FAIL] unhandled exception: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0)
    {
        std::cerr << g_failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "all ds_vk PBA user tests passed\n";
    return 0;
}
