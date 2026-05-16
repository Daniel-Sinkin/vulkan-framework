#include "app/pba/physics.hpp"

#include <cmath>
#include <iostream>
#include <string_view>

namespace
{
auto g_failures = 0;

auto check(const bool condition, const std::string_view message) -> void
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

auto test_pyramid_scene() -> void
{
    auto sim = ds_vk_app::pba::PyramidSimulation{};
    sim.reset();
    check(sim.bodies().size() == 56zu, "pyramid reset creates 55 stack bodies plus projectile");
    check(sim.step_count() == 0zu, "pyramid reset clears step count");
    check(sim.bodies().back().name == "Projectile cube", "pyramid reset names projectile");
    check(sim.bodies().back().velocity.x > 0.0f, "projectile starts moving");

    const auto initial_z = sim.bodies().back().position.z;
    for (auto i = 0; i < 16; ++i)
    {
        sim.step(ds_vk_app::pba::k_fixed_dt);
    }
    check(sim.step_count() == 16zu, "pyramid sim counts fixed steps");
    check(sim.bodies().back().position.z != initial_z, "projectile position changes over time");
    for (const auto& body : sim.bodies())
    {
        check(finite_vec3(body.position), "pyramid sim keeps positions finite");
        check(finite_vec3(body.velocity), "pyramid sim keeps velocities finite");
        check(
            body.position.z - body.half_extent.z >= -1.0e-4f,
            "pyramid sim keeps bodies above ground"
        );
    }
}

auto test_projectile_optional() -> void
{
    auto sim = ds_vk_app::pba::PyramidSimulation{};
    sim.reset({.base = 3, .add_projectile = false});
    check(sim.bodies().size() == 14zu, "pyramid reset can omit projectile");
}
}  // namespace

auto main() -> int
{
    test_pyramid_scene();
    test_projectile_optional();
    if (g_failures != 0)
    {
        std::cerr << g_failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "all ds_vk PBA user tests passed\n";
    return 0;
}
