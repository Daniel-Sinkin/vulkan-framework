#include "ds_vk/runtime.hpp"

#include <iostream>
#include <string_view>
#include <type_traits>

namespace
{
auto g_failures = 0;

struct PlainApp
{
    auto setup(ds_vk::Runtime& runtime) -> void;
    auto update(ds_vk::FrameContext& frame, ds_vk::f32 dt_seconds) -> void;
    auto draw_ui(ds_vk::FrameContext& frame) -> void;
    auto shutdown(ds_vk::Runtime& runtime) -> void;
};

struct SetupOnlyApp
{
    auto setup(ds_vk::Runtime& runtime) -> void;
};

struct NoHooksApp
{
    auto tick() -> void;
};

static_assert(ds_vk::detail::has_runtime_hook<PlainApp>);
static_assert(ds_vk::detail::has_runtime_hook<SetupOnlyApp>);
static_assert(!ds_vk::detail::has_runtime_hook<NoHooksApp>);
static_assert(!std::is_polymorphic_v<PlainApp>);

auto check(const bool condition, const std::string_view message) -> void
{
    if (!condition)
    {
        ++g_failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

auto test_draw_list() -> void
{
    auto draw = ds_vk::DrawList{};
    draw.draw_mesh(ds_vk::MeshHandle{}, ds_vk::Transform{});
    check(draw.mesh_commands().empty(), "invalid mesh handles are ignored");

    draw.draw_mesh(ds_vk::MeshHandle{.index = 0u}, ds_vk::Transform{});
    check(draw.mesh_commands().size() == 1u, "valid mesh draw is recorded");

    draw.debug_line({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, ds_vk::Vec4{1.0f});
    draw.debug_arrow({0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, ds_vk::Vec4{1.0f});
    check(draw.debug_segments().size() == 2u, "debug line and arrow are recorded");

    draw.debug_sphere({0.0f, 0.0f, 0.0f}, 1.0f, ds_vk::Vec4{1.0f}, 12u);
    check(draw.debug_segments().size() == 38u, "debug sphere records three circles");

    draw.debug_sphere({0.0f, 0.0f, 0.0f}, -1.0f, ds_vk::Vec4{1.0f}, 12u);
    check(draw.debug_segments().size() == 38u, "debug sphere ignores non-positive radius");

    draw.clear();
    check(draw.mesh_commands().empty(), "clear removes mesh commands");
    check(draw.debug_segments().empty(), "clear removes debug segments");
}
}  // namespace

auto main() -> int
{
    test_draw_list();
    if (g_failures != 0)
    {
        std::cerr << g_failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "all ds_vk runtime tests passed\n";
    return 0;
}
