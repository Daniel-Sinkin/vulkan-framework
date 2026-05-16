#include "ds_vk/runtime.hpp"

#include <iostream>
#include <limits>
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
static_assert(sizeof(ds_vk::MeshDebugMode) == sizeof(ds_vk::u8));

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
    draw.draw_mesh({.mesh = ds_vk::MeshHandle{}});
    check(draw.mesh_commands().empty(), "invalid mesh handles are ignored");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.index = 0u},
        .object_id = {.value = 42u},
        .material =
            {
                .base_color = ds_vk::Color{0.2f, 0.3f, 0.4f, 1.0f},
                .emissive_color = ds_vk::Color{0.05f, 0.02f, 0.01f, 1.0f},
                .metallic = 0.35f,
                .roughness = 0.47f,
                .ambient_occlusion = 0.82f,
                .textures = {.base_color = ds_vk::TextureHandle{.index = 5u}},
            },
        .debug = {
            .mode = ds_vk::MeshDebugMode::scalar_heatmap,
            .color = ds_vk::Color{1.0f, 0.0f, 0.8f, 0.75f},
            .scalar = 4.2f,
            .scalar_range = {0.0f, 10.0f},
            .selected = true,
        },
    });
    check(draw.mesh_commands().size() == 1u, "valid mesh draw is recorded");
    check(draw.mesh_commands().back().object_id.value == 42u, "mesh draw records object id");
    check(
        draw.mesh_commands().back().material.base_color.g() == 0.3f,
        "mesh draw records material base color"
    );
    check(
        draw.mesh_commands().back().material.emissive_color.r() == 0.05f,
        "mesh draw records material emissive color"
    );
    check(
        draw.mesh_commands().back().material.metallic == 0.35f,
        "mesh draw records material metallic"
    );
    check(
        draw.mesh_commands().back().material.roughness == 0.47f,
        "mesh draw records material roughness"
    );
    check(
        draw.mesh_commands().back().material.ambient_occlusion == 0.82f,
        "mesh draw records material ambient occlusion"
    );
    check(
        draw.mesh_commands().back().material.textures.base_color.index == 5u,
        "mesh draw records material base color texture"
    );
    check(
        draw.mesh_commands().back().debug.mode == ds_vk::MeshDebugMode::scalar_heatmap,
        "mesh draw records debug mode"
    );
    check(draw.mesh_commands().back().debug.selected, "mesh draw records selected debug flag");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.index = 3u},
        .debug = {.hidden = true},
    });
    check(draw.mesh_commands().size() == 1u, "hidden mesh draws are culled");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.index = 4u},
        .object_id = {.value = std::numeric_limits<ds_vk::u32>::max()},
    });
    check(
        !draw.mesh_commands().back().object_id.valid(), "max u32 object id is reserved as invalid"
    );

    draw.draw_basic_mesh({
        .mesh = ds_vk::MeshHandle{.index = 1u},
        .object_id = {.value = 7u},
        .color = ds_vk::Color{0.8f, 0.7f, 0.6f, 1.0f},
        .debug = {.mode = ds_vk::MeshDebugMode::color_override},
    });
    check(draw.mesh_commands().size() == 3u, "basic mesh draw is recorded");
    check(
        draw.mesh_commands().back().material.base_color.r() == 0.8f,
        "basic mesh draw maps color to material"
    );
    check(draw.mesh_commands().back().object_id.value == 7u, "basic mesh draw records object id");
    check(
        draw.mesh_commands().back().debug.mode == ds_vk::MeshDebugMode::color_override,
        "basic mesh draw records debug mode"
    );

    draw.debug_line({
        .start = {0.0f, 0.0f, 0.0f},
        .end = {1.0f, 0.0f, 0.0f},
        .color = ds_vk::Color::white,
    });
    draw.debug_arrow({
        .origin = {0.0f, 0.0f, 0.0f},
        .vector = {0.0f, 1.0f, 0.0f},
        .color = ds_vk::Color::white,
    });
    check(draw.debug_segments().size() == 2u, "debug line and arrow are recorded");

    draw.debug_sphere({
        .center = {0.0f, 0.0f, 0.0f},
        .radius = 1.0f,
        .color = ds_vk::Color::white,
        .segments = 12u,
    });
    check(draw.debug_segments().size() == 38u, "debug sphere records three circles");

    draw.debug_sphere({
        .center = {0.0f, 0.0f, 0.0f},
        .radius = -1.0f,
        .color = ds_vk::Color::white,
        .segments = 12u,
    });
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
