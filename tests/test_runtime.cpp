#include "ds_vk/runtime.hpp"

#include <concepts>
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

template <typename App>
concept has_prototype_runner = requires(ds_vk::Runtime& runtime, App& app) {
    { runtime.run_prototype(app) } -> std::same_as<int>;
};

static_assert(ds_vk::detail::has_runtime_hook<PlainApp>);
static_assert(ds_vk::detail::has_runtime_hook<SetupOnlyApp>);
static_assert(!ds_vk::detail::has_runtime_hook<NoHooksApp>);
static_assert(has_prototype_runner<PlainApp>);
static_assert(has_prototype_runner<SetupOnlyApp>);
static_assert(!std::is_polymorphic_v<PlainApp>);
static_assert(sizeof(ds_vk::MeshDebugMode) == sizeof(ds_vk::u8));
static_assert(sizeof(ds_vk::LightType) == sizeof(ds_vk::u8));

auto check(bool condition, const std::string_view message) -> void
{
    if (!condition)
    {
        ++g_failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
}

auto test_draw_list() -> void
{
    ds_vk::DrawList draw{};
    draw.draw_mesh({.mesh = ds_vk::MeshHandle{}});
    check(draw.mesh_commands().empty(), "invalid mesh handles are ignored");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.id = 0u},
        .object_id = {.value = 42u},
        .material =
            {
                .base_color = ds_vk::Color{0.2f, 0.3f, 0.4f, 1.0f},
                .emissive_color = ds_vk::Color{0.05f, 0.02f, 0.01f, 1.0f},
                .metallic = 0.35f,
                .roughness = 0.47f,
                .ambient_occlusion = 0.82f,
                .textures = {.base_color = ds_vk::TextureHandle{.id = 5u}},
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
        draw.mesh_commands().back().material.textures.base_color.id == 5u,
        "mesh draw records material base color texture"
    );
    check(
        draw.mesh_commands().back().debug.mode == ds_vk::MeshDebugMode::scalar_heatmap,
        "mesh draw records debug mode"
    );
    check(draw.mesh_commands().back().debug.selected, "mesh draw records selected debug flag");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.id = 8u},
        .mask = {.visible_to_camera = false, .shadow_producer = true},
    });
    check(draw.mesh_commands().size() == 2u, "shadow-only mesh draw is recorded");
    check(
        !draw.mesh_commands().back().mask.visible_to_camera,
        "mesh draw records camera visibility mask"
    );
    check(
        draw.mesh_commands().back().mask.shadow_producer, "mesh draw records shadow producer mask"
    );

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.id = 9u},
        .mask = {.visible_to_camera = false, .shadow_producer = false},
    });
    check(draw.mesh_commands().size() == 2u, "fully invisible mesh draw is culled");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.id = 3u},
        .debug = {.hidden = true},
    });
    check(draw.mesh_commands().size() == 2u, "hidden mesh draws are culled");

    draw.draw_mesh({
        .mesh = ds_vk::MeshHandle{.id = 4u},
        .object_id = {.value = std::numeric_limits<ds_vk::u32>::max()},
    });
    check(
        !draw.mesh_commands().back().object_id.valid(), "max u32 object id is reserved as invalid"
    );

    draw.draw_basic_mesh({
        .mesh = ds_vk::MeshHandle{.id = 1u},
        .object_id = {.value = 7u},
        .color = ds_vk::Color{0.8f, 0.7f, 0.6f, 1.0f},
        .debug = {
            .mode = ds_vk::MeshDebugMode::camera_depth,
            .scalar_range = {1.0f, 12.0f},
        },
    });
    check(draw.mesh_commands().size() == 4u, "basic mesh draw is recorded");
    check(
        draw.mesh_commands().back().material.base_color.r() == 0.8f,
        "basic mesh draw maps color to material"
    );
    check(draw.mesh_commands().back().object_id.value == 7u, "basic mesh draw records object id");
    check(
        draw.mesh_commands().back().debug.mode == ds_vk::MeshDebugMode::camera_depth,
        "basic mesh draw records camera depth debug mode"
    );
    check(
        draw.mesh_commands().back().debug.scalar_range.y == 12.0f,
        "basic mesh draw records depth debug range"
    );

    draw.set_ambient_light(ds_vk::Color{0.1f, 0.2f, 0.3f, 1.0f});
    check(draw.ambient_light().g() == 0.2f, "draw list records ambient light");
    draw.set_environment({
        .texture = {.id = 6u},
        .lighting_intensity = 0.42f,
        .background_intensity = 0.75f,
        .rotation_radians = 0.25f,
    });
    check(draw.environment().texture.id == 6u, "draw list records environment texture");
    check(
        draw.environment().lighting_intensity == 0.42f,
        "draw list records environment lighting intensity"
    );
    draw.directional_light({
        .direction = {-1.0f, -1.0f, -1.0f},
        .intensity = 2.0f,
        .shadow = {.enabled = true},
    });
    draw.radial_light({
        .position = {1.0f, 2.0f, 3.0f},
        .intensity = 12.0f,
        .range = 4.0f,
    });
    draw.spot_light({
        .position = {0.0f, 0.0f, 3.0f},
        .direction = -ds_vk::k_axis_z,
        .intensity = 20.0f,
        .range = 6.0f,
        .inner_cone_angle = 0.2f,
        .outer_cone_angle = 0.5f,
    });
    check(draw.lights().size() == 3u, "draw list records three light types");
    check(draw.lights()[0].type == ds_vk::LightType::directional, "directional light type");
    check(draw.lights()[0].shadow.enabled, "directional light records shadow config");
    check(draw.lights()[1].type == ds_vk::LightType::radial, "radial light type");
    check(draw.lights()[2].type == ds_vk::LightType::spot, "spot light type");
    draw.radial_light({.enabled = false});
    check(draw.lights().size() == 3u, "disabled lights are ignored");

    draw.debug_line({
        .start = {0.0f, 0.0f, 0.0f},
        .end = ds_vk::k_axis_x,
        .color = ds_vk::Color::white,
    });
    draw.debug_arrow({
        .origin = {0.0f, 0.0f, 0.0f},
        .vector = ds_vk::k_axis_y,
        .color = ds_vk::Color::white,
    });
    check(draw.debug_segments().size() == 2u, "debug line and arrow are recorded");
    draw.debug_arrow({
        .origin = {0.0f, 0.0f, 0.0f},
        .vector = 0.001f * ds_vk::k_axis_y,
        .color = ds_vk::Color::white,
    });
    check(draw.debug_segments().size() == 2u, "tiny debug arrows are ignored");
    draw.debug_arrow({
        .origin = {0.0f, 0.0f, 0.0f},
        .vector = ds_vk::k_axis_z,
        .color = ds_vk::Color::white,
        .draw_on_top = true,
    });
    check(draw.debug_segments().size() == 2u, "on-top debug arrows skip depth-tested list");
    check(draw.debug_on_top_segments().size() == 1u, "on-top debug arrows are recorded separately");

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
    check(draw.debug_on_top_segments().empty(), "clear removes on-top debug segments");
    check(draw.lights().empty(), "clear removes lights");
    check(draw.ambient_light().r() == 0.035f, "clear resets ambient light");
    check(!draw.environment().texture.valid(), "clear resets environment texture");
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
