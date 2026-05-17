#include "app/quake/quake_app.hpp"

#include "app/quake/background_music.hpp"
#include "app/quake/mdl.hpp"
#include "app/quake/paths.hpp"
#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/picker.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <imgui.h>
#include <limits>
#include <print>
#include <string>
#include <utility>
#include <vector>

namespace ds_vk_quake
{
using namespace ds_vk;

[[nodiscard]] auto transformed_aabb(const Aabb& aabb, const Transform& transform) -> Aabb
{
    const auto matrix = transform.matrix();
    const auto transform_point = [&](Vec3 point) -> Vec3
    { return Vec3{matrix * Vec4{point, 1.0f}}; };

    const std::array corners{
        Vec3{aabb.min.x, aabb.min.y, aabb.min.z},
        Vec3{aabb.max.x, aabb.min.y, aabb.min.z},
        Vec3{aabb.min.x, aabb.max.y, aabb.min.z},
        Vec3{aabb.max.x, aabb.max.y, aabb.min.z},
        Vec3{aabb.min.x, aabb.min.y, aabb.max.z},
        Vec3{aabb.max.x, aabb.min.y, aabb.max.z},
        Vec3{aabb.min.x, aabb.max.y, aabb.max.z},
        Vec3{aabb.max.x, aabb.max.y, aabb.max.z},
    };

    Aabb out{
        .min = Vec3{std::numeric_limits<f32>::max()},
        .max = Vec3{std::numeric_limits<f32>::lowest()},
    };
    for (const auto& corner : corners)
    {
        const auto world_corner = transform_point(corner);
        out.min = glm::min(out.min, world_corner);
        out.max = glm::max(out.max, world_corner);
    }
    return out;
}

[[nodiscard]] auto merge_aabb(const Aabb& a, const Aabb& b) -> Aabb
{
    return Aabb{
        .min = glm::min(a.min, b.min),
        .max = glm::max(a.max, b.max),
    };
}

auto run_quake_app() -> int
{
    namespace fs = std::filesystem;

    const auto mdl_files = [&]
    {
        std::vector<fs::path> out{};
        std::error_code ec{};
        for (const auto& entry : fs::directory_iterator{paths::progs_dir, ec})
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const auto& filepath = entry.path();
            if (filepath.extension() != ".mdl")
            {
                continue;
            }
            out.push_back(filepath);
        }
        std::ranges::sort(out);
        return out;
    }();

    if (mdl_files.empty())
    {
        std::println(stderr, "Failed to find MDL files in {}", paths::progs_dir.string());
        return EXIT_FAILURE;
    }

    struct ParsedModel
    {
        std::string name{};
        std::vector<MeshData> meshes{};
        std::vector<Aabb> pose_bounds{};
        Aabb bounds{};
        Vec3 extent{};
        f32 max_extent{};
        usize source_vertices{};
        usize source_triangles{};
        usize source_frames{};
        usize source_skins{};
        usize render_vertices{};
        usize render_triangles{};
    };

    std::vector<ParsedModel> parsed_models{};
    parsed_models.reserve(mdl_files.size());
    const auto dump_texel_indices = std::getenv("DS_VK_QUAKE_DUMP_TEXEL_INDICES") != nullptr;
    for (const auto& file : mdl_files)
    {
        const auto name = file.stem().string();
        auto binary = parse_mdl_binary(file);
        if (!binary)
        {
            continue;
        }
        if (dump_texel_indices)
        {
            save_mdl_skins_to_file(binary->skins, binary->header, name);
        }
        if (binary->frames.empty())
        {
            std::println(stderr, "{} has no renderable frames", file.filename().string());
            continue;
        }

        std::vector<MeshData> meshes{};
        std::vector<Aabb> pose_bounds{};
        meshes.reserve(binary->frames.size());
        pose_bounds.reserve(binary->frames.size());
        for (const auto& mdl_frame : binary->frames)
        {
            auto mesh =
                make_mesh_data(binary->header, binary->stverts, binary->triangles, mdl_frame);
            if (!mesh)
            {
                continue;
            }
            pose_bounds.push_back(aabb_of(*mesh));
            meshes.push_back(std::move(*mesh));
        }
        if (meshes.empty())
        {
            std::println(stderr, "Failed to convert {} to MeshData", file.filename().string());
            continue;
        }

        auto bounds = pose_bounds.front();
        for (const auto& pose_bound : pose_bounds)
        {
            bounds = merge_aabb(bounds, pose_bound);
        }
        const auto extent = bounds.max - bounds.min;
        const auto render_vertices = meshes.front().vertices.size();
        const auto render_triangles = triangle_count(meshes.front());
        parsed_models.push_back(
            ParsedModel{
                .name = name,
                .meshes = std::move(meshes),
                .pose_bounds = std::move(pose_bounds),
                .bounds = bounds,
                .extent = extent,
                .max_extent = std::max({extent.x, extent.y, extent.z}),
                .source_vertices = static_cast<usize>(binary->header.num_verts),
                .source_triangles = static_cast<usize>(binary->header.num_tris),
                .source_frames = binary->frames.size(),
                .source_skins = binary->skins.size(),
                .render_vertices = render_vertices,
                .render_triangles = render_triangles,
            }
        );
    }

    if (parsed_models.empty())
    {
        std::println(stderr, "No MDL files could be converted to renderable meshes");
        return EXIT_FAILURE;
    }
    std::println("Loaded {} Quake MDL meshes", parsed_models.size());

    RuntimeConfig config{
        .window_title = "ds_vk Quake demo",
        .initial_width = 1280u,
        .initial_height = 820u,
        .clear_color = Color{0.018f, 0.022f, 0.027f, 1.0f},
    };

    Runtime runtime{std::move(config)};
    runtime.initialize();

    auto background_music =
        BackgroundMusic::load_looped_ogg(paths::music_dir / "track02.ogg", 0.38f);

    struct RenderModel
    {
        std::vector<MeshHandle> meshes{};
        Transform transform{};
        Sphere pick_sphere{};
        ObjectId object_id{};
        usize parsed_model_index{};
    };
    std::vector<RenderModel> render_models{};
    render_models.reserve(parsed_models.size());

    constexpr auto quake_mesh_scale = 0.1f;
    constexpr auto model_spacing = 0.75f;
    constexpr auto row_spacing = 2.2f;

    struct LayoutRow
    {
        std::vector<usize> model_indices{};
        f32 width{};
        f32 depth{};
    };

    const auto scaled_width_of = [&](const usize model_idx) -> f32
    { return std::max(0.35f, parsed_models[model_idx].extent.x * quake_mesh_scale); };
    const auto scaled_depth_of = [&](const usize model_idx) -> f32
    { return std::max(0.35f, parsed_models[model_idx].extent.y * quake_mesh_scale); };

    auto append_to_row = [&](LayoutRow& row, usize model_idx) -> void
    {
        if (!row.model_indices.empty())
        {
            row.width += model_spacing;
        }
        row.model_indices.push_back(model_idx);
        row.width += scaled_width_of(model_idx);
        row.depth = std::max(row.depth, scaled_depth_of(model_idx));
    };

    std::vector<usize> model_indices(parsed_models.size());
    for (auto i = 0zu; i < model_indices.size(); ++i)
    {
        model_indices[i] = i;
    }
    std::ranges::sort(
        model_indices,
        [&](usize a, usize b) { return parsed_models[a].max_extent > parsed_models[b].max_extent; }
    );

    std::array<LayoutRow, 5> rows{};
    const auto giant_count = std::min(2zu, model_indices.size());
    for (auto i = 0zu; i < giant_count; ++i)
    {
        append_to_row(rows[i], model_indices[i]);
    }
    for (auto i = giant_count; i < model_indices.size(); ++i)
    {
        auto target_row = 2zu;
        for (auto row_idx = 3zu; row_idx < rows.size(); ++row_idx)
        {
            if (rows[row_idx].width < rows[target_row].width)
            {
                target_row = row_idx;
            }
        }
        append_to_row(rows[target_row], model_indices[i]);
    }

    auto layout_width = 1.0f;
    auto layout_depth = 0.0f;
    for (const auto& row : rows)
    {
        if (row.model_indices.empty())
        {
            continue;
        }
        layout_width = std::max(layout_width, row.width);
        if (layout_depth > 0.0f)
        {
            layout_depth += row_spacing;
        }
        layout_depth += row.depth;
    }
    layout_depth = std::max(layout_depth, 1.0f);

    auto row_cursor_y = 0.5f * layout_depth;
    for (const auto& row : rows)
    {
        if (row.model_indices.empty())
        {
            continue;
        }
        const auto row_y = row_cursor_y - 0.5f * row.depth;
        auto row_cursor_x = -0.5f * row.width;
        for (const auto model_idx : row.model_indices)
        {
            const auto& model = parsed_models[model_idx];
            const auto local_center = 0.5f * (model.bounds.min + model.bounds.max);
            const auto scaled_width = scaled_width_of(model_idx);
            const auto target_center = Vec3{
                row_cursor_x + 0.5f * scaled_width,
                row_y,
                quake_mesh_scale * (local_center.z - model.bounds.min.z),
            };
            const auto transform = Transform{
                .translation =
                    Vec3{
                        target_center.x - quake_mesh_scale * local_center.x,
                        target_center.y - quake_mesh_scale * local_center.y,
                        -quake_mesh_scale * model.bounds.min.z,
                    },
                .scale = Vec3{quake_mesh_scale},
            };
            std::vector<MeshHandle> mesh_handles{};
            mesh_handles.reserve(model.meshes.size());
            for (const auto& mesh : model.meshes)
            {
                mesh_handles.push_back(runtime.upload_mesh(mesh));
            }
            render_models.push_back(
                RenderModel{
                    .meshes = std::move(mesh_handles),
                    .transform = transform,
                    .pick_sphere =
                        Sphere{
                            .center = target_center,
                            .radius = std::max(0.20f, 0.5f * model.max_extent * quake_mesh_scale),
                        },
                    .object_id = ObjectId{.value = static_cast<u32>(1000zu + model_idx)},
                    .parsed_model_index = model_idx,
                }
            );
            row_cursor_x += scaled_width + model_spacing;
        }
        row_cursor_y -= row.depth + row_spacing;
    }

    const auto scene_extent = std::max(layout_width, layout_depth);
    runtime.camera({
        .pivot = Vec3{0.0f, 0.0f, 2.5f},
        .distance = std::max(18.0f, scene_extent * 0.74f),
        .yaw = glm::radians(42.0f),
        .pitch = glm::radians(24.0f),
        .z_far = 5000.0f,
    });

    const auto floor_mesh_handle =
        runtime.upload_mesh(make_quad(std::max(24.0f, scene_extent + 8.0f), Color::white));

    const Material mdl_material{
        .base_color = Color{0.72f, 0.64f, 0.48f, 1.0f},
        .roughness = 0.92f,
    };
    const Material selected_mdl_material{
        .base_color = Color{1.0f, 0.46f, 0.08f, 1.0f},
        .emissive_color = Color{1.0f, 0.32f, 0.04f, 1.0f},
        .roughness = 0.62f,
    };
    const Material floor_material{
        .base_color = Color{0.48f, 0.52f, 0.48f, 1.0f},
        .roughness = 0.88f,
    };
    const EnvironmentConfig environment{
        .lighting_intensity = 0.10f,
        .background_intensity = 0.75f,
        .rotation_radians = glm::radians(22.0f),
        .visible_to_camera = true,
    };
    const Color ambient_light{0.030f, 0.036f, 0.046f, 1.0f};
    const auto shadow_extent = std::max(12.0f, scene_extent * 0.62f);
    const DirectionalLightConfig front_sun{
        .direction = normalize_or(Vec3{0.08f, -0.58f, -1.0f}, -k_axis_z),
        .color = {1.0f, 0.94f, 0.84f, 1.0f},
        .intensity = 3.1f,
        .shadow = LightShadowConfig{
            .enabled = true,
            .bias = 0.0040f,
            .strength = 0.78f,
            .near_plane = 0.05f,
            .far_plane = 80.0f,
            .ortho_extent = shadow_extent,
        },
    };
    const DirectionalLightConfig back_sun{
        .direction = normalize_or(Vec3{-0.35f, 0.42f, -0.82f}, -k_axis_z),
        .color = {0.56f, 0.72f, 1.0f, 1.0f},
        .intensity = 1.7f,
        .shadow = LightShadowConfig{
            .enabled = true,
            .bias = 0.0045f,
            .strength = 0.45f,
            .near_plane = 0.05f,
            .far_plane = 80.0f,
            .ortho_extent = shadow_extent,
        },
    };

    Picker picker{};
    auto selected_render_model = k_invalid_index;
    auto selection_window_open = false;
    constexpr auto animation_step_seconds = 0.1f;
    auto animation_accumulator_seconds = 0.0f;
    auto animation_step = 0zu;

    while (auto* frame = runtime.begin_frame())
    {
        if (background_music)
        {
            background_music->pump();
        }
        animation_accumulator_seconds += frame->dt_seconds;
        while (animation_accumulator_seconds >= animation_step_seconds)
        {
            animation_accumulator_seconds -= animation_step_seconds;
            ++animation_step;
        }

        picker.clear();
        for (auto i = 0zu; i < render_models.size(); ++i)
        {
            (void) picker.add_sphere({
                .object_id = render_models[i].object_id,
                .sub_index = static_cast<u32>(i),
                .sphere = render_models[i].pick_sphere,
            });
        }
        if (frame->input.left_click.occurred)
        {
            const auto hit = picker.click({
                .camera = frame->camera,
                .mouse_px = frame->input.left_click.position_px,
                .viewport_px = Vec2{
                    static_cast<f32>(frame->extent.width),
                    static_cast<f32>(frame->extent.height),
                },
            });
            if (hit.has_value() and static_cast<usize>(hit->sub_index) < render_models.size())
            {
                selected_render_model = static_cast<usize>(hit->sub_index);
                selection_window_open = true;
            }
            else
            {
                selected_render_model = k_invalid_index;
                selection_window_open = false;
            }
        }

        frame->draw.set_ambient_light(ambient_light);
        frame->draw.set_environment(environment);
        frame->draw.directional_light(front_sun);
        frame->draw.directional_light(back_sun);

        const auto pose_index_for = [&](const RenderModel& render_model) -> usize
        {
            if (render_model.meshes.size() <= 1zu)
            {
                return 0zu;
            }
            return animation_step % render_model.meshes.size();
        };

        frame->draw.draw_mesh({
            .mesh = floor_mesh_handle,
            .material = floor_material,
            .mask = {.shadow_producer = false},
        });
        for (auto i = 0zu; i < render_models.size(); ++i)
        {
            const auto selected = i == selected_render_model;
            const auto& render_model = render_models[i];
            const auto pose_index = pose_index_for(render_model);
            frame->draw.draw_mesh({
                .mesh = render_model.meshes[pose_index],
                .transform = render_model.transform,
                .material = selected ? selected_mdl_material : mdl_material,
                .debug =
                    selected
                        ? MeshDebugConfig{
                              .mode = MeshDebugMode::selected_pulse,
                              .color = Color{1.0f, 0.42f, 0.04f, 0.92f},
                              .selected = true,
                          }
                        : MeshDebugConfig{},
            });
            if (selected)
            {
                const auto& model = parsed_models[render_model.parsed_model_index];
                (void) viz::draw_aabb(
                    frame->draw,
                    viz::AabbMarkerConfig{
                        .aabb =
                            transformed_aabb(model.pose_bounds[pose_index], render_model.transform),
                        .color = Color{1.0f, 0.58f, 0.05f, 0.95f},
                        .width = 0.050f,
                        .draw_on_top = true,
                    }
                );
            }
        }
        if (runtime.ui_visible())
        {
            runtime.draw_runtime_ui();
            if (selection_window_open and selected_render_model != k_invalid_index
                and selected_render_model < render_models.size())
            {
                const auto& render_model = render_models[selected_render_model];
                const auto& model = parsed_models[render_model.parsed_model_index];
                const auto pose_index = pose_index_for(render_model);
                auto open = selection_window_open;
                ImGui::SetNextWindowPos(ImVec2{24.0f, 72.0f}, ImGuiCond_Once);
                ImGui::SetNextWindowSize(ImVec2{360.0f, 250.0f}, ImGuiCond_Once);
                if (ImGui::Begin("MDL Mesh", &open))
                {
                    ImGui::Text("Name: %s", model.name.c_str());
                    ImGui::Separator();
                    ImGui::Text("Source verts: %zu", model.source_vertices);
                    ImGui::Text("Source tris: %zu", model.source_triangles);
                    ImGui::Text("Frames/poses: %zu", model.source_frames);
                    ImGui::Text("Current pose: %zu / %zu", pose_index + 1zu, model.meshes.size());
                    ImGui::Text("Step time: %.2fs", static_cast<f64>(animation_step_seconds));
                    ImGui::Text("Skins: %zu", model.source_skins);
                    ImGui::Text("Render verts: %zu", model.render_vertices);
                    ImGui::Text("Render tris: %zu", model.render_triangles);
                    ImGui::Separator();
                    ImGui::Text(
                        "Bounds min: %.2f %.2f %.2f",
                        static_cast<f64>(model.bounds.min.x),
                        static_cast<f64>(model.bounds.min.y),
                        static_cast<f64>(model.bounds.min.z)
                    );
                    ImGui::Text(
                        "Bounds max: %.2f %.2f %.2f",
                        static_cast<f64>(model.bounds.max.x),
                        static_cast<f64>(model.bounds.max.y),
                        static_cast<f64>(model.bounds.max.z)
                    );
                    ImGui::Text(
                        "Extent: %.2f %.2f %.2f",
                        static_cast<f64>(model.extent.x),
                        static_cast<f64>(model.extent.y),
                        static_cast<f64>(model.extent.z)
                    );
                    ImGui::Text("Display scale: %.2f", static_cast<f64>(quake_mesh_scale));
                    ImGui::Text(
                        "Pick sphere: center %.2f %.2f %.2f, r %.2f",
                        static_cast<f64>(render_model.pick_sphere.center.x),
                        static_cast<f64>(render_model.pick_sphere.center.y),
                        static_cast<f64>(render_model.pick_sphere.center.z),
                        static_cast<f64>(render_model.pick_sphere.radius)
                    );
                    if (ImGui::Button("Clear selection"))
                    {
                        open = false;
                    }
                }
                ImGui::End();
                selection_window_open = open;
                if (!selection_window_open)
                {
                    selected_render_model = k_invalid_index;
                }
            }
        }
        runtime.render_shadow_pass();
        runtime.begin_main_pass();
        runtime.render_draw_list();
        runtime.render_imgui();
        runtime.end_main_pass();
        runtime.end_frame();
    }

    return EXIT_SUCCESS;
}
}  // namespace ds_vk_quake
