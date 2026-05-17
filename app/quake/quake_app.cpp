#include "app/quake/quake_app.hpp"

#include "app/quake/background_music.hpp"
#include "app/quake/io.hpp"
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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <imgui.h>
#include <limits>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ds_vk_quake
{
using namespace ds_vk;

#ifndef DS_VK_ASSET_DIR
#    define DS_VK_ASSET_DIR "assets"
#endif

namespace
{
constexpr auto k_quake_mesh_scale = 0.1f;
constexpr auto k_default_pose_interval_seconds = 0.1f;
constexpr auto k_invalid_triangle = std::numeric_limits<u32>::max();
constexpr Color k_quake_selection_color{1.0f, 0.42f, 0.04f, 0.92f};

enum class SceneMode : u8
{
    overview = 0,
    model_viewer = 1,
};

struct PaletteVariant
{
    std::string name{};
    MdlPalette palette{};
};

struct SkinRecord
{
    usize model_index{};
    usize skin_index{};
    usize image_index{};
    std::string label{};
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    Vec2 uv_min{};
    Vec2 uv_max{};
};

struct SkinAtlasCpu
{
    u32 width{};
    u32 height{};
    std::vector<SkinRecord> records{};
    std::vector<std::vector<ColorU8>> pixels_by_palette{};
};

struct SkinAtlasGpu
{
    u32 width{};
    u32 height{};
    std::vector<SkinRecord> records{};
    std::vector<TextureHandle> textures{};
};

struct ParsedModel
{
    std::string name{};
    MdlHeader header{};
    std::vector<MdlSkin> skins{};
    std::vector<MeshData> meshes{};
    std::vector<Aabb> pose_bounds{};
    std::vector<std::string> pose_names{};
    std::vector<f32> pose_intervals{};
    std::vector<usize> skin_record_indices{};
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

struct RenderModel
{
    std::vector<MeshHandle> meshes{};
    Transform transform{};
    Sphere pick_sphere{};
    ObjectId object_id{};
    usize parsed_model_index{};
};

struct ViewerState
{
    usize model_index{};
    usize skin_record_index{};
    usize pose_index{};
    f32 pose_timer{};
    f32 playback_speed{1.0f};
    bool paused{};
    bool wireframe{};
    bool show_skin{true};
    int skin_magnification{2};
    std::vector<MeshHandle> pose_meshes{};
    usize pose_mesh_count{};
    usize pose_mesh_model_index{k_invalid_index};
    usize pose_mesh_skin_record_index{k_invalid_index};
    u32 selected_triangle{k_invalid_triangle};
};

struct ShowcaseMeshes
{
    MeshHandle floor{};
    MeshHandle sphere{};
    MeshHandle cube{};
};

[[nodiscard]] auto asset_path(const std::filesystem::path& relative) -> std::filesystem::path
{
    return std::filesystem::path{DS_VK_ASSET_DIR} / relative;
}

[[nodiscard]] auto transformed_aabb(const Aabb& aabb, const Transform& transform) -> Aabb
{
    const auto matrix = transform.matrix();
    const auto transform_point = [&](const Vec3 point) -> Vec3
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

[[nodiscard]] auto transform_point(const Transform& transform, const Vec3 point) -> Vec3
{
    return Vec3{transform.matrix() * Vec4{point, 1.0f}};
}

[[nodiscard]] auto merge_aabb(const Aabb& a, const Aabb& b) -> Aabb
{
    return Aabb{
        .min = glm::min(a.min, b.min),
        .max = glm::max(a.max, b.max),
    };
}

[[nodiscard]] auto frame_name(const MdlFrame& frame) -> std::string
{
    auto length = 0zu;
    while (length < frame.name.size() and frame.name[length] != '\0')
    {
        ++length;
    }
    return std::string{frame.name.data(), length};
}

[[nodiscard]] auto color_from_palette(const MdlPalette& palette, const u8 palette_index) -> ColorU8
{
    const auto& entry = palette[palette_index];
    return ColorU8{entry.r, entry.g, entry.b, 255u};
}

[[nodiscard]] auto scale_channel(const u8 value, const f32 multiplier, const f32 offset = 0.0f)
    -> u8
{
    return color_channel_to_u8(
        std::clamp(static_cast<f32>(value) / 255.0f * multiplier + offset, 0.0f, 1.0f)
    );
}

[[nodiscard]] auto make_warm_palette(const MdlPalette& source) -> MdlPalette
{
    auto out = source;
    for (auto& entry : out)
    {
        entry.r = scale_channel(entry.r, 1.12f, 0.010f);
        entry.g = scale_channel(entry.g, 0.98f);
        entry.b = scale_channel(entry.b, 0.84f);
    }
    return out;
}

[[nodiscard]] auto make_moonlit_palette(const MdlPalette& source) -> MdlPalette
{
    auto out = source;
    for (auto& entry : out)
    {
        entry.r = scale_channel(entry.r, 0.76f);
        entry.g = scale_channel(entry.g, 0.90f, 0.006f);
        entry.b = scale_channel(entry.b, 1.20f, 0.010f);
    }
    return out;
}

[[nodiscard]] auto make_high_contrast_palette(const MdlPalette& source) -> MdlPalette
{
    auto out = source;
    for (auto& entry : out)
    {
        const auto transform = [](const u8 value) -> u8
        {
            const auto linear = static_cast<f32>(value) / 255.0f;
            const auto contrasted = std::clamp((linear - 0.5f) * 1.22f + 0.5f, 0.0f, 1.0f);
            return color_channel_to_u8(std::pow(contrasted, 0.92f));
        };
        entry.r = transform(entry.r);
        entry.g = transform(entry.g);
        entry.b = transform(entry.b);
    }
    return out;
}

[[nodiscard]] auto load_palette_lmp(const std::filesystem::path& filepath)
    -> std::optional<MdlPalette>
{
    const auto buf = load_binary_file(filepath);
    if (!buf or buf->size() != sizeof(MdlPalette))
    {
        return std::nullopt;
    }

    MdlPalette palette{};
    std::memcpy(palette.data(), buf->data(), sizeof(MdlPalette));
    return palette;
}

[[nodiscard]] auto load_palette_variants(const MdlPalette& base_palette)
    -> std::vector<PaletteVariant>
{
    std::vector<PaletteVariant> out{
        PaletteVariant{.name = "Quake palette", .palette = base_palette},
    };

    std::error_code ec{};
    if (std::filesystem::exists(paths::palette_dir, ec))
    {
        for (const auto& entry : std::filesystem::directory_iterator{paths::palette_dir, ec})
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const auto path = entry.path();
            if (path.extension() != ".lmp" and path.extension() != ".pal")
            {
                continue;
            }
            if (auto palette = load_palette_lmp(path))
            {
                out.push_back(PaletteVariant{.name = path.stem().string(), .palette = *palette});
            }
        }
    }

    out.push_back(PaletteVariant{.name = "Warm remap", .palette = make_warm_palette(base_palette)});
    out.push_back(
        PaletteVariant{.name = "Moonlit remap", .palette = make_moonlit_palette(base_palette)}
    );
    out.push_back(
        PaletteVariant{
            .name = "High contrast remap",
            .palette = make_high_contrast_palette(base_palette),
        }
    );
    return out;
}

auto write_skin_atlas_padding(
    std::span<ColorU8> atlas,
    const u32 atlas_width,
    const SkinRecord& record,
    std::span<const u8> skin,
    const MdlPalette& palette
) -> void
{
    constexpr auto pad = 2;
    for (auto dy = -pad; dy < static_cast<int>(record.height) + pad; ++dy)
    {
        const auto src_y = std::clamp(dy, 0, static_cast<int>(record.height) - 1);
        const auto dst_y = static_cast<u32>(static_cast<int>(record.y) + dy);
        for (auto dx = -pad; dx < static_cast<int>(record.width) + pad; ++dx)
        {
            const auto src_x = std::clamp(dx, 0, static_cast<int>(record.width) - 1);
            const auto dst_x = static_cast<u32>(static_cast<int>(record.x) + dx);
            const auto src_idx = static_cast<usize>(src_y) * static_cast<usize>(record.width)
                                 + static_cast<usize>(src_x);
            const auto dst_idx = static_cast<usize>(dst_y) * static_cast<usize>(atlas_width)
                                 + static_cast<usize>(dst_x);
            atlas[dst_idx] = color_from_palette(palette, skin[src_idx]);
        }
    }
}

[[nodiscard]] auto build_skin_atlas_cpu(
    std::vector<ParsedModel>& models, std::span<const PaletteVariant> palette_variants
) -> SkinAtlasCpu
{
    constexpr auto atlas_width = 2048u;
    constexpr auto padding = 2u;

    SkinAtlasCpu atlas{.width = atlas_width};
    auto cursor_x = padding;
    auto cursor_y = padding;
    auto row_height = 0u;

    for (auto model_idx = 0zu; model_idx < models.size(); ++model_idx)
    {
        auto& model = models[model_idx];
        model.skin_record_indices.clear();
        for (auto skin_idx = 0zu; skin_idx < model.skins.size(); ++skin_idx)
        {
            const auto& skin = model.skins[skin_idx];
            for (auto image_idx = 0zu; image_idx < skin.images.size(); ++image_idx)
            {
                const auto width = static_cast<u32>(model.header.skin_width);
                const auto height = static_cast<u32>(model.header.skin_height);
                if (cursor_x + width + padding > atlas_width)
                {
                    cursor_x = padding;
                    cursor_y += row_height + 2u * padding;
                    row_height = 0u;
                }

                const auto record_index = atlas.records.size();
                model.skin_record_indices.push_back(record_index);
                atlas.records.push_back(
                    SkinRecord{
                        .model_index = model_idx,
                        .skin_index = skin_idx,
                        .image_index = image_idx,
                        .label = skin.images.size() == 1zu
                                     ? std::format("{} skin {}", model.name, skin_idx)
                                     : std::format(
                                           "{} skin {} frame {}", model.name, skin_idx, image_idx
                                       ),
                        .x = cursor_x,
                        .y = cursor_y,
                        .width = width,
                        .height = height,
                        .uv_min = Vec2{
                            static_cast<f32>(cursor_x) / static_cast<f32>(atlas_width),
                            static_cast<f32>(cursor_y) / static_cast<f32>(atlas_width),
                        },
                    }
                );
                cursor_x += width + 2u * padding;
                row_height = std::max(row_height, height);
            }
        }
    }

    atlas.height = cursor_y + row_height + padding;
    for (auto& record : atlas.records)
    {
        record.uv_min = Vec2{
            static_cast<f32>(record.x) / static_cast<f32>(atlas.width),
            static_cast<f32>(record.y) / static_cast<f32>(atlas.height),
        };
        record.uv_max = Vec2{
            static_cast<f32>(record.x + record.width) / static_cast<f32>(atlas.width),
            static_cast<f32>(record.y + record.height) / static_cast<f32>(atlas.height),
        };
    }

    atlas.pixels_by_palette.reserve(palette_variants.size());
    for (const auto& palette_variant : palette_variants)
    {
        auto& pixels = atlas.pixels_by_palette.emplace_back(
            static_cast<usize>(atlas.width) * static_cast<usize>(atlas.height),
            ColorU8{0u, 0u, 0u, 255u}
        );
        for (const auto& record : atlas.records)
        {
            const auto& model = models[record.model_index];
            const auto& skin = model.skins[record.skin_index].images[record.image_index];
            write_skin_atlas_padding(pixels, atlas.width, record, skin, palette_variant.palette);
        }
    }

    return atlas;
}

[[nodiscard]] auto upload_skin_atlas(Runtime& runtime, SkinAtlasCpu&& cpu) -> SkinAtlasGpu
{
    SkinAtlasGpu out{
        .width = cpu.width,
        .height = cpu.height,
        .records = std::move(cpu.records),
    };
    out.textures.reserve(cpu.pixels_by_palette.size());
    for (const auto& pixels : cpu.pixels_by_palette)
    {
        out.textures.push_back(runtime.upload_texture_rgba(
            pixels, out.width, out.height, TextureLoadConfig{.srgb = true}
        ));
    }
    return out;
}

[[nodiscard]] auto mesh_with_skin_record(const MeshData& source, const SkinRecord& record)
    -> MeshData
{
    auto out = source;
    const auto uv_scale = record.uv_max - record.uv_min;
    for (auto& vertex : out.vertices)
    {
        vertex.color = Color::white;
        vertex.texcoord = record.uv_min + vertex.texcoord * uv_scale;
    }
    return out;
}

[[nodiscard]] auto parse_models(const std::vector<std::filesystem::path>& mdl_files)
    -> std::vector<ParsedModel>
{
    std::vector<ParsedModel> parsed_models{};
    parsed_models.reserve(mdl_files.size());
    for (const auto& file : mdl_files)
    {
        const auto name = file.stem().string();
        auto binary = parse_mdl_binary(file);
        if (!binary)
        {
            continue;
        }
        if (binary->frames.empty())
        {
            std::println(stderr, "{} has no renderable frames", file.filename().string());
            continue;
        }

        std::vector<MeshData> meshes{};
        std::vector<Aabb> pose_bounds{};
        std::vector<std::string> pose_names{};
        std::vector<f32> pose_intervals{};
        meshes.reserve(binary->frames.size());
        pose_bounds.reserve(binary->frames.size());
        pose_names.reserve(binary->frames.size());
        pose_intervals.reserve(binary->frames.size());
        for (const auto& mdl_frame : binary->frames)
        {
            auto mesh =
                make_mesh_data(binary->header, binary->stverts, binary->triangles, mdl_frame);
            if (!mesh)
            {
                continue;
            }
            pose_bounds.push_back(aabb_of(*mesh));
            pose_names.push_back(frame_name(mdl_frame));
            pose_intervals.push_back(
                mdl_frame.interval > 0.0f ? mdl_frame.interval : k_default_pose_interval_seconds
            );
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
                .header = binary->header,
                .skins = std::move(binary->skins),
                .meshes = std::move(meshes),
                .pose_bounds = std::move(pose_bounds),
                .pose_names = std::move(pose_names),
                .pose_intervals = std::move(pose_intervals),
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
    return parsed_models;
}

[[nodiscard]] auto find_mdl_files() -> std::vector<std::filesystem::path>
{
    std::vector<std::filesystem::path> out{};
    std::error_code ec{};
    for (const auto& entry : std::filesystem::directory_iterator{paths::progs_dir, ec})
    {
        if (entry.is_regular_file() and entry.path().extension() == ".mdl")
        {
            out.push_back(entry.path());
        }
    }
    std::ranges::sort(out);
    return out;
}

auto dump_skins_only(const std::vector<std::filesystem::path>& mdl_files) -> int
{
    auto parsed_mdl_count = 0zu;
    auto dumped_skin_count = 0zu;
    auto dumped_palette = false;
    for (const auto& file : mdl_files)
    {
        const auto name = file.stem().string();
        auto binary = parse_mdl_binary(file);
        if (!binary)
        {
            continue;
        }
        ++parsed_mdl_count;
        dumped_skin_count +=
            save_mdl_skins_to_file(binary->skins, binary->header, binary->palette, name);
        if (!dumped_palette)
        {
            dumped_palette = save_quake_palette_to_file(binary->palette);
        }
    }

    std::println(
        "Dumped {} colored MDL skin image(s) from {} parsed MDL file(s) to {}",
        dumped_skin_count,
        parsed_mdl_count,
        paths::skin_output_dir.string()
    );
    if (dumped_palette)
    {
        std::println(
            "Wrote palette preview to {}", (paths::skin_output_dir / "quake_palette.ppm").string()
        );
    }
    return dumped_skin_count > 0zu and dumped_palette ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto dump_gltf_only(const std::vector<std::filesystem::path>& mdl_files) -> int
{
    const auto alias_normals = load_alias_normals(paths::anorms_path);
    if (!alias_normals)
    {
        return EXIT_FAILURE;
    }

    std::error_code ec{};
    std::filesystem::create_directories(paths::gltf_output_dir, ec);
    if (ec)
    {
        std::println(
            stderr,
            "Failed to create glTF output directory {}: {}",
            paths::gltf_output_dir.string(),
            ec.message()
        );
        return EXIT_FAILURE;
    }

    auto parsed_mdl_count = 0zu;
    auto written_gltf_count = 0zu;
    for (const auto& file : mdl_files)
    {
        auto binary = parse_mdl_binary(file);
        if (!binary)
        {
            continue;
        }
        ++parsed_mdl_count;
        const auto name = file.stem().string();
        if (save_mdl_as_gltf(
                *binary, name, paths::gltf_output_dir / std::format("{}.gltf", name), *alias_normals
            ))
        {
            ++written_gltf_count;
        }
    }

    std::println(
        "Wrote {} glTF model file(s) from {} parsed MDL file(s) to {}",
        written_gltf_count,
        parsed_mdl_count,
        paths::gltf_output_dir.string()
    );
    return written_gltf_count > 0zu ? EXIT_SUCCESS : EXIT_FAILURE;
}

[[nodiscard]] auto pose_duration(const ParsedModel& model, const usize pose_index) -> f32
{
    if (pose_index >= model.pose_intervals.size())
    {
        return k_default_pose_interval_seconds;
    }
    return std::max(0.001f, model.pose_intervals[pose_index]);
}

auto advance_pose(ViewerState& viewer, const ParsedModel& model, const f32 dt_seconds) -> void
{
    if (viewer.paused or model.meshes.size() <= 1zu)
    {
        return;
    }
    viewer.pose_timer += dt_seconds * std::max(0.0f, viewer.playback_speed);
    while (viewer.pose_timer >= pose_duration(model, viewer.pose_index))
    {
        viewer.pose_timer -= pose_duration(model, viewer.pose_index);
        viewer.pose_index = (viewer.pose_index + 1zu) % model.meshes.size();
    }
}

auto advance_overview_pose(usize& pose_step, f32& accumulator, const f32 dt_seconds) -> void
{
    accumulator += dt_seconds;
    while (accumulator >= k_default_pose_interval_seconds)
    {
        accumulator -= k_default_pose_interval_seconds;
        ++pose_step;
    }
}

[[nodiscard]] auto viewer_transform_for(const ParsedModel& model) -> Transform
{
    const auto center = 0.5f * (model.bounds.min + model.bounds.max);
    return Transform{
        .translation =
            Vec3{
                -k_quake_mesh_scale * center.x,
                -k_quake_mesh_scale * center.y,
                -k_quake_mesh_scale * model.bounds.min.z,
            },
        .scale = Vec3{k_quake_mesh_scale},
    };
}

auto configure_viewer_camera(Runtime& runtime, const ParsedModel& model) -> void
{
    const auto display_extent = std::max(0.5f, model.max_extent * k_quake_mesh_scale);
    const auto display_height = std::max(0.5f, model.extent.z * k_quake_mesh_scale);
    runtime.camera({
        .pivot = Vec3{0.0f, 0.0f, display_height * 0.52f},
        .distance = std::max(3.0f, display_extent * 2.3f),
        .yaw = glm::radians(44.0f),
        .pitch = glm::radians(20.0f),
        .z_near = 0.01f,
        .z_far = 5000.0f,
        .allow_pivot_move = false,
        .clamp_position_z_min = true,
        .min_position_z = 0.0f,
    });
}

auto configure_overview_camera(Runtime& runtime, const f32 scene_extent) -> void
{
    runtime.camera({
        .pivot = Vec3{0.0f, 0.0f, 2.5f},
        .distance = std::max(18.0f, scene_extent * 0.74f),
        .yaw = glm::radians(42.0f),
        .pitch = glm::radians(24.0f),
        .z_far = 5000.0f,
        .allow_pivot_move = true,
    });
}

[[nodiscard]] auto make_common_lighting(const f32 scene_extent)
{
    struct Lighting
    {
        Color ambient{};
        DirectionalLightConfig front_sun{};
        DirectionalLightConfig back_sun{};
    };

    const auto shadow_extent = std::max(12.0f, scene_extent * 0.62f);
    return Lighting{
        .ambient = Color{0.55f, 0.50f, 0.42f, 1.0f},
        .front_sun =
            DirectionalLightConfig{
                .direction = normalize_or(Vec3{0.10f, -0.58f, -1.0f}, -k_axis_z),
                .color = {1.0f, 0.97f, 0.90f, 1.0f},
                .intensity = 6.2f,
                .shadow =
                    LightShadowConfig{
                        .enabled = true,
                        .bias = 0.0040f,
                        .strength = 0.18f,
                        .near_plane = 0.05f,
                        .far_plane = 120.0f,
                        .ortho_extent = shadow_extent,
                    },
            },
        .back_sun = DirectionalLightConfig{
            .direction = normalize_or(Vec3{-0.42f, 0.36f, -0.88f}, -k_axis_z),
            .color = {0.82f, 0.90f, 1.0f, 1.0f},
            .intensity = 4.0f,
            .shadow = LightShadowConfig{
                .enabled = true,
                .bias = 0.0045f,
                .strength = 0.10f,
                .near_plane = 0.05f,
                .far_plane = 120.0f,
                .ortho_extent = shadow_extent,
            },
        },
    };
}

auto apply_lighting(DrawList& draw, const auto& lighting) -> void
{
    draw.set_ambient_light(lighting.ambient);
    draw.set_environment({});
    draw.directional_light(lighting.front_sun);
    draw.directional_light(lighting.back_sun);
}

auto draw_showcase_floor(
    DrawList& draw,
    const ShowcaseMeshes& meshes,
    const Material& floor_material,
    const f32 floor_scale = 1.0f
) -> void
{
    draw.draw_mesh({
        .mesh = meshes.floor,
        .transform = Transform{.scale = Vec3{floor_scale}},
        .material = floor_material,
        .mask = {.shadow_producer = false},
    });
}

auto draw_showcase_objects(
    DrawList& draw,
    const ShowcaseMeshes& meshes,
    const f32 scene_extent,
    const Material& floor_material,
    const Material& sphere_material,
    const Material& cube_material
) -> void
{
    draw_showcase_floor(draw, meshes, floor_material, 3.0f);
    draw.draw_mesh({
        .mesh = meshes.sphere,
        .transform =
            Transform{
                .translation =
                    Vec3{
                        std::max(2.2f, scene_extent * 0.18f),
                        -std::max(2.2f, scene_extent * 0.16f),
                        0.55f,
                    },
                .scale = Vec3{0.55f},
            },
        .material = sphere_material,
    });
    draw.draw_mesh({
        .mesh = meshes.cube,
        .transform =
            Transform{
                .translation =
                    Vec3{
                        -std::max(2.6f, scene_extent * 0.20f),
                        -std::max(2.0f, scene_extent * 0.14f),
                        0.34f,
                    },
                .rotation = glm::angleAxis(glm::radians(24.0f), k_axis_z),
                .scale = Vec3{0.36f},
            },
        .material = cube_material,
    });
}

[[nodiscard]] auto build_render_models(
    Runtime& runtime, const std::vector<ParsedModel>& parsed_models, const SkinAtlasGpu& atlas
) -> std::vector<RenderModel>
{
    constexpr auto model_spacing = 0.75f;
    constexpr auto row_spacing = 2.2f;

    struct LayoutRow
    {
        std::vector<usize> model_indices{};
        f32 width{};
        f32 depth{};
    };

    const auto scaled_width_of = [&](const usize model_idx) -> f32
    { return std::max(0.35f, parsed_models[model_idx].extent.x * k_quake_mesh_scale); };
    const auto scaled_depth_of = [&](const usize model_idx) -> f32
    { return std::max(0.35f, parsed_models[model_idx].extent.y * k_quake_mesh_scale); };

    auto append_to_row = [&](LayoutRow& row, const usize model_idx) -> void
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
        [&](const usize a, const usize b)
        { return parsed_models[a].max_extent > parsed_models[b].max_extent; }
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

    auto layout_depth = 0.0f;
    for (const auto& row : rows)
    {
        if (row.model_indices.empty())
        {
            continue;
        }
        if (layout_depth > 0.0f)
        {
            layout_depth += row_spacing;
        }
        layout_depth += row.depth;
    }
    layout_depth = std::max(layout_depth, 1.0f);

    std::vector<RenderModel> render_models{};
    render_models.reserve(parsed_models.size());

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
                k_quake_mesh_scale * (local_center.z - model.bounds.min.z),
            };
            const auto transform = Transform{
                .translation =
                    Vec3{
                        target_center.x - k_quake_mesh_scale * local_center.x,
                        target_center.y - k_quake_mesh_scale * local_center.y,
                        -k_quake_mesh_scale * model.bounds.min.z,
                    },
                .scale = Vec3{k_quake_mesh_scale},
            };
            const auto skin_record_index =
                model.skin_record_indices.empty() ? 0zu : model.skin_record_indices.front();
            const auto& skin_record = atlas.records[skin_record_index];
            std::vector<MeshHandle> mesh_handles{};
            mesh_handles.reserve(model.meshes.size());
            for (const auto& mesh : model.meshes)
            {
                mesh_handles.push_back(
                    runtime.upload_mesh(mesh_with_skin_record(mesh, skin_record))
                );
            }
            render_models.push_back(
                RenderModel{
                    .meshes = std::move(mesh_handles),
                    .transform = transform,
                    .pick_sphere =
                        Sphere{
                            .center = target_center,
                            .radius = std::max(0.20f, 0.5f * model.max_extent * k_quake_mesh_scale),
                        },
                    .object_id = ObjectId{.value = static_cast<u32>(1000zu + model_idx)},
                    .parsed_model_index = model_idx,
                }
            );
            row_cursor_x += scaled_width + model_spacing;
        }
        row_cursor_y -= row.depth + row_spacing;
    }

    return render_models;
}

[[nodiscard]] auto scene_extent_for(const std::vector<RenderModel>& render_models) -> f32
{
    if (render_models.empty())
    {
        return 8.0f;
    }
    Aabb bounds{
        .min = Vec3{std::numeric_limits<f32>::max()},
        .max = Vec3{std::numeric_limits<f32>::lowest()},
    };
    for (const auto& render_model : render_models)
    {
        bounds.min = glm::min(
            bounds.min, render_model.pick_sphere.center - Vec3{render_model.pick_sphere.radius}
        );
        bounds.max = glm::max(
            bounds.max, render_model.pick_sphere.center + Vec3{render_model.pick_sphere.radius}
        );
    }
    const auto extent = bounds.max - bounds.min;
    return std::max(8.0f, std::max(extent.x, extent.y));
}

auto draw_palette_combo(
    const std::vector<PaletteVariant>& palettes, usize& palette_index, bool& changed
) -> void
{
    changed = false;
    if (palettes.empty())
    {
        return;
    }
    const auto preview = palettes[palette_index].name.c_str();
    if (ImGui::BeginCombo("Palette", preview))
    {
        for (auto i = 0zu; i < palettes.size(); ++i)
        {
            const auto selected = i == palette_index;
            if (ImGui::Selectable(palettes[i].name.c_str(), selected))
            {
                palette_index = i;
                changed = true;
            }
            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

auto draw_overview_selection_window(
    bool& selection_window_open,
    usize& selected_render_model,
    const std::vector<RenderModel>& render_models,
    const std::vector<ParsedModel>& parsed_models,
    const usize pose_index
) -> void
{
    if (!selection_window_open or selected_render_model == k_invalid_index
        or selected_render_model >= render_models.size())
    {
        return;
    }

    const auto& render_model = render_models[selected_render_model];
    const auto& model = parsed_models[render_model.parsed_model_index];
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
        ImGui::Text("Display scale: %.2f", static_cast<f64>(k_quake_mesh_scale));
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

[[nodiscard]] auto triangle_intersection(const Ray& ray, Vec3 a, Vec3 b, Vec3 c)
    -> std::optional<f32>
{
    constexpr auto epsilon = 1.0e-7f;
    const auto edge1 = b - a;
    const auto edge2 = c - a;
    const auto h = glm::cross(ray.direction, edge2);
    const auto det = glm::dot(edge1, h);
    if (std::abs(det) < epsilon)
    {
        return std::nullopt;
    }
    const auto inv_det = 1.0f / det;
    const auto s = ray.origin - a;
    const auto u = inv_det * glm::dot(s, h);
    if (u < 0.0f or u > 1.0f)
    {
        return std::nullopt;
    }
    const auto q = glm::cross(s, edge1);
    const auto v = inv_det * glm::dot(ray.direction, q);
    if (v < 0.0f or u + v > 1.0f)
    {
        return std::nullopt;
    }
    const auto t = inv_det * glm::dot(edge2, q);
    if (t < 0.0f)
    {
        return std::nullopt;
    }
    return t;
}

auto select_viewer_triangle(
    ViewerState& viewer,
    const ParsedModel& model,
    const FrameContext& frame,
    const Transform& transform
) -> void
{
    if (!frame.input.left_click.occurred or frame.input.mouse_captured_by_ui or viewer.wireframe)
    {
        return;
    }
    const auto& mesh = model.meshes[viewer.pose_index];
    const auto ray = make_camera_ray(
        frame.camera,
        frame.input.left_click.position_px,
        Vec2{static_cast<f32>(frame.extent.width), static_cast<f32>(frame.extent.height)}
    );

    auto best_t = std::numeric_limits<f32>::max();
    auto best_triangle = k_invalid_triangle;
    const auto tri_count = mesh.indices.size() / 3zu;
    for (auto tri = 0zu; tri < tri_count; ++tri)
    {
        const auto i0 = mesh.indices[tri * 3zu + 0zu];
        const auto i1 = mesh.indices[tri * 3zu + 1zu];
        const auto i2 = mesh.indices[tri * 3zu + 2zu];
        const auto a = transform_point(transform, mesh.vertices[i0].position);
        const auto b = transform_point(transform, mesh.vertices[i1].position);
        const auto c = transform_point(transform, mesh.vertices[i2].position);
        const auto hit = triangle_intersection(ray, a, b, c);
        if (hit and *hit < best_t)
        {
            best_t = *hit;
            best_triangle = static_cast<u32>(tri);
        }
    }
    viewer.selected_triangle = best_triangle;
}

auto draw_wireframe_mesh(DrawList& draw, const MeshData& mesh, const Transform& transform) -> void
{
    const auto tri_count = mesh.indices.size() / 3zu;
    for (auto tri = 0zu; tri < tri_count; ++tri)
    {
        const std::array indices{
            mesh.indices[tri * 3zu + 0zu],
            mesh.indices[tri * 3zu + 1zu],
            mesh.indices[tri * 3zu + 2zu],
        };
        const std::array points{
            transform_point(transform, mesh.vertices[indices[0]].position),
            transform_point(transform, mesh.vertices[indices[1]].position),
            transform_point(transform, mesh.vertices[indices[2]].position),
        };
        draw.debug_line({
            .start = points[0],
            .end = points[1],
            .color = Color{0.85f, 0.90f, 1.0f, 1.0f},
            .width = 0.004f,
        });
        draw.debug_line({
            .start = points[1],
            .end = points[2],
            .color = Color{0.85f, 0.90f, 1.0f, 1.0f},
            .width = 0.004f,
        });
        draw.debug_line({
            .start = points[2],
            .end = points[0],
            .color = Color{0.85f, 0.90f, 1.0f, 1.0f},
            .width = 0.004f,
        });
    }
}

auto draw_skin_uv_overlay(
    const ParsedModel& model, const ViewerState& viewer, ImVec2 image_min, ImVec2 image_size
) -> void
{
    if (viewer.selected_triangle == k_invalid_triangle)
    {
        return;
    }
    const auto& mesh = model.meshes[viewer.pose_index];
    const auto base = static_cast<usize>(viewer.selected_triangle) * 3zu;
    if (base + 2zu >= mesh.indices.size())
    {
        return;
    }
    const auto uv_to_screen = [&](const Vec2 uv) -> ImVec2
    {
        return ImVec2{
            image_min.x + uv.x * image_size.x,
            image_min.y + uv.y * image_size.y,
        };
    };
    const auto p0 = uv_to_screen(mesh.vertices[mesh.indices[base + 0zu]].texcoord);
    const auto p1 = uv_to_screen(mesh.vertices[mesh.indices[base + 1zu]].texcoord);
    const auto p2 = uv_to_screen(mesh.vertices[mesh.indices[base + 2zu]].texcoord);
    auto* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddTriangleFilled(p0, p1, p2, IM_COL32(255, 122, 14, 110));
    draw_list->AddTriangle(p0, p1, p2, IM_COL32(255, 122, 14, 255), 2.0f);
}

[[nodiscard]] auto imgui_texture(TextureHandle handle, Runtime& runtime) -> ImTextureID
{
    return static_cast<ImTextureID>(runtime.imgui_texture_id(handle));
}

auto draw_skin_texture_window(
    Runtime& runtime,
    const SkinAtlasGpu& atlas,
    const std::vector<ParsedModel>& models,
    const ViewerState& viewer,
    const usize palette_index
) -> void
{
    if (atlas.records.empty() or viewer.skin_record_index >= atlas.records.size())
    {
        return;
    }
    const auto& record = atlas.records[viewer.skin_record_index];
    const auto& model = models[viewer.model_index];
    const auto texture = imgui_texture(atlas.textures[palette_index], runtime);
    const auto magnification = static_cast<f32>(std::max(1, viewer.skin_magnification));
    const ImVec2 size{
        static_cast<f32>(record.width) * magnification,
        static_cast<f32>(record.height) * magnification,
    };

    ImGui::SetNextWindowSize(ImVec2{420.0f, 440.0f}, ImGuiCond_Once);
    if (ImGui::Begin("Skin Texture"))
    {
        ImGui::Text("%s", record.label.c_str());
        ImGui::Text("%u x %u", record.width, record.height);
        ImGui::Separator();
        ImGui::Image(
            texture,
            size,
            ImVec2{record.uv_min.x, record.uv_min.y},
            ImVec2{record.uv_max.x, record.uv_max.y}
        );
        draw_skin_uv_overlay(model, viewer, ImGui::GetItemRectMin(), size);
    }
    ImGui::End();
}

auto ensure_viewer_pose_meshes(
    Runtime& runtime, ViewerState& viewer, const ParsedModel& model, const SkinRecord& skin_record
) -> void
{
    if (viewer.pose_mesh_model_index == viewer.model_index
        and viewer.pose_mesh_skin_record_index == viewer.skin_record_index
        and viewer.pose_mesh_count == model.meshes.size())
    {
        return;
    }

    if (viewer.pose_meshes.size() < model.meshes.size())
    {
        viewer.pose_meshes.resize(model.meshes.size());
    }
    for (auto i = 0zu; i < model.meshes.size(); ++i)
    {
        const auto mesh = mesh_with_skin_record(model.meshes[i], skin_record);
        viewer.pose_meshes[i] = viewer.pose_meshes[i].valid()
                                    ? runtime.replace_mesh(viewer.pose_meshes[i], mesh)
                                    : runtime.upload_mesh(mesh);
    }
    viewer.pose_mesh_count = model.meshes.size();
    viewer.pose_mesh_model_index = viewer.model_index;
    viewer.pose_mesh_skin_record_index = viewer.skin_record_index;
}

auto draw_model_viewer_window(
    Runtime& runtime,
    const SkinAtlasGpu& atlas,
    const std::vector<ParsedModel>& models,
    ViewerState& viewer,
    usize& palette_index,
    const std::vector<PaletteVariant>& palettes,
    bool& palette_changed,
    bool& viewer_camera_requested
) -> void
{
    const auto& model = models[viewer.model_index];
    ImGui::SetNextWindowPos(ImVec2{24.0f, 72.0f}, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2{470.0f, 560.0f}, ImGuiCond_Once);
    if (ImGui::Begin("Quake Model Viewer"))
    {
        ImGui::Text("Model: %s", model.name.c_str());
        ImGui::Text(
            "Verts %zu | tris %zu | poses %zu | skins %zu",
            model.source_vertices,
            model.source_triangles,
            model.meshes.size(),
            model.source_skins
        );
        ImGui::Separator();
        draw_palette_combo(palettes, palette_index, palette_changed);
        ImGui::Checkbox("Pause", &viewer.paused);
        ImGui::SameLine();
        ImGui::Checkbox("Wireframe", &viewer.wireframe);
        ImGui::SameLine();
        ImGui::Checkbox("Show skin", &viewer.show_skin);
        ImGui::SliderFloat("Playback speed", &viewer.playback_speed, 0.0f, 4.0f, "%.2fx");
        auto pose = static_cast<int>(viewer.pose_index);
        if (ImGui::SliderInt("Pose", &pose, 0, static_cast<int>(model.meshes.size() - 1zu)))
        {
            viewer.pose_index = static_cast<usize>(pose);
            viewer.pose_timer = 0.0f;
        }
        if (!model.pose_names.empty())
        {
            ImGui::Text("Pose name: %s", model.pose_names[viewer.pose_index].c_str());
        }
        ImGui::Text(
            "Pose interval: %.3fs", static_cast<f64>(pose_duration(model, viewer.pose_index))
        );
        const std::array mag_labels{"1x", "2x", "4x", "8x"};
        const std::array mag_values{1, 2, 4, 8};
        auto current_mag_idx = 0;
        for (auto i = 0; i < static_cast<int>(mag_values.size()); ++i)
        {
            if (viewer.skin_magnification == mag_values[static_cast<usize>(i)])
            {
                current_mag_idx = i;
            }
        }
        if (ImGui::Combo(
                "Skin magnification",
                &current_mag_idx,
                mag_labels.data(),
                static_cast<int>(mag_labels.size())
            ))
        {
            viewer.skin_magnification = mag_values[static_cast<usize>(current_mag_idx)];
        }
        if (ImGui::Button("Reset camera"))
        {
            viewer_camera_requested = true;
        }
        ImGui::Separator();
        ImGui::Text("Skins");
        const auto texture = imgui_texture(atlas.textures[palette_index], runtime);
        const auto available_width = std::max(64.0f, ImGui::GetContentRegionAvail().x);
        const auto columns = std::max(1, static_cast<int>(available_width / 78.0f));
        auto column = 0;
        for (auto record_idx = 0zu; record_idx < atlas.records.size(); ++record_idx)
        {
            const auto& record = atlas.records[record_idx];
            ImGui::PushID(static_cast<int>(record_idx));
            const auto active = record_idx == viewer.skin_record_index;
            if (active)
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{1.0f, 0.42f, 0.04f, 0.55f});
            }
            const auto clicked = ImGui::ImageButton(
                "skin",
                texture,
                ImVec2{64.0f, 64.0f},
                ImVec2{record.uv_min.x, record.uv_min.y},
                ImVec2{record.uv_max.x, record.uv_max.y}
            );
            if (active)
            {
                ImGui::PopStyleColor();
            }
            if (clicked)
            {
                viewer.model_index = record.model_index;
                viewer.skin_record_index = record_idx;
                viewer.pose_index = 0zu;
                viewer.pose_timer = 0.0f;
                viewer.selected_triangle = k_invalid_triangle;
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("%s", record.label.c_str());
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                {
                    viewer.model_index = record.model_index;
                    viewer.skin_record_index = record_idx;
                    viewer.pose_index = 0zu;
                    viewer.pose_timer = 0.0f;
                    viewer.selected_triangle = k_invalid_triangle;
                    viewer_camera_requested = true;
                }
            }
            ImGui::PopID();
            ++column;
            if (column < columns)
            {
                ImGui::SameLine();
            }
            else
            {
                column = 0;
            }
        }
    }
    ImGui::End();
}

auto draw_scene_selector(SceneMode& scene, bool& scene_changed) -> void
{
    scene_changed = false;
    ImGui::SetNextWindowPos(ImVec2{545.0f, 20.0f}, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2{280.0f, 110.0f}, ImGuiCond_Once);
    if (ImGui::Begin("Quake Scene"))
    {
        const std::array scene_names{"Overview", "Model viewer"};
        auto scene_index = scene == SceneMode::model_viewer ? 1 : 0;
        if (ImGui::Combo(
                "Scene", &scene_index, scene_names.data(), static_cast<int>(scene_names.size())
            ))
        {
            scene = scene_index == 1 ? SceneMode::model_viewer : SceneMode::overview;
            scene_changed = true;
        }
    }
    ImGui::End();
}

}  // namespace

auto run_quake_app() -> int
{
    const auto mdl_files = find_mdl_files();
    if (mdl_files.empty())
    {
        std::println(stderr, "Failed to find MDL files in {}", paths::progs_dir.string());
        return EXIT_FAILURE;
    }

    if (std::getenv("DS_VK_QUAKE_DUMP_TEXEL_INDICES") != nullptr)
    {
        return dump_skins_only(mdl_files);
    }
    if (std::getenv("DS_VK_QUAKE_DUMP_GLTF") != nullptr)
    {
        return dump_gltf_only(mdl_files);
    }

    auto parsed_models = parse_models(mdl_files);
    if (parsed_models.empty())
    {
        std::println(stderr, "No MDL files could be converted to renderable meshes");
        return EXIT_FAILURE;
    }

    const auto base_palette = load_quake_palette();
    if (!base_palette)
    {
        return EXIT_FAILURE;
    }
    auto palette_variants = load_palette_variants(*base_palette);
    auto skin_atlas_cpu = build_skin_atlas_cpu(parsed_models, palette_variants);

    std::println("Loaded {} Quake MDL meshes", parsed_models.size());

    RuntimeConfig config{
        .window_title = "ds_vk Quake demo",
        .initial_width = 1280u,
        .initial_height = 820u,
        .clear_color = Color{0.018f, 0.022f, 0.027f, 1.0f},
    };

    Runtime runtime{std::move(config)};
    runtime.initialize();

    auto skin_atlas = upload_skin_atlas(runtime, std::move(skin_atlas_cpu));
    auto floor_texture =
        runtime.load_texture(asset_path("textures/polyhaven/concrete_floor_diff_1k.jpg"));
    const auto showcase_meshes = ShowcaseMeshes{
        .floor = runtime.upload_mesh(make_quad(64.0f, Color::white)),
        .sphere = runtime.upload_mesh(
            make_uv_sphere(UvSphereConfig{.radius = 1.0f, .slices = 40u, .stacks = 20u})
        ),
        .cube = runtime.upload_mesh(make_cube(1.0f, Color::white)),
    };

    auto render_models = build_render_models(runtime, parsed_models, skin_atlas);
    const auto scene_extent = scene_extent_for(render_models);
    auto lighting = make_common_lighting(scene_extent);
    configure_overview_camera(runtime, scene_extent);

    auto background_music =
        BackgroundMusic::load_looped_ogg(paths::music_dir / "track02.ogg", 0.38f);

    const Material floor_material{
        .base_color = Color{0.55f, 0.58f, 0.53f, 1.0f},
        .roughness = 0.88f,
        .textures = {.base_color = floor_texture},
    };
    const Material sphere_material{
        .base_color = Color{0.18f, 0.52f, 0.95f, 1.0f},
        .metallic = 0.0f,
        .roughness = 0.34f,
    };
    const Material cube_material{
        .base_color = Color{0.90f, 0.46f, 0.20f, 1.0f},
        .roughness = 0.62f,
    };
    const Material unskinned_model_material{
        .base_color = Color{0.72f, 0.64f, 0.48f, 1.0f},
        .roughness = 0.92f,
    };
    Picker picker{};
    auto selected_render_model = k_invalid_index;
    auto selection_window_open = false;
    auto overview_pose_step = 0zu;
    auto overview_pose_accumulator = 0.0f;
    auto scene = SceneMode::overview;
    auto palette_index = 0zu;
    ViewerState viewer{
        .model_index = 0zu,
        .skin_record_index = parsed_models.front().skin_record_indices.empty()
                                 ? 0zu
                                 : parsed_models.front().skin_record_indices.front(),
    };

    while (auto* frame = runtime.begin_frame())
    {
        if (background_music)
        {
            background_music->pump();
        }

        if (scene == SceneMode::overview)
        {
            advance_overview_pose(overview_pose_step, overview_pose_accumulator, frame->dt_seconds);
        }
        else
        {
            frame->camera.set_allow_pivot_move(false)
                .set_clamp_position_z_min(true)
                .set_min_position_z(0.0f);
            if (frame->input.space_pressed)
            {
                viewer.paused = !viewer.paused;
            }
            advance_pose(viewer, parsed_models[viewer.model_index], frame->dt_seconds);
        }

        apply_lighting(frame->draw, lighting);

        const auto current_atlas_texture = skin_atlas.textures[palette_index];
        const auto textured_model_material = Material{
            .base_color = Color::white,
            .roughness = 0.92f,
            .textures = {.base_color = current_atlas_texture},
        };

        if (scene == SceneMode::overview)
        {
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
                if (hit and static_cast<usize>(hit->sub_index) < render_models.size())
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

            draw_showcase_objects(
                frame->draw,
                showcase_meshes,
                scene_extent,
                floor_material,
                sphere_material,
                cube_material
            );
            for (auto i = 0zu; i < render_models.size(); ++i)
            {
                const auto selected = i == selected_render_model;
                const auto& render_model = render_models[i];
                const auto pose_index = render_model.meshes.size() <= 1zu
                                            ? 0zu
                                            : overview_pose_step % render_model.meshes.size();
                frame->draw.draw_mesh({
                    .mesh = render_model.meshes[pose_index],
                    .transform = render_model.transform,
                    .material = textured_model_material,
                    .debug =
                        selected
                            ? MeshDebugConfig{
                                  .mode = MeshDebugMode::selected_pulse,
                                  .color = k_quake_selection_color,
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
                            .aabb = transformed_aabb(
                                model.pose_bounds[pose_index], render_model.transform
                            ),
                            .color = Color{1.0f, 0.58f, 0.05f, 0.95f},
                            .width = 0.050f,
                            .draw_on_top = true,
                        }
                    );
                }
            }
        }
        else
        {
            const auto& model = parsed_models[viewer.model_index];
            const auto& skin_record = skin_atlas.records[viewer.skin_record_index];
            const auto transform = viewer_transform_for(model);
            ensure_viewer_pose_meshes(runtime, viewer, model, skin_record);
            draw_showcase_floor(frame->draw, showcase_meshes, floor_material);
            select_viewer_triangle(viewer, model, *frame, transform);
            if (viewer.wireframe)
            {
                draw_wireframe_mesh(frame->draw, model.meshes[viewer.pose_index], transform);
            }
            else
            {
                const auto selected_triangle_debug =
                    viewer.selected_triangle == k_invalid_triangle
                        ? MeshDebugConfig{}
                        : MeshDebugConfig{
                              .mode = MeshDebugMode::triangle_selected_pulse,
                              .color = k_quake_selection_color,
                              .scalar = static_cast<f32>(viewer.selected_triangle),
                          };
                frame->draw.draw_mesh({
                    .mesh = viewer.pose_meshes[viewer.pose_index],
                    .transform = transform,
                    .material =
                        viewer.show_skin ? textured_model_material : unskinned_model_material,
                    .debug = selected_triangle_debug,
                });
            }
        }

        if (runtime.ui_visible())
        {
            runtime.draw_runtime_ui();
            auto scene_changed = false;
            draw_scene_selector(scene, scene_changed);

            auto palette_changed = false;
            auto viewer_camera_requested = false;
            if (scene == SceneMode::overview)
            {
                ImGui::SetNextWindowPos(ImVec2{545.0f, 140.0f}, ImGuiCond_Once);
                ImGui::SetNextWindowSize(ImVec2{300.0f, 110.0f}, ImGuiCond_Once);
                if (ImGui::Begin("Quake Shading"))
                {
                    draw_palette_combo(palette_variants, palette_index, palette_changed);
                    ImGui::Text("Skin atlas: %u x %u", skin_atlas.width, skin_atlas.height);
                }
                ImGui::End();
                if (selected_render_model != k_invalid_index
                    and selected_render_model < render_models.size())
                {
                    const auto& selected = render_models[selected_render_model];
                    const auto pose =
                        selected.meshes.empty() ? 0zu : overview_pose_step % selected.meshes.size();
                    draw_overview_selection_window(
                        selection_window_open,
                        selected_render_model,
                        render_models,
                        parsed_models,
                        pose
                    );
                }
            }
            else
            {
                draw_model_viewer_window(
                    runtime,
                    skin_atlas,
                    parsed_models,
                    viewer,
                    palette_index,
                    palette_variants,
                    palette_changed,
                    viewer_camera_requested
                );
                draw_skin_texture_window(runtime, skin_atlas, parsed_models, viewer, palette_index);
            }

            if (scene_changed)
            {
                if (scene == SceneMode::overview)
                {
                    configure_overview_camera(runtime, scene_extent);
                }
                else
                {
                    configure_viewer_camera(runtime, parsed_models[viewer.model_index]);
                }
            }
            if (viewer_camera_requested)
            {
                configure_viewer_camera(runtime, parsed_models[viewer.model_index]);
            }
            (void) palette_changed;
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
