// app/quake_main.cpp
#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mdspan>
#include <optional>
#include <print>
#include <queue>
#include <span>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ds_vk_quake
{
using namespace ds_vk;

constexpr auto k_kib = 1024zu;
constexpr auto k_mib = 1024zu * k_kib;
constexpr auto k_gib = 1024zu * k_mib;

template <std::unsigned_integral T>
auto format_byte(T bytes, bool verbose = false) -> std::string
{
    const auto bytes_d = static_cast<f64>(bytes);
    if (bytes < k_kib)
    {
        return std::format("{} bytes", bytes);
    }
    auto out = [&]
    {
        if (bytes < 10 * k_mib)
        {
            return std::format("{:.2f} KiB", bytes_d / static_cast<f64>(k_kib));
        }
        else if (bytes < 10 * k_gib)
        {
            return std::format("{:.2f} MiB", bytes_d / static_cast<f64>(k_mib));
        }
        else
        {
            return std::format("{:.2f} GiB", bytes_d / static_cast<f64>(k_gib));
        }
    }();
    if (verbose)
    {
        out = std::format("{} ({} bytes)", out, bytes);
    }
    return out;
}

auto load_binary_file(const std::filesystem::path& filepath)
    -> std::optional<std::vector<std::byte>>
{
    std::error_code ec;
    if (!std::filesystem::exists(filepath, ec))
    {
        std::println(stderr, "Path does not exist: {} ({})", filepath.string(), ec.message());
        return std::nullopt;
    }

    std::ifstream f{filepath, std::ios::binary};
    if (!f)
    {
        std::println(stderr, "Failed to open {}", filepath.string());
        return std::nullopt;
    }

    const auto model_size_bytes = std::filesystem::file_size(filepath);
    std::vector<std::byte> buf(model_size_bytes);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!f or f.gcount() != static_cast<std::streamsize>(buf.size()))
    {
        std::println(stderr, "Failed to read complete file {}", filepath.string());
        return std::nullopt;
    }
    return buf;
}

namespace paths
{
inline const std::filesystem::path quake_root{
    "/Users/danielsinkin/GitHub_private/vulkan-framework/app/quake"
};
inline auto asset_root = quake_root / "assets";
inline auto progs_dir = asset_root / "models" / "progs";
inline auto maps_dir = asset_root / "maps";
inline auto sounds_dir = asset_root / "sound";
inline auto texel_index_output_dir = quake_root / "outputs" / "texel_indices";
}  // namespace paths

// Identification marker for Id Software Binary File, little endian
static constexpr i32 k_idpoly = (('O' << 24) + ('P' << 16) + ('D' << 8) + 'I');

template <typename T, typename U, typename V>
    requires requires(T x, U lo, V hi) {
        std::cmp_less_equal(lo, x);
        std::cmp_less_equal(x, hi);
        std::cmp_less_equal(lo, hi);
    }
[[nodiscard]] constexpr auto in_interval(T x, U lo, V hi) -> bool
{
    assert(std::cmp_less_equal(lo, hi));
    return std::cmp_less_equal(lo, x) && std::cmp_less_equal(x, hi);
}

template <std::floating_point T>
[[nodiscard]] constexpr auto in_interval(T x, T lo, T hi) -> bool
{
    assert(lo <= hi);
    return lo <= x && x <= hi;
}

// NOLINTNEXTLINE(performance-enum-size): Quake MDL stores this field as a 32-bit int.
enum class MdlSyncType : i32
{
    Sync = 0,
    Rand = 1
};
auto to_string(MdlSyncType type) -> std::string_view
{
    switch (type)
    {
        case MdlSyncType::Sync:
            return "Sync";
        case MdlSyncType::Rand:
            return "Rand";
    }
    std::unreachable();
}

struct MdlHeader
{
    i32 identifier{};
    i32 version{};
    std::array<f32, 3> scale{};
    std::array<f32, 3> scale_origin{};
    f32 boundingradius{};
    std::array<f32, 3> eye_pos{};
    i32 num_skins{};
    i32 skin_width{};
    i32 skin_height{};
    i32 num_verts{};
    i32 num_tris{};
    i32 num_frames{};
    MdlSyncType sync_type{};
    i32 flags{};
    f32 size{};

    static constexpr i32 k_correct_ident{k_idpoly};
    static constexpr i32 k_correct_version{6};
    static constexpr i32 k_max_verts{2000};
    static constexpr i32 k_max_skinheight{480};
};
static_assert(sizeof(MdlHeader) == 84);
static_assert(std::is_trivially_copyable_v<MdlHeader>);
static_assert(std::is_standard_layout_v<MdlHeader>);
static_assert(alignof(MdlHeader) == 4);

enum class MdlHeaderValidity : u8
{
    Valid = 0,
    BadMagic,
    BadVersion,
    NumVertsOutOfRange,
    NumTrisNotPositive,
    SkinWidthInvalid,
    SkinHeightOutOfRange,
    NumSkinsNotPositive,
};
[[nodiscard]] constexpr auto to_string(MdlHeaderValidity v) -> std::string_view
{
    using enum MdlHeaderValidity;
    switch (v)
    {
        case Valid:
            return "Valid";
        case BadMagic:
            return "BadMagic (ident != IDPO)";
        case BadVersion:
            return "BadVersion (version != 6)";
        case NumVertsOutOfRange:
            return "NumVertsOutOfRange";
        case NumTrisNotPositive:
            return "NumTrisNotPositive";
        case SkinWidthInvalid:
            return "SkinWidthInvalid (must be positive multiple of 4)";
        case SkinHeightOutOfRange:
            return "SkinHeightOutOfRange";
        case NumSkinsNotPositive:
            return "NumSkinsNotPositive";
    }
    std::unreachable();
}
[[nodiscard]] auto validate(const MdlHeader& h) -> MdlHeaderValidity
{
    using enum MdlHeaderValidity;
    if (h.identifier != MdlHeader::k_correct_ident) return BadMagic;
    if (h.version != MdlHeader::k_correct_version) return BadVersion;
    if (!in_interval(h.num_verts, 1, MdlHeader::k_max_verts)) return NumVertsOutOfRange;
    if (h.num_tris <= 0) return NumTrisNotPositive;
    if (h.skin_width <= 0 || h.skin_width % 4 != 0) return SkinWidthInvalid;
    if (!in_interval(h.skin_height, 1, MdlHeader::k_max_skinheight)) return SkinHeightOutOfRange;
    if (h.num_skins <= 0) return NumSkinsNotPositive;
    return Valid;
}
[[nodiscard]] auto is_valid(const MdlHeader& h) -> bool
{
    return validate(h) == MdlHeaderValidity::Valid;
}

[[nodiscard]] auto to_string(const MdlHeader& header) -> std::string
{
    return std::format(
        "MdlHeader{{\n"
        "  ident         = 0x{:08x} ({:c}{:c}{:c}{:c})\n"
        "  version       = {}\n"
        "  scale         = ({:.4f}, {:.4f}, {:.4f})\n"
        "  scale_origin  = ({:.4f}, {:.4f}, {:.4f})\n"
        "  boundingradius= {:.4f}\n"
        "  eyeposition   = ({:.4f}, {:.4f}, {:.4f})\n"
        "  numskins      = {}\n"
        "  skinwidth     = {}\n"
        "  skinheight    = {}\n"
        "  numverts      = {}\n"
        "  numtris       = {}\n"
        "  numframes     = {}\n"
        "  synctype      = {}\n"
        "  flags         = 0x{:08x}\n"
        "  size          = {:.4f}\n"
        "}}",
        static_cast<u32>(header.identifier),
        (header.identifier) & 0xff,
        (header.identifier >> 8) & 0xff,
        (header.identifier >> 16) & 0xff,
        (header.identifier >> 24) & 0xff,
        header.version,
        header.scale[0],
        header.scale[1],
        header.scale[2],
        header.scale_origin[0],
        header.scale_origin[1],
        header.scale_origin[2],
        header.boundingradius,
        header.eye_pos[0],
        header.eye_pos[1],
        header.eye_pos[2],
        header.num_skins,
        header.skin_width,
        header.skin_height,
        header.num_verts,
        header.num_tris,
        header.num_frames,
        to_string(header.sync_type),
        static_cast<u32>(header.flags),
        header.size
    );
}

struct MdlVertex
{
    i32 onseam{};
    i32 s{};
    i32 t{};
};

struct MdlTriangle
{
    i32 faces_front{};
    std::array<i32, 3> vertex_indices{};
};

struct MdlTriVertex
{
    std::array<u8, 3> vertices{};
    u8 light_normal_index{};
};

struct MdlFrame
{
    MdlTriVertex bbox_min{};
    MdlTriVertex bbox_max{};
    std::array<char, 16> name{};
    std::vector<MdlTriVertex> vertices{};
};

struct MdlGroup
{
    i32 num_frames{};
    MdlTriVertex bbox_min{};
    MdlTriVertex bbox_max{};
};

struct MdlSkinGroup
{
    i32 num_skins{};
};

struct MdlInterval
{
    f32 interval{};
};

// NOLINTNEXTLINE(performance-enum-size): Quake MDL stores this field as a 32-bit int.
enum class MdlFrameType : i32
{
    Single = 0,
    Group = 1
};

// NOLINTNEXTLINE(performance-enum-size): Quake MDL stores this field as a 32-bit int.
enum class MdlSkinType : i32
{
    Single = 0,
    Group = 1
};

[[nodiscard]] auto dequantize_position(const MdlHeader& header, const MdlTriVertex& vertex) -> Vec3
{
    return Vec3{
        static_cast<f32>(vertex.vertices[0]) * header.scale[0] + header.scale_origin[0],
        static_cast<f32>(vertex.vertices[1]) * header.scale[1] + header.scale_origin[1],
        static_cast<f32>(vertex.vertices[2]) * header.scale[2] + header.scale_origin[2],
    };
}

[[nodiscard]] auto
texcoord_for_vertex(const MdlHeader& header, const MdlVertex& vertex, const MdlTriangle& triangle)
    -> Vec2
{
    auto s = vertex.s;
    if (triangle.faces_front == 0 and vertex.onseam != 0)
    {
        s += header.skin_width / 2;
    }

    return Vec2{
        (static_cast<f32>(s) + 0.5f) / static_cast<f32>(header.skin_width),
        (static_cast<f32>(vertex.t) + 0.5f) / static_cast<f32>(header.skin_height),
    };
}

[[nodiscard]] auto make_mesh_data(
    const MdlHeader& header,
    std::span<const MdlVertex> stverts,
    std::span<const MdlTriangle> triangles,
    const MdlFrame& frame
) -> std::optional<MeshData>
{
    if (stverts.size() != static_cast<usize>(header.num_verts)
        or frame.vertices.size() != static_cast<usize>(header.num_verts))
    {
        return std::nullopt;
    }

    constexpr Color color{0.72f, 0.64f, 0.48f, 1.0f};
    MeshData out{};
    out.vertices.reserve(triangles.size() * 3zu);
    out.indices.reserve(triangles.size() * 3zu);

    for (const auto& triangle : triangles)
    {
        std::array<Vec3, 3> positions{};
        std::array<Vec2, 3> texcoords{};
        for (auto corner = 0zu; corner < 3zu; ++corner)
        {
            const auto vertex_idx = triangle.vertex_indices[corner];
            if (!in_interval(vertex_idx, 0, header.num_verts - 1))
            {
                std::println(stderr, "MDL triangle references out-of-range vertex {}", vertex_idx);
                return std::nullopt;
            }

            const auto idx = static_cast<usize>(vertex_idx);
            positions[corner] = dequantize_position(header, frame.vertices[idx]);
            texcoords[corner] = texcoord_for_vertex(header, stverts[idx], triangle);
        }

        const auto normal = normalize_or(
            glm::cross(positions[1] - positions[0], positions[2] - positions[0]), k_axis_z
        );
        const auto base = static_cast<u32>(out.vertices.size());
        for (auto corner = 0zu; corner < 3zu; ++corner)
        {
            out.vertices.push_back(
                Vertex{
                    .position = positions[corner],
                    .normal = normal,
                    .color = color,
                    .texcoord = texcoords[corner],
                }
            );
            out.indices.push_back(base + static_cast<u32>(corner));
        }
    }

    return out;
}

auto parse_mdl_header(std::span<const std::byte>& buf) -> MdlHeader
{
    MdlHeader header{};
    std::memcpy(&header, buf.data(), sizeof(MdlHeader));
    std::println("{}", to_string(header));
    buf = buf.subspan(sizeof(MdlHeader));
    return header;
}

using MdlSkinData = std::vector<u8>;

namespace
{
auto parse_mdl_skins_single(std::span<const std::byte>& buf, const MdlHeader& header) -> MdlSkinData
{
    const auto pixel_count =
        static_cast<usize>(header.skin_width) * static_cast<usize>(header.skin_height);
    std::vector<u8> out(pixel_count);
    std::memcpy(out.data(), buf.data(), out.size());

    const auto fill_origin = out[0];
    const auto fill_target = 0;  // TODO: Do palette lookup here instead

    const auto sh = static_cast<usize>(header.skin_height);
    const auto sw = static_cast<usize>(header.skin_width);
    std::mdspan<u8, std::dextents<usize, 2>> texels{out.data(), sh, sw};

    std::queue<std::pair<int, int>> q{};
    q.emplace(0, 0);
    const auto try_neighbor = [&](int yp, int xp) -> void
    {
        const auto valid_y = in_interval(yp, 0, sh - 1);
        const auto valid_x = in_interval(xp, 0, sw - 1);
        if (valid_y and valid_x and texels[yp, xp] == fill_origin)
        {
            texels[yp, xp] = fill_target;
            q.emplace(yp, xp);
        }
    };
    while (!q.empty())
    {
        const auto [y, x] = q.front();
        q.pop();

        try_neighbor(y - 1, x);
        try_neighbor(y + 1, x);
        try_neighbor(y, x - 1);
        try_neighbor(y, x + 1);
    }

    buf = buf.subspan(pixel_count);
    return out;
}
}  // namespace

auto parse_mdl_skins(std::span<const std::byte>& buf, const MdlHeader& header)
    -> std::vector<MdlSkinData>
{
    const auto num_skins = static_cast<usize>(header.num_skins);
    std::vector<MdlSkinData> out(num_skins);
    for (auto skin_idx = 0zu; skin_idx < num_skins; ++skin_idx)
    {
        const auto type_val = [&]
        {
            i32 out{};
            std::memcpy(&out, buf.data(), sizeof(i32));
            buf = buf.subspan(sizeof(i32));
            return MdlSkinType{out};
        }();

        switch (type_val)
        {
            case MdlSkinType::Single:
                {
                    out[skin_idx] = parse_mdl_skins_single(buf, header);
                }
                break;
            case MdlSkinType::Group:
                {
                    std::println("Skin Type Group not supported yet");
                    std::terminate();
                }
                break;
        }
    }
    return out;
}

auto parse_mdl_frames(std::span<const std::byte>& buf, const MdlHeader& header)
    -> std::vector<MdlFrame>
{
    const auto num_frames = static_cast<usize>(header.num_frames);
    std::vector<MdlFrame> out(num_frames);
    for (auto i = 0zu; i < num_frames; ++i)
    {
        const auto type_val = [&]
        {
            i32 type_out{};
            std::memcpy(&type_out, buf.data(), sizeof(i32));
            buf = buf.subspan(sizeof(i32));
            return MdlFrameType{type_out};
        }();

        switch (type_val)
        {
            case MdlFrameType::Single:
                {
                    std::memcpy(&out[i].bbox_min, buf.data(), sizeof(MdlTriVertex));
                    buf = buf.subspan(sizeof(MdlTriVertex));
                    std::memcpy(&out[i].bbox_max, buf.data(), sizeof(MdlTriVertex));
                    buf = buf.subspan(sizeof(MdlTriVertex));
                    std::memcpy(out[i].name.data(), buf.data(), 16);
                    buf = buf.subspan(16);

                    const auto num_verts = static_cast<usize>(header.num_verts);
                    const auto verts_bytes = num_verts * sizeof(MdlTriVertex);
                    out[i].vertices.resize(num_verts);
                    std::memcpy(out[i].vertices.data(), buf.data(), verts_bytes);
                    buf = buf.subspan(verts_bytes);
                    break;
                }
            case MdlFrameType::Group:
                {
                    std::terminate();
                    break;
                }
        }
    }
    return out;
}

auto save_mdl_skins_to_file(
    const std::vector<MdlSkinData>& skins, const MdlHeader& header, std::string_view name
) -> void
{
    std::error_code ec;
    std::filesystem::create_directories(paths::texel_index_output_dir, ec);
    if (ec)
    {
        std::println(
            stderr,
            "Failed to create texel index output directory {}: {}",
            paths::texel_index_output_dir.string(),
            ec.message()
        );
        return;
    }

    for (auto i = 0zu; i < skins.size(); ++i)
    {
        const auto output_path =
            paths::texel_index_output_dir / std::format("{}_skin{}.pgm", name, i);
        std::ofstream dump{output_path, std::ios::binary};
        dump << "P5\n" << header.skin_width << " " << header.skin_height << "\n255\n";
        dump.write(
            reinterpret_cast<const char*>(skins[i].data()),
            static_cast<std::streamsize>(skins[i].size())
        );
    }
}

}  // namespace ds_vk_quake

auto main() -> int
{
    using namespace ds_vk_quake;
    namespace fs = std::filesystem;

    const auto mdl_files = [&]
    {
        std::unordered_map<std::string, fs::path> out{};
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
            out[filepath.stem().string()] = filepath;
        }
        return out;
    }();

    const auto suit_it = mdl_files.find("suit");
    if (suit_it == mdl_files.end())
    {
        std::println(stderr, "Failed to find suit.mdl in {}", paths::progs_dir.string());
        return EXIT_FAILURE;
    }

    const auto& [name, file] = *suit_it;
    const auto buf_res = load_binary_file(file);
    if (!buf_res)
    {
        std::println("Failed to load {}.", file.filename().string());
        return EXIT_FAILURE;
    }
    std::span<const std::byte> buf{*buf_res};
    std::println("Buffer has {} bytes remaining", buf.size());

    const auto header = parse_mdl_header(buf);
    if (const auto validity = validate(header); validity != MdlHeaderValidity::Valid)
    {
        std::println(stderr, "Invalid MDL header: {}", to_string(validity));
        return EXIT_FAILURE;
    }
    std::println("Buffer has {} bytes remaining", buf.size());
    const auto skins = parse_mdl_skins(buf, header);
    save_mdl_skins_to_file(skins, header, name);

    std::println("Buffer has {} bytes remaining", buf.size());
    const auto stverts = [&]
    {
        const auto num_verts = static_cast<usize>(header.num_verts);
        const auto bytes = num_verts * sizeof(MdlVertex);
        std::vector<MdlVertex> out(num_verts);
        std::memcpy(out.data(), buf.data(), bytes);
        buf = buf.subspan(bytes);
        return out;
    }();
    std::println("Buffer has {} bytes remaining", buf.size());
    const auto triangles = [&]
    {
        const auto num_triangles = static_cast<usize>(header.num_tris);
        const auto bytes = num_triangles * sizeof(MdlTriangle);
        std::vector<MdlTriangle> out(num_triangles);
        std::memcpy(out.data(), buf.data(), bytes);
        buf = buf.subspan(bytes);
        return out;
    }();
    std::println("Buffer has {} bytes remaining", buf.size());
    const auto frames = parse_mdl_frames(buf, header);
    std::println("Buffer has {} bytes remaining", buf.size());
    if (frames.empty())
    {
        std::println(stderr, "MDL has no renderable frames");
        return EXIT_FAILURE;
    }

    auto mesh = make_mesh_data(header, stverts, triangles, frames.front());
    if (!mesh)
    {
        std::println(stderr, "Failed to convert MDL frame to MeshData");
        return EXIT_FAILURE;
    }
    const auto bounds = aabb_of(*mesh);
    const auto center = 0.5f * (bounds.min + bounds.max);
    const auto radius = glm::length(0.5f * (bounds.max - bounds.min));

    RuntimeConfig config{
        .window_title = "ds_vk Quake demo",
        .initial_width = 1280u,
        .initial_height = 820u,
        .clear_color = Color{0.018f, 0.022f, 0.027f, 1.0f},
    };

    Runtime runtime{std::move(config)};
    runtime.initialize();
    runtime.camera({
        .pivot = center,
        .distance = std::max(12.0f, radius * 2.8f),
        .yaw = glm::radians(42.0f),
        .pitch = glm::radians(20.0f),
    });
    const auto mdl_mesh_handle = runtime.upload_mesh(*mesh);
    const auto floor_mesh_handle = runtime.upload_mesh(make_quad(80.0f, Color::white));
    const auto sphere_mesh_handle =
        runtime.upload_mesh(make_uv_sphere(UvSphereConfig{.radius = 1.0f}));
    const auto cube_mesh_handle = runtime.upload_mesh(make_cube(2.0f, Color::white));

    const Material mdl_material{
        .base_color = Color{0.72f, 0.64f, 0.48f, 1.0f},
        .roughness = 0.92f,
    };
    const Material floor_material{
        .base_color = Color{0.48f, 0.52f, 0.48f, 1.0f},
        .roughness = 0.88f,
    };
    const Material sphere_material{
        .base_color = Color{0.18f, 0.52f, 0.95f, 1.0f},
        .roughness = 0.34f,
    };
    const Material cube_material{
        .base_color = Color{0.9f, 0.46f, 0.2f, 1.0f},
        .roughness = 0.62f,
    };

    const EnvironmentConfig environment{
        .lighting_intensity = 0.32f,
        .background_intensity = 0.75f,
        .rotation_radians = glm::radians(22.0f),
        .visible_to_camera = true,
    };
    const Color ambient_light{0.030f, 0.036f, 0.046f, 1.0f};
    const DirectionalLightConfig sun{
        .direction = {-0.42f, -0.34f, -0.84f},
        .color = {1.0f, 0.94f, 0.84f, 1.0f},
        .intensity = 2.45f,
        .shadow = LightShadowConfig{
            .enabled = true,
            .bias = 0.0040f,
            .strength = 0.78f,
            .near_plane = 0.05f,
            .far_plane = 24.0f,
            .ortho_extent = 5.2f,
        },
    };
    const RadialLightConfig radial_light{
        .position = {-2.2f, -1.3f, 1.55f},
        .color = {0.24f, 0.78f, 1.0f, 1.0f},
        .intensity = 18.0f,
        .range = 5.0f,
    };
    const SpotLightConfig spot_light{
        .position = {2.4f, -2.2f, 2.65f},
        .direction = {-0.62f, 0.48f, -0.62f},
        .color = {1.0f, 0.44f, 0.22f, 1.0f},
        .intensity = 32.0f,
        .range = 7.0f,
        .inner_cone_angle = glm::radians(10.0f),
        .outer_cone_angle = glm::radians(24.0f),
    };

    while (auto* frame = runtime.begin_frame())
    {
        frame->draw.set_ambient_light(ambient_light);
        frame->draw.set_environment(environment);
        frame->draw.directional_light(sun);
        frame->draw.radial_light(radial_light);
        frame->draw.spot_light(spot_light);

        frame->draw.draw_mesh({
            .mesh = floor_mesh_handle,
            .material = floor_material,
            .mask = {.shadow_producer = false},
        });
        frame->draw.draw_mesh({
            .mesh = mdl_mesh_handle,
            .material = mdl_material,
        });
        frame->draw.draw_mesh({
            .mesh = sphere_mesh_handle,
            .transform =
                Transform{
                    .translation = center + Vec3{8.0f, -7.0f, 2.0f},
                    .scale = Vec3{2.0f},
                },
            .material = sphere_material,
        });
        frame->draw.draw_mesh({
            .mesh = cube_mesh_handle,
            .transform =
                Transform{
                    .translation = center + Vec3{-9.0f, 6.0f, 1.0f},
                    .rotation = glm::angleAxis(glm::radians(24.0f), k_axis_z),
                },
            .material = cube_material,
        });
        if (runtime.ui_visible())
        {
            runtime.draw_runtime_ui();
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
