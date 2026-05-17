#include "app/quake/mdl.hpp"

#include "app/quake/io.hpp"
#include "app/quake/paths.hpp"
#include "ds_vk/math.hpp"
#include "ds_vk/utility.hpp"

#include <cstring>
#include <format>
#include <fstream>
#include <mdspan>
#include <print>
#include <queue>
#include <system_error>
#include <type_traits>
#include <utility>

namespace ds_vk_quake
{
using namespace ds_vk;

static_assert(sizeof(MdlHeader) == 84);
static_assert(std::is_trivially_copyable_v<MdlHeader>);
static_assert(std::is_standard_layout_v<MdlHeader>);
static_assert(alignof(MdlHeader) == 4);

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

[[nodiscard]] auto to_string(MdlHeaderValidity v) -> std::string_view
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

template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] auto read_value(std::span<const std::byte>& buf) -> T
{
    T out{};
    std::memcpy(&out, buf.data(), sizeof(T));
    buf = buf.subspan(sizeof(T));
    return out;
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] auto read_values(std::span<const std::byte>& buf, usize count) -> std::vector<T>
{
    std::vector<T> out(count);
    const auto bytes = count * sizeof(T);
    std::memcpy(out.data(), buf.data(), bytes);
    buf = buf.subspan(bytes);
    return out;
}

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
    const auto header = read_value<MdlHeader>(buf);
    return header;
}

namespace
{
auto parse_mdl_skins_single(
    std::span<const std::byte>& buf, const MdlHeader& header, bool flood_fill = true
) -> MdlSkinData
{
    const auto sh = static_cast<usize>(header.skin_height);
    const auto sw = static_cast<usize>(header.skin_width);
    const auto pixel_count = sh * sw;
    std::vector<u8> out(pixel_count);
    std::memcpy(out.data(), buf.data(), out.size());

    const auto fill_origin = out[0];
    const auto fill_target = 0;  // TODO: Do palette lookup here instead

    std::mdspan<u8, std::dextents<usize, 2>> texels{out.data(), sh, sw};

    if (flood_fill)
    {
        std::queue<std::pair<int, int>> q{};
        texels[0, 0] = fill_target;
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
    }

    buf = buf.subspan(pixel_count);
    return out;
}
}  // namespace

auto parse_mdl_skins(
    std::span<const std::byte>& buf, const MdlHeader& header, bool flood_fill = true
) -> std::vector<MdlSkin>
{
    const auto num_skins = static_cast<usize>(header.num_skins);
    std::vector<MdlSkin> out(num_skins);
    for (auto skin_idx = 0zu; skin_idx < num_skins; ++skin_idx)
    {
        const auto type_val = MdlSkinType{read_value<i32>(buf)};

        switch (type_val)
        {
            case MdlSkinType::Single:
                {
                    out[skin_idx].images.push_back(parse_mdl_skins_single(buf, header, flood_fill));
                }
                break;
            case MdlSkinType::Group:
                {
                    const auto group = read_value<MdlSkinGroup>(buf);
                    const auto group_skins = static_cast<usize>(group.num_skins);
                    out[skin_idx].intervals.reserve(group_skins);
                    out[skin_idx].images.reserve(group_skins);
                    for (auto j = 0zu; j < group_skins; ++j)
                    {
                        out[skin_idx].intervals.push_back(read_value<MdlInterval>(buf).interval);
                    }
                    for (auto j = 0zu; j < group_skins; ++j)
                    {
                        out[skin_idx].images.push_back(
                            parse_mdl_skins_single(buf, header, flood_fill)
                        );
                    }
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
    std::vector<MdlFrame> out{};
    out.reserve(num_frames);
    for (auto i = 0zu; i < num_frames; ++i)
    {
        const auto type_val = MdlFrameType{read_value<i32>(buf)};

        switch (type_val)
        {
            case MdlFrameType::Single:
                {
                    MdlFrame frame{};
                    frame.bbox_min = read_value<MdlTriVertex>(buf);
                    frame.bbox_max = read_value<MdlTriVertex>(buf);
                    std::memcpy(frame.name.data(), buf.data(), frame.name.size());
                    buf = buf.subspan(frame.name.size());
                    frame.vertices =
                        read_values<MdlTriVertex>(buf, static_cast<usize>(header.num_verts));
                    out.push_back(std::move(frame));
                    break;
                }
            case MdlFrameType::Group:
                {
                    const auto group = read_value<MdlGroup>(buf);
                    const auto group_frames = static_cast<usize>(group.num_frames);
                    std::vector<f32> intervals{};
                    intervals.reserve(group_frames);
                    for (auto j = 0zu; j < group_frames; ++j)
                    {
                        intervals.push_back(read_value<MdlInterval>(buf).interval);
                    }
                    for (auto j = 0zu; j < group_frames; ++j)
                    {
                        MdlFrame frame{};
                        frame.bbox_min = read_value<MdlTriVertex>(buf);
                        frame.bbox_max = read_value<MdlTriVertex>(buf);
                        std::memcpy(frame.name.data(), buf.data(), frame.name.size());
                        buf = buf.subspan(frame.name.size());
                        frame.vertices =
                            read_values<MdlTriVertex>(buf, static_cast<usize>(header.num_verts));
                        frame.interval = intervals[j];
                        out.push_back(std::move(frame));
                    }
                    break;
                }
        }
    }
    return out;
}

auto save_mdl_skins_to_file(
    const std::vector<MdlSkin>& skins, const MdlHeader& header, std::string_view name
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

    for (auto skin_idx = 0zu; skin_idx < skins.size(); ++skin_idx)
    {
        for (auto image_idx = 0zu; image_idx < skins[skin_idx].images.size(); ++image_idx)
        {
            const auto output_name =
                skins[skin_idx].images.size() == 1zu
                    ? std::format("{}_skin{}.pgm", name, skin_idx)
                    : std::format("{}_skin{}_pose{}.pgm", name, skin_idx, image_idx);
            const auto output_path = paths::texel_index_output_dir / output_name;
            std::ofstream dump{output_path, std::ios::binary};
            dump << "P5\n" << header.skin_width << " " << header.skin_height << "\n255\n";
            dump.write(
                reinterpret_cast<const char*>(skins[skin_idx].images[image_idx].data()),
                static_cast<std::streamsize>(skins[skin_idx].images[image_idx].size())
            );
        }
    }
}

auto parse_mdl_binary(const std::filesystem::path& filepath) -> std::optional<MdlBinary>
{
    const auto buf_res = load_binary_file(filepath);
    if (!buf_res)
    {
        std::println(stderr, "Failed to load {}.", filepath.filename().string());
        return std::nullopt;
    }

    std::span<const std::byte> buf{*buf_res};
    MdlBinary out{};
    out.header = parse_mdl_header(buf);
    if (const auto validity = validate(out.header); validity != MdlHeaderValidity::Valid)
    {
        std::println(
            stderr,
            "Invalid MDL header in {}: {}",
            filepath.filename().string(),
            to_string(validity)
        );
        return std::nullopt;
    }

    out.skins = parse_mdl_skins(buf, out.header);
    out.stverts = read_values<MdlVertex>(buf, static_cast<usize>(out.header.num_verts));
    out.triangles = read_values<MdlTriangle>(buf, static_cast<usize>(out.header.num_tris));
    out.frames = parse_mdl_frames(buf, out.header);

    if (!buf.empty())
    {
        std::println(
            stderr, "{} has {} unparsed trailing bytes", filepath.filename().string(), buf.size()
        );
    }
    return out;
}

}  // namespace ds_vk_quake
