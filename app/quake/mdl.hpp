#pragma once

#include "ds_vk/geometry.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/types.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ds_vk_quake
{
using namespace ds_vk;

// Identification marker for Id Software Binary File, little endian.
inline constexpr i32 k_idpoly = (('O' << 24) + ('P' << 16) + ('D' << 8) + 'I');

// NOLINTNEXTLINE(performance-enum-size): Quake MDL stores this field as a 32-bit int.
enum class MdlSyncType : i32
{
    Sync = 0,
    Rand = 1
};

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

struct MdlPaletteEntry
{
    u8 r{};
    u8 g{};
    u8 b{};
};
static_assert(sizeof(MdlPaletteEntry) == 3zu);

using MdlPalette = std::array<MdlPaletteEntry, 256>;

struct MdlFrame
{
    MdlTriVertex bbox_min{};
    MdlTriVertex bbox_max{};
    std::array<char, 16> name{};
    f32 interval{};
    std::vector<MdlTriVertex> vertices{};
};

using MdlSkinData = std::vector<u8>;

struct MdlSkin
{
    std::vector<f32> intervals{};
    std::vector<MdlSkinData> images{};
};

struct MdlBinary
{
    MdlHeader header{};
    MdlPalette palette{};
    std::vector<MdlSkin> skins{};
    std::vector<MdlVertex> stverts{};
    std::vector<MdlTriangle> triangles{};
    std::vector<MdlFrame> frames{};
};

[[nodiscard]] auto to_string(MdlSyncType) -> std::string_view;
[[nodiscard]] auto to_string(MdlHeaderValidity) -> std::string_view;
[[nodiscard]] auto validate(const MdlHeader&) -> MdlHeaderValidity;
[[nodiscard]] auto is_valid(const MdlHeader&) -> bool;
[[nodiscard]] auto to_string(const MdlHeader&) -> std::string;
[[nodiscard]] auto load_quake_palette() -> std::optional<MdlPalette>;
[[nodiscard]] auto find_black_palette_index(const MdlPalette&) -> std::optional<u8>;
[[nodiscard]] auto load_alias_normals(const std::filesystem::path&)
    -> std::optional<std::vector<Vec3>>;

[[nodiscard]] auto make_mesh_data(
    const MdlHeader&,
    std::span<const MdlVertex>,
    std::span<const MdlTriangle>,
    const MdlFrame&,
    std::span<const Vec3> alias_normals = {}
) -> std::optional<MeshData>;

[[nodiscard]] auto parse_mdl_binary(const std::filesystem::path&) -> std::optional<MdlBinary>;
[[nodiscard]] auto save_mdl_skins_to_file(
    const std::vector<MdlSkin>&, const MdlHeader&, const MdlPalette&, std::string_view
) -> usize;
[[nodiscard]] auto save_quake_palette_to_file(const MdlPalette&) -> bool;
[[nodiscard]] auto save_mdl_as_gltf(
    const MdlBinary&,
    std::string_view name,
    const std::filesystem::path&,
    std::span<const Vec3> alias_normals
) -> bool;
}  // namespace ds_vk_quake
