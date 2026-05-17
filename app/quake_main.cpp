// app/quake_main.cpp
#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/picker.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <SDL3/SDL.h>
#include <imgui.h>

#define OV_EXCLUDE_STATIC_CALLBACKS
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mdspan>
#include <optional>
#include <print>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vorbis/vorbisfile.h>

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
inline auto progs_dir = asset_root / "progs";
inline auto maps_dir = asset_root / "maps";
inline auto sounds_dir = asset_root / "sound";
inline auto music_dir = asset_root / "music";
inline auto texel_index_output_dir = quake_root / "outputs" / "texel_indices";
}  // namespace paths

class BackgroundMusic
{
  public:
    BackgroundMusic() = default;
    BackgroundMusic(const BackgroundMusic&) = delete;
    auto operator=(const BackgroundMusic&) -> BackgroundMusic& = delete;

    BackgroundMusic(BackgroundMusic&& other) noexcept
        : spec_{other.spec_}, pcm_{std::move(other.pcm_)},
          stream_{std::exchange(other.stream_, nullptr)},
          cursor_{std::exchange(other.cursor_, 0zu)},
          owns_audio_subsystem_{std::exchange(other.owns_audio_subsystem_, false)}
    {
    }

    auto operator=(BackgroundMusic&& other) noexcept -> BackgroundMusic&
    {
        if (this == &other)
        {
            return *this;
        }
        reset();
        spec_ = other.spec_;
        pcm_ = std::move(other.pcm_);
        stream_ = std::exchange(other.stream_, nullptr);
        cursor_ = std::exchange(other.cursor_, 0zu);
        owns_audio_subsystem_ = std::exchange(other.owns_audio_subsystem_, false);
        return *this;
    }

    ~BackgroundMusic()
    {
        reset();
    }

    [[nodiscard]] static auto load_looped_ogg(const std::filesystem::path& filepath, f32 gain)
        -> std::optional<BackgroundMusic>
    {
        const auto should_own_audio_subsystem = !SDL_WasInit(SDL_INIT_AUDIO);
        if (should_own_audio_subsystem && !SDL_InitSubSystem(SDL_INIT_AUDIO))
        {
            std::println(stderr, "Failed to initialize SDL audio: {}", SDL_GetError());
            return std::nullopt;
        }
        struct AudioSubsystemGuard
        {
            bool owns{};
            ~AudioSubsystemGuard()
            {
                if (owns)
                {
                    SDL_QuitSubSystem(SDL_INIT_AUDIO);
                }
            }

            [[nodiscard]] auto release() -> bool
            {
                return std::exchange(owns, false);
            }
        } audio_guard{.owns = should_own_audio_subsystem};

        const auto filepath_string = filepath.string();
        OggVorbis_File vorbis{};
        if (const auto open_result = ov_fopen(filepath_string.c_str(), &vorbis); open_result != 0)
        {
            std::println(
                stderr,
                "Failed to open OGG music {}: ov_fopen returned {}",
                filepath_string,
                open_result
            );
            return std::nullopt;
        }

        struct VorbisGuard
        {
            OggVorbis_File* file{};
            ~VorbisGuard()
            {
                if (file != nullptr)
                {
                    ov_clear(file);
                }
            }
        } vorbis_guard{.file = &vorbis};

        const auto* info = ov_info(&vorbis, -1);
        if (info == nullptr || info->channels <= 0 || info->rate <= 0
            || info->rate > std::numeric_limits<int>::max())
        {
            std::println(stderr, "Invalid Vorbis stream info in {}", filepath_string);
            return std::nullopt;
        }

        std::vector<std::byte> pcm{};
        if (const auto frame_count = ov_pcm_total(&vorbis, -1); frame_count > 0)
        {
            constexpr auto bytes_per_sample = 2;
            const auto bytes_per_frame =
                static_cast<ogg_int64_t>(info->channels) * bytes_per_sample;
            const auto byte_count = frame_count * bytes_per_frame;
            if (byte_count > 0
                && byte_count <= static_cast<ogg_int64_t>(std::numeric_limits<usize>::max()))
            {
                pcm.reserve(static_cast<usize>(byte_count));
            }
        }

        std::array<char, 32 * 1024> decode_buf{};
        auto bitstream = 0;
        while (true)
        {
            const auto bytes_read = ov_read(
                &vorbis, decode_buf.data(), static_cast<int>(decode_buf.size()), 0, 2, 1, &bitstream
            );
            if (bytes_read == 0)
            {
                break;
            }
            if (bytes_read < 0)
            {
                std::println(
                    stderr,
                    "Failed while decoding OGG music {}: ov_read returned {}",
                    filepath_string,
                    bytes_read
                );
                return std::nullopt;
            }

            const auto old_size = pcm.size();
            pcm.resize(old_size + static_cast<usize>(bytes_read));
            std::memcpy(pcm.data() + old_size, decode_buf.data(), static_cast<usize>(bytes_read));
        }

        if (pcm.empty())
        {
            std::println(stderr, "Decoded no PCM data from {}", filepath_string);
            return std::nullopt;
        }

        BackgroundMusic out{};
        out.spec_ = SDL_AudioSpec{
            .format = SDL_AUDIO_S16LE,
            .channels = info->channels,
            .freq = static_cast<int>(info->rate),
        };
        out.pcm_ = std::move(pcm);
        out.owns_audio_subsystem_ = audio_guard.release();
        out.stream_ = SDL_OpenAudioDeviceStream(
            SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &out.spec_, nullptr, nullptr
        );
        if (out.stream_ == nullptr)
        {
            std::println(stderr, "Failed to open SDL audio stream: {}", SDL_GetError());
            return std::nullopt;
        }
        if (!SDL_SetAudioStreamGain(out.stream_, gain))
        {
            std::println(stderr, "Failed to set music gain: {}", SDL_GetError());
        }
        if (!SDL_ResumeAudioStreamDevice(out.stream_))
        {
            std::println(stderr, "Failed to start SDL audio stream: {}", SDL_GetError());
            return std::nullopt;
        }
        out.pump();
        std::println("Looping background music: {}", filepath.filename().string());
        return out;
    }

    auto pump() -> void
    {
        if (stream_ == nullptr || pcm_.empty())
        {
            return;
        }

        constexpr auto target_queued_bytes = 256 * 1024;
        constexpr auto max_chunk_bytes = 64zu * 1024zu;

        auto queued_bytes = SDL_GetAudioStreamQueued(stream_);
        if (queued_bytes < 0)
        {
            std::println(stderr, "Failed to query SDL audio stream queue: {}", SDL_GetError());
            return;
        }

        while (queued_bytes < target_queued_bytes)
        {
            const auto bytes_available = pcm_.size() - cursor_;
            const auto chunk_size = std::min(max_chunk_bytes, bytes_available);
            if (chunk_size == 0)
            {
                cursor_ = 0;
                continue;
            }

            const auto chunk_size_int = static_cast<int>(chunk_size);
            if (!SDL_PutAudioStreamData(stream_, pcm_.data() + cursor_, chunk_size_int))
            {
                std::println(stderr, "Failed to queue background music: {}", SDL_GetError());
                return;
            }
            cursor_ = (cursor_ + chunk_size) % pcm_.size();
            queued_bytes += chunk_size_int;
        }
    }

  private:
    auto reset() -> void
    {
        if (stream_ != nullptr)
        {
            SDL_DestroyAudioStream(stream_);
            stream_ = nullptr;
        }
        if (owns_audio_subsystem_)
        {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            owns_audio_subsystem_ = false;
        }
    }

    SDL_AudioSpec spec_{};
    std::vector<std::byte> pcm_{};
    SDL_AudioStream* stream_{};
    usize cursor_{};
    bool owns_audio_subsystem_{};
};

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
    f32 interval{};
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

using MdlSkinData = std::vector<u8>;

struct MdlSkin
{
    std::vector<f32> intervals{};
    std::vector<MdlSkinData> images{};
};

struct MdlBinary
{
    MdlHeader header{};
    std::vector<MdlSkin> skins{};
    std::vector<MdlVertex> stverts{};
    std::vector<MdlTriangle> triangles{};
    std::vector<MdlFrame> frames{};
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

}  // namespace ds_vk_quake

auto main() -> int
{
    using namespace ds_vk_quake;
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
