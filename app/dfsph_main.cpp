#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <zlib.h>

namespace
{
using namespace ds_vk;

#ifndef DS_VK_ASSET_DIR
#    define DS_VK_ASSET_DIR "assets"
#endif

constexpr auto k_particle_radius = 0.025f;
constexpr auto k_frame_dt = 1.0f / 60.0f;
constexpr u32 k_particle_object_base{100000u};
constexpr u32 k_rigid_body_object_base{200000u};
constexpr u32 k_surface_object_base{300000u};
constexpr std::array<char, 8> k_surface_magic{'D', 'F', 'S', 'U', 'R', 'F', '1', '\0'};
constexpr u32 k_surface_cache_version{3u};
constexpr u32 k_min_surface_cache_version{1u};
constexpr auto k_surface_frame_dir = "frames";
constexpr auto k_gzip_read_chunk_size = 1zu << 20zu;
constexpr auto k_max_surface_preload_workers = 8zu;
constexpr auto k_position_quantization_max = 65535.0f;
constexpr auto k_normal_quantization_max = 32767.0f;

struct ParticleSample
{
    Vec3 position{};
    Vec3 velocity{};
    u32 id{};
    f32 density{};
    f32 advected_density{};
};

struct ParticleFrame
{
    u32 index{};
    f32 time_seconds{};
    std::vector<ParticleSample> particles{};
    f32 max_speed{};
    f32 max_density{};
};

struct SurfaceFrame
{
    u32 index{};
    f32 time_seconds{};
    MeshData mesh{};
};

struct SurfaceFrameLoadResult
{
    SurfaceFrame frame{};
    f32 decompress_ms{};
    f32 decode_ms{};
};

struct SurfacePreloadWorkerStats
{
    usize loaded_frames{};
    usize max_vertices{};
    usize max_triangles{};
};

struct ProfileTimer
{
    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};

    [[nodiscard]] auto elapsed_ms() const -> f32
    {
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<f32, std::milli>(end - start).count();
    }
};

struct ParticleHit
{
    usize particle_index{k_invalid_index};
    f32 depth{};
    f32 screen_distance{};
    Vec3 position{};
};

struct RigidBodyVtkId
{
    u32 body_id{};
    u32 frame_number{};
};

struct RigidBodySlot
{
    u32 body_id{};
    MeshHandle mesh{};
    Material material{};
    usize cached_frame_index{k_invalid_index};
    bool available{};
};

struct BinaryCursor
{
    std::vector<char> data{};
    usize offset{};
};

struct DfsphSceneConfig
{
    const char* id{};
    const char* label{};
    const char* vtk_dir{};
    const char* surface_dir{};
    Aabb bounds{};
    f32 particle_radius{k_particle_radius};
    f32 support_radius{4.0f * k_particle_radius};
    usize default_velocity_stride{24zu};
    usize default_max_velocity_arrows{600zu};
    CameraConfig camera{};
};

[[nodiscard]] auto asset_path(const std::filesystem::path& relative) -> std::filesystem::path
{
    return std::filesystem::path{DS_VK_ASSET_DIR} / relative;
}

[[nodiscard]] auto available_scenes() noexcept -> std::span<const DfsphSceneConfig>
{
    static const std::array scenes{
        DfsphSceneConfig{
            .id = "dambreak_small_iisph_v1",
            .label = "Dam Break Small IISPH",
            .vtk_dir = "dfsph/dambreak_small_iisph_v1/vtk",
            .surface_dir = "dfsph/dambreak_small_iisph_v1/surface",
            .bounds = {.min = {-2.2f, -0.8f, -0.05f}, .max = {2.2f, 0.8f, 1.35f}},
            .particle_radius = 0.025f,
            .support_radius = 0.100f,
            .default_velocity_stride = 24zu,
            .default_max_velocity_arrows = 600zu,
            .camera =
                {
                    .pivot = {0.0f, 0.0f, 0.5f},
                    .distance = 4.7f,
                    .yaw = glm::radians(42.0f),
                    .pitch = glm::radians(24.0f),
                },
        },
        DfsphSceneConfig{
            .id = "dambreak_20k_600f_v1",
            .label = "Dam Break 20k",
            .vtk_dir = "dfsph/dambreak_20k_600f_v1/vtk",
            .surface_dir = "dfsph/dambreak_20k_600f_v1/surface",
            .bounds = {.min = {-3.5f, -2.2f, -0.05f}, .max = {3.0f, 1.8f, 1.75f}},
            .particle_radius = 0.025f,
            .support_radius = 0.100f,
            .default_velocity_stride = 48zu,
            .default_max_velocity_arrows = 900zu,
            .camera =
                {
                    .pivot = {-0.25f, -0.2f, 0.82f},
                    .distance = 6.7f,
                    .yaw = glm::radians(42.0f),
                    .pitch = glm::radians(24.0f),
                },
        },
        DfsphSceneConfig{
            .id = "dambreak_50k_600f_dfsph_v2",
            .label = "Dam Break 50k",
            .vtk_dir = "dfsph/dambreak_50k_600f_dfsph_v2/vtk",
            .surface_dir = "dfsph/dambreak_50k_600f_dfsph_v2/surface",
            .bounds = {.min = {-3.5f, -2.2f, -0.05f}, .max = {3.7f, 2.4f, 2.3f}},
            .particle_radius = 0.025f,
            .support_radius = 0.100f,
            .default_velocity_stride = 80zu,
            .default_max_velocity_arrows = 1200zu,
            .camera =
                {
                    .pivot = {0.1f, 0.1f, 1.1f},
                    .distance = 7.8f,
                    .yaw = glm::radians(42.0f),
                    .pitch = glm::radians(24.0f),
                },
        },
        DfsphSceneConfig{
            .id = "twoway_rigidbody_50k_4bodies_dfsph_v1",
            .label = "Two-Way Rigid Bodies 50k",
            .vtk_dir = "dfsph/twoway_rigidbody_50k_4bodies_dfsph_v1/vtk",
            .surface_dir = "dfsph/twoway_rigidbody_50k_4bodies_dfsph_v1/surface",
            .bounds = {.min = {-2.0f, -1.25f, -0.05f}, .max = {2.0f, 1.25f, 4.0f}},
            .particle_radius = 0.025f,
            .support_radius = 0.100f,
            .default_velocity_stride = 80zu,
            .default_max_velocity_arrows = 1200zu,
            .camera =
                {
                    .pivot = {0.0f, 0.0f, 1.7f},
                    .distance = 6.2f,
                    .yaw = glm::radians(38.0f),
                    .pitch = glm::radians(22.0f),
                },
        },
        DfsphSceneConfig{
            .id = "viscous_bunny_dfsph_bender2017_nu_000_v2",
            .label = "Viscous Bunny nu=0",
            .vtk_dir = "dfsph/viscous_bunny_dfsph_bender2017_nu_000_v2/vtk",
            .surface_dir = "dfsph/viscous_bunny_dfsph_bender2017_nu_000_v2/surface",
            .bounds = {.min = {-3.0f, -3.0f, -0.25f}, .max = {3.0f, 3.0f, 3.6f}},
            .particle_radius = 0.01285f,
            .support_radius = 0.0514f,
            .default_velocity_stride = 72zu,
            .default_max_velocity_arrows = 1000zu,
            .camera =
                {
                    .pivot = {0.0f, 0.0f, 1.5f},
                    .distance = 7.1f,
                    .yaw = glm::radians(42.0f),
                    .pitch = glm::radians(21.0f),
                },
        },
        DfsphSceneConfig{
            .id = "viscous_bunny_dfsph_bender2017_nu_025_v2",
            .label = "Viscous Bunny nu=0.25",
            .vtk_dir = "dfsph/viscous_bunny_dfsph_bender2017_nu_025_v2/vtk",
            .surface_dir = "dfsph/viscous_bunny_dfsph_bender2017_nu_025_v2/surface",
            .bounds = {.min = {-3.0f, -3.0f, -0.25f}, .max = {3.0f, 3.0f, 3.6f}},
            .particle_radius = 0.01285f,
            .support_radius = 0.0514f,
            .default_velocity_stride = 72zu,
            .default_max_velocity_arrows = 1000zu,
            .camera =
                {
                    .pivot = {0.0f, 0.0f, 1.5f},
                    .distance = 7.1f,
                    .yaw = glm::radians(42.0f),
                    .pitch = glm::radians(21.0f),
                },
        },
        DfsphSceneConfig{
            .id = "viscous_bunny_dfsph_bender2017_nu_050_v2",
            .label = "Viscous Bunny nu=0.5",
            .vtk_dir = "dfsph/viscous_bunny_dfsph_bender2017_nu_050_v2/vtk",
            .surface_dir = "dfsph/viscous_bunny_dfsph_bender2017_nu_050_v2/surface",
            .bounds = {.min = {-3.0f, -3.0f, -0.25f}, .max = {3.0f, 3.0f, 3.6f}},
            .particle_radius = 0.01285f,
            .support_radius = 0.0514f,
            .default_velocity_stride = 72zu,
            .default_max_velocity_arrows = 1000zu,
            .camera = {
                .pivot = {0.0f, 0.0f, 1.5f},
                .distance = 7.1f,
                .yaw = glm::radians(42.0f),
                .pitch = glm::radians(21.0f),
            },
        },
    };
    return scenes;
}

[[nodiscard]] auto scene_index_for_id(std::string_view id) noexcept -> std::optional<usize>
{
    const auto scenes = available_scenes();
    for (auto i = 0zu; i < scenes.size(); ++i)
    {
        if (id == scenes[i].id)
        {
            return i;
        }
    }
    return std::nullopt;
}

auto print_usage(const char* program) -> void
{
    std::cerr << "usage: " << program
              << " [--smoke-frames N] [--screenshot PATH] [--hide-ui]"
                 " [--transparent-screenshot] [--scene-id ID] [--show-mesh]"
                 " [--hide-mesh] [--show-particles] [--hide-particles]"
                 " [--playback-speed X] [--profile]\n";
    std::cerr << "available scene IDs:\n";
    for (const auto& scene : available_scenes())
    {
        std::cerr << "  " << scene.id << '\n';
    }
}

[[nodiscard]] auto parse_u32(std::string_view text, u32 fallback) noexcept -> u32
{
    try
    {
        return static_cast<u32>(std::stoul(std::string{text}));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

[[nodiscard]] auto parse_f32(std::string_view text, f32 fallback) noexcept -> f32
{
    try
    {
        return std::stof(std::string{text});
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

[[nodiscard]] auto read_all_bytes(const std::filesystem::path& path) -> std::vector<char>
{
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in)
    {
        throw std::runtime_error(std::format("failed to open VTK file: {}", path.string()));
    }
    const auto end = in.tellg();
    if (end <= 0)
    {
        throw std::runtime_error(std::format("empty VTK file: {}", path.string()));
    }
    std::vector<char> bytes(static_cast<usize>(end));
    in.seekg(0, std::ios::beg);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!in)
    {
        throw std::runtime_error(std::format("failed to read VTK file: {}", path.string()));
    }
    return bytes;
}

[[nodiscard]] auto read_line(BinaryCursor& cursor) -> std::string
{
    if (cursor.offset >= cursor.data.size())
    {
        throw std::runtime_error("unexpected end of VTK file while reading line");
    }
    const auto begin = cursor.offset;
    while (cursor.offset < cursor.data.size() and cursor.data[cursor.offset] != '\n')
    {
        ++cursor.offset;
    }
    if (cursor.offset >= cursor.data.size())
    {
        throw std::runtime_error("unterminated VTK line");
    }
    std::string line{cursor.data.data() + begin, cursor.offset - begin};
    ++cursor.offset;
    if (!line.empty() and line.back() == '\r')
    {
        line.pop_back();
    }
    return line;
}

auto consume_optional_newline(BinaryCursor& cursor) -> void
{
    if (cursor.offset < cursor.data.size() and cursor.data[cursor.offset] == '\r')
    {
        ++cursor.offset;
    }
    if (cursor.offset < cursor.data.size() and cursor.data[cursor.offset] == '\n')
    {
        ++cursor.offset;
    }
}

auto require_available(const BinaryCursor& cursor, usize count) -> void
{
    if (cursor.offset + count > cursor.data.size())
    {
        throw std::runtime_error("unexpected end of VTK binary payload");
    }
}

[[nodiscard]] auto read_be_u32(BinaryCursor& cursor) -> u32
{
    require_available(cursor, 4zu);
    const auto b0 = static_cast<u32>(static_cast<unsigned char>(cursor.data[cursor.offset + 0zu]));
    const auto b1 = static_cast<u32>(static_cast<unsigned char>(cursor.data[cursor.offset + 1zu]));
    const auto b2 = static_cast<u32>(static_cast<unsigned char>(cursor.data[cursor.offset + 2zu]));
    const auto b3 = static_cast<u32>(static_cast<unsigned char>(cursor.data[cursor.offset + 3zu]));
    cursor.offset += 4zu;
    return (b0 << 24u) | (b1 << 16u) | (b2 << 8u) | b3;
}

[[nodiscard]] auto read_be_f32(BinaryCursor& cursor) -> f32
{
    return std::bit_cast<f32>(read_be_u32(cursor));
}

[[nodiscard]] auto read_be_f64_as_f32(BinaryCursor& cursor) -> f32
{
    require_available(cursor, 8zu);
    auto bits = 0ull;
    for (auto i = 0zu; i < 8zu; ++i)
    {
        bits = (bits << 8u)
               | static_cast<u64>(static_cast<unsigned char>(cursor.data[cursor.offset + i]));
    }
    cursor.offset += 8zu;
    return static_cast<f32>(std::bit_cast<f64>(bits));
}

[[nodiscard]] auto read_be_real(BinaryCursor& cursor, std::string_view type) -> f32
{
    if (type == "float") return read_be_f32(cursor);
    if (type == "double") return read_be_f64_as_f32(cursor);
    throw std::runtime_error(std::format("unsupported VTK real type: {}", type));
}

[[nodiscard]] auto splishsplash_y_up_to_z_up(Vec3 value) noexcept -> Vec3
{
    return {value.x, value.z, value.y};
}

[[nodiscard]] auto split_words(std::string_view line) -> std::vector<std::string>
{
    std::istringstream in{std::string{line}};
    std::vector<std::string> words{};
    std::string word{};
    while (in >> word)
    {
        words.push_back(word);
    }
    return words;
}

[[nodiscard]] auto contains_id(std::span<const u32> ids, u32 id) noexcept -> bool
{
    return std::ranges::find(ids, id) != ids.end();
}

auto append_unique_id(std::vector<u32>& ids, u32 id) -> void
{
    if (!contains_id(std::span<const u32>{ids}, id))
    {
        ids.push_back(id);
    }
}

[[nodiscard]] auto parse_u32_word(std::string_view word, std::string_view context) -> u32
{
    try
    {
        return static_cast<u32>(std::stoul(std::string{word}));
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::format("failed to parse {} from '{}'", context, word));
    }
}

auto skip_be_u32_values(BinaryCursor& cursor, u32 count) -> void
{
    require_available(cursor, static_cast<usize>(count) * 4zu);
    cursor.offset += static_cast<usize>(count) * 4zu;
    consume_optional_newline(cursor);
}

[[nodiscard]] auto read_points(BinaryCursor& cursor, ParticleFrame& frame) -> u32
{
    const auto words = split_words(read_line(cursor));
    if (words.size() != 3zu or words[0] != "POINTS")
    {
        throw std::runtime_error("expected VTK POINTS line");
    }
    const auto point_count_u32 = parse_u32_word(words[1], "POINTS count");
    const auto point_count = static_cast<usize>(point_count_u32);
    frame.particles.assign(point_count, {});
    for (auto i = 0zu; i < point_count; ++i)
    {
        auto& particle = frame.particles[i];
        particle.position = {
            read_be_real(cursor, words[2]),
            read_be_real(cursor, words[2]),
            read_be_real(cursor, words[2]),
        };
        particle.position = splishsplash_y_up_to_z_up(particle.position);
        particle.id = static_cast<u32>(i);
    }
    consume_optional_newline(cursor);
    return point_count_u32;
}

auto read_cells(BinaryCursor& cursor, u32 expected_points) -> void
{
    const auto words = split_words(read_line(cursor));
    if (words.size() != 3zu or words[0] != "CELLS")
    {
        throw std::runtime_error("expected VTK CELLS line");
    }
    const auto cell_count = parse_u32_word(words[1], "CELLS count");
    const auto cell_entries = parse_u32_word(words[2], "CELLS entries");
    if (cell_count != expected_points)
    {
        throw std::runtime_error("VTK CELLS count does not match POINTS count");
    }
    skip_be_u32_values(cursor, cell_entries);

    const auto type_words = split_words(read_line(cursor));
    if (type_words.size() != 2zu or type_words[0] != "CELL_TYPES")
    {
        throw std::runtime_error("expected VTK CELL_TYPES line");
    }
    const auto type_count = parse_u32_word(type_words[1], "CELL_TYPES count");
    if (type_count != expected_points)
    {
        throw std::runtime_error("VTK CELL_TYPES count does not match POINTS count");
    }
    skip_be_u32_values(cursor, type_count);
}

[[nodiscard]] auto parse_rigid_body_vtk_id(const std::filesystem::path& path)
    -> std::optional<RigidBodyVtkId>
{
    if (path.extension() != ".vtk")
    {
        return std::nullopt;
    }

    const auto stem = path.stem().string();
    constexpr std::string_view prefix{"rb_data_"};
    if (!stem.starts_with(prefix))
    {
        return std::nullopt;
    }

    const auto rest = stem.substr(prefix.size());
    const auto separator = rest.find('_');
    if (separator == std::string::npos or separator + 1zu >= rest.size())
    {
        return std::nullopt;
    }

    try
    {
        return RigidBodyVtkId{
            .body_id = static_cast<u32>(std::stoul(rest.substr(0zu, separator))),
            .frame_number = static_cast<u32>(std::stoul(rest.substr(separator + 1zu))),
        };
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

[[nodiscard]] auto discover_rigid_body_ids(const std::filesystem::path& vtk_dir) -> std::vector<u32>
{
    std::vector<u32> body_ids{};
    if (!std::filesystem::is_directory(vtk_dir))
    {
        return body_ids;
    }

    for (const auto& entry : std::filesystem::directory_iterator(vtk_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const auto vtk_id = parse_rigid_body_vtk_id(entry.path());
        if (!vtk_id.has_value() or vtk_id->body_id == 0u)
        {
            continue;
        }
        body_ids.push_back(vtk_id->body_id);
    }

    std::ranges::sort(body_ids);
    body_ids.erase(std::ranges::unique(body_ids).begin(), body_ids.end());
    return body_ids;
}

[[nodiscard]] auto
rigid_body_vtk_path(const std::filesystem::path& vtk_dir, u32 body_id, u32 frame_number)
    -> std::filesystem::path
{
    return vtk_dir / std::format("rb_data_{}_{}.vtk", body_id, frame_number);
}

[[nodiscard]] auto
rigid_body_vtk_path_for_frame(const std::filesystem::path& vtk_dir, u32 body_id, usize frame_index)
    -> std::filesystem::path
{
    const auto frame_number = static_cast<u32>(frame_index);
    const std::array candidates{frame_number + 1u, frame_number, 0u};
    for (const auto candidate : candidates)
    {
        const auto path = rigid_body_vtk_path(vtk_dir, body_id, candidate);
        if (std::filesystem::is_regular_file(path))
        {
            return path;
        }
    }
    return {};
}

[[nodiscard]] auto read_mesh_points(BinaryCursor& cursor) -> std::vector<Vec3>
{
    const auto words = split_words(read_line(cursor));
    if (words.size() != 3zu or words[0] != "POINTS")
    {
        throw std::runtime_error("expected VTK POINTS line");
    }

    const auto point_count = static_cast<usize>(parse_u32_word(words[1], "POINTS count"));
    std::vector<Vec3> points(point_count);
    for (auto& point : points)
    {
        point = {
            read_be_real(cursor, words[2]),
            read_be_real(cursor, words[2]),
            read_be_real(cursor, words[2]),
        };
        point = splishsplash_y_up_to_z_up(point);
    }
    consume_optional_newline(cursor);
    return points;
}

[[nodiscard]] auto read_triangle_cells(BinaryCursor& cursor, usize point_count) -> std::vector<u32>
{
    const auto words = split_words(read_line(cursor));
    if (words.size() != 3zu or words[0] != "CELLS")
    {
        throw std::runtime_error("expected VTK CELLS line");
    }

    const auto cell_count = static_cast<usize>(parse_u32_word(words[1], "CELLS count"));
    const auto cell_entries = static_cast<usize>(parse_u32_word(words[2], "CELLS entries"));
    if (cell_entries != cell_count * 4zu)
    {
        throw std::runtime_error("VTK rigid-body mesh expects triangle CELLS");
    }

    std::vector<u32> indices{};
    indices.reserve(cell_count * 3zu);
    for (auto i = 0zu; i < cell_count; ++i)
    {
        const auto vertices_in_cell = read_be_u32(cursor);
        if (vertices_in_cell != 3u)
        {
            throw std::runtime_error("VTK rigid-body mesh contains a non-triangle cell");
        }

        const auto a = read_be_u32(cursor);
        const auto b = read_be_u32(cursor);
        const auto c = read_be_u32(cursor);
        if (static_cast<usize>(a) >= point_count or static_cast<usize>(b) >= point_count
            or static_cast<usize>(c) >= point_count)
        {
            throw std::runtime_error("VTK rigid-body mesh index out of range");
        }

        indices.push_back(a);
        indices.push_back(c);
        indices.push_back(b);
    }
    consume_optional_newline(cursor);

    const auto type_words = split_words(read_line(cursor));
    if (type_words.size() != 2zu or type_words[0] != "CELL_TYPES")
    {
        throw std::runtime_error("expected VTK CELL_TYPES line");
    }
    const auto type_count = static_cast<usize>(parse_u32_word(type_words[1], "CELL_TYPES count"));
    if (type_count != cell_count)
    {
        throw std::runtime_error("VTK CELL_TYPES count does not match rigid-body cell count");
    }
    for (auto i = 0zu; i < type_count; ++i)
    {
        if (read_be_u32(cursor) != 5u)
        {
            throw std::runtime_error("VTK rigid-body mesh expects triangle cell type");
        }
    }
    consume_optional_newline(cursor);
    return indices;
}

[[nodiscard]] auto finite_normal_or_z(Vec3 value) noexcept -> Vec3
{
    const auto length_squared = glm::dot(value, value);
    if (!std::isfinite(length_squared) or length_squared <= 1.0e-12f)
    {
        return k_axis_z;
    }
    return value * glm::inversesqrt(length_squared);
}

[[nodiscard]] auto read_gzip_file(const std::filesystem::path& path) -> std::vector<u8>
{
    const auto path_string = path.string();
    auto* file = gzopen(path_string.c_str(), "rb");
    if (file == nullptr)
    {
        throw std::runtime_error(
            std::format("failed to open compressed surface frame: {}", path_string)
        );
    }

    std::vector<u8> payload{};
    std::vector<u8> buffer(k_gzip_read_chunk_size);
    try
    {
        while (true)
        {
            const auto result =
                gzread(file, buffer.data(), static_cast<unsigned int>(buffer.size()));
            if (result < 0)
            {
                auto errnum = Z_OK;
                const auto* message = gzerror(file, &errnum);
                throw std::runtime_error(
                    std::format(
                        "failed to read compressed surface cache: {}",
                        message == nullptr ? "zlib error" : message
                    )
                );
            }
            if (result == 0)
            {
                break;
            }
            payload.insert(payload.end(), buffer.begin(), buffer.begin() + result);
        }

        const auto close_result = gzclose(file);
        file = nullptr;
        if (close_result != Z_OK)
        {
            throw std::runtime_error(
                std::format("failed to close compressed surface frame: {}", path_string)
            );
        }
    }
    catch (...)
    {
        if (file != nullptr)
        {
            (void) gzclose(file);
        }
        throw;
    }
    return payload;
}

template <typename T, typename ReadBytes>
[[nodiscard]] auto read_surface_scalar(ReadBytes& read_bytes) -> T
{
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    read_bytes(&value, sizeof(T));
    return value;
}

template <typename ReadBytes>
[[nodiscard]] auto read_surface_vec3(ReadBytes& read_bytes) -> Vec3
{
    return {
        read_surface_scalar<f32>(read_bytes),
        read_surface_scalar<f32>(read_bytes),
        read_surface_scalar<f32>(read_bytes),
    };
}

[[nodiscard]] auto decode_position_component(u16 value, f32 origin, f32 extent) noexcept -> f32
{
    return origin + extent * (static_cast<f32>(value) / k_position_quantization_max);
}

[[nodiscard]] auto decode_normal_component(i16 value) noexcept -> f32
{
    return static_cast<f32>(value) / k_normal_quantization_max;
}

template <typename ReadBytes>
[[nodiscard]] auto read_surface_payload_header(ReadBytes& read_bytes) -> u32
{
    std::array<char, 8> magic{};
    read_bytes(magic.data(), magic.size());
    if (magic != k_surface_magic)
    {
        throw std::runtime_error("invalid surface cache magic");
    }

    const auto version = read_surface_scalar<u32>(read_bytes);
    if (version < k_min_surface_cache_version or version > k_surface_cache_version)
    {
        throw std::runtime_error(std::format("unsupported surface cache version: {}", version));
    }
    return version;
}

template <typename ReadBytes>
[[nodiscard]] auto read_surface_frame_payload(ReadBytes& read_bytes, u32 version) -> SurfaceFrame
{
    SurfaceFrame frame{};
    frame.index = read_surface_scalar<u32>(read_bytes);
    frame.time_seconds = read_surface_scalar<f32>(read_bytes);
    const auto vertex_count = read_surface_scalar<u32>(read_bytes);
    const auto index_count = read_surface_scalar<u32>(read_bytes);
    frame.mesh.vertices.resize(static_cast<usize>(vertex_count));
    frame.mesh.indices.resize(static_cast<usize>(index_count));

    if (version == 1u)
    {
        for (auto& vertex : frame.mesh.vertices)
        {
            vertex = Vertex{
                .position = read_surface_vec3(read_bytes),
                .normal = finite_normal_or_z(read_surface_vec3(read_bytes)),
                .color = Color::white,
                .texcoord = {0.0f, 0.0f},
            };
        }
    }
    else
    {
        const auto bounds_min = read_surface_vec3(read_bytes);
        const auto extent = glm::max(read_surface_vec3(read_bytes), Vec3{0.0f});
        for (auto& vertex : frame.mesh.vertices)
        {
            const auto qx = read_surface_scalar<u16>(read_bytes);
            const auto qy = read_surface_scalar<u16>(read_bytes);
            const auto qz = read_surface_scalar<u16>(read_bytes);
            const auto nx = read_surface_scalar<i16>(read_bytes);
            const auto ny = read_surface_scalar<i16>(read_bytes);
            const auto nz = read_surface_scalar<i16>(read_bytes);
            vertex = Vertex{
                .position =
                    {
                        decode_position_component(qx, bounds_min.x, extent.x),
                        decode_position_component(qy, bounds_min.y, extent.y),
                        decode_position_component(qz, bounds_min.z, extent.z),
                    },
                .normal = finite_normal_or_z({
                    decode_normal_component(nx),
                    decode_normal_component(ny),
                    decode_normal_component(nz),
                }),
                .color = Color::white,
                .texcoord = {0.0f, 0.0f},
            };
        }
    }

    for (auto& index : frame.mesh.indices)
    {
        index = read_surface_scalar<u32>(read_bytes);
        if (index >= vertex_count)
        {
            throw std::runtime_error("surface cache index out of range");
        }
    }
    if (frame.mesh.vertices.empty() or frame.mesh.indices.empty()
        or frame.mesh.indices.size() % 3zu != 0zu)
    {
        throw std::runtime_error("surface cache frame has invalid mesh shape");
    }
    return frame;
}

[[nodiscard]] auto
surface_frame_payload_path(const std::filesystem::path& surface_dir, u32 frame_index)
    -> std::filesystem::path
{
    return surface_dir / k_surface_frame_dir / std::format("frame_{:06}.mesh.gz", frame_index);
}

[[nodiscard]] auto read_surface_mesh_frame(const std::filesystem::path& path)
    -> SurfaceFrameLoadResult
{
    if constexpr (std::endian::native != std::endian::little)
    {
        throw std::runtime_error("surface cache currently requires little-endian host");
    }

    ProfileTimer decompress_timer{};
    const auto payload = read_gzip_file(path);
    const auto decompress_ms = decompress_timer.elapsed_ms();
    ProfileTimer decode_timer{};
    auto offset = 0zu;
    auto read_bytes = [&](void* data, usize size) -> void
    {
        if (offset + size > payload.size())
        {
            throw std::runtime_error(
                std::format("unexpected end of surface cache payload: {}", path.string())
            );
        }
        std::memcpy(data, payload.data() + offset, size);
        offset += size;
    };

    const auto version = read_surface_payload_header(read_bytes);
    const auto frame_count = read_surface_scalar<u32>(read_bytes);
    if (frame_count != 1u)
    {
        throw std::runtime_error(
            std::format(
                "per-frame surface payload must contain exactly one frame, got {}", frame_count
            )
        );
    }
    return SurfaceFrameLoadResult{
        .frame = read_surface_frame_payload(read_bytes, version),
        .decompress_ms = decompress_ms,
        .decode_ms = decode_timer.elapsed_ms(),
    };
}

[[nodiscard]] auto surface_preload_worker_count(usize frame_count) noexcept -> usize
{
    if (frame_count == 0zu)
    {
        return 0zu;
    }
    const auto hardware_thread_count =
        static_cast<usize>(std::max(1u, std::thread::hardware_concurrency()));
    return std::clamp(
        std::min(hardware_thread_count, frame_count), 1zu, k_max_surface_preload_workers
    );
}

[[nodiscard]] auto exception_message(const std::exception_ptr& exception) -> std::string
{
    if (exception == nullptr)
    {
        return "unknown error";
    }

    try
    {
        std::rethrow_exception(exception);
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    catch (...)
    {
        return "unknown error";
    }
}

[[nodiscard]] auto read_rigid_body_vtk_mesh(const std::filesystem::path& path) -> MeshData
{
    BinaryCursor cursor{.data = read_all_bytes(path)};
    const auto header = read_line(cursor);
    if (!header.starts_with("# vtk DataFile"))
    {
        throw std::runtime_error("expected VTK DataFile header");
    }
    (void) read_line(cursor);
    if (read_line(cursor) != "BINARY")
    {
        throw std::runtime_error("only binary VTK rigid-body mesh files are supported");
    }
    if (read_line(cursor) != "DATASET UNSTRUCTURED_GRID")
    {
        throw std::runtime_error("expected VTK UNSTRUCTURED_GRID dataset");
    }

    const auto points = read_mesh_points(cursor);
    MeshData mesh{};
    mesh.indices = read_triangle_cells(cursor, points.size());
    mesh.vertices.resize(points.size());

    std::vector<Vec3> normals(points.size(), Vec3{0.0f});
    for (auto i = 0zu; i + 2zu < mesh.indices.size(); i += 3zu)
    {
        const auto ia = static_cast<usize>(mesh.indices[i + 0zu]);
        const auto ib = static_cast<usize>(mesh.indices[i + 1zu]);
        const auto ic = static_cast<usize>(mesh.indices[i + 2zu]);
        const auto normal = glm::cross(points[ib] - points[ia], points[ic] - points[ia]);
        normals[ia] += normal;
        normals[ib] += normal;
        normals[ic] += normal;
    }
    for (auto i = 0zu; i < points.size(); ++i)
    {
        mesh.vertices[i] = Vertex{
            .position = points[i],
            .normal = finite_normal_or_z(normals[i]),
            .color = Color::white,
            .texcoord = {0.0f, 0.0f},
        };
    }
    return mesh;
}

[[nodiscard]] auto rigid_body_material(u32 body_id) noexcept -> Material
{
    Color base_color{0.62f, 0.50f, 0.36f, 1.0f};
    switch (body_id)
    {
        case 1u:
            base_color = Color{0.18f, 0.70f, 0.36f, 1.0f};
            break;
        case 2u:
            base_color = Color{0.94f, 0.78f, 0.26f, 1.0f};
            break;
        case 3u:
            base_color = Color{0.86f, 0.22f, 0.18f, 1.0f};
            break;
        default:
            break;
    }

    return Material{
        .base_color = base_color,
        .metallic = 0.0f,
        .roughness = 0.52f,
        .ambient_occlusion = 1.0f,
    };
}

auto read_ids(BinaryCursor& cursor, ParticleFrame& frame) -> void
{
    const auto scalar_words = split_words(read_line(cursor));
    if (scalar_words.size() < 4zu or scalar_words[0] != "SCALARS" or scalar_words[1] != "id")
    {
        throw std::runtime_error("expected VTK id SCALARS block");
    }
    if (scalar_words[2] != "unsigned_int")
    {
        throw std::runtime_error("expected VTK id data as unsigned_int");
    }
    const auto lookup = read_line(cursor);
    if (!lookup.starts_with("LOOKUP_TABLE"))
    {
        throw std::runtime_error("expected VTK LOOKUP_TABLE line");
    }
    for (auto& particle : frame.particles)
    {
        particle.id = read_be_u32(cursor);
    }
    consume_optional_newline(cursor);
}

auto skip_field_payload(BinaryCursor& cursor, u32 components, u32 tuples, std::string_view type)
    -> void
{
    const auto count = static_cast<usize>(components) * static_cast<usize>(tuples);
    if (type == "float" or type == "unsigned_int")
    {
        require_available(cursor, count * 4zu);
        cursor.offset += count * 4zu;
    }
    else if (type == "double")
    {
        require_available(cursor, count * 8zu);
        cursor.offset += count * 8zu;
    }
    else
    {
        throw std::runtime_error(std::format("unsupported VTK field type: {}", type));
    }
    consume_optional_newline(cursor);
}

[[nodiscard]] auto read_field(BinaryCursor& cursor, ParticleFrame& frame, u32 point_count) -> bool
{
    const auto words = split_words(read_line(cursor));
    if (words.size() != 4zu)
    {
        throw std::runtime_error("expected VTK FIELD entry line");
    }
    const auto& name = words[0];
    const auto components = parse_u32_word(words[1], "FIELD components");
    const auto tuples = parse_u32_word(words[2], "FIELD tuples");
    const auto& type = words[3];
    if (tuples != point_count)
    {
        throw std::runtime_error("VTK FIELD tuple count does not match POINTS count");
    }

    if (name == "density" and components == 1u)
    {
        for (auto& particle : frame.particles)
        {
            particle.density = read_be_real(cursor, type);
        }
        consume_optional_newline(cursor);
        return false;
    }
    if (name == "advected_density" and components == 1u)
    {
        for (auto& particle : frame.particles)
        {
            particle.advected_density = read_be_real(cursor, type);
        }
        consume_optional_newline(cursor);
        return true;
    }
    if (name == "velocity" and components == 3u)
    {
        for (auto& particle : frame.particles)
        {
            particle.velocity = {
                read_be_real(cursor, type),
                read_be_real(cursor, type),
                read_be_real(cursor, type),
            };
            particle.velocity = splishsplash_y_up_to_z_up(particle.velocity);
        }
        consume_optional_newline(cursor);
        return false;
    }

    skip_field_payload(cursor, components, tuples, type);
    return false;
}

[[nodiscard]] auto trailing_frame_number(const std::filesystem::path& path) -> u32
{
    const auto stem = path.stem().string();
    const auto pos = stem.find_last_of('_');
    if (pos == std::string::npos or pos + 1zu >= stem.size())
    {
        return 0u;
    }
    try
    {
        return static_cast<u32>(std::stoul(stem.substr(pos + 1zu)));
    }
    catch (const std::exception&)
    {
        return 0u;
    }
}

[[nodiscard]] auto discover_particle_vtk_files(const std::filesystem::path& vtk_dir)
    -> std::vector<std::filesystem::path>
{
    std::vector<std::filesystem::path> files{};
    if (!std::filesystem::is_directory(vtk_dir))
    {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(vtk_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const auto& path = entry.path();
        if (path.extension() == ".vtk" and path.filename().string().starts_with("ParticleData_"))
        {
            files.push_back(path);
        }
    }
    std::ranges::sort(
        files,
        [](const std::filesystem::path& a, const std::filesystem::path& b) -> bool
        {
            const auto af = trailing_frame_number(a);
            const auto bf = trailing_frame_number(b);
            if (af != bf)
            {
                return af < bf;
            }
            return a.string() < b.string();
        }
    );
    return files;
}

[[nodiscard]] auto
read_particle_vtk(const std::filesystem::path& path, u32 frame_index, f32 time_seconds)
    -> ParticleFrame
{
    BinaryCursor cursor{.data = read_all_bytes(path)};
    const auto header = read_line(cursor);
    if (!header.starts_with("# vtk DataFile"))
    {
        throw std::runtime_error("expected VTK DataFile header");
    }
    (void) read_line(cursor);
    if (read_line(cursor) != "BINARY")
    {
        throw std::runtime_error("only binary VTK particle files are supported");
    }
    if (read_line(cursor) != "DATASET UNSTRUCTURED_GRID")
    {
        throw std::runtime_error("expected VTK UNSTRUCTURED_GRID dataset");
    }

    ParticleFrame frame{.index = frame_index, .time_seconds = time_seconds};
    const auto point_count = read_points(cursor, frame);
    read_cells(cursor, point_count);

    const auto point_data_words = split_words(read_line(cursor));
    if (point_data_words.size() != 2zu or point_data_words[0] != "POINT_DATA")
    {
        throw std::runtime_error("expected VTK POINT_DATA line");
    }
    if (parse_u32_word(point_data_words[1], "POINT_DATA count") != point_count)
    {
        throw std::runtime_error("VTK POINT_DATA count does not match POINTS count");
    }
    read_ids(cursor, frame);

    const auto field_words = split_words(read_line(cursor));
    if (field_words.size() != 3zu or field_words[0] != "FIELD")
    {
        throw std::runtime_error("expected VTK FIELD block");
    }
    const auto field_count = parse_u32_word(field_words[2], "FIELD count");
    auto has_advected_density = false;
    for (auto i = 0u; i < field_count; ++i)
    {
        has_advected_density = read_field(cursor, frame, point_count) or has_advected_density;
    }
    for (auto& particle : frame.particles)
    {
        if (!has_advected_density)
        {
            particle.advected_density = particle.density;
        }
        frame.max_speed = std::max(frame.max_speed, glm::length(particle.velocity));
        frame.max_density = std::max(frame.max_density, particle.density);
    }
    return frame;
}

[[nodiscard]] auto select_particle_projected(
    const ParticleFrame& frame,
    const Camera& camera,
    Vec2 mouse_px,
    Vec2 viewport_px,
    f32 particle_radius
) noexcept -> std::optional<ParticleHit>
{
    const auto viewport = glm::max(viewport_px, Vec2{1.0f});
    const auto view_projection = camera.view_projection_matrix(viewport.x / viewport.y);
    const auto radius_axis = camera.right() * particle_radius;

    auto best_depth = std::numeric_limits<f32>::max();
    auto best_screen_distance = std::numeric_limits<f32>::max();
    auto best_index = k_invalid_index;
    Vec3 best_position{};

    for (auto i = 0zu; i < frame.particles.size(); ++i)
    {
        const auto center = frame.particles[i].position;
        const auto clip = view_projection * Vec4{center, 1.0f};
        if (clip.w <= 0.0f)
        {
            continue;
        }

        const Vec3 ndc{Vec3{clip} / clip.w};
        if (ndc.z < 0.0f or ndc.z > 1.0f)
        {
            continue;
        }

        const auto radius_clip = view_projection * Vec4{center + radius_axis, 1.0f};
        if (radius_clip.w <= 0.0f)
        {
            continue;
        }

        const Vec3 radius_ndc{Vec3{radius_clip} / radius_clip.w};
        const Vec2 center_px{
            (ndc.x * 0.5f + 0.5f) * viewport.x,
            (ndc.y * 0.5f + 0.5f) * viewport.y,
        };
        const Vec2 radius_px{
            (radius_ndc.x * 0.5f + 0.5f) * viewport.x,
            (radius_ndc.y * 0.5f + 0.5f) * viewport.y,
        };
        const auto pick_radius_px = std::max(4.0f, glm::length(radius_px - center_px) * 1.15f);
        const auto screen_distance = glm::length(mouse_px - center_px);
        if (screen_distance > pick_radius_px)
        {
            continue;
        }

        if (ndc.z < best_depth - 1.0e-5f
            or (std::abs(ndc.z - best_depth) <= 1.0e-5f and screen_distance < best_screen_distance))
        {
            best_depth = ndc.z;
            best_screen_distance = screen_distance;
            best_index = i;
            best_position = center;
        }
    }

    if (best_index == k_invalid_index)
    {
        return std::nullopt;
    }

    return ParticleHit{
        .particle_index = best_index,
        .depth = best_depth,
        .screen_distance = best_screen_distance,
        .position = best_position,
    };
}

[[nodiscard]] auto find_particle_index_by_id(const ParticleFrame& frame, u32 particle_id) noexcept
    -> usize
{
    for (auto i = 0zu; i < frame.particles.size(); ++i)
    {
        if (frame.particles[i].id == particle_id)
        {
            return i;
        }
    }
    return k_invalid_index;
}

[[nodiscard]] auto
find_particle_neighbors(const ParticleFrame& frame, usize particle_index, f32 support_radius)
    -> std::vector<usize>
{
    if (particle_index >= frame.particles.size() or support_radius <= 0.0f)
    {
        return {};
    }

    const auto center = frame.particles[particle_index].position;
    const auto radius_squared = support_radius * support_radius;
    std::vector<usize> neighbors{};
    for (auto i = 0zu; i < frame.particles.size(); ++i)
    {
        if (i == particle_index)
        {
            continue;
        }

        const auto delta = frame.particles[i].position - center;
        if (glm::dot(delta, delta) <= radius_squared)
        {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

[[nodiscard]] auto load_vtk_history(const std::filesystem::path& vtk_dir)
    -> std::vector<ParticleFrame>
{
    const auto files = discover_particle_vtk_files(vtk_dir);
    if (files.empty())
    {
        throw std::runtime_error(std::format("no DFSPH VTK frames found in: {}", vtk_dir.string()));
    }

    std::vector<ParticleFrame> frames{};
    frames.reserve(files.size());
    for (auto i = 0zu; i < files.size(); ++i)
    {
        frames.push_back(
            read_particle_vtk(files[i], static_cast<u32>(i), static_cast<f32>(i) * k_frame_dt)
        );
    }
    const auto particle_count = frames.front().particles.size();
    for (const auto& frame : frames)
    {
        if (frame.particles.size() != particle_count)
        {
            throw std::runtime_error("VTK sequence particle count changed between frames");
        }
    }
    return frames;
}

struct DfsphPlaybackConfig
{
    usize initial_scene_index{};
    f32 playback_speed{1.0f};
    bool show_particles{true};
    bool show_surface_mesh{};
    bool profile{};
};

struct ProfileCounter
{
    usize samples{};
    f32 total_ms{};
    f32 max_ms{};

    auto add(f32 ms) noexcept -> void
    {
        ++samples;
        total_ms += ms;
        max_ms = std::max(max_ms, ms);
    }

    [[nodiscard]] auto mean_ms() const noexcept -> f32
    {
        return samples == 0zu ? 0.0f : total_ms / static_cast<f32>(samples);
    }
};

struct DfsphProfile
{
    ProfileCounter runtime_frame{};
    ProfileCounter runtime_update{};
    ProfileCounter runtime_render{};
    ProfileCounter surface_read{};
    ProfileCounter surface_decompress{};
    ProfileCounter surface_decode{};
    ProfileCounter surface_upload{};
    ProfileCounter surface_total{};
    ProfileCounter surface_preload{};
    usize surface_frames{};
    usize surface_preloaded_frames{};
    usize max_surface_vertices{};
    usize max_surface_triangles{};

    auto add_runtime(const RuntimeStats& stats) noexcept -> void
    {
        if (stats.last_frame_ms <= 0.0f)
        {
            return;
        }
        runtime_frame.add(stats.last_frame_ms);
        runtime_update.add(stats.last_update_ms);
        runtime_render.add(stats.last_render_ms);
    }
};

class DfsphPlaybackApp
{
  public:
    explicit DfsphPlaybackApp(const DfsphPlaybackConfig& cfg) noexcept
        : scene_index_(cfg.initial_scene_index), show_particles_(cfg.show_particles),
          show_surface_mesh_(cfg.show_surface_mesh), profile_enabled_(cfg.profile)
    {
        playback_speed_ = cfg.playback_speed;
    }

    auto setup(Runtime& runtime) -> void
    {
        runtime_ = &runtime;
        particle_mesh_ = runtime.upload_mesh(make_uv_sphere({
            .radius = 1.0f,
            .slices = 8u,
            .stacks = 4u,
            .color = Color::white,
        }));
        floor_mesh_ = runtime.upload_mesh(make_quad(1.0f, Color::white));
        load_scene(scene_index_, runtime);
    }

    auto update(FrameContext& frame, f32 dt_seconds) -> void
    {
        poll_surface_mesh_preload();
        if (pending_scene_index_.has_value())
        {
            if (runtime_ == nullptr) return;
            show_surface_mesh_ = false;
            load_scene(*pending_scene_index_, *runtime_);
            pending_scene_index_.reset();
        }

        if (frame.input.space_pressed)
        {
            paused_ = !paused_;
        }
        if (profile_enabled_)
        {
            profile_.add_runtime(frame.stats);
        }

        configure_lighting(frame.draw);
        update_playback(dt_seconds);

        frame.draw.draw_mesh({
            .mesh = floor_mesh_,
            .transform =
                {
                    .translation = floor_center() - 0.012f * k_axis_z,
                    .scale = Vec3{floor_side_length(), floor_side_length(), 1.0f},
                },
            .material =
                Material{
                    .base_color = Color{0.16f, 0.18f, 0.16f, 1.0f},
                    .roughness = 0.92f,
                },
            .mask = {.shadow_producer = false},
        });
        (void) viz::draw_aabb(
            frame.draw,
            viz::AabbMarkerConfig{
                .aabb = current_scene().bounds,
                .color = Color{0.36f, 0.52f, 0.68f, 0.75f},
                .width = 0.006f,
            }
        );

        if (frames_.empty())
        {
            return;
        }

        const auto& vtk_frame = frames_[current_frame_];
        handle_selection_click(frame);
        sync_selection_to_current_frame();
        if (runtime_ != nullptr)
        {
            sync_rigid_body_meshes(*runtime_, current_frame_);
            if (show_surface_mesh_)
            {
                sync_surface_mesh(*runtime_, current_frame_);
            }
        }
        draw_surface_mesh(frame.draw);
        draw_rigid_bodies(frame.draw);

        if (show_particles_ or show_velocity_arrows_)
        {
            velocity_positions_.clear();
            velocity_vectors_.clear();
            const auto velocity_arrow_reserve =
                velocity_arrows_selection_only_
                    ? selected_particle_indices_.size() + neighbor_particle_indices_.size()
                    : vtk_frame.particles.size() / std::max(1zu, velocity_stride_);
            velocity_positions_.reserve(velocity_arrow_reserve);
            velocity_vectors_.reserve(velocity_positions_.capacity());
            for (auto i = 0zu; i < vtk_frame.particles.size(); ++i)
            {
                const auto& particle = vtk_frame.particles[i];
                if (show_particles_)
                {
                    frame.draw.draw_basic_mesh({
                        .mesh = particle_mesh_,
                        .object_id = {.value = k_particle_object_base + particle.id},
                        .transform =
                            Transform{
                                .translation = particle.position,
                                .scale = Vec3{particle_draw_radius(i)},
                            },
                        .color = particle_color(i, particle),
                        .mask = {.shadow_producer = false},
                    });
                }

                if (show_velocity_arrows_ and velocity_arrow_particle(i))
                {
                    velocity_positions_.push_back(particle.position);
                    velocity_vectors_.push_back(particle.velocity);
                }
            }
            if (show_velocity_arrows_)
            {
                (void) viz::draw_vector_field(
                    frame.draw,
                    viz::VectorFieldConfig{
                        .positions = std::span<const Vec3>{velocity_positions_},
                        .vectors = std::span<const Vec3>{velocity_vectors_},
                        .scale = velocity_arrow_scale_,
                        .width = 0.004f,
                        .color_by_magnitude = true,
                        .color_ramp = speed_ramp_,
                        .max_vectors = max_velocity_arrows_,
                        .draw_on_top = velocity_arrows_on_top_,
                    }
                );
            }
        }
        draw_selection_debug(frame.draw, vtk_frame);
    }

    auto draw_ui(FrameContext&) -> void
    {
        const auto ui_disabled = surface_preload_running();
        if (ui_disabled)
        {
            ImGui::BeginDisabled();
        }

        ImGui::SetNextWindowPos(ImVec2{18.0f, 280.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{500.0f, 520.0f}, ImGuiCond_Once);
        if (ImGui::Begin("DFSPH Viewer"))
        {
            const auto scenes = available_scenes();
            if (ImGui::BeginCombo("Scene", current_scene().label))
            {
                for (auto i = 0zu; i < scenes.size(); ++i)
                {
                    const auto selected = i == scene_index_;
                    if (ImGui::Selectable(scenes[i].label, selected))
                    {
                        if (!selected)
                        {
                            show_surface_mesh_ = false;
                            pending_scene_index_ = i;
                        }
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::Text("Scene ID: %s", current_scene().id);
            ImGui::Text("Frames: %zu", frames_.size());
            if (!frames_.empty())
            {
                auto frame_i = static_cast<int>(current_frame_);
                if (ImGui::SliderInt("Frame", &frame_i, 0, static_cast<int>(frames_.size() - 1zu)))
                {
                    current_frame_ = static_cast<usize>(std::max(0, frame_i));
                    playback_seconds_ = static_cast<f32>(current_frame_) * k_frame_dt;
                    playback_accumulator_ = 0.0f;
                    mark_selection_dirty();
                }
                if (ImGui::IsItemActivated())
                {
                    frame_slider_resume_after_release_ = !paused_;
                    frame_slider_scrubbing_ = true;
                }
                if (ImGui::IsItemActive())
                {
                    paused_ = true;
                    playback_accumulator_ = 0.0f;
                }
                if (frame_slider_scrubbing_ and ImGui::IsItemDeactivated())
                {
                    if (frame_slider_resume_after_release_)
                    {
                        paused_ = false;
                    }
                    frame_slider_resume_after_release_ = false;
                    frame_slider_scrubbing_ = false;
                }
                const auto& frame = frames_[current_frame_];
                ImGui::Text("Particles: %zu", frame.particles.size());
                ImGui::Text("Rigid bodies: %zu", rigid_bodies_.size());
                ImGui::Text("Time: %.3f s", static_cast<f64>(frame.time_seconds));
            }
            ImGui::Checkbox("Paused", &paused_);
            ImGui::Checkbox("Loop", &loop_playback_);
            ImGui::SliderFloat("Playback speed", &playback_speed_, 0.0f, 6.0f, "%.2f");
            ImGui::Checkbox("Particles", &show_particles_);
            ImGui::SameLine();
            ImGui::Checkbox("Surface mesh", &show_surface_mesh_);
            if (show_surface_mesh_ and surface_frame_available_)
            {
                ImGui::Text(
                    "Surface: %zu vertices, %zu triangles",
                    surface_vertex_count_,
                    surface_triangle_count_
                );
            }
            if (show_surface_mesh_ and surface_preload_finished_)
            {
                ImGui::Text(
                    "Surface CPU cache: %zu/%zu frames, %.1f ms",
                    surface_preloaded_frames_,
                    frames_.size(),
                    static_cast<f64>(surface_preload_ms_)
                );
            }
            else if (show_surface_mesh_ and surface_preload_running())
            {
                ImGui::TextDisabled("Surface: preloading CPU mesh cache");
            }
            else if (show_surface_mesh_ and !surface_preload_error_.empty())
            {
                ImGui::TextDisabled("Surface: preload failed");
            }
            else if (!surface_cache_present_)
            {
                ImGui::TextDisabled("Surface: cache missing");
            }
            else if (show_surface_mesh_)
            {
                ImGui::TextDisabled("Surface: current frame unavailable");
            }
            if (ImGui::Checkbox("Speed coloring", &color_by_speed_) and color_by_speed_)
            {
                color_by_density_ = false;
            }
            if (ImGui::Checkbox("Density coloring", &color_by_density_) and color_by_density_)
            {
                color_by_speed_ = false;
            }
            ImGui::Checkbox("Rigid bodies", &show_rigid_bodies_);
            ImGui::Checkbox("Velocity arrows", &show_velocity_arrows_);
            ImGui::Checkbox("Velocity arrows on top", &velocity_arrows_on_top_);
            ImGui::Checkbox("Selected/neighbors arrows only", &velocity_arrows_selection_only_);
            ImGui::SliderFloat("Arrow scale", &velocity_arrow_scale_, 0.0f, 0.08f, "%.3f");
            auto arrow_stride = static_cast<int>(velocity_stride_);
            if (velocity_arrows_selection_only_)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::SliderInt("Arrow stride", &arrow_stride, 1, 80))
            {
                velocity_stride_ = static_cast<usize>(std::max(1, arrow_stride));
            }
            if (velocity_arrows_selection_only_)
            {
                ImGui::EndDisabled();
            }
            auto max_arrows = static_cast<int>(max_velocity_arrows_);
            if (ImGui::SliderInt("Max arrows", &max_arrows, 0, 2000))
            {
                max_velocity_arrows_ = static_cast<usize>(std::max(0, max_arrows));
            }
            ImGui::Separator();
            draw_selection_ui();
        }
        ImGui::End();

        if (ui_disabled)
        {
            ImGui::EndDisabled();
        }
        draw_surface_preload_overlay();
    }

    auto shutdown(Runtime&) -> void
    {
        if (!profile_enabled_)
        {
            return;
        }

        const auto print_counter = [](std::string_view label, const ProfileCounter& counter)
        {
            std::cout << std::format(
                "[dfsph-profile] {:<16} samples={} mean={:.3f} ms max={:.3f} ms\n",
                label,
                counter.samples,
                counter.mean_ms(),
                counter.max_ms
            );
        };

        std::cout << std::format(
            "[dfsph-profile] scene={} history_frames={} surface_frames={} max_vertices={} "
            "max_triangles={}\n",
            current_scene().id,
            frames_.size(),
            profile_.surface_frames,
            profile_.max_surface_vertices,
            profile_.max_surface_triangles
        );
        std::cout << std::format(
            "[dfsph-profile] surface_preloaded_frames={}\n", profile_.surface_preloaded_frames
        );
        print_counter("frame", profile_.runtime_frame);
        print_counter("update", profile_.runtime_update);
        print_counter("render", profile_.runtime_render);
        print_counter("surface_read", profile_.surface_read);
        print_counter("surface_gzip", profile_.surface_decompress);
        print_counter("surface_decode", profile_.surface_decode);
        print_counter("surface_upload", profile_.surface_upload);
        print_counter("surface_total", profile_.surface_total);
        print_counter("surface_preload", profile_.surface_preload);
    }

  private:
    auto update_playback(f32 dt_seconds) -> void
    {
        if (paused_ or frames_.empty())
        {
            return;
        }
        if (frames_.size() < 2zu)
        {
            playback_accumulator_ = 0.0f;
            return;
        }
        if (current_frame_ + 1zu >= frames_.size() and !loop_playback_)
        {
            paused_ = true;
            playback_accumulator_ = 0.0f;
            return;
        }

        playback_accumulator_ += dt_seconds * std::max(0.0f, playback_speed_);
        constexpr auto max_steps_per_update = 12zu;
        auto steps = 0zu;
        while (playback_accumulator_ >= k_frame_dt and steps < max_steps_per_update)
        {
            playback_accumulator_ -= k_frame_dt;
            ++steps;
            if (current_frame_ + 1zu >= frames_.size())
            {
                if (!loop_playback_)
                {
                    paused_ = true;
                    playback_accumulator_ = 0.0f;
                    break;
                }
                current_frame_ = 0zu;
            }
            else
            {
                ++current_frame_;
            }
            mark_selection_dirty();
        }
        playback_seconds_ = static_cast<f32>(current_frame_) * k_frame_dt + playback_accumulator_;
    }

    auto handle_selection_click(const FrameContext& frame) -> void
    {
        if (!frame.input.left_click.occurred or frame.input.mouse_captured_by_ui or frames_.empty())
        {
            return;
        }

        const auto& vtk_frame = frames_[current_frame_];
        const auto hit = select_particle_projected(
            vtk_frame,
            frame.camera,
            frame.input.left_click.position_px,
            Vec2{static_cast<f32>(frame.extent.width), static_cast<f32>(frame.extent.height)},
            particle_radius_
        );
        if (!hit.has_value())
        {
            if (!frame.input.left_click.modifiers.shift)
            {
                clear_particle_selection();
            }
            return;
        }

        if (frame.input.left_click.modifiers.shift)
        {
            toggle_particle_selection(hit->particle_index);
            return;
        }
        select_particle(hit->particle_index);
    }

    auto select_particle(usize particle_index) -> void
    {
        if (frames_.empty() or particle_index >= frames_[current_frame_].particles.size())
        {
            clear_particle_selection();
            return;
        }

        selected_particle_ids_ = {frames_[current_frame_].particles[particle_index].id};
        mark_selection_dirty();
        sync_selection_to_current_frame();
    }

    auto toggle_particle_selection(usize particle_index) -> void
    {
        if (frames_.empty() or particle_index >= frames_[current_frame_].particles.size())
        {
            return;
        }

        const auto particle_id = frames_[current_frame_].particles[particle_index].id;
        const auto existing = std::ranges::find(selected_particle_ids_, particle_id);
        if (existing != selected_particle_ids_.end())
        {
            selected_particle_ids_.erase(existing);
        }
        else
        {
            selected_particle_ids_.push_back(particle_id);
        }
        mark_selection_dirty();
        sync_selection_to_current_frame();
    }

    auto clear_particle_selection() -> void
    {
        selected_particle_ids_.clear();
        mark_selection_dirty();
        sync_selection_to_current_frame();
    }

    auto select_neighbors_of_selection() -> void
    {
        if (frames_.empty() or selected_particle_ids_.empty())
        {
            return;
        }

        sync_selection_to_current_frame();
        const auto& frame = frames_[current_frame_];
        auto expanded_ids = selected_particle_ids_;
        const auto source_indices = selected_particle_indices_;
        for (const auto selected_index : source_indices)
        {
            for (const auto neighbor_index :
                 find_particle_neighbors(frame, selected_index, support_radius_))
            {
                append_unique_id(expanded_ids, frame.particles[neighbor_index].id);
            }
        }

        selected_particle_ids_ = std::move(expanded_ids);
        mark_selection_dirty();
        sync_selection_to_current_frame();
    }

    auto mark_selection_dirty() noexcept -> void
    {
        selection_dirty_ = true;
    }

    auto sync_selection_to_current_frame() -> void
    {
        if (!selection_dirty_ and selection_synced_frame_ == current_frame_)
        {
            return;
        }

        selected_particle_indices_.clear();
        neighbor_particle_indices_.clear();
        selection_flags_.clear();
        neighbor_flags_.clear();

        if (frames_.empty())
        {
            selection_dirty_ = false;
            selection_synced_frame_ = k_invalid_index;
            return;
        }

        const auto& frame = frames_[current_frame_];
        selection_flags_.assign(frame.particles.size(), 0u);
        neighbor_flags_.assign(frame.particles.size(), 0u);

        for (const auto particle_id : selected_particle_ids_)
        {
            const auto particle_index = find_particle_index_by_id(frame, particle_id);
            if (particle_index == k_invalid_index)
            {
                continue;
            }
            selected_particle_indices_.push_back(particle_index);
            selection_flags_[particle_index] = 1u;
        }

        for (const auto selected_index : selected_particle_indices_)
        {
            for (const auto neighbor_index :
                 find_particle_neighbors(frame, selected_index, support_radius_))
            {
                if (selection_flags_[neighbor_index] != 0u or neighbor_flags_[neighbor_index] != 0u)
                {
                    continue;
                }
                neighbor_flags_[neighbor_index] = 1u;
                neighbor_particle_indices_.push_back(neighbor_index);
            }
        }

        selection_dirty_ = false;
        selection_synced_frame_ = current_frame_;
    }

    [[nodiscard]] auto particle_selected(usize particle_index) const noexcept -> bool
    {
        return particle_index < selection_flags_.size() and selection_flags_[particle_index] != 0u;
    }

    [[nodiscard]] auto particle_neighbor(usize particle_index) const noexcept -> bool
    {
        return particle_index < neighbor_flags_.size() and neighbor_flags_[particle_index] != 0u;
    }

    [[nodiscard]] auto velocity_arrow_particle(usize particle_index) const noexcept -> bool
    {
        if (velocity_arrows_selection_only_)
        {
            return particle_selected(particle_index) or particle_neighbor(particle_index);
        }
        return (particle_index % std::max(1zu, velocity_stride_)) == 0zu;
    }

    [[nodiscard]] auto particle_draw_radius(usize particle_index) const noexcept -> f32
    {
        if (particle_selected(particle_index))
        {
            return 1.85f * particle_radius_;
        }
        if (show_neighbor_markers_ and particle_neighbor(particle_index))
        {
            return 1.35f * particle_radius_;
        }
        return particle_radius_;
    }

    auto load_scene(usize scene_index, Runtime& runtime) -> void
    {
        const auto scenes = available_scenes();
        if (scene_index >= scenes.size())
        {
            throw std::runtime_error(
                std::format("DFSPH scene index out of range: {}", scene_index)
            );
        }

        const auto& scene = scenes[scene_index];
        frames_.clear();
        frames_.shrink_to_fit();
        velocity_positions_.clear();
        velocity_vectors_.clear();
        rigid_bodies_.clear();
        selected_particle_ids_.clear();
        cached_surface_frame_index_ = k_invalid_index;
        surface_cache_present_ = false;
        surface_frame_available_ = false;
        surface_vertex_count_ = 0zu;
        surface_triangle_count_ = 0zu;
        clear_surface_preload();
        mark_selection_dirty();

        frames_ = load_vtk_history(asset_path(scene.vtk_dir));
        global_max_speed_ = 0.0f;
        global_max_density_ = 0.0f;
        for (const auto& frame : frames_)
        {
            global_max_speed_ = std::max(global_max_speed_, frame.max_speed);
            global_max_density_ = std::max(global_max_density_, frame.max_density);
        }
        speed_ramp_.configure({
            .preset = viz::ColorPreset::turbo,
            .range = {.min = 0.0f, .max = std::max(0.1f, global_max_speed_)},
        });
        density_ramp_.configure({
            .preset = viz::ColorPreset::viridis,
            .range = {.min = 0.0f, .max = std::max(1.0f, global_max_density_)},
        });

        scene_index_ = scene_index;
        current_frame_ = 0zu;
        playback_seconds_ = 0.0f;
        playback_accumulator_ = 0.0f;
        particle_radius_ = scene.particle_radius;
        support_radius_ = scene.support_radius;
        velocity_stride_ = scene.default_velocity_stride;
        max_velocity_arrows_ = scene.default_max_velocity_arrows;
        surface_cache_present_ =
            std::filesystem::is_directory(asset_path(scene.surface_dir) / k_surface_frame_dir);
        configure_rigid_body_slots(asset_path(scene.vtk_dir));
        sync_rigid_body_meshes(runtime, current_frame_);
        if (show_surface_mesh_)
        {
            start_surface_mesh_preload();
        }
        runtime.camera().configure(scene.camera);
    }

    auto clear_surface_preload() -> void
    {
        if (surface_preload_thread_.joinable())
        {
            surface_preload_thread_.request_stop();
            surface_preload_thread_.join();
        }
        preloaded_surface_frames_.clear();
        preloaded_surface_frames_.shrink_to_fit();
        surface_preload_completed_.store(0zu, std::memory_order_relaxed);
        surface_preload_loaded_.store(0zu, std::memory_order_relaxed);
        surface_preload_done_.store(false, std::memory_order_relaxed);
        surface_preload_failed_.store(false, std::memory_order_relaxed);
        surface_preload_total_ = 0zu;
        surface_preload_worker_count_ = 0zu;
        surface_preload_result_max_vertices_ = 0zu;
        surface_preload_result_max_triangles_ = 0zu;
        surface_preload_finished_ = false;
        surface_preloaded_frames_ = 0zu;
        surface_preload_ms_ = 0.0f;
        surface_preload_error_.clear();
    }

    auto start_surface_mesh_preload() -> void
    {
        if (!surface_cache_present_ or surface_preload_finished_ or frames_.empty()
            or surface_preload_thread_.joinable() or !surface_preload_error_.empty())
        {
            return;
        }

        preloaded_surface_frames_.clear();
        preloaded_surface_frames_.shrink_to_fit();
        preloaded_surface_frames_.resize(frames_.size());
        surface_frame_available_ = false;
        surface_preload_total_ = frames_.size();
        surface_preload_worker_count_ = surface_preload_worker_count(surface_preload_total_);
        surface_preload_completed_.store(0zu, std::memory_order_relaxed);
        surface_preload_loaded_.store(0zu, std::memory_order_relaxed);
        surface_preload_done_.store(false, std::memory_order_relaxed);
        surface_preload_failed_.store(false, std::memory_order_relaxed);
        surface_preload_result_max_vertices_ = 0zu;
        surface_preload_result_max_triangles_ = 0zu;
        surface_preloaded_frames_ = 0zu;
        surface_preload_ms_ = 0.0f;
        surface_preload_error_.clear();
        surface_preload_started_ = std::chrono::steady_clock::now();

        const auto surface_dir =
            std::make_shared<const std::filesystem::path>(asset_path(current_scene().surface_dir));
        surface_preload_thread_ = std::jthread{
            [this, surface_dir](const std::stop_token& stop_token) noexcept -> void
            {
                try
                {
                    run_surface_mesh_preload(*surface_dir, stop_token);
                }
                catch (...)
                {
                    record_surface_preload_failure("surface preload thread failed");
                }
            },
        };
    }

    auto record_surface_preload_failure(std::string_view message) noexcept -> void
    {
        try
        {
            surface_preload_error_ = message;
        }
        catch (...)
        {
            surface_preload_failed_.store(true, std::memory_order_release);
        }
        surface_preload_failed_.store(true, std::memory_order_release);
        surface_preload_done_.store(true, std::memory_order_release);
    }

    auto run_surface_mesh_preload(
        const std::filesystem::path& surface_dir, const std::stop_token& stop_token
    ) -> void
    {
        ProfileTimer preload_timer{};
        std::atomic<usize> next_frame{};
        std::atomic<bool> worker_failed{};
        std::mutex aggregate_mutex{};
        std::exception_ptr preload_exception{};
        SurfacePreloadWorkerStats aggregate{};

        auto store_exception = [&](std::exception_ptr exception) -> void
        {
            worker_failed.store(true, std::memory_order_relaxed);
            std::scoped_lock lock{aggregate_mutex};
            if (preload_exception == nullptr)
            {
                preload_exception = std::move(exception);
            }
        };

        try
        {
            std::vector<std::jthread> workers{};
            workers.reserve(surface_preload_worker_count_);
            for (auto worker = 0zu; worker < surface_preload_worker_count_; ++worker)
            {
                workers.emplace_back(
                    [this,
                     &surface_dir,
                     stop_token,
                     &next_frame,
                     &worker_failed,
                     &aggregate_mutex,
                     &aggregate,
                     &store_exception]() -> void
                    {
                        SurfacePreloadWorkerStats stats{};
                        try
                        {
                            while (!stop_token.stop_requested()
                                   and !worker_failed.load(std::memory_order_relaxed))
                            {
                                const auto frame_slot =
                                    next_frame.fetch_add(1zu, std::memory_order_relaxed);
                                if (frame_slot >= surface_preload_total_)
                                {
                                    break;
                                }
                                if (frame_slot
                                    > static_cast<usize>(std::numeric_limits<u32>::max()))
                                {
                                    throw std::runtime_error(
                                        std::format(
                                            "surface frame index out of range: {}", frame_slot
                                        )
                                    );
                                }

                                const auto frame_index = static_cast<u32>(frame_slot);
                                const auto path =
                                    surface_frame_payload_path(surface_dir, frame_index);
                                if (std::filesystem::is_regular_file(path))
                                {
                                    auto load_result = read_surface_mesh_frame(path);
                                    auto& frame = load_result.frame;
                                    if (frame.index != frame_index)
                                    {
                                        throw std::runtime_error(
                                            std::format(
                                                "surface frame payload index mismatch: expected "
                                                "{}, "
                                                "got {}",
                                                frame_index,
                                                frame.index
                                            )
                                        );
                                    }

                                    stats.max_vertices =
                                        std::max(stats.max_vertices, frame.mesh.vertices.size());
                                    stats.max_triangles = std::max(
                                        stats.max_triangles, frame.mesh.indices.size() / 3zu
                                    );
                                    preloaded_surface_frames_[frame_slot] = std::move(frame);
                                    ++stats.loaded_frames;
                                    surface_preload_loaded_.fetch_add(
                                        1zu, std::memory_order_relaxed
                                    );
                                }
                                surface_preload_completed_.fetch_add(
                                    1zu, std::memory_order_relaxed
                                );
                            }
                        }
                        catch (...)
                        {
                            store_exception(std::current_exception());
                        }

                        std::scoped_lock lock{aggregate_mutex};
                        aggregate.loaded_frames += stats.loaded_frames;
                        aggregate.max_vertices =
                            std::max(aggregate.max_vertices, stats.max_vertices);
                        aggregate.max_triangles =
                            std::max(aggregate.max_triangles, stats.max_triangles);
                    }
                );
            }
        }
        catch (...)
        {
            store_exception(std::current_exception());
        }

        if (stop_token.stop_requested())
        {
            surface_preload_done_.store(true, std::memory_order_release);
            return;
        }

        surface_preload_ms_ = preload_timer.elapsed_ms();
        surface_preloaded_frames_ = aggregate.loaded_frames;
        surface_preload_result_max_vertices_ = aggregate.max_vertices;
        surface_preload_result_max_triangles_ = aggregate.max_triangles;

        if (preload_exception != nullptr)
        {
            try
            {
                record_surface_preload_failure(exception_message(preload_exception));
            }
            catch (...)
            {
                record_surface_preload_failure("surface preload failed");
            }
            return;
        }
        surface_preload_done_.store(true, std::memory_order_release);
    }

    auto poll_surface_mesh_preload() -> void
    {
        if (!surface_preload_thread_.joinable())
        {
            return;
        }
        if (!surface_preload_done_.load(std::memory_order_acquire))
        {
            return;
        }

        surface_preload_thread_.join();
        if (surface_preload_failed_.load(std::memory_order_acquire))
        {
            surface_preload_finished_ = false;
            surface_frame_available_ = false;
            preloaded_surface_frames_.clear();
            preloaded_surface_frames_.shrink_to_fit();
            std::cerr << "[dfsph] surface preload failed: " << surface_preload_error_ << '\n';
            return;
        }

        surface_preload_finished_ = true;
        if (profile_enabled_)
        {
            profile_.surface_preloaded_frames = surface_preloaded_frames_;
            profile_.surface_preload.add(surface_preload_ms_);
            profile_.max_surface_vertices =
                std::max(profile_.max_surface_vertices, surface_preload_result_max_vertices_);
            profile_.max_surface_triangles =
                std::max(profile_.max_surface_triangles, surface_preload_result_max_triangles_);
        }
    }

    [[nodiscard]] auto surface_preload_running() const noexcept -> bool
    {
        return surface_preload_thread_.joinable();
    }

    auto configure_rigid_body_slots(const std::filesystem::path& vtk_dir) -> void
    {
        const auto body_ids = discover_rigid_body_ids(vtk_dir);
        rigid_bodies_.resize(body_ids.size());
        for (auto i = 0zu; i < body_ids.size(); ++i)
        {
            auto& slot = rigid_bodies_[i];
            slot.body_id = body_ids[i];
            slot.material = rigid_body_material(slot.body_id);
            slot.cached_frame_index = k_invalid_index;
            slot.available = false;
        }
    }

    auto sync_rigid_body_meshes(Runtime& runtime, usize frame_index) -> void
    {
        if (rigid_bodies_.empty())
        {
            return;
        }

        const auto vtk_dir = asset_path(current_scene().vtk_dir);
        for (auto& slot : rigid_bodies_)
        {
            if (slot.cached_frame_index == frame_index)
            {
                continue;
            }

            slot.cached_frame_index = frame_index;
            const auto path = rigid_body_vtk_path_for_frame(vtk_dir, slot.body_id, frame_index);
            if (path.empty())
            {
                slot.available = false;
                continue;
            }

            auto mesh = read_rigid_body_vtk_mesh(path);
            slot.mesh = runtime.replace_mesh(slot.mesh, mesh);
            slot.available = true;
        }
    }

    auto sync_surface_mesh(Runtime& runtime, usize frame_index) -> void
    {
        if (!surface_cache_present_)
        {
            surface_frame_available_ = false;
            return;
        }
        if (cached_surface_frame_index_ == frame_index and surface_frame_available_)
        {
            return;
        }
        if (frame_index > static_cast<usize>(std::numeric_limits<u32>::max()))
        {
            throw std::runtime_error(
                std::format("surface frame index out of range: {}", frame_index)
            );
        }

        start_surface_mesh_preload();

        cached_surface_frame_index_ = frame_index;
        if (!surface_preload_finished_)
        {
            surface_frame_available_ = false;
            surface_vertex_count_ = 0zu;
            surface_triangle_count_ = 0zu;
            return;
        }
        if (frame_index >= preloaded_surface_frames_.size())
        {
            surface_frame_available_ = false;
            surface_vertex_count_ = 0zu;
            surface_triangle_count_ = 0zu;
            return;
        }
        const auto& preloaded_frame = preloaded_surface_frames_[frame_index];
        if (!preloaded_frame.has_value())
        {
            surface_frame_available_ = false;
            surface_vertex_count_ = 0zu;
            surface_triangle_count_ = 0zu;
            return;
        }

        ProfileTimer total_timer{};
        upload_surface_frame(runtime, *preloaded_frame, 0.0f, 0.0f, 0.0f, total_timer);
    }

    auto upload_surface_frame(
        Runtime& runtime,
        const SurfaceFrame& frame,
        f32 read_ms,
        f32 decompress_ms,
        f32 decode_ms,
        const ProfileTimer& total_timer
    ) -> void
    {
        surface_vertex_count_ = frame.mesh.vertices.size();
        surface_triangle_count_ = frame.mesh.indices.size() / 3zu;
        ProfileTimer upload_timer{};
        surface_mesh_ = runtime.replace_mesh(surface_mesh_, frame.mesh);
        const auto upload_ms = upload_timer.elapsed_ms();
        surface_frame_available_ = true;
        if (profile_enabled_)
        {
            ++profile_.surface_frames;
            profile_.surface_read.add(read_ms);
            profile_.surface_decompress.add(decompress_ms);
            profile_.surface_decode.add(decode_ms);
            profile_.surface_upload.add(upload_ms);
            profile_.surface_total.add(total_timer.elapsed_ms());
            profile_.max_surface_vertices =
                std::max(profile_.max_surface_vertices, surface_vertex_count_);
            profile_.max_surface_triangles =
                std::max(profile_.max_surface_triangles, surface_triangle_count_);
        }
    }

    auto draw_surface_mesh(DrawList& draw) const -> void
    {
        if (!show_surface_mesh_ or !surface_frame_available_)
        {
            return;
        }

        draw.draw_mesh({
            .mesh = surface_mesh_,
            .object_id = {.value = k_surface_object_base},
            .transform = {},
            .material = surface_material_,
            .mask = {.shadow_producer = false},
        });
    }

    auto draw_rigid_bodies(DrawList& draw) const -> void
    {
        if (!show_rigid_bodies_)
        {
            return;
        }

        for (const auto& slot : rigid_bodies_)
        {
            if (!slot.available)
            {
                continue;
            }
            draw.draw_mesh({
                .mesh = slot.mesh,
                .object_id = {.value = k_rigid_body_object_base + slot.body_id},
                .transform = {},
                .material = slot.material,
            });
        }
    }

    [[nodiscard]] auto current_scene() const noexcept -> const DfsphSceneConfig&
    {
        const auto scenes = available_scenes();
        return scenes[std::min(scene_index_, scenes.size() - 1zu)];
    }

    [[nodiscard]] auto floor_center() const noexcept -> Vec3
    {
        const auto& bounds = current_scene().bounds;
        return Vec3{
            0.5f * (bounds.min.x + bounds.max.x),
            0.5f * (bounds.min.y + bounds.max.y),
            bounds.min.z,
        };
    }

    [[nodiscard]] auto floor_side_length() const noexcept -> f32
    {
        const auto& bounds = current_scene().bounds;
        const auto extent = glm::abs(bounds.max - bounds.min);
        return 1.15f * std::max(extent.x, extent.y);
    }

    auto draw_selection_ui() -> void
    {
        ImGui::Text("Selected particles: %zu", selected_particle_indices_.size());
        ImGui::Text("Marked neighbors: %zu", neighbor_particle_indices_.size());
        const auto has_selection = !selected_particle_indices_.empty();
        if (!has_selection)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Select neighbors"))
        {
            select_neighbors_of_selection();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear selection"))
        {
            clear_particle_selection();
        }
        if (!has_selection)
        {
            ImGui::EndDisabled();
        }

        if (ImGui::SliderFloat("Support radius", &support_radius_, 0.005f, 0.20f, "%.4f"))
        {
            mark_selection_dirty();
        }
        if (ImGui::Checkbox("Neighbor markers", &show_neighbor_markers_))
        {
            mark_selection_dirty();
        }
        ImGui::Checkbox("Neighbor lines", &show_neighbor_lines_);
        ImGui::Checkbox("Selected velocity", &show_selected_velocity_);
        auto max_lines = static_cast<int>(max_neighbor_lines_);
        if (ImGui::SliderInt("Max neighbor lines", &max_lines, 0, 512))
        {
            max_neighbor_lines_ = static_cast<usize>(std::max(0, max_lines));
        }

        if (has_selection and !frames_.empty())
        {
            const auto& frame = frames_[current_frame_];
            const auto selected_index = selected_particle_indices_.front();
            if (selected_index < frame.particles.size())
            {
                const auto& particle = frame.particles[selected_index];
                ImGui::Text("Primary ID: %u", particle.id);
                ImGui::Text(
                    "Position: %.3f %.3f %.3f",
                    static_cast<f64>(particle.position.x),
                    static_cast<f64>(particle.position.y),
                    static_cast<f64>(particle.position.z)
                );
                ImGui::Text(
                    "Velocity: %.3f %.3f %.3f | speed %.3f",
                    static_cast<f64>(particle.velocity.x),
                    static_cast<f64>(particle.velocity.y),
                    static_cast<f64>(particle.velocity.z),
                    static_cast<f64>(glm::length(particle.velocity))
                );
                ImGui::Text(
                    "Density: %.2f advected %.2f",
                    static_cast<f64>(particle.density),
                    static_cast<f64>(particle.advected_density)
                );
            }
        }
    }

    auto draw_selection_debug(DrawList& draw, const ParticleFrame& frame) const -> void
    {
        if (selected_particle_indices_.empty())
        {
            return;
        }

        if (show_selected_velocity_)
        {
            for (const auto selected_index : selected_particle_indices_)
            {
                if (selected_index >= frame.particles.size())
                {
                    continue;
                }
                const auto& particle = frame.particles[selected_index];
                draw.debug_arrow({
                    .origin = particle.position,
                    .vector = velocity_arrow_scale_ * particle.velocity,
                    .color = Color{1.0f, 0.18f, 0.95f, 1.0f},
                    .width = 0.008f,
                    .draw_on_top = velocity_arrows_on_top_,
                });
            }
        }

        if (!show_neighbor_lines_ or max_neighbor_lines_ == 0zu)
        {
            return;
        }

        auto line_count = 0zu;
        for (const auto selected_index : selected_particle_indices_)
        {
            if (selected_index >= frame.particles.size())
            {
                continue;
            }
            const auto selected_position = frame.particles[selected_index].position;
            for (const auto neighbor_index :
                 find_particle_neighbors(frame, selected_index, support_radius_))
            {
                if (neighbor_index >= frame.particles.size())
                {
                    continue;
                }
                draw.debug_line({
                    .start = selected_position,
                    .end = frame.particles[neighbor_index].position,
                    .color = Color{1.0f, 0.72f, 0.16f, 0.72f},
                    .width = 0.0025f,
                });
                ++line_count;
                if (line_count >= max_neighbor_lines_)
                {
                    return;
                }
            }
        }
    }

    [[nodiscard]] auto surface_preload_elapsed_seconds() const noexcept -> f32
    {
        if (surface_preload_started_ == std::chrono::steady_clock::time_point{})
        {
            return 0.0f;
        }
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<f32>(now - surface_preload_started_).count();
    }

    auto draw_surface_preload_overlay() const -> void
    {
        if (!surface_preload_running())
        {
            return;
        }

        const auto* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f, 0.0f});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.0f, 0.0f, 0.0f, 0.0f});

        constexpr auto flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove
                               | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar
                               | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::Begin("##surface_preload_overlay", nullptr, flags);

        auto* draw_list = ImGui::GetWindowDrawList();
        const auto viewport_min = viewport->Pos;
        const auto viewport_max =
            ImVec2{viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y};
        draw_list->AddRectFilled(
            viewport_min, viewport_max, ImGui::GetColorU32(ImVec4{0.18f, 0.18f, 0.19f, 0.70f})
        );

        ImGui::SetCursorScreenPos(viewport_min);
        ImGui::InvisibleButton("##surface_preload_input_capture", viewport->Size);

        const auto panel_width = std::clamp(viewport->Size.x - 64.0f, 360.0f, 680.0f);
        const ImVec2 panel_size{panel_width, 214.0f};
        const ImVec2 panel_min{
            viewport->Pos.x + 0.5f * (viewport->Size.x - panel_size.x),
            viewport->Pos.y + 0.5f * (viewport->Size.y - panel_size.y),
        };
        const ImVec2 panel_max{panel_min.x + panel_size.x, panel_min.y + panel_size.y};
        draw_list->AddRectFilled(
            panel_min, panel_max, ImGui::GetColorU32(ImVec4{0.075f, 0.080f, 0.090f, 0.96f}), 8.0f
        );
        draw_list->AddRect(
            panel_min, panel_max, ImGui::GetColorU32(ImVec4{0.72f, 0.78f, 0.88f, 0.55f}), 8.0f
        );

        const auto elapsed_seconds = surface_preload_elapsed_seconds();
        const auto completed_frames = std::min(
            surface_preload_completed_.load(std::memory_order_relaxed), surface_preload_total_
        );
        const auto loaded_frames = std::min(
            surface_preload_loaded_.load(std::memory_order_relaxed), surface_preload_total_
        );
        const auto progress =
            surface_preload_total_ == 0zu
                ? 0.0f
                : static_cast<f32>(completed_frames) / static_cast<f32>(surface_preload_total_);

        const auto title = std::string{"Loading DFSPH surface mesh cache"};
        const auto detail = std::format(
            "{} workers decoding CPU meshes for {}",
            surface_preload_worker_count_,
            current_scene().label
        );
        const auto elapsed_text = std::format("elapsed {:.1f}s", static_cast<f64>(elapsed_seconds));
        const auto progress_text = std::format(
            "checked {}/{} frames, decoded {}",
            completed_frames,
            surface_preload_total_,
            loaded_frames
        );

        const ImVec2 title_pos{panel_min.x + 28.0f, panel_min.y + 24.0f};
        draw_list->AddText(
            title_pos, ImGui::GetColorU32(ImVec4{0.94f, 0.96f, 1.0f, 1.0f}), title.c_str()
        );
        draw_list->AddText(
            ImVec2{title_pos.x, title_pos.y + 32.0f},
            ImGui::GetColorU32(ImVec4{0.78f, 0.82f, 0.90f, 1.0f}),
            detail.c_str()
        );

        const auto elapsed_size = ImGui::CalcTextSize(elapsed_text.c_str());
        draw_list->AddText(
            ImVec2{panel_max.x - elapsed_size.x - 28.0f, title_pos.y},
            ImGui::GetColorU32(ImVec4{0.64f, 0.70f, 0.82f, 1.0f}),
            elapsed_text.c_str()
        );
        draw_list->AddText(
            ImVec2{title_pos.x, panel_max.y - 86.0f},
            ImGui::GetColorU32(ImVec4{0.72f, 0.76f, 0.86f, 1.0f}),
            progress_text.c_str()
        );

        const ImVec2 bar_min{panel_min.x + 28.0f, panel_max.y - 58.0f};
        const ImVec2 bar_max{panel_max.x - 28.0f, panel_max.y - 28.0f};
        const auto bar_width = bar_max.x - bar_min.x;
        const auto filled_width = bar_width * std::clamp(progress, 0.0f, 1.0f);
        draw_list->AddRectFilled(
            bar_min, bar_max, ImGui::GetColorU32(ImVec4{0.02f, 0.025f, 0.035f, 1.0f}), 5.0f
        );
        draw_list->AddRectFilled(
            bar_min,
            ImVec2{bar_min.x + filled_width, bar_max.y},
            ImGui::GetColorU32(ImVec4{0.38f, 0.70f, 1.0f, 1.0f}),
            5.0f
        );
        draw_list->AddRect(
            bar_min, bar_max, ImGui::GetColorU32(ImVec4{0.82f, 0.86f, 0.96f, 0.60f}), 5.0f
        );

        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    [[nodiscard]] auto
    particle_color(usize particle_index, const ParticleSample& particle) const noexcept -> Color
    {
        if (particle_selected(particle_index))
        {
            return Color{1.0f, 0.0f, 0.85f, 1.0f};
        }
        if (show_neighbor_markers_ and particle_neighbor(particle_index))
        {
            return Color{1.0f, 0.82f, 0.18f, 1.0f};
        }
        if (color_by_density_)
        {
            return density_ramp_.sample(particle.density);
        }
        if (color_by_speed_)
        {
            return speed_ramp_.sample(glm::length(particle.velocity));
        }
        return Color{0.30f, 0.64f, 0.98f, 1.0f};
    }

    auto configure_lighting(DrawList& draw) const -> void
    {
        draw.set_ambient_light(Color{0.05f, 0.06f, 0.075f, 1.0f});
        draw.directional_light({
            .direction = normalize_or(Vec3{-0.45f, -0.35f, -1.0f}, -k_axis_z),
            .color = Color{1.0f, 0.95f, 0.86f, 1.0f},
            .intensity = 2.2f,
            .shadow = {.enabled = true, .ortho_extent = 5.5f},
        });
        draw.radial_light({
            .position = {-1.4f, -0.9f, 1.2f},
            .color = Color{0.24f, 0.55f, 1.0f, 1.0f},
            .intensity = 5.0f,
            .range = 4.0f,
        });
    }

    MeshHandle particle_mesh_{};
    MeshHandle floor_mesh_{};
    MeshHandle surface_mesh_{};
    Material surface_material_{
        .base_color = Color{0.08f, 0.44f, 0.82f, 1.0f},
        .metallic = 0.0f,
        .roughness = 0.38f,
        .ambient_occlusion = 1.0f,
    };
    Runtime* runtime_{};
    std::vector<ParticleFrame> frames_{};
    std::vector<RigidBodySlot> rigid_bodies_{};
    std::vector<Vec3> velocity_positions_{};
    std::vector<Vec3> velocity_vectors_{};
    std::vector<u32> selected_particle_ids_{};
    std::vector<usize> selected_particle_indices_{};
    std::vector<usize> neighbor_particle_indices_{};
    std::vector<u8> selection_flags_{};
    std::vector<u8> neighbor_flags_{};
    viz::ColorRamp speed_ramp_{};
    viz::ColorRamp density_ramp_{};
    std::optional<usize> pending_scene_index_{};
    std::vector<std::optional<SurfaceFrame>> preloaded_surface_frames_{};
    std::atomic<usize> surface_preload_completed_{};
    std::atomic<usize> surface_preload_loaded_{};
    std::atomic<bool> surface_preload_done_{};
    std::atomic<bool> surface_preload_failed_{};
    std::chrono::steady_clock::time_point surface_preload_started_{};
    std::string surface_preload_error_{};
    usize scene_index_{};
    usize selection_synced_frame_{k_invalid_index};
    usize cached_surface_frame_index_{k_invalid_index};
    usize surface_vertex_count_{};
    usize surface_triangle_count_{};
    usize surface_preload_total_{};
    usize surface_preload_worker_count_{};
    usize surface_preload_result_max_vertices_{};
    usize surface_preload_result_max_triangles_{};
    usize surface_preloaded_frames_{};
    f32 surface_preload_ms_{};
    f32 global_max_speed_{};
    f32 global_max_density_{};
    usize current_frame_{};
    f32 playback_seconds_{};
    f32 playback_accumulator_{};
    f32 playback_speed_{1.0f};
    f32 particle_radius_{k_particle_radius};
    f32 support_radius_{4.0f * k_particle_radius};
    f32 velocity_arrow_scale_{0.018f};
    usize velocity_stride_{24zu};
    usize max_velocity_arrows_{600zu};
    usize max_neighbor_lines_{128zu};
    bool paused_{false};
    bool loop_playback_{false};
    bool color_by_speed_{true};
    bool color_by_density_{false};
    bool show_particles_{true};
    bool show_surface_mesh_{false};
    bool surface_cache_present_{false};
    bool surface_frame_available_{false};
    bool surface_preload_finished_{false};
    bool show_rigid_bodies_{true};
    bool show_velocity_arrows_{false};
    bool velocity_arrows_on_top_{true};
    bool velocity_arrows_selection_only_{false};
    bool show_neighbor_markers_{true};
    bool show_neighbor_lines_{true};
    bool show_selected_velocity_{true};
    bool selection_dirty_{true};
    bool frame_slider_scrubbing_{false};
    bool frame_slider_resume_after_release_{false};
    bool profile_enabled_{};
    DfsphProfile profile_{};
    std::jthread surface_preload_thread_{};
};
}  // namespace

auto main(int argc, char** argv) -> int
{
    try
    {
        RuntimeConfig config{
            .window_title = "ds_vk DFSPH dambreak",
            .initial_width = 1280u,
            .initial_height = 820u,
            .clear_color = Color{0.018f, 0.022f, 0.027f, 1.0f},
        };
        DfsphPlaybackConfig app_cfg{};
        for (auto i = 1; i < argc; ++i)
        {
            const std::string_view arg{argv[i]};
            if (arg == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--smoke-frames" and i + 1 < argc)
            {
                config.smoke_frames = parse_u32(argv[++i], 0u);
            }
            else if (arg == "--screenshot" and i + 1 < argc)
            {
                config.screenshot_path = argv[++i];
            }
            else if (arg == "--hide-ui")
            {
                config.hide_ui = true;
            }
            else if (arg == "--transparent-screenshot")
            {
                config.transparent_screenshot = true;
            }
            else if (arg == "--scene-id" and i + 1 < argc)
            {
                const std::string_view scene_id{argv[++i]};
                const auto scene_index = scene_index_for_id(scene_id);
                if (!scene_index.has_value())
                {
                    throw std::runtime_error(std::format("unknown DFSPH scene id: {}", scene_id));
                }
                app_cfg.initial_scene_index = *scene_index;
            }
            else if (arg == "--show-mesh")
            {
                app_cfg.show_surface_mesh = true;
            }
            else if (arg == "--hide-mesh")
            {
                app_cfg.show_surface_mesh = false;
            }
            else if (arg == "--show-particles")
            {
                app_cfg.show_particles = true;
            }
            else if (arg == "--hide-particles")
            {
                app_cfg.show_particles = false;
            }
            else if (arg == "--playback-speed" and i + 1 < argc)
            {
                app_cfg.playback_speed = parse_f32(argv[++i], app_cfg.playback_speed);
            }
            else if (arg == "--profile")
            {
                app_cfg.profile = true;
            }
            else
            {
                print_usage(argv[0]);
                return 2;
            }
        }

        Runtime runtime{std::move(config)};
        DfsphPlaybackApp app{app_cfg};
        return runtime.run(app);
    }
    catch (const std::exception& error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
