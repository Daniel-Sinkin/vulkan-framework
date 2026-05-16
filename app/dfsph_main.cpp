#include "ds_vk/math.hpp"
#include "ds_vk/mesh.hpp"
#include "ds_vk/plugins/viz.hpp"
#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <bit>
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
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace ds_vk;

#ifndef DS_VK_ASSET_DIR
#    define DS_VK_ASSET_DIR "assets"
#endif

constexpr auto k_particle_radius = 0.025f;
constexpr auto k_frame_dt = 1.0f / 60.0f;
constexpr auto k_particle_object_base = u32{100000u};

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

struct BinaryCursor
{
    std::vector<char> data{};
    usize offset{};
};

[[nodiscard]] auto asset_path(const std::filesystem::path& relative) -> std::filesystem::path
{
    return std::filesystem::path{DS_VK_ASSET_DIR} / relative;
}

auto print_usage(const char* program) -> void
{
    std::cerr << "usage: " << program
              << " [--smoke-frames N] [--screenshot PATH] [--hide-ui]"
                 " [--transparent-screenshot]\n";
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

[[nodiscard]] auto read_all_bytes(const std::filesystem::path& path) -> std::vector<char>
{
    auto in = std::ifstream{path, std::ios::binary | std::ios::ate};
    if (!in)
    {
        throw std::runtime_error(std::format("failed to open VTK file: {}", path.string()));
    }
    const auto end = in.tellg();
    if (end <= 0)
    {
        throw std::runtime_error(std::format("empty VTK file: {}", path.string()));
    }
    auto bytes = std::vector<char>(static_cast<usize>(end));
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
    auto line = std::string{cursor.data.data() + begin, cursor.offset - begin};
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
    auto in = std::istringstream{std::string{line}};
    auto words = std::vector<std::string>{};
    auto word = std::string{};
    while (in >> word)
    {
        words.push_back(word);
    }
    return words;
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

auto read_points(BinaryCursor& cursor, ParticleFrame& frame) -> void
{
    const auto words = split_words(read_line(cursor));
    if (words.size() != 3zu or words[0] != "POINTS")
    {
        throw std::runtime_error("expected VTK POINTS line");
    }
    const auto point_count = parse_u32_word(words[1], "POINTS count");
    frame.particles.assign(static_cast<usize>(point_count), {});
    for (auto i = 0u; i < point_count; ++i)
    {
        auto& particle = frame.particles[static_cast<usize>(i)];
        particle.position = {
            read_be_real(cursor, words[2]),
            read_be_real(cursor, words[2]),
            read_be_real(cursor, words[2]),
        };
        particle.position = splishsplash_y_up_to_z_up(particle.position);
        particle.id = i;
    }
    consume_optional_newline(cursor);
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
    auto files = std::vector<std::filesystem::path>{};
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
    auto cursor = BinaryCursor{.data = read_all_bytes(path)};
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

    auto frame = ParticleFrame{.index = frame_index, .time_seconds = time_seconds};
    read_points(cursor, frame);
    const auto point_count = static_cast<u32>(frame.particles.size());
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

[[nodiscard]] auto load_vtk_history(const std::filesystem::path& vtk_dir)
    -> std::vector<ParticleFrame>
{
    const auto files = discover_particle_vtk_files(vtk_dir);
    if (files.empty())
    {
        throw std::runtime_error(std::format("no DFSPH VTK frames found in: {}", vtk_dir.string()));
    }

    auto frames = std::vector<ParticleFrame>{};
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

class DfsphPlaybackApp
{
  public:
    auto setup(Runtime& runtime) -> void
    {
        particle_mesh_ = runtime.upload_mesh(make_uv_sphere({
            .radius = 1.0f,
            .slices = 8u,
            .stacks = 4u,
            .color = Color::white,
        }));
        floor_mesh_ = runtime.upload_mesh(make_quad(5.0f, Color::white));
        frames_ = load_vtk_history(asset_path("dfsph/dambreak_small_iisph_v1/vtk"));

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

        runtime.camera({
            .pivot = {0.0f, 0.0f, 0.5f},
            .distance = 4.7f,
            .yaw = glm::radians(42.0f),
            .pitch = glm::radians(24.0f),
        });
    }

    auto update(FrameContext& frame, f32 dt_seconds) -> void
    {
        configure_lighting(frame.draw);
        if (!paused_ and !frames_.empty())
        {
            playback_seconds_ += dt_seconds * playback_speed_;
            const auto frame_count = static_cast<f32>(frames_.size());
            const auto wrapped = std::fmod(playback_seconds_ / k_frame_dt, frame_count);
            current_frame_ = static_cast<usize>(std::max(0.0f, wrapped)) % frames_.size();
        }

        frame.draw.draw_mesh({
            .mesh = floor_mesh_,
            .transform = {.translation = -0.012f * k_axis_z},
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
                .aabb = {.min = {-2.2f, -0.8f, -0.02f}, .max = {2.2f, 0.8f, 1.35f}},
                .color = Color{0.36f, 0.52f, 0.68f, 0.75f},
                .width = 0.006f,
            }
        );

        if (frames_.empty())
        {
            return;
        }

        const auto& vtk_frame = frames_[current_frame_];
        velocity_positions_.clear();
        velocity_vectors_.clear();
        velocity_positions_.reserve(vtk_frame.particles.size() / std::max(1zu, velocity_stride_));
        velocity_vectors_.reserve(velocity_positions_.capacity());
        for (auto i = 0zu; i < vtk_frame.particles.size(); i += particle_stride_)
        {
            const auto& particle = vtk_frame.particles[i];
            const auto color = particle_color(particle);
            frame.draw.draw_basic_mesh({
                .mesh = particle_mesh_,
                .object_id = {.value = k_particle_object_base + particle.id},
                .transform =
                    Transform{
                        .translation = particle.position,
                        .scale = Vec3{particle_radius_},
                    },
                .color = color,
                .mask = {.shadow_producer = false},
            });

            if (show_velocity_arrows_ and (i % velocity_stride_) == 0zu)
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
                }
            );
        }
    }

    auto draw_ui(FrameContext&) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{18.0f, 280.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{420.0f, 300.0f}, ImGuiCond_Once);
        if (ImGui::Begin("DFSPH Dam Break"))
        {
            ImGui::Text("Frames: %zu", frames_.size());
            if (!frames_.empty())
            {
                auto frame_i = static_cast<int>(current_frame_);
                if (ImGui::SliderInt("Frame", &frame_i, 0, static_cast<int>(frames_.size() - 1zu)))
                {
                    current_frame_ = static_cast<usize>(std::max(0, frame_i));
                    playback_seconds_ = static_cast<f32>(current_frame_) * k_frame_dt;
                }
                const auto& frame = frames_[current_frame_];
                ImGui::Text("Particles: %zu", frame.particles.size());
                ImGui::Text("Time: %.3f s", static_cast<f64>(frame.time_seconds));
            }
            ImGui::Checkbox("Paused", &paused_);
            ImGui::SliderFloat("Playback speed", &playback_speed_, 0.0f, 6.0f, "%.2f");
            ImGui::SliderFloat("Particle radius", &particle_radius_, 0.005f, 0.06f, "%.3f");
            auto stride = static_cast<int>(particle_stride_);
            if (ImGui::SliderInt("Particle stride", &stride, 1, 8))
            {
                particle_stride_ = static_cast<usize>(std::max(1, stride));
            }
            ImGui::Checkbox("Speed coloring", &color_by_speed_);
            ImGui::Checkbox("Density coloring", &color_by_density_);
            ImGui::Checkbox("Velocity arrows", &show_velocity_arrows_);
            ImGui::SliderFloat("Arrow scale", &velocity_arrow_scale_, 0.0f, 0.08f, "%.3f");
            auto arrow_stride = static_cast<int>(velocity_stride_);
            if (ImGui::SliderInt("Arrow stride", &arrow_stride, 1, 80))
            {
                velocity_stride_ = static_cast<usize>(std::max(1, arrow_stride));
            }
            auto max_arrows = static_cast<int>(max_velocity_arrows_);
            if (ImGui::SliderInt("Max arrows", &max_arrows, 0, 2000))
            {
                max_velocity_arrows_ = static_cast<usize>(std::max(0, max_arrows));
            }
        }
        ImGui::End();
    }

  private:
    [[nodiscard]] auto particle_color(const ParticleSample& particle) const noexcept -> Color
    {
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
    std::vector<ParticleFrame> frames_{};
    std::vector<Vec3> velocity_positions_{};
    std::vector<Vec3> velocity_vectors_{};
    viz::ColorRamp speed_ramp_{};
    viz::ColorRamp density_ramp_{};
    f32 global_max_speed_{};
    f32 global_max_density_{};
    usize current_frame_{};
    f32 playback_seconds_{};
    f32 playback_speed_{1.0f};
    f32 particle_radius_{k_particle_radius};
    f32 velocity_arrow_scale_{0.018f};
    usize particle_stride_{1zu};
    usize velocity_stride_{24zu};
    usize max_velocity_arrows_{600zu};
    bool paused_{false};
    bool color_by_speed_{true};
    bool color_by_density_{false};
    bool show_velocity_arrows_{false};
};
}  // namespace

auto main(const int argc, char** argv) -> int
{
    try
    {
        auto config = RuntimeConfig{
            .window_title = "ds_vk DFSPH dambreak",
            .initial_width = 1280u,
            .initial_height = 820u,
            .clear_color = Color{0.018f, 0.022f, 0.027f, 1.0f},
        };
        for (auto i = 1; i < argc; ++i)
        {
            const auto arg = std::string_view{argv[i]};
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
            else
            {
                print_usage(argv[0]);
                return 2;
            }
        }

        auto runtime = Runtime{std::move(config)};
        auto app = DfsphPlaybackApp{};
        return runtime.run(app);
    }
    catch (const std::exception& error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
