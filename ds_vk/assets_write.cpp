#include "ds_vk/assets.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ds_vk
{
namespace
{
constexpr auto k_component_u32 = 5125;
constexpr auto k_component_f32 = 5126;
constexpr auto k_buffer_target_array = 34962;
constexpr auto k_buffer_target_element_array = 34963;
constexpr auto k_mode_triangles = 4;
constexpr auto k_gltf_alignment = 4zu;

struct GltfVec2
{
    f32 x{};
    f32 y{};
};

struct GltfVec3
{
    f32 x{};
    f32 y{};
    f32 z{};
};

static_assert(sizeof(GltfVec2) == 2zu * sizeof(f32));
static_assert(sizeof(GltfVec3) == 3zu * sizeof(f32));

[[nodiscard]] auto sanitize_filename(std::string_view name, std::string_view fallback)
    -> std::string
{
    std::string out{};
    out.reserve(name.size());
    for (const auto c : name)
    {
        const auto valid = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')
                           or (c >= '0' and c <= '9') or c == '-' or c == '_';
        out.push_back(valid ? c : '_');
    }
    if (out.empty())
    {
        out = fallback;
    }
    return out;
}

auto write_all_bytes(const std::filesystem::path& path, std::span<const u8> bytes) -> void
{
    std::ofstream out{path, std::ios::binary};
    if (!out)
    {
        throw std::runtime_error(std::format("failed to open output file: {}", path.string()));
    }
    out.write(
        reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())
    );
    if (!out)
    {
        throw std::runtime_error(std::format("failed to write output file: {}", path.string()));
    }
}

auto write_text(const std::filesystem::path& path, std::string_view text) -> void
{
    std::ofstream out{path, std::ios::binary};
    if (!out)
    {
        throw std::runtime_error(std::format("failed to open output file: {}", path.string()));
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out)
    {
        throw std::runtime_error(std::format("failed to write output file: {}", path.string()));
    }
}

auto align_buffer(std::vector<u8>& buffer) -> void
{
    while ((buffer.size() % k_gltf_alignment) != 0zu)
    {
        buffer.push_back(0u);
    }
}

template <typename T>
[[nodiscard]] auto append_values(std::vector<u8>& buffer, std::span<const T> values) -> usize
{
    align_buffer(buffer);
    const auto offset = buffer.size();
    const auto bytes = values.size_bytes();
    const auto* begin = reinterpret_cast<const u8*>(values.data());
    buffer.insert(buffer.end(), begin, begin + static_cast<std::ptrdiff_t>(bytes));
    return offset;
}

[[nodiscard]] auto color_factor(Color color) -> std::array<f32, 4>
{
    return {color.r(), color.g(), color.b(), color.a()};
}

[[nodiscard]] auto transform_identity(const Transform& transform) -> bool
{
    constexpr auto eps = 1.0e-6f;
    return glm::length(transform.translation) < eps and std::abs(transform.rotation.w - 1.0f) < eps
           and std::abs(transform.rotation.x) < eps and std::abs(transform.rotation.y) < eps
           and std::abs(transform.rotation.z) < eps
           and glm::length(transform.scale - Vec3{1.0f}) < eps;
}

auto apply_node_transform(nlohmann::json& node, const Transform& transform) -> void
{
    if (transform_identity(transform))
    {
        return;
    }
    node["translation"] =
        std::array{transform.translation.x, transform.translation.y, transform.translation.z};
    node["rotation"] = std::array{
        transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w
    };
    node["scale"] = std::array{transform.scale.x, transform.scale.y, transform.scale.z};
}

auto write_png(const std::filesystem::path& path, const GltfImageData& image) -> void
{
    if (image.width == 0u or image.height == 0u
        or image.pixels.size() != static_cast<usize>(image.width) * image.height)
    {
        throw std::runtime_error(std::format("invalid glTF image data: {}", image.name));
    }
    const auto result = stbi_write_png(
        path.string().c_str(),
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        4,
        image.pixels.data(),
        static_cast<int>(image.width * sizeof(ColorU8))
    );
    if (result == 0)
    {
        throw std::runtime_error(std::format("failed to write PNG image: {}", path.string()));
    }
}

[[nodiscard]] auto
make_relative_uri(const std::filesystem::path& base_dir, const std::filesystem::path& path)
    -> std::string
{
    std::error_code ec{};
    auto relative = std::filesystem::relative(path, base_dir, ec);
    if (ec)
    {
        relative = path.filename();
    }
    return relative.generic_string();
}
}  // namespace

auto write_gltf_scene(
    const std::filesystem::path& path,
    std::span<const GltfMeshData> meshes,
    std::span<const GltfMaterialData> materials,
    std::span<const GltfImageData> images,
    const GltfWriteConfig& cfg
) -> void
{
    if (meshes.empty())
    {
        throw std::runtime_error("cannot write glTF scene with no meshes");
    }

    const auto output_dir =
        path.parent_path().empty() ? std::filesystem::path{"."} : path.parent_path();
    std::filesystem::create_directories(output_dir);

    const auto stem = sanitize_filename(path.stem().string(), "scene");
    const auto bin_path = output_dir / (stem + ".bin");
    const auto image_dir = output_dir / (stem + "_images");

    nlohmann::json document{
        {"asset", {{"version", "2.0"}, {"generator", cfg.generator}}},
        {"scene", 0},
        {"scenes", nlohmann::json::array({{{"nodes", nlohmann::json::array()}}})},
        {"nodes", nlohmann::json::array()},
        {"meshes", nlohmann::json::array()},
        {"buffers", nlohmann::json::array()},
        {"bufferViews", nlohmann::json::array()},
        {"accessors", nlohmann::json::array()},
    };

    if (!images.empty())
    {
        std::filesystem::create_directories(image_dir);
        document["images"] = nlohmann::json::array();
        document["textures"] = nlohmann::json::array();
        for (auto i = 0zu; i < images.size(); ++i)
        {
            const auto image_name = sanitize_filename(images[i].name, std::format("image_{}", i));
            const auto image_path = image_dir / (image_name + ".png");
            write_png(image_path, images[i]);
            document["images"].push_back(
                {{"name", images[i].name},
                 {"uri", make_relative_uri(output_dir, image_path)},
                 {"mimeType", "image/png"}}
            );
            document["textures"].push_back({{"name", images[i].name}, {"source", i}});
        }
    }

    document["materials"] = nlohmann::json::array();
    if (materials.empty())
    {
        document["materials"].push_back(
            {{"name", "default"},
             {"doubleSided", true},
             {"pbrMetallicRoughness",
              {{"baseColorFactor", color_factor(Color::white)},
               {"metallicFactor", 0.0f},
               {"roughnessFactor", 1.0f}}}}
        );
    }
    else
    {
        for (const auto& material : materials)
        {
            nlohmann::json pbr{
                {"baseColorFactor", color_factor(material.base_color)},
                {"metallicFactor", material.metallic},
                {"roughnessFactor", material.roughness},
            };
            if (material.base_color_image != k_invalid_index)
            {
                if (material.base_color_image >= images.size())
                {
                    throw std::runtime_error(
                        std::format("glTF material references missing image: {}", material.name)
                    );
                }
                pbr["baseColorTexture"] = {{"index", material.base_color_image}};
            }
            document["materials"].push_back(
                {{"name", material.name},
                 {"doubleSided", material.double_sided},
                 {"pbrMetallicRoughness", std::move(pbr)}}
            );
        }
    }

    std::vector<u8> binary{};
    const auto add_buffer_view =
        [&](const usize byte_offset, const usize byte_length, const int target) -> int
    {
        auto view = nlohmann::json{
            {"buffer", 0},
            {"byteOffset", byte_offset},
            {"byteLength", byte_length},
            {"target", target},
        };
        const auto index = static_cast<int>(document["bufferViews"].size());
        document["bufferViews"].push_back(std::move(view));
        return index;
    };
    const auto add_accessor = [&](const int buffer_view,
                                  const int component_type,
                                  const usize count,
                                  std::string_view type,
                                  const nlohmann::json& min_value = nullptr,
                                  const nlohmann::json& max_value = nullptr) -> int
    {
        auto accessor = nlohmann::json{
            {"bufferView", buffer_view},
            {"componentType", component_type},
            {"count", count},
            {"type", type},
        };
        if (!min_value.is_null())
        {
            accessor["min"] = min_value;
        }
        if (!max_value.is_null())
        {
            accessor["max"] = max_value;
        }
        const auto index = static_cast<int>(document["accessors"].size());
        document["accessors"].push_back(std::move(accessor));
        return index;
    };

    for (auto mesh_index = 0zu; mesh_index < meshes.size(); ++mesh_index)
    {
        const auto& mesh_asset = meshes[mesh_index];
        const auto& mesh = mesh_asset.mesh;
        if (mesh.vertices.empty() or mesh.indices.empty())
        {
            throw std::runtime_error(
                std::format("cannot write empty glTF mesh: {}", mesh_asset.name)
            );
        }

        std::vector<GltfVec3> positions{};
        std::vector<GltfVec3> normals{};
        std::vector<GltfVec2> texcoords{};
        positions.reserve(mesh.vertices.size());
        normals.reserve(mesh.vertices.size());
        texcoords.reserve(mesh.vertices.size());

        Vec3 min_position{std::numeric_limits<f32>::max()};
        Vec3 max_position{std::numeric_limits<f32>::lowest()};
        for (const auto& vertex : mesh.vertices)
        {
            positions.push_back({vertex.position.x, vertex.position.y, vertex.position.z});
            normals.push_back({vertex.normal.x, vertex.normal.y, vertex.normal.z});
            texcoords.push_back({vertex.texcoord.x, vertex.texcoord.y});
            min_position = glm::min(min_position, vertex.position);
            max_position = glm::max(max_position, vertex.position);
        }

        const auto position_offset = append_values(binary, std::span<const GltfVec3>{positions});
        const auto position_view = add_buffer_view(
            position_offset, positions.size() * sizeof(GltfVec3), k_buffer_target_array
        );
        const auto position_accessor = add_accessor(
            position_view,
            k_component_f32,
            positions.size(),
            "VEC3",
            std::array{min_position.x, min_position.y, min_position.z},
            std::array{max_position.x, max_position.y, max_position.z}
        );

        const auto normal_offset = append_values(binary, std::span<const GltfVec3>{normals});
        const auto normal_view = add_buffer_view(
            normal_offset, normals.size() * sizeof(GltfVec3), k_buffer_target_array
        );
        const auto normal_accessor =
            add_accessor(normal_view, k_component_f32, normals.size(), "VEC3");

        const auto texcoord_offset = append_values(binary, std::span<const GltfVec2>{texcoords});
        const auto texcoord_view = add_buffer_view(
            texcoord_offset, texcoords.size() * sizeof(GltfVec2), k_buffer_target_array
        );
        const auto texcoord_accessor =
            add_accessor(texcoord_view, k_component_f32, texcoords.size(), "VEC2");

        const auto index_offset = append_values(binary, std::span<const u32>{mesh.indices});
        const auto index_view = add_buffer_view(
            index_offset, mesh.indices.size() * sizeof(u32), k_buffer_target_element_array
        );
        const auto index_accessor =
            add_accessor(index_view, k_component_u32, mesh.indices.size(), "SCALAR");

        const auto material_index =
            mesh_asset.material == k_invalid_index ? 0zu : mesh_asset.material;
        if (material_index >= document["materials"].size())
        {
            throw std::runtime_error(
                std::format("glTF mesh references missing material: {}", mesh_asset.name)
            );
        }

        document["meshes"].push_back(
            {{"name", mesh_asset.name},
             {"primitives",
              nlohmann::json::array(
                  {{{"attributes",
                     {{"POSITION", position_accessor},
                      {"NORMAL", normal_accessor},
                      {"TEXCOORD_0", texcoord_accessor}}},
                    {"indices", index_accessor},
                    {"material", material_index},
                    {"mode", k_mode_triangles}}}
              )}}
        );

        auto node = nlohmann::json{{"name", mesh_asset.name}, {"mesh", mesh_index}};
        apply_node_transform(node, mesh_asset.transform);
        if (mesh_asset.visible_in_default_scene)
        {
            document["scenes"][0]["nodes"].push_back(document["nodes"].size());
        }
        document["nodes"].push_back(std::move(node));
    }

    align_buffer(binary);
    document["buffers"].push_back(
        {{"byteLength", binary.size()}, {"uri", make_relative_uri(output_dir, bin_path)}}
    );

    write_all_bytes(bin_path, binary);
    write_text(path, cfg.pretty ? document.dump(2) : document.dump());
}
}  // namespace ds_vk
