#include "ds_vk/assets.hpp"

#include "ds_vk/math.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ds_vk
{
namespace
{
constexpr auto k_component_u8 = 5121;
constexpr auto k_component_u16 = 5123;
constexpr auto k_component_u32 = 5125;
constexpr auto k_component_f32 = 5126;
constexpr auto k_mode_triangles = 4;
constexpr u32 k_glb_magic{0x46546c67u};
constexpr u32 k_glb_version_2{2u};
constexpr auto k_glb_header_bytes = 12zu;
constexpr auto k_glb_chunk_header_bytes = 8zu;
constexpr auto k_glb_min_bytes = k_glb_header_bytes + k_glb_chunk_header_bytes;
constexpr u32 k_glb_json_chunk_type{0x4e4f534au};
constexpr u32 k_glb_binary_chunk_type{0x004e4942u};
constexpr auto k_glb_chunk_alignment = 4zu;

struct GltfDocument
{
    nlohmann::json json{};
    std::vector<std::vector<u8>> buffers{};
};

struct AccessorView
{
    const u8* data{};
    usize count{};
    usize stride{};
    usize component_size{};
    int component_type{};
};

[[nodiscard]] auto read_file_bytes(const std::filesystem::path& path) -> std::vector<u8>
{
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in)
    {
        throw std::runtime_error(std::format("failed to open glTF file: {}", path.string()));
    }
    const auto size = in.tellg();
    if (size < 0)
    {
        throw std::runtime_error(std::format("failed to size glTF file: {}", path.string()));
    }
    std::vector<u8> bytes(static_cast<usize>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in)
    {
        throw std::runtime_error(std::format("failed to read glTF file: {}", path.string()));
    }
    return bytes;
}

template <typename T>
[[nodiscard]] auto read_unaligned(const u8* data) noexcept -> T
{
    T value{};
    std::memcpy(&value, data, sizeof(T));
    return value;
}

[[nodiscard]] auto json_usize(const nlohmann::json& object, const char* key, usize fallback)
    -> usize
{
    const auto found = object.find(key);
    if (found == object.end())
    {
        return fallback;
    }
    if (found->is_number_unsigned())
    {
        return static_cast<usize>(found->get<u64>());
    }
    if (found->is_number_integer())
    {
        const auto value = found->get<i64>();
        if (value >= 0)
        {
            return static_cast<usize>(value);
        }
    }
    return fallback;
}

[[nodiscard]] auto json_int(const nlohmann::json& object, const char* key, int fallback) -> int
{
    const auto found = object.find(key);
    if (found == object.end() or !found->is_number_integer())
    {
        return fallback;
    }
    return found->get<int>();
}

[[nodiscard]] auto component_size(int component_type) -> usize
{
    switch (component_type)
    {
        case k_component_u8:
            return sizeof(u8);
        case k_component_u16:
            return sizeof(u16);
        case k_component_u32:
            return sizeof(u32);
        case k_component_f32:
            return sizeof(f32);
        default:
            throw std::runtime_error(
                std::format("unsupported glTF component type: {}", component_type)
            );
    }
}

[[nodiscard]] auto type_component_count(std::string_view type) -> usize
{
    if (type == "SCALAR") return 1zu;
    if (type == "VEC2") return 2zu;
    if (type == "VEC3") return 3zu;
    if (type == "VEC4") return 4zu;
    throw std::runtime_error(std::format("unsupported glTF accessor type: {}", type));
}

[[nodiscard]] auto decode_base64(std::string_view text) -> std::vector<u8>
{
    constexpr int invalid{-1};
    std::array<int, 256> table{};
    table.fill(invalid);
    constexpr auto alphabet =
        std::string_view{"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};
    for (auto i = 0zu; i < alphabet.size(); ++i)
    {
        table[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }

    std::vector<u8> out{};
    auto value = 0;
    auto bits = -8;
    for (const char c : text)
    {
        if (c == '=')
        {
            break;
        }
        const auto decoded = table[static_cast<unsigned char>(c)];
        if (decoded == invalid)
        {
            continue;
        }
        value = (value << 6) + decoded;
        bits += 6;
        if (bits >= 0)
        {
            out.push_back(static_cast<u8>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

[[nodiscard]] auto read_buffer_uri(const std::filesystem::path& base_dir, std::string_view uri)
    -> std::vector<u8>
{
    constexpr std::string_view data_prefix{"data:"};
    if (uri.starts_with(data_prefix))
    {
        const auto comma = uri.find(',');
        if (comma == std::string::npos)
        {
            throw std::runtime_error("invalid glTF data URI");
        }
        return decode_base64(std::string_view{uri}.substr(comma + 1zu));
    }
    return read_file_bytes(base_dir / std::filesystem::path{uri});
}

[[nodiscard]] auto parse_glb(const std::filesystem::path& path, std::span<const u8> bytes)
    -> GltfDocument
{
    // glTF 2.0 GLB container rules:
    // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#glb-file-format-specification
    if (bytes.size() < k_glb_min_bytes)
    {
        throw std::runtime_error(std::format("GLB file is too small: {}", path.string()));
    }
    const auto magic = read_unaligned<u32>(bytes.data());
    const auto version = read_unaligned<u32>(bytes.data() + 4zu);
    const auto container_length = static_cast<usize>(read_unaligned<u32>(bytes.data() + 8zu));
    if (magic != k_glb_magic or version != k_glb_version_2 or container_length != bytes.size())
    {
        throw std::runtime_error(std::format("invalid GLB header: {}", path.string()));
    }

    GltfDocument document{};
    auto cursor = k_glb_header_bytes;

    const auto json_chunk_length = static_cast<usize>(read_unaligned<u32>(bytes.data() + cursor));
    const auto json_chunk_type = read_unaligned<u32>(bytes.data() + cursor + 4zu);
    cursor += k_glb_chunk_header_bytes;
    // The JSON chunk must be the first GLB chunk according to the spec's chunk table.
    if (json_chunk_type != k_glb_json_chunk_type)
    {
        throw std::runtime_error(std::format("first GLB chunk is not JSON: {}", path.string()));
    }
    if ((json_chunk_length % k_glb_chunk_alignment) != 0zu)
    {
        throw std::runtime_error(std::format("misaligned GLB JSON chunk: {}", path.string()));
    }
    if (cursor + json_chunk_length > container_length)
    {
        throw std::runtime_error(std::format("invalid GLB JSON chunk length: {}", path.string()));
    }
    const auto json_text =
        std::string{reinterpret_cast<const char*>(bytes.data() + cursor), json_chunk_length};
    document.json = nlohmann::json::parse(json_text);
    cursor += json_chunk_length;

    while (cursor + k_glb_chunk_header_bytes <= container_length)
    {
        const auto chunk_length = static_cast<usize>(read_unaligned<u32>(bytes.data() + cursor));
        const auto chunk_type = read_unaligned<u32>(bytes.data() + cursor + 4zu);
        cursor += k_glb_chunk_header_bytes;
        if ((chunk_length % k_glb_chunk_alignment) != 0zu)
        {
            throw std::runtime_error(std::format("misaligned GLB chunk length: {}", path.string()));
        }
        if (cursor + chunk_length > container_length)
        {
            throw std::runtime_error(std::format("invalid GLB chunk length: {}", path.string()));
        }
        // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#chunks-overview
        // Unknown chunk types are intentionally ignored after validation.
        if (chunk_type == k_glb_binary_chunk_type)
        {
            document.buffers.emplace_back(
                bytes.begin() + static_cast<isize>(cursor),
                bytes.begin() + static_cast<isize>(cursor + chunk_length)
            );
        }
        cursor += chunk_length;
    }
    if (cursor != container_length)
    {
        throw std::runtime_error(std::format("truncated GLB chunk header: {}", path.string()));
    }
    return document;
}

[[nodiscard]] auto parse_gltf(const std::filesystem::path& path) -> GltfDocument
{
    const auto bytes = read_file_bytes(path);
    if (path.extension() == ".glb")
    {
        return parse_glb(path, bytes);
    }

    GltfDocument document{};
    const std::string json_text{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    document.json = nlohmann::json::parse(json_text);

    const auto buffers = document.json.find("buffers");
    if (buffers == document.json.end() or !buffers->is_array())
    {
        throw std::runtime_error(std::format("glTF has no buffers: {}", path.string()));
    }
    document.buffers.reserve(buffers->size());
    for (const auto& buffer : *buffers)
    {
        const auto uri = buffer.value("uri", std::string{});
        if (uri.empty())
        {
            throw std::runtime_error(
                std::format("non-GLB glTF buffer is missing uri: {}", path.string())
            );
        }
        document.buffers.push_back(read_buffer_uri(path.parent_path(), uri));
    }
    return document;
}

[[nodiscard]] auto accessor_view(const GltfDocument& document, int accessor_index) -> AccessorView
{
    if (accessor_index < 0)
    {
        return {};
    }
    const auto& accessors = document.json.at("accessors");
    if (static_cast<usize>(accessor_index) >= accessors.size())
    {
        throw std::runtime_error("glTF accessor index out of range");
    }
    const auto& accessor = accessors[static_cast<usize>(accessor_index)];
    const auto buffer_view_index = json_usize(accessor, "bufferView", k_invalid_index);
    const auto& buffer_views = document.json.at("bufferViews");
    if (buffer_view_index >= buffer_views.size())
    {
        throw std::runtime_error("glTF bufferView index out of range");
    }
    const auto& buffer_view = buffer_views[buffer_view_index];
    const auto buffer_index = json_usize(buffer_view, "buffer", k_invalid_index);
    if (buffer_index >= document.buffers.size())
    {
        throw std::runtime_error("glTF buffer index out of range");
    }
    const auto& buffer = document.buffers[buffer_index];
    const auto accessor_offset = json_usize(accessor, "byteOffset", 0zu);
    const auto view_offset = json_usize(buffer_view, "byteOffset", 0zu);
    const auto component_type = json_int(accessor, "componentType", 0);
    const auto count = json_usize(accessor, "count", 0zu);
    const auto component_bytes = component_size(component_type);
    const auto type = accessor.value("type", std::string{});
    const auto components = type_component_count(type);
    const auto element_bytes = component_bytes * components;
    const auto stride = json_usize(buffer_view, "byteStride", element_bytes);
    const auto offset = view_offset + accessor_offset;
    if (stride < element_bytes)
    {
        throw std::runtime_error("glTF accessor stride is smaller than element size");
    }
    if (
        count != 0zu
        and (offset > buffer.size() or (count - 1zu) * stride + element_bytes > buffer.size() - offset)
    )
    {
        throw std::runtime_error("glTF accessor points outside buffer");
    }
    return AccessorView{
        .data = buffer.data() + offset,
        .count = count,
        .stride = stride,
        .component_size = component_bytes,
        .component_type = component_type,
    };
}

[[nodiscard]] auto read_vec3(const AccessorView& view, usize index) -> Vec3
{
    if (view.component_type != k_component_f32 or view.component_size != sizeof(f32))
    {
        throw std::runtime_error(
            std::format("glTF VEC3 accessor is not f32; component type is {}", view.component_type)
        );
    }
    const auto* data = view.data + index * view.stride;
    return {
        read_unaligned<f32>(data + 0zu),
        read_unaligned<f32>(data + 4zu),
        read_unaligned<f32>(data + 8zu),
    };
}

[[nodiscard]] auto read_vec2(const AccessorView& view, usize index) -> Vec2
{
    if (view.component_type != k_component_f32 or view.component_size != sizeof(f32))
    {
        throw std::runtime_error(
            std::format("glTF VEC2 accessor is not f32; component type is {}", view.component_type)
        );
    }
    const auto* data = view.data + index * view.stride;
    return {read_unaligned<f32>(data + 0zu), read_unaligned<f32>(data + 4zu)};
}

[[nodiscard]] auto read_index(const AccessorView& view, usize index) -> usize
{
    const auto* data = view.data + index * view.stride;
    switch (view.component_type)
    {
        case k_component_u8:
            return static_cast<usize>(read_unaligned<u8>(data));
        case k_component_u16:
            return static_cast<usize>(read_unaligned<u16>(data));
        case k_component_u32:
            return static_cast<usize>(read_unaligned<u32>(data));
        default:
            throw std::runtime_error(
                std::format("unsupported glTF index component type: {}", view.component_type)
            );
    }
}

auto generate_smooth_normals(MeshData& mesh) -> void
{
    std::vector<Vec3> normals(mesh.vertices.size(), Vec3{0.0f});
    for (auto i = 0zu; i + 2zu < mesh.indices.size(); i += 3zu)
    {
        const auto ia = mesh.indices[i + 0zu];
        const auto ib = mesh.indices[i + 1zu];
        const auto ic = mesh.indices[i + 2zu];
        if (ia >= mesh.vertices.size() or ib >= mesh.vertices.size() or ic >= mesh.vertices.size())
        {
            continue;
        }
        const auto a = mesh.vertices[ia].position;
        const auto b = mesh.vertices[ib].position;
        const auto c = mesh.vertices[ic].position;
        const auto face_normal = glm::cross(b - a, c - a);
        normals[ia] += face_normal;
        normals[ib] += face_normal;
        normals[ic] += face_normal;
    }
    for (auto i = 0zu; i < mesh.vertices.size(); ++i)
    {
        mesh.vertices[i].normal = normalize_or(normals[i], k_axis_z);
    }
}
}  // namespace

auto load_gltf_mesh(const std::filesystem::path& path, const GltfMeshLoadConfig& cfg) -> MeshData
{
    const auto document = parse_gltf(path);
    const auto& mesh = document.json.at("meshes").at(0zu);
    const auto& primitive = mesh.at("primitives").at(0zu);
    if (json_int(primitive, "mode", k_mode_triangles) != k_mode_triangles)
    {
        throw std::runtime_error("only glTF triangle primitives are supported");
    }

    const auto& attributes = primitive.at("attributes");
    const auto positions = accessor_view(document, attributes.at("POSITION").get<int>());
    const auto normal_attribute = attributes.find("NORMAL");
    const auto normals = normal_attribute == attributes.end()
                             ? AccessorView{}
                             : accessor_view(document, normal_attribute->get<int>());
    const auto texcoord_attribute = attributes.find("TEXCOORD_0");
    const auto texcoords = texcoord_attribute == attributes.end()
                               ? AccessorView{}
                               : accessor_view(document, texcoord_attribute->get<int>());
    const auto indices = primitive.find("indices") == primitive.end()
                             ? AccessorView{}
                             : accessor_view(document, primitive.at("indices").get<int>());

    MeshData mesh_data{};
    mesh_data.vertices.reserve(positions.count);
    const auto model = cfg.transform.matrix();
    const auto normal_matrix = glm::transpose(glm::inverse(Mat4{model}));
    const auto has_normals = normals.data and normals.count == positions.count;
    const auto has_texcoords = texcoords.data and texcoords.count == positions.count;
    for (auto i = 0zu; i < positions.count; ++i)
    {
        const auto position = model * Vec4{read_vec3(positions, i), 1.0f};
        const auto normal =
            has_normals
                ? normalize_or(Vec3{normal_matrix * Vec4{read_vec3(normals, i), 0.0f}}, k_axis_z)
                : k_axis_z;
        mesh_data.vertices.push_back(
            Vertex{
                .position = Vec3{position},
                .normal = normal,
                .color = cfg.color,
                .texcoord = has_texcoords ? read_vec2(texcoords, i) : Vec2{},
            }
        );
    }

    if (indices.data)
    {
        mesh_data.indices.reserve(indices.count);
        for (auto i = 0zu; i < indices.count; ++i)
        {
            const auto index = read_index(indices, i);
            if (index >= mesh_data.vertices.size())
            {
                throw std::runtime_error("glTF index out of range");
            }
            mesh_data.indices.push_back(static_cast<u32>(index));
        }
    }
    else
    {
        if ((positions.count % 3zu) != 0zu)
        {
            throw std::runtime_error("non-indexed glTF primitive has non-triangle vertex count");
        }
        if (positions.count > static_cast<usize>(std::numeric_limits<u32>::max()))
        {
            throw std::runtime_error("non-indexed glTF primitive has too many vertices");
        }
        mesh_data.indices.reserve(positions.count);
        const auto index_count = static_cast<u32>(positions.count);
        for (auto i = 0u; i < index_count; ++i)
        {
            mesh_data.indices.push_back(i);
        }
    }

    if (!has_normals and cfg.generate_normals_if_missing)
    {
        generate_smooth_normals(mesh_data);
    }
    return mesh_data;
}
}  // namespace ds_vk
