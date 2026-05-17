#include "app/quake/io.hpp"

#include <fstream>
#include <print>
#include <system_error>

namespace ds_vk_quake
{
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
}  // namespace ds_vk_quake
