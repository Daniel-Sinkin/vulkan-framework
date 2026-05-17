#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

namespace ds_vk_quake
{
[[nodiscard]] auto load_binary_file(const std::filesystem::path&)
    -> std::optional<std::vector<std::byte>>;
}  // namespace ds_vk_quake
