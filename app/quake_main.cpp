#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>

namespace ds_vk_quake
{
}

auto main() -> int
{
    using namespace ds_vk_quake;
    namespace fs = std::filesystem;

    const auto p = fs::current_path();

    std::cout << "The current path is " << p << "\n";
}
