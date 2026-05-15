#include "ds_vk/runtime.hpp"
#include "ds_vk/types.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <iostream>
#include <string_view>
#include <utility>

namespace
{
using namespace ds_vk;
class BasicViewerApp final
{
  public:
    auto setup(ds_vk::Runtime& runtime) -> void
    {
        runtime_ = &runtime;
        floor_mesh_ =
            runtime.upload_mesh(ds_vk::make_quad(8.0f, ds_vk::Vec4{0.55f, 0.58f, 0.53f, 1.0f}));
        cube_mesh_ =
            runtime.upload_mesh(ds_vk::make_cube(1.0f, ds_vk::Vec4{0.82f, 0.48f, 0.25f, 1.0f}));
        rebuild_sphere();

        auto& camera = runtime.camera();
        camera.pivot = ds_vk::Vec3{0.0f, 0.0f, 0.7f};
        camera.distance = 5.4f;
        camera.yaw = glm::radians(42.0f);
        camera.pitch = glm::radians(25.0f);
    }

    auto update(ds_vk::FrameContext& frame, const ds_vk::f32) -> void
    {
        frame.draw.draw_mesh(
            floor_mesh_, ds_vk::Transform{}, ds_vk::Vec4{0.48f, 0.52f, 0.48f, 1.0f}
        );
        frame.draw.draw_mesh(
            sphere_mesh_,
            {
                .translation = sphere_position_,
                .scale = ds_vk::Vec3{sphere_radius_},
            },
            sphere_color_
        );
        if (show_sphere_wire_)
        {
            frame.draw.debug_sphere(
                sphere_position_,
                sphere_radius_,
                ds_vk::Vec4{0.55f, 0.84f, 1.0f, 0.9f},
                std::clamp(sphere_slices_, 12u, 64u),
                0.006f
            );
        }
        frame.draw.draw_mesh(
            cube_mesh_,
            {
                .translation = {-1.6f, 1.1f, 0.5f},
                .rotation = glm::angleAxis(glm::radians(24.0f), ds_vk::k_axis_z),
                .scale = {0.5f, 0.5f, 0.5f},
            },
            {0.9f, 0.46f, 0.2f, 1.0f}
        );

        if (show_debug_axes_)
        {
            const auto& cfg = debug_axis_cfg_;
            frame.draw.debug_arrow(cfg.pos, cfg.arrow_length * k_axis_x, cfg.color_x);
            frame.draw.debug_arrow(cfg.pos, cfg.arrow_length * k_axis_y, cfg.color_y);
            frame.draw.debug_arrow(cfg.pos, cfg.arrow_length * k_axis_z, cfg.color_z);
        }
        if (show_floor_grid_)
        {
            const auto& cfg = floor_grid_cfg_;
            const auto dx = cfg.half_extent * k_axis_x;
            const auto dy = cfg.half_extent * k_axis_y;

            const auto draw_line = [&](int i) -> void
            {
                const auto color = (i == 0) ? cfg.color_major : cfg.color_minor;
                const auto p = static_cast<f32>(i);
                const Vec3 x_base{p, 0, cfg.width};
                const Vec3 y_base{0, p, cfg.width};

                frame.draw.debug_line(x_base - dy, x_base + dy, color, cfg.width);
                frame.draw.debug_line(y_base - dx, y_base + dx, color, cfg.width);
            };
            for (int i = cfg.count_per_side; i <= cfg.count_per_side; ++i)
            {
                draw_line(i);
            }
        }
    }

    auto draw_ui(ds_vk::FrameContext& frame) -> void
    {
        ImGui::SetNextWindowPos(ImVec2{20.0f, 290.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2{500.0f, 360.0f}, ImGuiCond_Once);
        if (!ImGui::Begin("Basic Viewer"))
        {
            ImGui::End();
            return;
        }
        ImGui::PushItemWidth(300.0f);

        ImGui::Checkbox("Debug axes", &show_debug_axes_);
        ImGui::Checkbox("Floor grid", &show_floor_grid_);
        ImGui::Checkbox("Sphere wire", &show_sphere_wire_);
        ImGui::Separator();
        ImGui::DragFloat3("Sphere position", &sphere_position_.x, 0.025f);
        ImGui::SliderFloat("Sphere radius", &sphere_radius_, 0.1f, 2.0f, "%.2f");
        ImGui::ColorEdit4("Sphere color", &sphere_color_.x, ImGuiColorEditFlags_NoInputs);
        auto slices = static_cast<int>(sphere_slices_);
        auto stacks = static_cast<int>(sphere_stacks_);
        auto changed = false;
        changed |= ImGui::SliderInt("Sphere slices", &slices, 8, 96);
        changed |= ImGui::SliderInt("Sphere stacks", &stacks, 4, 64);
        sphere_slices_ = static_cast<ds_vk::u32>(std::clamp(slices, 8, 96));
        sphere_stacks_ = static_cast<ds_vk::u32>(std::clamp(stacks, 4, 64));
        if (changed || ImGui::Button("Rebuild sphere mesh"))
        {
            rebuild_sphere();
        }
        ImGui::Separator();
        ImGui::Text(
            "Framebuffer: %ux%u",
            static_cast<unsigned>(frame.extent.width),
            static_cast<unsigned>(frame.extent.height)
        );
        ImGui::Text("Vulkan command buffer: %p", static_cast<void*>(frame.command_buffer));
        ImGui::PopItemWidth();
        ImGui::End();
    }

  private:
    auto rebuild_sphere() -> void
    {
        if (runtime_ == nullptr)
        {
            return;
        }
        const auto sphere =
            ds_vk::make_uv_sphere(1.0f, sphere_slices_, sphere_stacks_, ds_vk::Vec4{1.0f});
        sphere_mesh_ = sphere_mesh_.valid() ? runtime_->replace_mesh(sphere_mesh_, sphere)
                                            : runtime_->upload_mesh(sphere);
    }

    Runtime* runtime_{};
    MeshHandle floor_mesh_{};
    MeshHandle sphere_mesh_{};
    MeshHandle cube_mesh_{};
    Vec3 sphere_position_{0.0f, 0.0f, 1.0f};
    Vec4 sphere_color_{0.18f, 0.52f, 0.95f, 1.0f};
    f32 sphere_radius_{0.75f};
    u32 sphere_slices_{40u};
    u32 sphere_stacks_{20u};
    bool show_debug_axes_{true};
    bool show_floor_grid_{true};
    bool show_sphere_wire_{true};

    struct DebugAxisConfig
    {
        f32 arrow_length{1.4f};
        Vec3 pos{0.0f, 0.0f, 0.04f};
        Vec4 color_x{0.96f, 0.18f, 0.14f, 1.0f};
        Vec4 color_y{0.20f, 0.78f, 0.24f, 1.0f};
        Vec4 color_z{0.20f, 0.42f, 1.00f, 1.0f};
    };
    DebugAxisConfig debug_axis_cfg_{};

    struct FloorGridConfig
    {
        Vec4 color_major{0.18f, 0.22f, 0.25f, 1.0f};
        Vec4 color_minor{0.18f, 0.22f, 0.25f, 1.0f};
        int count_per_side{4};
        f32 half_extent{4.0f};
        f32 width{0.012f};
    };
    FloorGridConfig floor_grid_cfg_{};
};

auto parse_u32(const char* text, const ds_vk::u32 fallback) -> ds_vk::u32
{
    char* end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text)
    {
        return fallback;
    }
    return static_cast<ds_vk::u32>(value);
}

auto print_usage(const char* executable) -> void
{
    std::cout << "usage: " << executable
              << " [--smoke-frames N] [--screenshot PATH] [--hide-ui]"
                 " [--transparent-screenshot]\n";
}
}  // namespace

auto main(const int argc, char** argv) -> int
{
    try
    {
        auto config = ds_vk::RuntimeConfig{.window_title = "ds_vk Basic Viewer"};
        for (auto i = 1; i < argc; ++i)
        {
            const auto arg = std::string_view{argv[i]};
            if (arg == "--help")
            {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--smoke-frames" && i + 1 < argc)
            {
                config.smoke_frames = parse_u32(argv[++i], 0u);
            }
            else if (arg == "--screenshot" && i + 1 < argc)
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
                std::cerr << "unknown or incomplete argument: " << arg << '\n';
                print_usage(argv[0]);
                return 2;
            }
        }

        auto app = BasicViewerApp{};
        auto runtime = ds_vk::Runtime{std::move(config)};
        return runtime.run(app);
    }
    catch (const std::exception& error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
