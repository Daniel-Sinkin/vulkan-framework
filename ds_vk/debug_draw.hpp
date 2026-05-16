#pragma once

#include "ds_vk/types.hpp"

namespace ds_vk
{

struct DebugLineConfig
{
    Vec3 start{};
    Vec3 end{};
    Color color{Color::white};
    f32 width{0.012f};
    bool draw_on_top{};
};

struct DebugArrowConfig
{
    Vec3 origin{};
    Vec3 vector{};
    Color color{Color::white};
    f32 width{0.016f};
    bool draw_on_top{};
};

struct DebugSphereConfig
{
    Vec3 center{};
    f32 radius{1.0f};
    Color color{Color::white};
    u32 segments{32u};
    f32 width{0.010f};
    bool draw_on_top{};
};

struct DebugSegment
{
    Vec3 start{};
    f32 width{0.012f};
    Vec3 end{};
    f32 arrow_tip{};
    Color color{Color::white};
};

}  // namespace ds_vk
