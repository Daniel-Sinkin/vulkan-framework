#version 450

layout(push_constant) uniform PushConstants
{
    mat4 inverse_view_projection;
    vec4 camera_position;
    vec4 params;
    vec4 background_color;
    vec4 background_top_color;
}
pc;

layout(location = 0) out vec3 out_world_direction;
layout(location = 1) out float out_background_intensity;
layout(location = 2) flat out uint out_environment_texture_index;
layout(location = 3) out float out_screen_y;
layout(location = 4) out vec4 out_background_color;
layout(location = 5) out vec4 out_background_top_color;
layout(location = 6) flat out uint out_environment_mode;

vec3 rotate_z(vec3 value, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * value.x - s * value.y, s * value.x + c * value.y, value.z);
}

void main()
{
    const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 position = positions[gl_VertexIndex];
    vec4 far_world = pc.inverse_view_projection * vec4(position, 1.0, 1.0);
    far_world /= far_world.w;
    out_world_direction = rotate_z(normalize(far_world.xyz - pc.camera_position.xyz), pc.params.y);
    out_background_intensity = pc.params.x;
    out_environment_texture_index = uint(pc.params.z + 0.5);
    out_screen_y = position.y * 0.5 + 0.5;
    out_background_color = pc.background_color;
    out_background_top_color = pc.background_top_color;
    out_environment_mode = uint(pc.params.w + 0.5);
    gl_Position = vec4(position, 1.0, 1.0);
}
