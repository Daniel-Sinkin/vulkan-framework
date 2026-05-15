#version 450

layout(push_constant) uniform PushConstants
{
    mat4 model_view_projection;
    vec4 normal_x;
    vec4 normal_y;
    vec4 normal_z;
    vec4 color;
} pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec4 out_color;

void main()
{
    mat3 normal_matrix = mat3(pc.normal_x.xyz, pc.normal_y.xyz, pc.normal_z.xyz);
    gl_Position = pc.model_view_projection * vec4(in_position, 1.0);
    out_normal = normalize(normal_matrix * in_normal);
    out_color = in_color * pc.color;
}
