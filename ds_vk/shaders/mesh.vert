#version 450

layout(push_constant) uniform PushConstants
{
    mat4 view_projection;
    mat4 model;
}
pc;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec2 in_texcoord;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec4 out_color;
layout(location = 2) flat out uint out_material_index;
layout(location = 3) out vec3 out_world_position;
layout(location = 4) out vec2 out_texcoord;

void main()
{
    vec4 world_position = pc.model * vec4(in_position, 1.0);
    mat3 normal_matrix = transpose(inverse(mat3(pc.model)));
    gl_Position = pc.view_projection * world_position;
    out_normal = normalize(normal_matrix * in_normal);
    out_color = in_color;
    out_material_index = uint(gl_InstanceIndex);
    out_world_position = world_position.xyz;
    out_texcoord = in_texcoord;
}
