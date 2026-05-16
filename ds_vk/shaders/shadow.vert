#version 450

layout(push_constant) uniform PushConstants
{
    mat4 view_projection;
    mat4 model;
}
pc;

layout(location = 0) in vec3 in_position;

void main()
{
    gl_Position = pc.view_projection * pc.model * vec4(in_position, 1.0);
}
