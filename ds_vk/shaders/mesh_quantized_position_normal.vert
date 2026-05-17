#version 450

layout(push_constant) uniform PushConstants
{
    mat4 view_projection;
    mat4 model;
}
pc;

struct MeshInstance
{
    mat4 model;
    mat4 normal_model;
    uint material_index;
};

layout(set = 0, binding = 4) readonly buffer MeshInstanceBuffer
{
    MeshInstance instances[];
}
instance_buffer;

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_normal_oct;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec4 out_color;
layout(location = 2) flat out uint out_material_index;
layout(location = 3) out vec3 out_world_position;
layout(location = 4) out vec2 out_texcoord;
layout(location = 5) flat out vec3 out_facet_seed;

vec2 oct_wrap(vec2 value)
{
    return (1.0 - abs(value.yx)) * vec2(value.x < 0.0 ? -1.0 : 1.0, value.y < 0.0 ? -1.0 : 1.0);
}

vec3 decode_octahedral_normal(vec2 encoded)
{
    vec2 value = encoded * 2.0 - 1.0;
    vec3 normal = vec3(value.x, value.y, 1.0 - abs(value.x) - abs(value.y));
    if (normal.z < 0.0)
    {
        normal.xy = oct_wrap(normal.xy);
    }
    return normalize(normal);
}

void main()
{
    MeshInstance instance = instance_buffer.instances[gl_InstanceIndex];
    vec4 world_position = instance.model * vec4(in_position.xyz, 1.0);
    gl_Position = pc.view_projection * world_position;
    out_normal = normalize(mat3(instance.normal_model) * decode_octahedral_normal(in_normal_oct));
    out_color = vec4(1.0);
    out_material_index = instance.material_index;
    out_world_position = world_position.xyz;
    out_texcoord = vec2(0.0);
    out_facet_seed = world_position.xyz + out_normal * 0.173;
}
