#version 450

layout(location = 0) in vec3 in_world_direction;
layout(location = 1) in float in_background_intensity;
layout(location = 2) flat in uint in_environment_texture_index;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler2D material_textures[15];

const float PI = 3.14159265359;

vec2 equirect_uv(vec3 direction)
{
    vec3 safe_direction = normalize(direction);
    float u = atan(safe_direction.y, safe_direction.x) / (2.0 * PI) + 0.5;
    float v = 0.5 - asin(clamp(safe_direction.z, -1.0, 1.0)) / PI;
    return vec2(fract(u), clamp(v, 0.0, 1.0));
}

vec3 reinhard_tonemap(vec3 color)
{
    return color / (color + vec3(1.0));
}

void main()
{
    vec3 hdr_color =
        texture(material_textures[in_environment_texture_index], equirect_uv(in_world_direction))
            .rgb
        * in_background_intensity;
    out_color = vec4(reinhard_tonemap(hdr_color), 1.0);
}
