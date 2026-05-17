#version 450

layout(location = 0) in vec3 in_world_direction;
layout(location = 1) in float in_background_intensity;
layout(location = 2) flat in uint in_environment_texture_index;
layout(location = 3) in float in_screen_y;
layout(location = 4) in vec4 in_background_color;
layout(location = 5) in vec4 in_background_top_color;
layout(location = 6) flat in uint in_environment_mode;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler2D material_textures[15];

const float PI = 3.14159265359;
const uint ENVIRONMENT_HDRI = 0;
const uint ENVIRONMENT_SOLID = 1;
const uint ENVIRONMENT_GRADIENT = 2;

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
    if (in_environment_mode == ENVIRONMENT_SOLID)
    {
        out_color = vec4(in_background_color.rgb, 1.0);
        return;
    }
    if (in_environment_mode == ENVIRONMENT_GRADIENT)
    {
        out_color =
            vec4(mix(in_background_color.rgb, in_background_top_color.rgb, in_screen_y), 1.0);
        return;
    }

    vec3 hdr_color =
        texture(material_textures[in_environment_texture_index], equirect_uv(in_world_direction))
            .rgb
        * in_background_intensity;
    out_color = vec4(reinhard_tonemap(hdr_color), 1.0);
}
