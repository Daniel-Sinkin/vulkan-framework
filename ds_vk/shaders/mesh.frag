#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec4 in_color;
layout(location = 2) flat in uint in_material_index;
layout(location = 3) in vec3 in_world_position;
layout(location = 4) in vec2 in_texcoord;

layout(location = 0) out vec4 out_color;

struct Material
{
    vec4 base_color;
    vec4 emissive_color;
    vec4 pbr_params;
    vec4 texture_params;
    vec4 debug_color;
    vec4 debug_params;
    vec4 debug_params2;
    vec4 camera_position;
};

layout(set = 0, binding = 0) readonly buffer MaterialBuffer
{
    Material materials[];
}
material_buffer;

layout(set = 0, binding = 1) uniform sampler2D material_textures[16];

const float PI = 3.14159265359;
const uint DEBUG_NONE = 0;
const uint DEBUG_COLOR_OVERRIDE = 1;
const uint DEBUG_SELECTED_PULSE = 2;
const uint DEBUG_SCALAR_HEATMAP = 3;
const uint DEBUG_NORMAL = 4;
const uint DEBUG_OBJECT_ID = 5;

float distribution_ggx(vec3 normal, vec3 half_vector, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 0.0001);
}

float geometry_schlick_ggx(float n_dot_v, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

float geometry_smith(vec3 normal, vec3 view, vec3 light, float roughness)
{
    float n_dot_v = max(dot(normal, view), 0.0);
    float n_dot_l = max(dot(normal, light), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

vec3 normalize_or(vec3 value, vec3 fallback)
{
    float length_squared = dot(value, value);
    if (length_squared <= 0.0000000001)
    {
        return normalize(fallback);
    }
    return value * inversesqrt(length_squared);
}

vec3 heatmap(float value)
{
    float t = clamp(value, 0.0, 1.0);
    vec3 blue = vec3(0.10, 0.22, 1.00);
    vec3 cyan = vec3(0.00, 0.85, 1.00);
    vec3 yellow = vec3(1.00, 0.86, 0.10);
    vec3 red = vec3(1.00, 0.05, 0.02);
    if (t < 0.33)
    {
        return mix(blue, cyan, t / 0.33);
    }
    if (t < 0.66)
    {
        return mix(cyan, yellow, (t - 0.33) / 0.33);
    }
    return mix(yellow, red, (t - 0.66) / 0.34);
}

vec3 object_id_color(uint object_id)
{
    uint x = object_id * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return vec3(
        float((x >> 0u) & 255u) / 255.0,
        float((x >> 8u) & 255u) / 255.0,
        float((x >> 16u) & 255u) / 255.0
    );
}

vec3 apply_debug(vec3 shaded_color, vec3 normal, Material material)
{
    uint mode = uint(material.debug_params.x + 0.5);
    vec3 result = shaded_color;
    if (mode == DEBUG_COLOR_OVERRIDE)
    {
        result = mix(shaded_color, material.debug_color.rgb, material.debug_color.a);
    }
    else if (mode == DEBUG_SCALAR_HEATMAP)
    {
        float range_min = material.debug_params.z;
        float range_max = material.debug_params.w;
        float denom = max(range_max - range_min, 0.0001);
        result = heatmap((material.debug_params.y - range_min) / denom);
    }
    else if (mode == DEBUG_NORMAL)
    {
        result = normalize(normal) * 0.5 + 0.5;
    }
    else if (mode == DEBUG_OBJECT_ID)
    {
        result = object_id_color(uint(material.debug_params2.y + 0.5));
    }

    if (mode == DEBUG_SELECTED_PULSE || material.debug_params2.z > 0.5)
    {
        float time = material.debug_params2.x;
        float wave = 0.5 + 0.5 * sin((gl_FragCoord.x + gl_FragCoord.y) * 0.045 + time * 5.5);
        float amount = clamp(0.28 + 0.42 * wave, 0.0, 1.0) * material.debug_color.a;
        result = mix(result, material.debug_color.rgb, amount);
    }
    return result;
}

void main()
{
    Material material = material_buffer.materials[in_material_index];
    vec4 texture_color = vec4(1.0);
    if (material.texture_params.y > 0.5)
    {
        uint texture_index = uint(material.texture_params.x + 0.5);
        texture_color = texture(material_textures[texture_index], in_texcoord);
    }
    vec4 surface_color = in_color * material.base_color * texture_color;
    vec3 albedo = surface_color.rgb;
    float metallic = clamp(material.pbr_params.x, 0.0, 1.0);
    float roughness = clamp(material.pbr_params.y, 0.04, 1.0);
    float ambient_occlusion = clamp(material.pbr_params.z, 0.0, 1.0);
    vec3 normal = normalize_or(in_normal, vec3(0.0, 0.0, 1.0));
    vec3 light = normalize(vec3(0.35, 0.45, 0.82));
    vec3 view = normalize_or(material.camera_position.xyz - in_world_position, vec3(0.0, 0.0, 1.0));
    vec3 half_vector = normalize_or(light + view, normal);
    vec3 radiance = vec3(1.85);
    float n_dot_l = max(dot(normal, light), 0.0);
    float n_dot_v = max(dot(normal, view), 0.0);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 fresnel = fresnel_schlick(max(dot(half_vector, view), 0.0), f0);
    float normal_distribution = distribution_ggx(normal, half_vector, roughness);
    float geometry = geometry_smith(normal, view, light, roughness);
    vec3 numerator = normal_distribution * geometry * fresnel;
    float denominator = max(4.0 * n_dot_v * n_dot_l, 0.0001);
    vec3 specular = numerator / denominator;
    vec3 k_s = fresnel;
    vec3 k_d = (vec3(1.0) - k_s) * (1.0 - metallic);
    vec3 diffuse = k_d * albedo / PI;
    vec3 ambient = albedo * 0.045 * ambient_occlusion;
    vec3 color = ambient + (diffuse + specular) * radiance * n_dot_l + material.emissive_color.rgb;
    out_color = vec4(apply_debug(color, normal, material), surface_color.a);
}
