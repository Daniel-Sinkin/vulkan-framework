#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec4 in_color;
layout(location = 2) flat in uint in_material_index;
layout(location = 3) in vec3 in_world_position;
layout(location = 4) in vec2 in_texcoord;
layout(location = 5) flat in vec3 in_facet_seed;

layout(location = 0) out vec4 out_color;

struct Material
{
    vec4 base_color;
    vec4 emissive_color;
    vec4 pbr_params;
    vec4 texture_params;
    vec4 render_params;
    vec4 debug_color;
    vec4 debug_params;
    vec4 debug_params2;
    vec4 camera_position;
    vec4 camera_forward;
};

struct Light
{
    vec4 position_range;
    vec4 direction_type;
    vec4 color_intensity;
    vec4 spot_shadow;
};

struct Lighting
{
    vec4 ambient_light_count;
    mat4 shadow_view_projection;
    vec4 shadow_params;
    vec4 environment_params;
    Light lights[16];
};

layout(set = 0, binding = 0) readonly buffer MaterialBuffer
{
    Material materials[];
}
material_buffer;

layout(set = 0, binding = 1) uniform sampler2D material_textures[15];
layout(set = 0, binding = 2) readonly buffer LightingBuffer
{
    Lighting lighting;
}
lighting_buffer;
layout(set = 0, binding = 3) uniform sampler2D shadow_map;

const float PI = 3.14159265359;
const uint DEBUG_NONE = 0;
const uint DEBUG_COLOR_OVERRIDE = 1;
const uint DEBUG_SELECTED_PULSE = 2;
const uint DEBUG_SCALAR_HEATMAP = 3;
const uint DEBUG_NORMAL = 4;
const uint DEBUG_OBJECT_ID = 5;
const uint DEBUG_CAMERA_DEPTH = 6;
const uint DEBUG_TRIANGLE_SELECTED_PULSE = 7;
const uint DEBUG_WORLD_Z_RAMP = 8;
const uint DEBUG_FACET_COLOR = 9;
const uint DEBUG_ANGLE_SHADED = 10;
const uint LIGHT_DIRECTIONAL = 0;
const uint LIGHT_RADIAL = 1;
const uint LIGHT_SPOT = 2;

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

vec2 equirect_uv(vec3 direction)
{
    vec3 safe_direction = normalize(direction);
    float u = atan(safe_direction.y, safe_direction.x) / (2.0 * PI) + 0.5;
    float v = 0.5 - asin(clamp(safe_direction.z, -1.0, 1.0)) / PI;
    return vec2(fract(u), clamp(v, 0.0, 1.0));
}

vec3 rotate_z(vec3 value, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * value.x - s * value.y, s * value.x + c * value.y, value.z);
}

vec3 sample_environment(vec3 direction, float rotation, uint texture_index)
{
    return texture(material_textures[texture_index], equirect_uv(rotate_z(direction, rotation)))
        .rgb;
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

vec3 mesh_height_ramp(float value)
{
    float t = clamp(value, 0.0, 1.0);
    vec3 low = vec3(0.015, 0.055, 0.085);
    vec3 mid = vec3(0.045, 0.56, 0.78);
    vec3 high = vec3(0.86, 0.98, 1.0);
    return t < 0.68 ? mix(low, mid, t / 0.68) : mix(mid, high, (t - 0.68) / 0.32);
}

uint hash_u32(uint value)
{
    value ^= value >> 16u;
    value *= 2246822519u;
    value ^= value >> 13u;
    value *= 3266489917u;
    value ^= value >> 16u;
    return value;
}

float hash01(uint value)
{
    return float(hash_u32(value) & 16777215u) / 16777215.0;
}

vec3 facet_color(vec3 seed)
{
    uvec3 bits = floatBitsToUint(seed);
    uint h = hash_u32(bits.x ^ hash_u32(bits.y + 1013904223u) ^ hash_u32(bits.z + 1664525u));
    vec3 deep = vec3(0.025, 0.23, 0.34);
    vec3 blue = vec3(0.02, 0.52, 0.75);
    vec3 foam = vec3(0.72, 0.94, 0.98);
    float t = hash01(h);
    vec3 color = t < 0.72 ? mix(deep, blue, t / 0.72) : mix(blue, foam, (t - 0.72) / 0.28);
    return color * (0.82 + 0.28 * hash01(h + 747796405u));
}

vec3 angle_shaded_surface(vec3 normal, vec3 view)
{
    float facing = pow(max(dot(normal, view), 0.0), 0.55);
    float upward = clamp(normal.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 low = vec3(0.025, 0.22, 0.30);
    vec3 high = vec3(0.18, 0.70, 0.84);
    vec3 base = mix(low, high, upward);
    vec3 rim = vec3(0.72, 0.95, 1.0) * pow(1.0 - facing, 2.4) * 0.45;
    return base * (0.44 + 0.68 * facing) + rim;
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

float range_attenuation(float distance_to_light, float range)
{
    float distance_squared = max(distance_to_light * distance_to_light, 0.01);
    if (range <= 0.0)
    {
        return 1.0 / distance_squared;
    }
    float x = clamp(distance_to_light / range, 0.0, 1.0);
    float smooth_cutoff = clamp(1.0 - x * x * x * x, 0.0, 1.0);
    return smooth_cutoff / distance_squared;
}

float shadow_visibility(Material material)
{
    if (lighting_buffer.lighting.shadow_params.x < 0.5 || material.render_params.y < 0.5)
    {
        return 1.0;
    }

    vec4 shadow_clip =
        lighting_buffer.lighting.shadow_view_projection * vec4(in_world_position, 1.0);
    if (shadow_clip.w <= 0.0)
    {
        return 1.0;
    }

    vec3 shadow_ndc = shadow_clip.xyz / shadow_clip.w;
    vec2 uv = shadow_ndc.xy * 0.5 + 0.5;
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 || shadow_ndc.z <= 0.0
        || shadow_ndc.z >= 1.0)
    {
        return 1.0;
    }

    float bias = lighting_buffer.lighting.shadow_params.y;
    float map_size = max(lighting_buffer.lighting.shadow_params.w, 1.0);
    vec2 texel = vec2(1.0 / map_size);
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            float closest = texture(shadow_map, uv + vec2(x, y) * texel).r;
            visibility += (shadow_ndc.z - bias <= closest) ? 1.0 : 0.0;
        }
    }
    return visibility / 9.0;
}

vec3 pbr_light(
    vec3 albedo,
    float metallic,
    float roughness,
    vec3 normal,
    vec3 view,
    vec3 light_direction,
    vec3 radiance
)
{
    vec3 half_vector = normalize_or(light_direction + view, normal);
    float n_dot_l = max(dot(normal, light_direction), 0.0);
    float n_dot_v = max(dot(normal, view), 0.0);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 fresnel = fresnel_schlick(max(dot(half_vector, view), 0.0), f0);
    float normal_distribution = distribution_ggx(normal, half_vector, roughness);
    float geometry = geometry_smith(normal, view, light_direction, roughness);
    vec3 numerator = normal_distribution * geometry * fresnel;
    float denominator = max(4.0 * n_dot_v * n_dot_l, 0.0001);
    vec3 specular = numerator / denominator;
    vec3 k_s = fresnel;
    vec3 k_d = (vec3(1.0) - k_s) * (1.0 - metallic);
    vec3 diffuse = k_d * albedo / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

vec3 evaluate_light(
    Light light,
    Material material,
    vec3 albedo,
    float metallic,
    float roughness,
    vec3 normal,
    vec3 view
)
{
    uint type = uint(light.direction_type.w + 0.5);
    vec3 light_direction = vec3(0.0, 0.0, 1.0);
    float attenuation = 1.0;
    if (type == LIGHT_DIRECTIONAL)
    {
        light_direction = normalize_or(-light.direction_type.xyz, vec3(0.0, 0.0, 1.0));
    }
    else
    {
        vec3 to_light = light.position_range.xyz - in_world_position;
        float distance_to_light = length(to_light);
        light_direction = normalize_or(to_light, vec3(0.0, 0.0, 1.0));
        attenuation = range_attenuation(distance_to_light, light.position_range.w);
        if (type == LIGHT_SPOT)
        {
            vec3 from_light = -light_direction;
            float cd =
                dot(normalize_or(light.direction_type.xyz, vec3(0.0, 0.0, -1.0)), from_light);
            float angular_attenuation =
                clamp(cd * light.spot_shadow.x + light.spot_shadow.y, 0.0, 1.0);
            attenuation *= angular_attenuation * angular_attenuation;
        }
    }

    float visibility = 1.0;
    if (light.spot_shadow.z > 0.5)
    {
        visibility = mix(1.0, shadow_visibility(material), clamp(light.spot_shadow.w, 0.0, 1.0));
    }
    vec3 radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation * visibility;
    return pbr_light(albedo, metallic, roughness, normal, view, light_direction, radiance);
}

vec3 environment_light(
    vec3 albedo, float metallic, float roughness, float ambient_occlusion, vec3 normal, vec3 view
)
{
    vec4 params = lighting_buffer.lighting.environment_params;
    if (params.w < 0.0 || params.x <= 0.0)
    {
        return vec3(0.0);
    }

    float rotation = params.z;
    uint texture_index = uint(params.w + 0.5);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 fresnel = fresnel_schlick(max(dot(normal, view), 0.0), f0);
    vec3 diffuse_env = sample_environment(normal, rotation, texture_index);
    vec3 reflection = reflect(-view, normal);
    vec3 specular_env = sample_environment(reflection, rotation, texture_index);
    vec3 diffuse = (vec3(1.0) - fresnel) * (1.0 - metallic) * albedo * diffuse_env;
    vec3 specular = fresnel * specular_env * max(0.0, 1.0 - 0.72 * roughness);
    return (diffuse + specular) * params.x * ambient_occlusion;
}

vec3 apply_selected_pulse(vec3 shaded_color, Material material)
{
    float time = material.debug_params2.x;
    float wave = 0.5 + 0.5 * sin((gl_FragCoord.x + gl_FragCoord.y) * 0.045 + time * 5.5);
    float amount = clamp(0.28 + 0.42 * wave, 0.0, 1.0) * material.debug_color.a;
    return mix(shaded_color, material.debug_color.rgb, amount);
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
    else if (mode == DEBUG_CAMERA_DEPTH)
    {
        float near_depth = material.debug_params.z;
        float far_depth = max(material.debug_params.w, near_depth + 0.0001);
        vec3 camera_forward = normalize_or(material.camera_forward.xyz, vec3(0.0, 0.0, -1.0));
        float camera_depth = dot(in_world_position - material.camera_position.xyz, camera_forward);
        float depth = smoothstep(near_depth, far_depth, camera_depth);
        result = vec3(depth);
    }
    else if (mode == DEBUG_WORLD_Z_RAMP)
    {
        float range_min = material.debug_params.z;
        float range_max = max(material.debug_params.w, range_min + 0.0001);
        result = mesh_height_ramp((in_world_position.z - range_min) / (range_max - range_min));
    }
    else if (mode == DEBUG_FACET_COLOR)
    {
        result = facet_color(in_facet_seed);
    }
    else if (mode == DEBUG_ANGLE_SHADED)
    {
        vec3 view = normalize_or(material.camera_position.xyz - in_world_position, vec3(0.0, 0.0, 1.0));
        result = angle_shaded_surface(normal, view);
    }

    if (mode == DEBUG_SELECTED_PULSE || material.debug_params2.z > 0.5)
    {
        result = apply_selected_pulse(result, material);
    }
    else if (mode == DEBUG_TRIANGLE_SELECTED_PULSE)
    {
        result = apply_selected_pulse(result, material);
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
    vec3 view = normalize_or(material.camera_position.xyz - in_world_position, vec3(0.0, 0.0, 1.0));
    vec3 ambient = albedo * lighting_buffer.lighting.ambient_light_count.rgb * ambient_occlusion;
    vec3 color = ambient + material.emissive_color.rgb;
    if (material.render_params.x > 0.5)
    {
        color += environment_light(albedo, metallic, roughness, ambient_occlusion, normal, view);
        uint light_count = min(uint(lighting_buffer.lighting.ambient_light_count.w + 0.5), 16u);
        for (uint i = 0u; i < light_count; ++i)
        {
            color += evaluate_light(
                lighting_buffer.lighting.lights[i],
                material,
                albedo,
                metallic,
                roughness,
                normal,
                view
            );
        }
    }
    else
    {
        color = albedo + material.emissive_color.rgb;
    }
    out_color = vec4(apply_debug(color, normal, material), surface_color.a);
}
