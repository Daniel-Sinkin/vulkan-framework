#version 450

struct RenderParams
{
    mat4 view_projection;
    mat4 view;
    mat4 projection;
    vec4 camera_right;
    vec4 camera_up;
    vec4 camera_forward;
    vec4 base_color;
    vec4 light_direction;
    vec4 options;
};

layout(std430, set = 0, binding = 2) readonly buffer RenderParamsBuffer
{
    RenderParams render_params;
};

layout(location = 0) in vec2 v_local;
layout(location = 1) in vec3 v_center_view;
layout(location = 2) in float v_radius;
layout(location = 3) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main()
{
    const float r2 = dot(v_local, v_local);
    if (r2 > 1.0)
    {
        discard;
    }

    const float nz = sqrt(max(1.0 - r2, 0.0));
    const vec3 normal_view = normalize(vec3(v_local, nz));
    const vec3 view_pos = v_center_view + vec3(v_local * v_radius, nz * v_radius);
    const vec4 clip = render_params.projection * vec4(view_pos, 1.0);
    gl_FragDepth = clip.z / clip.w;

    const vec3 light_view = normalize((render_params.view * vec4(normalize(-render_params.light_direction.xyz), 0.0)).xyz);
    const float ndl = max(dot(normal_view, light_view), 0.0);
    const float rim = pow(1.0 - max(normal_view.z, 0.0), 2.0);
    const vec3 color = v_color.rgb * (0.34 + 0.66 * ndl) + vec3(0.18, 0.28, 0.32) * rim;
    out_color = vec4(color, v_color.a);
}
