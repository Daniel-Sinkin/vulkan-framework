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

layout(std430, set = 0, binding = 0) readonly buffer Positions
{
    vec4 positions_radius[];
};
layout(std430, set = 0, binding = 1) readonly buffer VelocityLambda
{
    vec4 velocity_lambda[];
};
layout(std430, set = 0, binding = 2) readonly buffer RenderParamsBuffer
{
    RenderParams render_params;
};

layout(location = 0) out vec2 v_local;
layout(location = 1) out vec3 v_center_view;
layout(location = 2) out float v_radius;
layout(location = 3) out vec4 v_color;

const vec2 k_corners[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0)
);

vec3 ramp(float t)
{
    t = clamp(t, 0.0, 1.0);
    const vec3 deep = vec3(0.02, 0.15, 0.28);
    const vec3 mid = vec3(0.03, 0.50, 0.78);
    const vec3 foam = vec3(0.75, 0.96, 1.0);
    if (t < 0.55)
    {
        return mix(deep, mid, t / 0.55);
    }
    return mix(mid, foam, (t - 0.55) / 0.45);
}

void main()
{
    const vec4 position_radius = positions_radius[gl_InstanceIndex];
    const vec4 particle_velocity_lambda = velocity_lambda[gl_InstanceIndex];
    const vec2 local = k_corners[gl_VertexIndex];
    const float radius = position_radius.w * render_params.options.x;
    const vec3 world =
        position_radius.xyz +
        (render_params.camera_right.xyz * local.x + render_params.camera_up.xyz * local.y) * radius;

    const float speed_t = clamp(length(particle_velocity_lambda.xyz) / max(render_params.options.y, 0.001), 0.0, 1.0);
    v_local = local;
    v_center_view = (render_params.view * vec4(position_radius.xyz, 1.0)).xyz;
    v_radius = radius;
    v_color = vec4(mix(render_params.base_color.rgb, ramp(speed_t), render_params.options.z), render_params.base_color.a);
    gl_Position = render_params.view_projection * vec4(world, 1.0);
}
