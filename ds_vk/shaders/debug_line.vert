#version 450

layout(push_constant) uniform PushConstants
{
    mat4 view_projection;
    vec4 camera_position;
    vec4 camera_right;
} pc;

layout(location = 0) in vec3 in_start;
layout(location = 1) in float in_width;
layout(location = 2) in vec3 in_end;
layout(location = 3) in float in_arrow_tip;
layout(location = 4) in vec4 in_color;

layout(location = 0) out vec4 out_color;

vec3 safe_normalize(vec3 value, vec3 fallback)
{
    float len2 = dot(value, value);
    if (len2 < 1.0e-12)
    {
        return normalize(fallback);
    }
    return value * inversesqrt(len2);
}

vec3 segment_side(vec3 a, vec3 b, float width)
{
    vec3 segment = b - a;
    vec3 direction = safe_normalize(segment, vec3(1.0, 0.0, 0.0));
    vec3 midpoint = 0.5 * (a + b);
    vec3 to_camera = safe_normalize(pc.camera_position.xyz - midpoint, vec3(0.0, 0.0, 1.0));
    return safe_normalize(cross(direction, to_camera), pc.camera_right.xyz) * (0.5 * width);
}

vec3 quad_vertex(vec3 a, vec3 b, float width, int local_vertex)
{
    vec3 side = segment_side(a, b, width);

    if (local_vertex == 0) { return a - side; }
    if (local_vertex == 1) { return b - side; }
    if (local_vertex == 2) { return b + side; }
    if (local_vertex == 3) { return a - side; }
    if (local_vertex == 4) { return b + side; }
    return a + side;
}

void main()
{
    vec3 segment = in_end - in_start;
    float segment_length = length(segment);
    if (segment_length < 1.0e-6 || in_width <= 0.0)
    {
        gl_Position = pc.view_projection * vec4(in_start, 1.0);
        out_color = in_color;
        return;
    }

    vec3 position = in_end;
    if (in_arrow_tip < 0.5)
    {
        if (gl_VertexIndex < 6)
        {
            position = quad_vertex(in_start, in_end, in_width, gl_VertexIndex);
        }
        gl_Position = pc.view_projection * vec4(position, 1.0);
        out_color = in_color;
        return;
    }

    vec3 direction = segment / segment_length;
    float head_length = clamp(segment_length * 0.30, in_width * 3.0, in_width * 7.0);
    vec3 head_base = in_end - direction * head_length;
    float head_half_width = max(in_width * 2.2, head_length * 0.42);
    vec3 head_side = normalize(segment_side(head_base, in_end, 2.0 * head_half_width));

    if (gl_VertexIndex < 6)
    {
        position = quad_vertex(in_start, head_base, in_width, gl_VertexIndex);
    }
    else if (gl_VertexIndex == 6)
    {
        position = in_end;
    }
    else if (gl_VertexIndex == 7)
    {
        position = head_base + head_side * head_half_width;
    }
    else
    {
        position = head_base - head_side * head_half_width;
    }

    gl_Position = pc.view_projection * vec4(position, 1.0);
    out_color = in_color;
}
