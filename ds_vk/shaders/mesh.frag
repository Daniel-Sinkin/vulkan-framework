#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

void main()
{
    vec3 normal = normalize(in_normal);
    vec3 light = normalize(vec3(0.35, 0.45, 0.82));
    vec3 view = normalize(vec3(0.25, -0.35, 0.90));
    vec3 half_vector = normalize(light + view);
    float ambient = 0.18;
    float diffuse = max(dot(normal, light), 0.0);
    float back = 0.20 * max(dot(-normal, light), 0.0);
    float specular = pow(max(dot(normal, half_vector), 0.0), 48.0);
    float shade = clamp(ambient + 0.76 * diffuse + back, 0.0, 1.15);
    vec3 highlight = vec3(0.55, 0.68, 0.90) * specular * 0.28;
    out_color = vec4(in_color.rgb * shade + highlight, in_color.a);
}
