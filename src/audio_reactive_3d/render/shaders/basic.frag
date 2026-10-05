#version 330

// See basic.vert for the full feature-buffer layout reminder.
layout(std140) uniform Features {
    vec4 data[32];
};

in vec3 v_world_normal;
in vec3 v_view_dir;

uniform vec3 u_base_color;
uniform float u_ambient;

out vec4 frag_color;

void main() {
    float mid_band = data[16].y;
    float high_band = data[16].z;

    // Idle-visible base tone (tunable via config.BASE_COLOR); treble pushes
    // warm/bright hues, mids push blue/violet on top of it.
    vec3 color = u_base_color + vec3(high_band, 0.55 * mid_band + 0.25 * high_band, mid_band);

    vec3 normal = normalize(v_world_normal);
    vec3 view_dir = normalize(v_view_dir);

    vec3 key_light_dir = normalize(vec3(0.4, 0.7, 0.6));
    float diffuse = max(dot(normal, key_light_dir), 0.0);

    // Fresnel-style rim light so the silhouette reads clearly even when the
    // key light is dim or facing away -- keeps the sphere legible instead of
    // silhouetting to black against the dark background.
    float rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);

    float lighting = u_ambient + (1.0 - u_ambient) * diffuse + 0.35 * rim;
    frag_color = vec4(color * lighting, 1.0);
}
