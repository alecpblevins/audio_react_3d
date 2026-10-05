#version 330

// See basic.vert for the full feature-buffer layout reminder.
layout(std140) uniform Features {
    vec4 data[32];
};

in vec3 v_world_normal;

out vec4 frag_color;

void main() {
    float mid_band = data[16].y;
    float high_band = data[16].z;

    // Dark base tone; treble pushes warm/bright hues, mids push blue/violet.
    vec3 base_color = vec3(0.04, 0.05, 0.09);
    vec3 color = base_color + vec3(high_band, 0.5 * mid_band + 0.2 * high_band, mid_band);

    vec3 light_dir = normalize(vec3(0.4, 0.7, 0.6));
    float diffuse = max(dot(normalize(v_world_normal), light_dir), 0.0);
    float ambient = 0.25;

    frag_color = vec4(color * (ambient + (1.0 - ambient) * diffuse), 1.0);
}
