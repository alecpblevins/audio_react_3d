#version 330

in float v_bin_value;

out vec4 frag_color;

void main() {
    // Render each point as a soft circular sprite instead of a square.
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist_sq = dot(coord, coord);
    if (dist_sq > 0.25) {
        discard;
    }

    // Cool-to-hot colormap driven by this particle's spectrum bin energy.
    vec3 cold = vec3(0.10, 0.25, 0.95);
    vec3 hot = vec3(1.0, 0.65, 0.15);
    vec3 color = mix(cold, hot, clamp(v_bin_value, 0.0, 1.0));

    float alpha = smoothstep(0.25, 0.0, dist_sq);
    frag_color = vec4(color, alpha);
}
