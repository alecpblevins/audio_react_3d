#version 330

// Live feature vector sampled as a (width=128, height=1) float texture
// rather than the UBO basic.vert/basic.frag use. Each particle needs to
// dynamically pick its own spectrum bin at runtime (texelFetch supports an
// arbitrary per-invocation integer index); this is the "and/or 1D texture"
// upload path the project spec calls out, implemented as a 1-row 2D
// texture since moderngl/GL has no dedicated 1D texture target.
uniform sampler2D u_features_tex;

uniform mat4 u_mvp;
uniform float u_displacement_scale;
uniform float u_point_base_size;
uniform float u_point_size_scale;
uniform float u_point_rms_scale;

//: Flat index of the RMS element in the feature vector (see features.py).
const int RMS_TEXEL_INDEX = 67;

in vec3 in_direction;   // unit vector: this particle's "home" position
in float in_bin_index;  // which spectrum bin (0..63) this particle visualizes

out float v_bin_value;

void main() {
    float bin_value = texelFetch(u_features_tex, ivec2(int(in_bin_index), 0), 0).r;
    float rms = texelFetch(u_features_tex, ivec2(RMS_TEXEL_INDEX, 0), 0).r;

    vec3 pos = in_direction * (1.0 + bin_value * u_displacement_scale);
    gl_Position = u_mvp * vec4(pos, 1.0);
    gl_PointSize = u_point_base_size + bin_value * u_point_size_scale + rms * u_point_rms_scale;

    v_bin_value = bin_value;
}
