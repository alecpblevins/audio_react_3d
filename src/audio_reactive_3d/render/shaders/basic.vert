#version 330

// Live audio feature vector, uploaded once per frame as a uniform buffer.
//
// std140 packs an array of vec4 with a 16-byte (4-float) stride and no
// inter-element padding, so this block maps exactly onto the flat
// float32[128] vector produced by audio_reactive_3d.audio.features -- no
// manual padding/repacking is required on the Python side.
//
// Layout reminder (see features.py for the authoritative docs):
//   data[0..15]   = elements 0..63   = 64-bin log-spaced spectrum
//   data[16]      = elements 64..67  = (low_band, mid_band, high_band, rms)
//   data[17].x    = element 68       = onset flag (0/1)
//   data[17].y    = element 69       = smoothed BPM
layout(std140) uniform Features {
    vec4 data[32];
};

uniform mat4 u_model;
uniform mat4 u_mvp;
uniform float u_displacement_scale;
uniform vec3 u_camera_pos;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_world_normal;
out vec3 v_view_dir;

void main() {
    float low_band = data[16].x;

    // Displace each vertex outward along its normal by the smoothed bass
    // energy -- this is what makes the sphere visibly "pulse" with the beat.
    vec3 displaced = in_position + in_normal * (low_band * u_displacement_scale);
    vec3 world_pos = vec3(u_model * vec4(displaced, 1.0));

    gl_Position = u_mvp * vec4(displaced, 1.0);
    v_world_normal = normalize(mat3(u_model) * in_normal);
    v_view_dir = u_camera_pos - world_pos;
}
