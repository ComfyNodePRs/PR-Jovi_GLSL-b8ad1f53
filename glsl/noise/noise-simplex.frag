// name: NOISE SIMPLEX
// desc: Simplex noise, simply
// category: NOISE
// control: wh,

#include .lib/noise/noise_param.lib
#include .lib/noise/noise_simplex.lib

uniform float frequency;   // 1.;  1.; 100.; 0.01  | Base frequency multiplier
uniform float amplitude;   // 1.;  1.; 100.; 0.01  | Base amplitude multiplier
uniform int octaves;       // 4;   1;   12;  1     | Number of octaves
uniform float lacunarity;  // 2.;  0.; 100.; 0.01  | Frequency multiplier per octave
uniform float persistence; // 0.5; 0.; 100.; 0.01  | Amplitude multiplier per octave (same as 'gain' in some functions)
uniform float offset;      // 0.;  0.; 100.; 0.01  | For ridge noise

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = fragCoord / iResolution.xy;
    NoiseParams nparam = defaultNoiseParams();
    nparam.frequency = frequency;
    nparam.amplitude = amplitude;
    nparam.octaves = octaves;
    nparam.lacunarity = lacunarity;
    nparam.persistence = persistence;
    nparam.offset = offset;
    vec3 shit3 = vec3(uv.x, uv.y, uv.x * uv.y);
    vec4 shit4 = vec4(uv.x, uv.y, uv.x * uv.y, 5.01);
    float simplex = noise_simplex(shit4, nparam);
    fragColor = vec4(simplex, simplex, simplex, 1.);
}