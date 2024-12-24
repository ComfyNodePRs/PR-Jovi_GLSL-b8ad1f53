// name: MIN-MAX
// desc: Gets the minimum and maximum of an image
// category: FILTER

uniform sampler2D image; // | RGB(A) image

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 color = texture(image, uv);
    fragColor = vec4(
        min(min(color.r, color.g), color.b),
        max(max(color.r, color.g), color.b),
        0.0,
        1.0
    );
}