// name: PIXELATE
// desc: Pixelate input image
// category: COLOR

uniform sampler2D image; //              | RGB(A) image
uniform float amount;    // 32;0;255;0.1 | Pixel data range allowed

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    float d = 1.0 / amount;
    float ar = iResolution.x / iResolution.y;
	float u = floor( uv.x / d ) * d;
	float v = floor( uv.y / d ) * (ar / amount);
    fragColor = texture( image, vec2( u, v ) );
}
