// name: SOBEL
// desc: Finds the edges of an image
// category: FILTER

uniform sampler2D image; // | RGB(A) image

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    float x = 1.0 / iResolution.x;
	float y = 1.0 / iResolution.y;
	vec2 uv = fragCoord / iResolution.xy;

    vec4 horizEdge = vec4( 0.0 );
	horizEdge -= texture( image, vec2( uv.x - x, uv.y - y ) ) * 1.0;
	horizEdge -= texture( image, vec2( uv.x - x, uv.y     ) ) * 2.0;
	horizEdge -= texture( image, vec2( uv.x - x, uv.y + y ) ) * 1.0;
	horizEdge += texture( image, vec2( uv.x + x, uv.y - y ) ) * 1.0;
	horizEdge += texture( image, vec2( uv.x + x, uv.y     ) ) * 2.0;
	horizEdge += texture( image, vec2( uv.x + x, uv.y + y ) ) * 1.0;

    vec4 vertEdge = vec4( 0.0 );
	vertEdge -= texture( image, vec2( uv.x - x, uv.y - y ) ) * 1.0;
	vertEdge -= texture( image, vec2( uv.x    , uv.y - y ) ) * 2.0;
	vertEdge -= texture( image, vec2( uv.x + x, uv.y - y ) ) * 1.0;
	vertEdge += texture( image, vec2( uv.x - x, uv.y + y ) ) * 1.0;
	vertEdge += texture( image, vec2( uv.x    , uv.y + y ) ) * 2.0;
	vertEdge += texture( image, vec2( uv.x + x, uv.y + y ) ) * 1.0;
	vec3 edge = sqrt((horizEdge.rgb * horizEdge.rgb) + (vertEdge.rgb * vertEdge.rgb));

	fragColor = vec4( edge, texture( image, uv ).a );
}