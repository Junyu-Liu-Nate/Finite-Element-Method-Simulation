#version 330 core

// Create a UV coordinate in variable
in vec2 uvCoord;

// Add a sampler2D uniform
uniform sampler2D textureImg;

uniform bool isFBO;
uniform bool isFXAA;

uniform float screenWidth;
uniform float screenHeight;

out vec4 fragColor;

float luminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

void main()
{
    if (isFXAA) {
        vec2 inverseScreenSize = vec2(1.0 / screenWidth, 1.0 / screenHeight);

        vec4 color = texture(textureImg, uvCoord);
        float lumaCenter = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        // Sample neighboring pixels and compute their luminance
        vec2 offsets[8] = vec2[](
            vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(1.0, -1.0),
            vec2(-1.0,  0.0),                  vec2(1.0,  0.0),
            vec2(-1.0,  1.0), vec2(0.0,  1.0), vec2(1.0,  1.0)
        );

        float lumaMin = lumaCenter;
        float lumaMax = lumaCenter;
        for (int i = 0; i < 8; ++i) {
            float lumaNeighbor = dot(texture(textureImg, uvCoord + offsets[i] * inverseScreenSize).rgb, vec3(0.299, 0.587, 0.114));
            lumaMin = min(lumaMin, lumaNeighbor);
            lumaMax = max(lumaMax, lumaNeighbor);
        }

        // Determine edge strength
        float lumaRange = lumaMax - lumaMin;
        float edgeStrength = max(lumaRange / lumaMax, 0.0);

        // Sub-pixel anti-aliasing: Adjust edge strength based on local contrast
        edgeStrength *= (1.0 + 8.0 * dot(color.rgb - vec3(lumaCenter), color.rgb - vec3(lumaCenter)));

        // Edge detection thresholds
        float edgeThresholdMin = 0.0312;
        float edgeThresholdMax = 0.125;

        if (edgeStrength > max(edgeThresholdMin, lumaMax * edgeThresholdMax)) {
            // Calculate gradient and blend factor
            vec2 gradient = vec2(
                lumaCenter - dot(texture(textureImg, uvCoord + vec2(-1.0, 0.0) * inverseScreenSize).rgb, vec3(0.299, 0.587, 0.114)),
                lumaCenter - dot(texture(textureImg, uvCoord + vec2(0.0, -1.0) * inverseScreenSize).rgb, vec3(0.299, 0.587, 0.114))
            );

            float gradientLength = length(gradient);
            vec2 gradientDirection = normalize(gradient) * inverseScreenSize;

            // Sample along the gradient to find the end of the edge
            float endLuma = dot(texture(textureImg, uvCoord + gradientDirection).rgb, vec3(0.299, 0.587, 0.114));

            // Blend factor calculation
            float blendFactor = max(0.0, (endLuma - lumaMin) / (lumaMax - lumaMin));
            blendFactor = min(max(edgeStrength, 0.1), blendFactor);

            // Blend the color
            vec4 blendedColor = mix(color, texture(textureImg, uvCoord + gradientDirection), blendFactor);
            fragColor = blendedColor;
        } else {
            fragColor = color;
        }
    }
    else {
        fragColor = texture(textureImg, uvCoord);
    }

    if (isFBO) {
        // Calculate the current pixel's luminance
        float luma = luminance(fragColor.rgb);

        // Check if the pixel is black or near black
        if (luma > 0.999) { // Threshold can be adjusted as needed
            // Apply the gradient color to the white pixel
            fragColor = vec4(uvCoord, 0.5, 1.0);
        }
        else {
            fragColor = texture(textureImg, uvCoord);
        }
    }
    else {
        fragColor = texture(textureImg, uvCoord);
    }
}
