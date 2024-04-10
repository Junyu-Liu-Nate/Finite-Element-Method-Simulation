#version 330 core

// Add a second layout variable representing a UV coordinate
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

// Create an "out" variable representing a UV coordinate
out vec2 uvCoord;

void main() {
    // Assign the UV layout variable to the UV "out" variable
    uvCoord = uv;

    gl_Position = vec4(position, 1.0);
}
