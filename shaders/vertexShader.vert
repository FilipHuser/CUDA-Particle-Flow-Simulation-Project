#version 330 core

layout(location = 0) in vec2 inPosition; // Particle position from VBO

void main()
{
    // Pass the position to the clip space (no transformation for now)
    gl_Position = vec4(inPosition, 0.0, 1.0); // 2D position with homogeneous coordinates
    gl_PointSize = 0.1; // Set the size of the points (particles)
}
