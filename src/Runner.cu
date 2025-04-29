#include <glew.h>
#include <freeglut.h>

// includes, cudaimageWidth
#include <cudaDefs.h>
#include <cuda_gl_interop.h>
#include <helper_math.h>			// normalize method

#include <curand_kernel.h>


#include <imageManager.h>
#include <imageUtils.cuh>
#include <benchmark.h>

#include "Map.h"
#include "FlowField.h"

#define TPB_1D 32                                  
#define TPB_2D TPB_1D*TPB_1D      
cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();
using DT = uchar4;

int GRID_SIZE = 64;
int NUMBER_OF_PARTICLES = 1024;

constexpr uchar3 backgroundCollor = { 0 , 0 ,0 };
const uchar3 wallCollor = { 0 , 255 , 0 };

const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPosition;
uniform float uPointSize;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    gl_PointSize = uPointSize;
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;
    FragColor = uColor;
}
)";

GLuint shaderProgram;
GLuint particleVBO;
cudaGraphicsResource* cudaVBOResource;


// Struktura pro uchování údajů o OpenGL datech
struct GLData
{
    unsigned int imageWidth;         // Šířka textury
    unsigned int imageHeight;        // Výška textury
    unsigned int imageBPP;           // Počet bitů na pixel (8, 16, 24, nebo 32 bitů)
    unsigned int imagePitch;         // Výška řádku v bitech (počet bajtů na jeden řádek)

    unsigned int pboID;              // ID Pixel Buffer Object (PBO) pro práci s daty textury
    unsigned int textureID;          // ID textury pro OpenGL
    unsigned int viewportWidth = 1024; // Šířka viewportu
    unsigned int viewportHeight = 1024; // Výška viewportu
};

GLData FloatFieldTexture;
GLData RenderTexture;

// Struktura pro uchování údajů pro CUDA texturu a PBO (Pixel Buffer Object)
struct CudaData
{
    cudaTextureDesc texDesc;            // Popis textury pro CUDA, obsahuje parametry textury
    cudaArray_t texArrayData;           // Data textury ve formátu CUDA
    cudaResourceDesc resDesc;           // Popis prostředku pro získání dat z textury
    cudaChannelFormatDesc texChannelDesc; // Popis kanálů textury (např. jaké jsou velikosti jednotlivých kanálů)
    cudaTextureObject_t texObj;         // CUDA texturový objekt, který bude vytvořen
    cudaGraphicsResource_t texResource; // CUDA grafický prostředek pro texturu
    cudaGraphicsResource_t pboResource; // CUDA grafický prostředek pro PBO (pro zápis)

    CudaData()
    {
        memset(this, 0, sizeof(CudaData)); // Inicializace všech členů struktury na nulu
    }
};
CudaData FloatFieldCudaData;
CudaData RenderTextureCudaData;


void saveOpenGLTexture(GLuint textureID, int width, int height, const char* filename) {
    glBindTexture(GL_TEXTURE_2D, textureID);
    unsigned char* pixels = new unsigned char[width * height * 4]; // RGBA
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    // Flip vertically (OpenGL and FreeImage have opposite Y-axis)
    FIBITMAP* image = FreeImage_ConvertFromRawBits(
        pixels, width, height, width * 4, 32,
        0xFF000000, 0x00FF0000, 0x0000FF00, true
    );

    if (FreeImage_Save(FIF_PNG, image, filename)) {
        std::cout << "Texture saved to: " << filename << std::endl;
    }
    else {
        std::cerr << "Failed to save texture." << std::endl;
    }
    FreeImage_Unload(image);
    delete[] pixels;
}

void display()
{
    // OpenGL Rendering
    glClear(GL_COLOR_BUFFER_BIT);

    // Define the color as a float array (RGBA)
    float color[4] = { 1.0f, 0.0f, 0.0f, 1.0f }; // Red color (RGBA)

    // Get the location of the uniform 'uColor' in the shader
    GLuint colorLoc = glGetUniformLocation(shaderProgram, "uColor");
    GLuint uPointSizeLocation = glGetUniformLocation(shaderProgram, "uPointSize");

    //glUseProgram(shaderProgram);

    // Set the uniform color value using the float array
    glUniform4fv(colorLoc, 1, color);
    glUniform1f(uPointSizeLocation, NUMBER_OF_PARTICLES);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, RenderTexture.textureID);

    // Draw the quad
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0);          glVertex2d(0, 0);
    glTexCoord2d(1, 0);          glVertex2d(RenderTexture.viewportWidth, 0);
    glTexCoord2d(1, 1);          glVertex2d(RenderTexture.viewportWidth, RenderTexture.viewportHeight);
    glTexCoord2d(0, 1);          glVertex2d(0, RenderTexture.viewportHeight);
    glEnd();

    glDisable(GL_TEXTURE_2D);  // Disable texturing after drawing the quad

    // Swap buffers to display the updated texture
    glutSwapBuffers();
}


void my_resize(GLsizei w, GLsizei h)
{
    FloatFieldTexture.viewportWidth = w;
    FloatFieldTexture.viewportHeight = h;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, FloatFieldTexture.viewportWidth, FloatFieldTexture.viewportHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, FloatFieldTexture.viewportWidth, 0, FloatFieldTexture.viewportHeight);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

__global__ void ConstructMap(cudaTextureObject_t texObj, int width, int height, uint8_t* pboData) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < width && y < height) {


        const uchar4 texel = tex2D<uchar4>(texObj, x, y);

        if (texel.z == 1)
        {
            int idx = (y * width + x) * 4;
            pboData[idx] = 0;
            pboData[idx + 1] = 0;
            pboData[idx + 2] = 0; // Red channel (125)
            pboData[idx + 3] = 255; // Alpha channel (255)
        }
        else
        {
            int idx = (y * width + x) * 4;
            pboData[idx] = 255;
            pboData[idx + 1] = 255;
            pboData[idx + 2] = 255; // Red channel (125)
            pboData[idx + 3] = 255; // Alpha channel (255)
        }

        return;
    }
}

void cudaWorker()
{
    // Step 1: Map the CUDA resources (input texture and output PBO)
    cudaGraphicsResource_t resources[2] = {
        FloatFieldCudaData.texResource,   // Input texture resource
        RenderTextureCudaData.pboResource // Output PBO resource
    };

    checkCudaErrors(cudaGraphicsMapResources(2, resources, 0));

    // Step 2: Get mapped CUDA array and pointer to PBO memory
    uint8_t* pboData = nullptr;
    size_t pboSize = 0;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void**)&pboData, &pboSize, RenderTextureCudaData.pboResource));

    if (pboData == nullptr) {
        std::cerr << "Error: PBO pointer is null!" << std::endl;
        return;
    }

    // Step 3: Launch CUDA kernel to modify data in the PBO
    dim3 dimBlock(TPB_1D, TPB_1D, 1);
    dim3 dimGrid((FloatFieldTexture.imageWidth + TPB_1D - 1) / TPB_1D,
        (FloatFieldTexture.imageHeight + TPB_1D - 1) / TPB_1D, 1);

    
 ConstructMap << <dimGrid, dimBlock >> > (
     FloatFieldCudaData.texObj,
     FloatFieldTexture.imageWidth,
     FloatFieldTexture.imageHeight,
     pboData
     );
 

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 4: Unmap CUDA resources
    checkCudaErrors(cudaGraphicsUnmapResources(2, resources, 0));

    // Step 5: Transfer data from PBO to OpenGL texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, RenderTexture.pboID);
    glBindTexture(GL_TEXTURE_2D, RenderTexture.textureID);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
        RenderTexture.imageWidth, RenderTexture.imageHeight,
        GL_RGBA, GL_UNSIGNED_BYTE, (void*)0);  // Offset into PBO

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error during texture update: " << error << std::endl;
    }

    // Step 6: Unbind PBO and texture from OpenGL
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void my_idle()
{
    cudaWorker();

    //saveOpenGLTexture(FloatFieldTexture.textureID, FloatFieldTexture.imageWidth, FloatFieldTexture.imageHeight, "flowField.png");
    //saveOpenGLTexture(RenderTexture.textureID, RenderTexture.imageWidth, RenderTexture.imageHeight, "render.png");

    glutPostRedisplay();
}

GLuint createShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    // Optional: error checking
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compile Error:\n" << infoLog << std::endl;
    }

    return shader;
}

GLuint createShaderProgram() {
    GLuint vs = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    // Optional: program link error checking
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader Link Error:\n" << infoLog << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

void initGL(int argc, char** argv)
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(FloatFieldTexture.viewportWidth, FloatFieldTexture.viewportHeight);
    glutInitWindowPosition(0, 0);
    glutSetOption(GLUT_RENDERING_CONTEXT, false ? GLUT_USE_CURRENT_CONTEXT : GLUT_CREATE_NEW_CONTEXT);
    glutCreateWindow(0);

    char m_windowsTitle[512];
    sprintf_s(m_windowsTitle, 512, "SimpleView | context %s | renderer %s | vendor %s ",
        (const char*)glGetString(GL_VERSION),
        (const char*)glGetString(GL_RENDERER),
        (const char*)glGetString(GL_VENDOR));
    glutSetWindowTitle(m_windowsTitle);

    glutDisplayFunc(display);
    glutReshapeFunc(my_resize);
    glutIdleFunc(my_idle);
    glutSetCursor(GLUT_CURSOR_CROSSHAIR);

    // initialize necessary OpenGL extensions
    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glShadeModel(GL_SMOOTH);
    glViewport(0, 0, FloatFieldTexture.viewportWidth, FloatFieldTexture.viewportHeight);

    glFlush();
}

void generateFlowFieldTexture(FlowField* flowfield, Map* map)
{
    FloatFieldTexture.imageWidth = flowfield->getSize();
    FloatFieldTexture.imageHeight = flowfield->getSize();

    // Alokování pole pro data textury
    unsigned char* textureData = new unsigned char[FloatFieldTexture.imageWidth * FloatFieldTexture.imageHeight * 4];  // 4 pro RGBA
    for (int y = 0; y < FloatFieldTexture.imageHeight; ++y)
    {
        for (int x = 0; x < FloatFieldTexture.imageWidth; ++x)
        {
            std::array<uint8_t, 2> datapoint = flowfield->getData(x, y);
            int wall = (int)map->getCell(x, y);
            int idx = (y * FloatFieldTexture.imageWidth + x) * 4;  // 4 pro RGBA
            textureData[idx] = datapoint.at(0);
            textureData[idx + 1] = datapoint.at(1);
            textureData[idx + 2] = wall;
            textureData[idx + 3] = 255; // Alpha channel (plně průhledný)
        }
    }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &FloatFieldTexture.textureID);
    glBindTexture(GL_TEXTURE_2D, FloatFieldTexture.textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FloatFieldTexture.imageWidth, FloatFieldTexture.imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);

    // Nastavení parametrů textury
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // Uvolnění alokované paměti pro texturu
    delete[] textureData;

    // Vytvoření PBO pro OpenGL
    glGenBuffers(1, &FloatFieldTexture.pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, FloatFieldTexture.pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, FloatFieldTexture.imageWidth * FloatFieldTexture.imageHeight * 4, NULL, GL_DYNAMIC_COPY);    // Alokace dat pro PBO
}

void generateRenderTexture() {
    // Initialize the render texture size (same as the flowfield texture in your case)
    RenderTexture.imageWidth = FloatFieldTexture.imageWidth;
    RenderTexture.imageHeight = FloatFieldTexture.imageHeight;

    // Create the OpenGL texture for the render texture
    glGenTextures(1, &RenderTexture.textureID);
    glBindTexture(GL_TEXTURE_2D, RenderTexture.textureID);

    // Create the texture with no initial data, OpenGL will allocate it
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RenderTexture.imageWidth, RenderTexture.imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Set texture parameters for proper behavior
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // Linear filtering for minification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  // Linear filtering for magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // Clamp texture horizontally
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Clamp texture vertically

    // Create a PBO (Pixel Buffer Object) for OpenGL to use with the texture
    glGenBuffers(1, &RenderTexture.pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, RenderTexture.pboID);

    // Allocate memory for the PBO, where the texture data will be copied into
    glBufferData(GL_PIXEL_UNPACK_BUFFER, RenderTexture.imageWidth * RenderTexture.imageHeight * 4, NULL, GL_DYNAMIC_COPY); // NULL for unused initial data

    // Unbind the PBO to ensure OpenGL operations can proceed safely
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // After this, the texture and PBO are ready for CUDA integration or further OpenGL operations
}

void initCUDAObjects()
{
    checkCudaErrors(cudaGraphicsGLRegisterImage(&FloatFieldCudaData.texResource, FloatFieldTexture.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));  // Registrace textury pro CUDA (pro čtení)
    checkCudaErrors(cudaGraphicsMapResources(1, &FloatFieldCudaData.texResource, 0));     // Mapování textury pro CUDA
    cudaGraphicsSubResourceGetMappedArray(&FloatFieldCudaData.texArrayData, FloatFieldCudaData.texResource, 0, 0);  // Získání datového pole z mapované textury

    // Nastavení popisu pro texturu
    FloatFieldCudaData.resDesc.resType = cudaResourceTypeArray;
    FloatFieldCudaData.resDesc.res.array.array = FloatFieldCudaData.texArrayData;
    FloatFieldCudaData.texDesc.readMode = cudaReadModeElementType;
    FloatFieldCudaData.texDesc.normalizedCoords = false;
    FloatFieldCudaData.texDesc.filterMode = cudaFilterModePoint;
    FloatFieldCudaData.texDesc.addressMode[0] = cudaAddressModeClamp;
    FloatFieldCudaData.texDesc.addressMode[1] = cudaAddressModeClamp;

    // Získání popisu kanálu textury
    checkCudaErrors(cudaGetChannelDesc(&FloatFieldCudaData.texChannelDesc, FloatFieldCudaData.texArrayData));
    // Vytvoření objektu textury v CUDA
    checkCudaErrors(cudaCreateTextureObject(&FloatFieldCudaData.texObj, &FloatFieldCudaData.resDesc, &FloatFieldCudaData.texDesc, NULL));
    // Unmapping textury po registraci
    checkCudaErrors(cudaGraphicsUnmapResources(1, &FloatFieldCudaData.texResource, 0));
    // Registrace PBO pro zápis do CUDA
    cudaGraphicsGLRegisterBuffer(&FloatFieldCudaData.pboResource, FloatFieldTexture.pboID, cudaGraphicsRegisterFlagsWriteDiscard);


    //-----------------------


    // Register the RenderTexture texture for CUDA (read-write)
    checkCudaErrors(cudaGraphicsGLRegisterImage(&RenderTextureCudaData.texResource, RenderTexture.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(1, &RenderTextureCudaData.texResource, 0));  // Map the RenderTexture for CUDA
    cudaGraphicsSubResourceGetMappedArray(&RenderTextureCudaData.texArrayData, RenderTextureCudaData.texResource, 0, 0);  // Get the texture data array

    // Set up the texture description for the RenderTexture
    RenderTextureCudaData.resDesc.resType = cudaResourceTypeArray;
    RenderTextureCudaData.resDesc.res.array.array = RenderTextureCudaData.texArrayData;
    RenderTextureCudaData.texDesc.readMode = cudaReadModeElementType;
    RenderTextureCudaData.texDesc.normalizedCoords = false;
    RenderTextureCudaData.texDesc.filterMode = cudaFilterModePoint;
    RenderTextureCudaData.texDesc.addressMode[0] = cudaAddressModeClamp;
    RenderTextureCudaData.texDesc.addressMode[1] = cudaAddressModeClamp;

    // Get the channel description for the RenderTexture
    checkCudaErrors(cudaGetChannelDesc(&RenderTextureCudaData.texChannelDesc, RenderTextureCudaData.texArrayData));

    // Create the texture object in CUDA for RenderTexture
    checkCudaErrors(cudaCreateTextureObject(&RenderTextureCudaData.texObj, &RenderTextureCudaData.resDesc, &RenderTextureCudaData.texDesc, NULL));

    // Unmap the RenderTexture after registration
    checkCudaErrors(cudaGraphicsUnmapResources(1, &RenderTextureCudaData.texResource, 0));

    // Register the PBO for writing to RenderTexture in CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&RenderTextureCudaData.pboResource, RenderTexture.pboID, cudaGraphicsRegisterFlagsWriteDiscard));
}

void createSharedVBO(int numberOfParticles) {
    glGenBuffers(1, &particleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * numberOfParticles, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, particleVBO, cudaGraphicsMapFlagsWriteDiscard);
}

__global__ void randomizeParticles(float2* particles, int count, float minX, float maxX, float minY, float maxY, unsigned int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    float x = minX + (maxX - minX) * curand_uniform(&state);
    float y = minY + (maxY - minY) * curand_uniform(&state);

    particles[i] = make_float2(x, y);

}
void fillParticlesWithCUDA(int numberOfParticles, float minX, float maxX, float minY, float maxY, unsigned int seed)
{
    float2* dptr;
    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    size_t bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &bytes, cudaVBOResource);
    int threadsPerBlock = 256;
    int blocks = (numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;
    randomizeParticles << <blocks, threadsPerBlock >> > (dptr, numberOfParticles, minX, maxX, minY, maxY, seed);
    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
}

void allocateParticles(float2** Dparticles, int numberOfParticles)
{
    cudaMalloc((void**)Dparticles, numberOfParticles * sizeof(float2));
}

void releaseOpenGL()
{
    if (FloatFieldTexture.textureID > 0)
        glDeleteTextures(1, &FloatFieldTexture.textureID);
    if (FloatFieldTexture.pboID > 0)
        glDeleteBuffers(1, &FloatFieldTexture.pboID);
}

void releaseCUDA()
{
    checkCudaErrors(cudaGraphicsUnregisterResource(FloatFieldCudaData.pboResource));
    checkCudaErrors(cudaGraphicsUnregisterResource(FloatFieldCudaData.texResource));
}

void releaseResources()
{
    releaseCUDA();
    releaseOpenGL();
}

int main(int argc, char* argv[])
{
    initializeCUDA(deviceProp);
    FreeImage_Initialise();
    initGL(argc, argv);

    shaderProgram = createShaderProgram();

    Map* map = new Map(32);
    map->setWall(10, 10);
    map->setWall(9, 9);
    map->setWall(8, 8);
    map->setWall(7, 7);
    map->setWall(6, 6);
    map->setWall(5, 5);
    map->setWall(4, 4);
    map->setWall(3, 3);
    map->setStart(2, 2);
    map->setWall(1, 1);
    map->setStart(0, 0);
    map->setGoal(24, 24);

    FlowField* flowfield = new FlowField(32);
    flowfield->generateFlowFieldForMap(map);
    flowfield->printFlowField();

    generateFlowFieldTexture(flowfield, map);
    generateRenderTexture();

    //TEST
    saveOpenGLTexture(FloatFieldTexture.textureID, FloatFieldTexture.imageWidth, FloatFieldTexture.imageHeight, "flowField.png");
    saveOpenGLTexture(RenderTexture.textureID, RenderTexture.imageWidth, RenderTexture.imageHeight, "render.png");

    initCUDAObjects();

    const int PARTICLE_ALPHA = 255;
    const int PARTICLE_WEIGHT = 20;
    const int PARTICLE_RADIUS = 5;

    bool running = true;
    int threadsPerBlock = 256;
    int blocks = (NUMBER_OF_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int seed = 10;

    float2* Dparticles = nullptr;

    createSharedVBO(NUMBER_OF_PARTICLES);
    fillParticlesWithCUDA(NUMBER_OF_PARTICLES, 0, 10, 0, 10, 10);

    allocateParticles(&Dparticles, NUMBER_OF_PARTICLES);
    randomizeParticles <<<blocks,threadsPerBlock>>> (Dparticles, NUMBER_OF_PARTICLES,-10.0f, 10.0f,-5.0f, 5.0f,seed);
    cudaDeviceSynchronize();

    glutMainLoop();

    FreeImage_DeInitialise();
    atexit(releaseResources);

    return 0;
}