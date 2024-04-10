#include "graphics/shader.h"
#include "graphics/graphicsdebug.h"

#include <QFile>
#include <QString>
#include <QTextStream>
#include <iostream>

Shader::Shader(const std::string &vertexPath,
               const std::string &fragmentPath)
{
    createProgramID();

    std::vector<GLuint> shaders = {
        createShaderFromString(getFileContents(vertexPath),   GL_VERTEX_SHADER),
        createShaderFromString(getFileContents(fragmentPath), GL_FRAGMENT_SHADER)
    };

    buildShaderProgramFromShaders(shaders);
    discoverShaderData();
}

Shader::~Shader()
{
    glDeleteProgram(m_programID);
}

Shader::Shader(Shader &&that)
    : m_programID(that.m_programID),
      m_attributes(std::move(that.m_attributes)),
      m_uniforms(std::move(that.m_uniforms))
{
    that.m_programID = 0;
}

Shader& Shader::operator=(Shader &&that)
{
    this->~Shader();

    m_programID = that.m_programID;
    m_attributes = std::move(that.m_attributes);
    m_uniforms = std::move(that.m_uniforms);

    that.m_programID = 0;

    return *this;
}

// ================== Regular Use

void Shader::bind()   const { glUseProgram(m_programID); }

void Shader::unbind() const { glUseProgram(0); }

GLuint Shader::getUniformLocation(std::string name)
{
    return glGetUniformLocation(m_programID, name.c_str());
}

GLuint Shader::getEnumeratedUniformLocation(std::string name, int index)
{
    std::string n = name + "[" + std::to_string(index) + "]";
    return glGetUniformLocation(m_programID, n.c_str());
}

// ================== Setting Uniforms

// Note: the overload to set matrix uniforms is in the .h file

void Shader::setUniform(const std::string &name, float f)
{
    glUniform1f(m_uniforms[name], f);
}

void Shader::setUniform(const std::string &name, int i)
{
    glUniform1i(m_uniforms[name], i);
}

void Shader::setUniform(const std::string &name, bool b)
{
    glUniform1i(m_uniforms[name], static_cast<GLint>(b));
}

// ================== Creating the Program

void Shader::createProgramID()
{
    m_programID = glCreateProgram();
}

void Shader::buildShaderProgramFromShaders(const std::vector<GLuint> &shaderHandles)
{
    // Attach shaders
    for (const GLuint &shaderHandle : shaderHandles) {
        glAttachShader(m_programID, shaderHandle);
    }

    // Link program
    glLinkProgram(m_programID);
    checkShaderLinkStatus(m_programID);

    // Detach and delete shaders
    for (const GLuint &shaderHandle : shaderHandles) {
        glDetachShader(m_programID, shaderHandle);
        glDeleteShader(shaderHandle);
    }
}

// ================== Creating Shaders From Filepaths

std::string Shader::getFileContents(std::string filepath)
{
    QString filepathStr = QString::fromStdString(filepath);
    QFile file(filepathStr);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(std::string("Failed to open shader: ") + filepath);
    }

    QTextStream stream(&file);
    QString contents = stream.readAll();
    file.close();

    return contents.toStdString();
}

GLuint Shader::createShaderFromString(const std::string &str, GLenum shaderType)
{
    GLuint shaderHandle = glCreateShader(shaderType);

    // Compile shader code
    const char *codePtr = str.c_str();
    glShaderSource(shaderHandle, 1, &codePtr, nullptr); // Assumes code is null terminated
    glCompileShader(shaderHandle);

    checkShaderCompilationStatus(shaderHandle);

    return shaderHandle;
}

// ================== Discovering Attributes/Uniforms/Textures

void Shader::discoverShaderData() {
    discoverAttributes();
    discoverUniforms();
}

void Shader::discoverAttributes() {
    bind();
    GLint attribCount;
    glGetProgramiv(m_programID, GL_ACTIVE_ATTRIBUTES, &attribCount);
    for (int i = 0; i < attribCount; i++) {
        const GLsizei bufSize = 256;
        GLsizei nameLength = 0;
        GLint arraySize = 0;
        GLenum type;
        GLchar name[bufSize];
        glGetActiveAttrib(m_programID, i, bufSize, &nameLength, &arraySize, &type, name);
        name[std::min(nameLength, bufSize - 1)] = 0;
        m_attributes[std::string(name)] = glGetAttribLocation(m_programID, name);
    }
    unbind();
}

void Shader::discoverUniforms() {
    bind();
    GLint uniformCount;
    glGetProgramiv(m_programID, GL_ACTIVE_UNIFORMS, &uniformCount);
    for (int i = 0; i < uniformCount; i++) {
        const GLsizei bufSize = 256;
        GLsizei nameLength = 0;
        GLint arraySize = 0;
        GLenum type;
        GLchar name[bufSize];
        glGetActiveUniform(m_programID, i, bufSize, &nameLength, &arraySize, &type, name);
        name[std::min(nameLength, bufSize - 1)] = 0;

        std::string strname(name);
        if (isUniformArray(name, nameLength)) {
            addUniformArray(strname, arraySize);
        } else if (isTexture(type)) {
            addTexture(strname);
        } else {
            addUniform(strname);
        }
    }
    unbind();
}

bool Shader::isUniformArray(const GLchar *name, GLsizei nameLength) {
    // Check if the last 3 characters are '[0]'
    return (name[nameLength - 3] == '[') &&
            (name[nameLength - 2] == '0') &&
            (name[nameLength - 1] == ']');
}

bool Shader::isTexture(GLenum type) {
    return (type == GL_SAMPLER_2D) ||
            (type == GL_SAMPLER_CUBE);
}

void Shader::addUniformArray(const std::string &name, size_t size) {
    std::string cleanName = name.substr(0, name.length() - 3);
    for (auto i = static_cast<size_t>(0); i < size; i++) {
        std::string enumeratedName = name;
        enumeratedName[enumeratedName.length() - 2] = static_cast<char>('0' + i);
        std::tuple< std::string, size_t > nameIndexTuple = std::make_tuple(cleanName, i);
        m_uniformArrays[nameIndexTuple] = glGetUniformLocation(m_programID, enumeratedName.c_str());
    }

#if GRAPHICS_DEBUG_LEVEL > 0
    m_trackedUniformArrays[name] = false;
#endif
}

void Shader::addTexture(const std::string &name) {
    GLint location = glGetUniformLocation(m_programID, name.c_str());
    m_textureLocations[name] = location;
    GLint slot = m_textureSlots.size();
    m_textureSlots[location] = slot; // Assign slots in increasing order.

#if GRAPHICS_DEBUG_LEVEL > 0
    m_trackedTextures[name] = false;
#endif
}

void Shader::addUniform(const std::string &name) {
    m_uniforms[name] = glGetUniformLocation(m_programID, name.c_str());

#if GRAPHICS_DEBUG_LEVEL > 0
    m_trackedUniforms[name] = false;
#endif
}
