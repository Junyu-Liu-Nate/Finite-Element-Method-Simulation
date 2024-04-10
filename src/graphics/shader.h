#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <GL/glew.h>
#include "Eigen/Dense"
#include "util/unsupportedeigenthing/OpenGLSupport"


class Shader {
public:
    Shader(const std::string &vertexPath,
           const std::string &fragmentPath);

    virtual ~Shader();

    Shader(Shader &that)            = delete;
    Shader& operator=(Shader &that) = delete;
    Shader(Shader &&that);
    Shader& operator=(Shader &&that);

    // Basic Usage
    GLuint getProgramID() const { return m_programID; }
    void   bind()   const;
    void   unbind() const;
    GLuint getUniformLocation(std::string name);
    GLuint getEnumeratedUniformLocation(std::string name, int index);

    // Setting Uniforms
    void setUniform(const std::string &name, float f);
    void setUniform(const std::string &name, int   i);
    void setUniform(const std::string &name, bool  b);
    template<typename type, int n, int m>
    void setUniform(const std::string &name, const Eigen::Matrix<type, n, m> &mat)
    {
        glUniform(m_uniforms[name], mat);
    }


private:
    // Creating the Program
    void createProgramID();
    void buildShaderProgramFromShaders(const std::vector<GLuint> &shaders);

    // Creating Shaders From Filepaths
    std::string getFileContents(std::string path);
    GLuint createShaderFromString(const std::string &str, GLenum shaderType);

    // Discovering attributes/uniforms/textures
    void discoverShaderData();
    void discoverAttributes();
    void discoverUniforms();
    bool isUniformArray(const GLchar *name , GLsizei nameLength);
    bool isTexture(GLenum type);
    void addUniform(const std::string &name);
    void addUniformArray(const std::string &name, size_t size);
    void addTexture(const std::string &name);

    // Identifies the shader program associated with this shader
    GLuint m_programID;

    // Collections of known attributes/uniforms/textures
    std::map<std::string, GLuint>                     m_attributes;
    std::map<std::string, GLuint>                     m_uniforms;
    std::map<std::tuple<std::string, size_t>, GLuint> m_uniformArrays;
    std::map<std::string, GLuint>                     m_textureLocations; // name to uniform location
    std::map<GLuint, GLuint>                          m_textureSlots;     // uniform location to texture slot
};
