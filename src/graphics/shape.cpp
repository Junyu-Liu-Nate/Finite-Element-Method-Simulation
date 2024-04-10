#include "shape.h"

#include <iostream>

#include "graphics/shader.h"

using namespace Eigen;

Shape::Shape()
    : m_tetVao(-1),
      m_numSurfaceVertices(),
      m_verticesSize(),
      m_modelMatrix(Eigen::Matrix4f::Identity()),
      m_wireframe(false)
{
}

void Shape::init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3d> &normals, const std::vector<Eigen::Vector3i> &triangles)
{
    if(vertices.size() != normals.size()) {
        std::cerr << "Vertices and normals are not the same size" << std::endl;
        return;
    }
    glGenBuffers(1, &m_surfaceVbo);
    glGenBuffers(1, &m_surfaceIbo);
    glGenVertexArrays(1, &m_surfaceVao);

    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3 * 2, nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * vertices.size() * 3, static_cast<const void *>(vertices.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3, sizeof(double) * vertices.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 3 * triangles.size(), static_cast<const void *>(triangles.data()), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(m_surfaceVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, reinterpret_cast<GLvoid *>(sizeof(double) * vertices.size() * 3));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_numSurfaceVertices = triangles.size() * 3;
    m_verticesSize = vertices.size();
    m_faces = triangles;
}

void Shape::init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3i> &triangles)
{
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3i> faces;
    verts.reserve(triangles.size() * 3);
    normals.reserve(triangles.size() * 3);
    for(auto& f : triangles) {
        auto& v1 = vertices[f[0]];
        auto& v2 = vertices[f[1]];
        auto& v3 = vertices[f[2]];
        auto& e1 = v2 - v1;
        auto& e2 = v3 - v1;
        auto n = e1.cross(e2);
        int s = verts.size();
        faces.push_back(Eigen::Vector3i(s, s + 1, s + 2));
        normals.push_back(n);
        normals.push_back(n);
        normals.push_back(n);
        verts.push_back(v1);
        verts.push_back(v2);
        verts.push_back(v3);
    }
    glGenBuffers(1, &m_surfaceVbo);
    glGenBuffers(1, &m_surfaceIbo);
    glGenVertexArrays(1, &m_surfaceVao);

    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(double) * verts.size() * 3 * 2, nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * verts.size() * 3, static_cast<const void *>(verts.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * verts.size() * 3, sizeof(double) * verts.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 3 * faces.size(), static_cast<const void *>(faces.data()), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(m_surfaceVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, reinterpret_cast<GLvoid *>(sizeof(double) * verts.size() * 3));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_numSurfaceVertices = faces.size() * 3;
    m_verticesSize = vertices.size();
    m_faces = triangles;

    if (vertices.size() > 4) { //shape
        m_red = 0.93;
        m_green = 0.8;
        m_blue = 1.f;
        m_alpha = 1.f;
    } else { //ground
        m_red = 1;
        m_green = 1;
        m_blue = 1;
        m_alpha = 1.f;
    }
//    m_red = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//    m_blue = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//    m_green = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//    m_alpha = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void Shape::init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3i> &triangles, const std::vector<Eigen::Vector4i> &tetIndices)
{
    init(vertices, triangles);

    std::vector<Eigen::Vector2i> lines;
    for(Vector4i tet : tetIndices) {
        lines.emplace_back(tet[0], tet[1]);
        lines.emplace_back(tet[0], tet[2]);
        lines.emplace_back(tet[0], tet[3]);
        lines.emplace_back(tet[1], tet[2]);
        lines.emplace_back(tet[1], tet[3]);
        lines.emplace_back(tet[2], tet[3]);
    }
    glGenBuffers(1, &m_tetVbo);
    glGenBuffers(1, &m_tetIbo);
    glGenVertexArrays(1, &m_tetVao);

    glBindBuffer(GL_ARRAY_BUFFER, m_tetVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3, vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_tetIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 2 * lines.size(), static_cast<const void *>(lines.data()), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(m_tetVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_tetVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_tetIbo);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_numTetVertices = lines.size() * 2;
}

void Shape::setVertices(const std::vector<Eigen::Vector3d> &vertices)
{
    if(vertices.size() != m_verticesSize) {
        std::cerr << "You can't set vertices to a vector that is a different length that what shape was inited with" << std::endl;
        return;
    }
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3d> normals;
    verts.reserve(m_faces.size() * 3);
    normals.reserve(m_faces.size() * 3);
    for(auto& f : m_faces) {
        auto& v1 = vertices[f[0]];
        auto& v2 = vertices[f[1]];
        auto& v3 = vertices[f[2]];
        auto& e1 = v2 - v1;
        auto& e2 = v3 - v1;
        auto n = e1.cross(e2);
        normals.push_back(n);
        normals.push_back(n);
        normals.push_back(n);
        verts.push_back(v1);
        verts.push_back(v2);
        verts.push_back(v3);
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * verts.size() * 3, static_cast<const void *>(verts.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * verts.size() * 3, sizeof(double) * verts.size() * 3, static_cast<const void *>(normals.data()));
    if(m_tetVao != static_cast<GLuint>(-1)) {
        glBindBuffer(GL_ARRAY_BUFFER, m_tetVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * vertices.size() * 3, static_cast<const void *>(vertices.data()));
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::setModelMatrix(const Eigen::Affine3f &model)
{
    m_modelMatrix = model.matrix();
}

void Shape::toggleWireframe()
{
    m_wireframe = !m_wireframe;
}

void Shape::setVertices(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3d> &normals)
{
    if(vertices.size() != normals.size()) {
        std::cerr << "Vertices and normals are not the same size" << std::endl;
        return;
    }
    if(vertices.size() != m_verticesSize) {
        std::cerr << "You can't set vertices to a vector that is a different length that what shape was inited with" << std::endl;
        return;
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * vertices.size() * 3, static_cast<const void *>(vertices.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3, sizeof(double) * vertices.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::draw(Shader *shader)
{
    Eigen::Matrix3f m3 = m_modelMatrix.topLeftCorner(3, 3);
    Eigen::Matrix3f inverseTransposeModel = m3.inverse().transpose();

    if(m_wireframe && m_tetVao != static_cast<GLuint>(-1)) {
        shader->setUniform("wire", 1);
        shader->setUniform("model", m_modelMatrix);
        shader->setUniform("inverseTransposeModel", inverseTransposeModel);
        shader->setUniform("red",   1);
        shader->setUniform("green", 1);
        shader->setUniform("blue",  1);
        shader->setUniform("alpha", 1);
        glBindVertexArray(m_tetVao);
        glDrawElements(GL_LINES, m_numTetVertices, GL_UNSIGNED_INT, reinterpret_cast<GLvoid *>(0));
        glBindVertexArray(0);
    } else {
        shader->setUniform("wire", 0);
        shader->setUniform("model", m_modelMatrix);
        shader->setUniform("inverseTransposeModel", inverseTransposeModel);
        shader->setUniform("red",   m_red);
        shader->setUniform("green", m_green);
        shader->setUniform("blue",  m_blue);
        shader->setUniform("alpha", m_alpha);
        glBindVertexArray(m_surfaceVao);
        glDrawElements(GL_TRIANGLES, m_numSurfaceVertices, GL_UNSIGNED_INT, reinterpret_cast<GLvoid *>(0));
        glBindVertexArray(0);
    }
}
