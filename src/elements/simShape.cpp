#include "simShape.h"

SimShape::SimShape()
{
}

// Copy constructor
SimShape::SimShape(const SimShape& other)
    : vertices(other.vertices), tets(other.tets), faces(other.faces),
    simVertices(other.simVertices), simTets(other.simTets), modelMatrix(other.modelMatrix) {}

// Copy assignment operator
SimShape& SimShape::operator=(const SimShape& other) {
    if (this != &other) { // Protect against self-assignment
        vertices = other.vertices;
        tets = other.tets;
        faces = other.faces;
        simVertices = other.simVertices;
        simTets = other.simTets;
        modelMatrix = other.modelMatrix;
    }
    return *this;
}

void SimShape::setupVertexInfo(float density) {
    simVertices.clear();

    double totalVolume = 0.0;
    for (auto& tet : simTets) {
        double volume = computeTetrahedronVolume(vertices, tet);
        totalVolume += volume;
    }
    double totalMass = totalVolume * density;

    for (auto& rawVertex : vertices) {
        Vertex vertex = Vertex();
        vertex.position = modelMatrix * rawVertex; // Convert to world space
        vertex.restPosition = modelMatrix * rawVertex; // Convert to world space
        vertex.stepStartPosition = modelMatrix * rawVertex;
        vertex.mass = Matrix3d::Identity() * (totalMass / vertices.size());

        simVertices.push_back(vertex);
    }
}

void SimShape::setupTetInfo() {
    simTets.clear();
    for (auto& rawTet : tets) {
        Tetrahedron tet = Tetrahedron();
        tet.vertexIndices = rawTet;

        simTets.push_back(tet);
    }
}

double SimShape::computeTetrahedronVolume(const std::vector<Vector3d>& vertices, const Tetrahedron& tet) {
    // Assuming Tetrahedron has a member named vertexIndices which is an array of indices.
    const Vector3d& p1 = vertices[tet.vertexIndices[0]];
    const Vector3d& p2 = vertices[tet.vertexIndices[1]];
    const Vector3d& p3 = vertices[tet.vertexIndices[2]];
    const Vector3d& p4 = vertices[tet.vertexIndices[3]];

    // Form the matrix from the vectors p2 - p1, p3 - p1, p4 - p1
    Matrix3d mat;
    mat.col(0) = p2 - p1;
    mat.col(1) = p3 - p1;
    mat.col(2) = p4 - p1;

    // Compute the determinant and the volume
    double volume = std::abs(mat.determinant()) / 6.0;

    return volume;
}
