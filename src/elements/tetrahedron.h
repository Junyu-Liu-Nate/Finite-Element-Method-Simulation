#ifndef TETRAHEDRON_H
#define TETRAHEDRON_H

#include <Eigen/Dense>

using namespace Eigen;

class Tetrahedron
{
public:
    Tetrahedron();

    Vector4i vertexIndices; // Indices into the vertex list
    Matrix3d initialInverseMatrix = Matrix3d::Identity(); // For computing strains
};

#endif // TETRAHEDRON_H
