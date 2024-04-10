#ifndef SIMSHAPE_H
#define SIMSHAPE_H

#include "elements/vertex.h"
#include "elements/tetrahedron.h"

class SimShape
{
public:
    SimShape();
    SimShape(const SimShape& other); // Copy constructor
    SimShape& operator=(const SimShape& other); // Copy assignment operator

    std::vector<Vector3d> vertices;
    std::vector<Vector4i> tets;
    std::vector<Vector3i> faces;

    std::vector<Vertex> simVertices;
    std::vector<Tetrahedron> simTets;
    Affine3d modelMatrix;

    void setupVertexInfo(float density);
    void setupTetInfo();
    double computeTetrahedronVolume(const std::vector<Vector3d>& vertices, const Tetrahedron& tet);
};

#endif // SIMSHAPE_H
