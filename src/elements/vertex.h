#ifndef VERTEX_H
#define VERTEX_H

#include <Eigen/Dense>

using namespace Eigen;

class Vertex
{
public:
    Vertex();

    Vector3d position = Vector3d(0.0, 0.0, 0.0);
    Vector3d restPosition = Vector3d(0.0, 0.0, 0.0);
    Vector3d stepStartPosition = Vector3d(0.0, 0.0, 0.0);
    Vector3d stepStartVelocity = Vector3d(0.0, 0.0, 0.0);
    Vector3d velocity = Vector3d(0.0, 0.0, 0.0);
    Vector3d force = Vector3d(0.0, 0.0, 0.0);
    Vector3d acceleration = Vector3d(0.0, 0.0, 0.0);
//    Matrix3d mass = Matrix3d::Identity();
    Matrix3d mass = Matrix3d::Zero();
};

#endif // VERTEX_H
