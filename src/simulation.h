#pragma once

#include "QtCore/qstring.h"
#include "graphics/shape.h"
#include "elements/vertex.h"
#include "elements/tetrahedron.h"
#include "elements/simShape.h"

#include <unordered_map>
#include <QtConcurrent>
#include <QVector>
#include <QFuture>

using namespace Eigen;

struct Settings {
    QString inputMeshPath;
    QString obstacleMeshPath;

    int integrateMethod;

    Eigen::Vector3d _g;         //penalty for node collision
    double _kFloor;       //penalty for node collision
    double _lambda;       //incompressibility for the whole material
    double _mu;           //rigidity for the whole material
    double _phi;          //coefficients of viscosity
    double _psi;
    double _density;   //density

    Eigen::Vector3d translation; // Translation vector
    double rotationZ; // Rotation around Z axis in radians

    bool isCustomizeTimeStep;
    double integrationTimeStep;
    bool isAdaptiveTimeStep;

    bool isParallelize;

    bool isFBO;
    bool isFXAA;
};

// A simple hash function for Vector3i to use it as a key in unordered_map
struct hash_fn {
    std::size_t operator() (const Vector3i& vec) const {
        std::size_t h1 = std::hash<int>()(vec[0]);
        std::size_t h2 = std::hash<int>()(vec[1]);
        std::size_t h3 = std::hash<int>()(vec[2]);
        return h1 ^ h2 ^ h3;
    }
};

class Shader;

class Simulation
{
public:
    explicit Simulation(const Settings& settings);
    Simulation();

    void init();

    void update(double seconds);

    void draw(Shader *shader);

    void toggleWire();

    Settings m_settings;

private:

    Shape m_shape;
    Shape m_shape_static;

    std::vector<Vector3d> vertices;
    std::vector<Vector4i> tets;

    Shape m_ground;
    void initGround();

    // Timestep update and integration
    void fixedTimeStepUpdate(double seconds);
    void adaptiveTimeStepUpdate(double seconds);
    void integrate(double timeStep, SimShape& shape);

    // Setup and update vetices and tets info
    SimShape dynamicShape;
    SimShape staticShape;

    void updateVerticesInfo();
    bool isSameFace(Vector3i face1, Vector3i face2);
    std::vector<Vector3i> extractSurfaceMesh(const std::vector<Vector3d>& vertices, const std::vector<Vector4i>& tets);
    Eigen::Vector3d computeNormal(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
    void ensureCounterClockwise(const std::vector<Vector3d>& vertices, std::vector<Vector3i>& faces);

    // Force calculation
    void applyGravity(SimShape& simShape);
    void applyInternalForces(SimShape& simShape);

    Matrix3d calculateP(Tetrahedron& tet, SimShape& simShape);
    Matrix3d calculateV(Tetrahedron& tet, SimShape& simShape);
    Matrix3d calculateB(Tetrahedron& tet, SimShape& simShape);
    Vector3d calculateFaceNormal(std::vector<Vector3d> faceVertices, Vector3d insideVertex);
    double calculateFaceArea(const std::vector<Vector3d>& faceVertices);

    // Collision detection
    void collisionDetection(Vertex& simVertex);
    void collisionDetection(Vertex& simVertex, const SimShape& mesh);
    bool isPointInTriangle(const Vector3d& point, const Vector3d& a, const Vector3d& b, const Vector3d& c);
    bool checkLineTriangleIntersection(
        const Vector3d& lineStart, const Vector3d& lineEnd,
        const Vector3d& v0, const Vector3d& v1, const Vector3d& v2,
        Vector3d& intersectionPoint);

    // Solver
    void eulerIntegrate(double deltaT, SimShape& simShape);
    void eulerIntegrateParallel(double deltaT, SimShape& simShape);
    void midPointIntegrate(double deltaT, SimShape& simShape);
    void midPointIntegrateParallel(double deltaT, SimShape& simShape);
    void rk4Integrate(double deltaT, SimShape& simShape);

    double calculateMaxPositionError(const SimShape& simShape1, const SimShape& simShape2);
    double calculateMaxPositionErrorParallel(const SimShape& simShape1, const SimShape& simShape2);
};
