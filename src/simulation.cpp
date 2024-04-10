#include "simulation.h"
#include "graphics/meshloader.h"
//#include <omp.h>

#include <iostream>

using namespace Eigen;

Simulation::Simulation() {}

Simulation::Simulation(const Settings& settings) : m_settings(settings) {}

void Simulation::init() {
    // Load the dynamically simulated mesh
    std::string dynamicShapeMeshPath = m_settings.inputMeshPath.toStdString();
    if (MeshLoader::loadTetMesh(dynamicShapeMeshPath, dynamicShape.vertices, dynamicShape.tets)) {
        dynamicShape.faces = extractSurfaceMesh(dynamicShape.vertices, dynamicShape.tets);

        Eigen::Translation3d translationD(m_settings.translation);
        Eigen::AngleAxisd rotationD(m_settings.rotationZ, Eigen::Vector3d::UnitZ());
        Eigen::Affine3d transformationD = translationD * rotationD;

        dynamicShape.modelMatrix = transformationD;
        dynamicShape.setupTetInfo();
        dynamicShape.setupVertexInfo(m_settings._density);

        m_shape.init(dynamicShape.vertices, dynamicShape.faces, dynamicShape.tets);

        Eigen::Translation3f translation(m_settings.translation.cast<float>());
        Eigen::AngleAxisf rotation(m_settings.rotationZ, Eigen::Vector3f::UnitZ());
        Eigen::Affine3f transformation = translation * rotation;
        m_shape.setModelMatrix(transformation);

        this->vertices = dynamicShape.vertices; // Store all dynamically updated vertices
        this->tets = dynamicShape.tets;
    }

    // Load the static mesh as obstacles
    std::string staticShapeMeshPath = m_settings.obstacleMeshPath.toStdString();
    if (!staticShapeMeshPath.empty() && MeshLoader::loadTetMesh(staticShapeMeshPath, staticShape.vertices, staticShape.tets)) {
        staticShape.faces = extractSurfaceMesh(staticShape.vertices, staticShape.tets);
        staticShape.modelMatrix = Affine3d(Eigen::Translation3d(0, 0, 0));
        staticShape.setupTetInfo();
        staticShape.setupVertexInfo(m_settings._density);

        m_shape_static.init(staticShape.vertices, staticShape.faces, staticShape.tets);
        m_shape_static.setModelMatrix(Affine3f(Eigen::Translation3f(0, 0, 0)));
    }

    initGround();

    collisionDetection(dynamicShape.simVertices.at(0), staticShape);
}

void Simulation::update(double seconds)
{
    if (m_settings.isAdaptiveTimeStep) {
        adaptiveTimeStepUpdate(seconds);
    }
    else {
        fixedTimeStepUpdate(seconds);
    }
}

void Simulation::fixedTimeStepUpdate(double seconds) {
    double timeStep;
    if (m_settings.isCustomizeTimeStep) {
        timeStep = m_settings.integrationTimeStep;
    }
    else {
        timeStep = seconds;
    }

    int timeStepNum = int(seconds / timeStep);
    for (int i = 0; i < timeStepNum; i++) {
        integrate(timeStep, dynamicShape);
        updateVerticesInfo();
    }
    m_shape.setVertices(vertices);
}

// Adaptive time-step update (if using euler, the tetwild will crash without using adaptive-time)
void Simulation::adaptiveTimeStepUpdate(double seconds) {
    double remainingTime = seconds;
    double currentStepSize = std::min(seconds, 0.01); // Start with a reasonable initial step size
    double errorThreshold = 0.01; // Define an acceptable error threshold

    while (remainingTime > 0) {
        SimShape originalShape = dynamicShape; // Make a deep copy of the original state

        // Integrate using the current time step
        integrate(currentStepSize, dynamicShape); // Full step integration
        SimShape afterFullStepShape = dynamicShape; // Save the state after the full step

        dynamicShape = originalShape; // Reset to original state before the half steps

        // Perform two half-step integrations
        integrate(currentStepSize / 2, dynamicShape); // First half-step
        integrate(currentStepSize / 2, dynamicShape); // Second half-step

        // Calculate the error between the full step and the two half steps
        double error = calculateMaxPositionError(afterFullStepShape, dynamicShape);

        // Adjust the time step based on the error
        if (error > errorThreshold) {
            currentStepSize /= 2; // Decrease step size if error is too large
            dynamicShape = originalShape; // Reset the shape for retry
        } else {
            // If the error is acceptable, accept the two half-steps as the new state
            if (error < errorThreshold / 4) {
                currentStepSize = std::min(currentStepSize * 2, remainingTime); // Increase step size if error is very small
            }
            remainingTime -= currentStepSize; // Decrease the remaining time
            updateVerticesInfo(); // Update vertices information after a successful integration
        }
    }

    m_shape.setVertices(vertices); // Update the main shape vertices after completing all time steps
}

void Simulation::integrate(double timeStep, SimShape& shape) {
    // Integration method selection
    switch (m_settings.integrateMethod) {
        case 1:
            if (m_settings.isParallelize) {
                eulerIntegrateParallel(timeStep, shape);
            }
            else {
                eulerIntegrate(timeStep, shape);
            }
            break;
        case 2:
            if (m_settings.isParallelize) {
                midPointIntegrateParallel(timeStep, shape);
            }
            else {
                midPointIntegrate(timeStep, shape);
            }
            break;
        case 3:
            rk4Integrate(timeStep, shape);
            break;
        default:
            rk4Integrate(timeStep, shape);
            break;
    }
}

//----------------------- Setup/update vetices and tets info  -----------------------//
void Simulation::updateVerticesInfo() {
    int vertexNum = dynamicShape.simVertices.size();
    for (int i = 0; i < vertexNum; i++) {
        vertices[i] = dynamicShape.modelMatrix.inverse() * dynamicShape.simVertices.at(i).position; // Convert back to object space
    }
}

// Function to check if two faces are the same irrespective of their winding order
bool Simulation::isSameFace(Vector3i face1, Vector3i face2) {
    std::sort(face1.data(), face1.data() + face1.size());
    std::sort(face2.data(), face2.data() + face2.size());
    return face1 == face2;
}

std::string generateFaceKey(const Vector3i& face, const std::vector<Vector3d>& vertices) {
    // Generate all permutations of the face indices and convert to a string representation
    std::vector<std::string> permutations;

    permutations.push_back(std::to_string(face[0]) + "," + std::to_string(face[1]) + "," + std::to_string(face[2]));
    permutations.push_back(std::to_string(face[1]) + "," + std::to_string(face[2]) + "," + std::to_string(face[0]));
    permutations.push_back(std::to_string(face[2]) + "," + std::to_string(face[0]) + "," + std::to_string(face[1]));

    // Sort the permutations and choose the first as the unique key
    std::sort(permutations.begin(), permutations.end());
    return permutations.front();
}

// Function to extract surface meshes from a given tetrahedral mesh
std::vector<Vector3i> Simulation::extractSurfaceMesh(const std::vector<Vector3d>& vertices, const std::vector<Vector4i>& tets) {
    std::unordered_map<std::string, std::pair<Vector3i, int>> faceCount; // Map from face key to face and count
    std::vector<Vector3i> faces;

    for (const auto& tet : tets) {
        // Compute the centroid of the tetrahedron
        Eigen::Vector3d centroid = (vertices[tet[0]] + vertices[tet[1]] + vertices[tet[2]] + vertices[tet[3]]) / 4.0;

        std::vector<Vector3i> tetFaces = {
            Vector3i(tet[0], tet[1], tet[2]),
            Vector3i(tet[0], tet[2], tet[3]),
            Vector3i(tet[0], tet[3], tet[1]),
            Vector3i(tet[1], tet[2], tet[3])
        };

        for (auto& face : tetFaces) {
            // Ensure the face has the correct winding order before generating the key
            Eigen::Vector3d v0 = vertices[face[0]];
            Eigen::Vector3d v1 = vertices[face[1]];
            Eigen::Vector3d v2 = vertices[face[2]];
            Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0);
            Eigen::Vector3d faceCentroid = (v0 + v1 + v2) / 3.0;
            Eigen::Vector3d toCentroid = centroid - faceCentroid;

            if (normal.dot(toCentroid) > 0) {
                std::swap(face[1], face[2]); // Correct the winding order
            }

            // Use a unique key for the face
            std::string key = generateFaceKey(face, vertices);
            if (faceCount.find(key) == faceCount.end()) {
                faceCount[key] = std::make_pair(face, 1);
            } else {
                faceCount[key].second += 1;
            }
        }
    }

    // Select faces that are unique
    for (const auto& item : faceCount) {
        if (item.second.second == 1) {
            faces.push_back(item.second.first);
        }
    }

    return faces;
}

// Helper function to compute the normal of a triangle face
Eigen::Vector3d Simulation::computeNormal(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    return (v1 - v0).cross(v2 - v0).normalized();
}

// Function to ensure counter-clockwise winding of faces
void Simulation::ensureCounterClockwise(const std::vector<Vector3d>& vertices, std::vector<Vector3i>& faces) {
    // Example heuristic: Use the mesh's centroid to determine the outward direction
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& vertex : vertices) {
        centroid += vertex;
    }
    centroid /= vertices.size();

    for (auto& face : faces) {
        Eigen::Vector3d v0 = vertices[face[0]];
        Eigen::Vector3d v1 = vertices[face[1]];
        Eigen::Vector3d v2 = vertices[face[2]];

        Eigen::Vector3d normal = computeNormal(v0, v1, v2);
        Eigen::Vector3d faceCentroid = (v0 + v1 + v2) / 3.0;
        Eigen::Vector3d toCentroid = centroid - faceCentroid;

        // If the dot product is positive, the normal points towards the centroid, so the face is wound clockwise when seen from outside
        if (normal.dot(toCentroid) > 0) {
            // Swap two vertices to change winding order to counter-clockwise
            std::swap(face[0], face[1]);
        }
    }
}

//----------------------- Force calculation -----------------------//
void Simulation::applyGravity(SimShape& simShape) {
    Vector3d gravity = m_settings._g;
    #pragma omp parallel for
    for (auto& simVertex : simShape.simVertices) {
        simVertex.force += simVertex.mass * gravity;
    }
}

void Simulation::applyInternalForces(SimShape& simShape) {
//    #pragma omp parallel for
    for (auto& tet : simShape.simTets) {
        //----- Compute the strain
        // Construct deformation P and B for the current tetrahedron
        Matrix3d P = calculateP(tet, simShape);
        Matrix3d V = calculateV(tet, simShape);
        Matrix3d B = calculateB(tet, simShape);

        // Calculate the elastic strain
        Matrix3d F_Elastic;
        F_Elastic = P * B;
        Matrix3d epsilonElastic;
        epsilonElastic = F_Elastic.transpose() * F_Elastic - Matrix3d::Identity();

        //----- Compute the elastic stress
        double lambda = m_settings._lambda; // Material-specific constant
        double mu = m_settings._mu; // Material-specific constant
        Matrix3d sigmaElastic = lambda * epsilonElastic.trace() * Matrix3d::Identity() + 2 * mu * epsilonElastic;

        // Calculate the damping strain
        Matrix3d F_Damp;
        F_Damp = V * B;
        Matrix3d epsilonDamp;
        epsilonDamp = F_Elastic.transpose() * F_Damp + F_Damp.transpose() * F_Elastic;

        //----- Compute the damping stress
        double phi = m_settings._phi; // Material-specific constant
        double si = m_settings._psi; // Material-specific constant
        Matrix3d sigmaDamp = phi * epsilonDamp.trace() * Matrix3d::Identity() + 2 * si * epsilonDamp;

        Matrix3d sigma = sigmaElastic + sigmaDamp;

        //----- Compute the internal forces
        for (auto& vertexIndex: tet.vertexIndices) {
            std::vector<Vector3d> oppositeFaceVertices;
            for (auto& searchVertexIndex: tet.vertexIndices) {
                if (searchVertexIndex != vertexIndex) {
                    oppositeFaceVertices.push_back(simShape.simVertices[searchVertexIndex].restPosition);
                }
            }
            Vector3d oppositeFaceNormal = calculateFaceNormal(oppositeFaceVertices, simShape.simVertices[vertexIndex].restPosition);
            double faceArea = calculateFaceArea(oppositeFaceVertices);
            simShape.simVertices[vertexIndex].force += F_Elastic * sigma * faceArea * oppositeFaceNormal;
        }
    }
}

Matrix3d Simulation::calculateP(Tetrahedron& tet, SimShape& simShape) {
    // Extract vertex positions in world space
    Vector3d p1 = simShape.simVertices[tet.vertexIndices[0]].position;
    Vector3d p2 = simShape.simVertices[tet.vertexIndices[1]].position;
    Vector3d p3 = simShape.simVertices[tet.vertexIndices[2]].position;
    Vector3d p4 = simShape.simVertices[tet.vertexIndices[3]].position;

    // Construct deformation P for the current tetrahedron
    Matrix3d P;
    P.col(0) = p1 - p4;
    P.col(1) = p2 - p4;
    P.col(2) = p3 - p4;

    return P;
}

Matrix3d Simulation::calculateV(Tetrahedron& tet, SimShape& simShape) {
    // Extract vertex positions in world space
    Vector3d v1 = simShape.simVertices[tet.vertexIndices[0]].velocity;
    Vector3d v2 = simShape.simVertices[tet.vertexIndices[1]].velocity;
    Vector3d v3 = simShape.simVertices[tet.vertexIndices[2]].velocity;
    Vector3d v4 = simShape.simVertices[tet.vertexIndices[3]].velocity;

    // Construct deformation V for the current tetrahedron
    Matrix3d V;
    V.col(0) = v1 - v4;
    V.col(1) = v2 - v4;
    V.col(2) = v3 - v4;

    return V;
}

Matrix3d Simulation::calculateB(Tetrahedron& tet, SimShape& simShape) {
    // Extract vertex positions in material space
    Vector3d m1 = simShape.simVertices[tet.vertexIndices[0]].restPosition;
    Vector3d m2 = simShape.simVertices[tet.vertexIndices[1]].restPosition;
    Vector3d m3 = simShape.simVertices[tet.vertexIndices[2]].restPosition;
    Vector3d m4 = simShape.simVertices[tet.vertexIndices[3]].restPosition;

    // Construct deformation beta for the current tetrahedron
    Matrix3d B;
    B.col(0) = m1 - m4;
    B.col(1) = m2 - m4;
    B.col(2) = m3 - m4;
    B = B.inverse();

    return B;
}

Vector3d Simulation::calculateFaceNormal(std::vector<Vector3d> faceVertices, Vector3d insideVertex) {
    Vector3d v1 = faceVertices.at(0);
    Vector3d v2 = faceVertices.at(1);
    Vector3d v3 = faceVertices.at(2);

    Vector3d normal = (v2 - v1).cross(v3 - v1).normalized();

    // Check if normal is pointing outwards by dotting with vector from inside point to a face point
    Vector3d insideToFaceVector = v1 - insideVertex; // Vector from inside vertex to one of the face vertices
    if (normal.dot(insideToFaceVector) < 0) {
        normal = -normal; // Reverse the normal if it's pointing inwards
    }

    return normal;
}

double Simulation::calculateFaceArea(const std::vector<Vector3d>& faceVertices) {
    if (faceVertices.size() != 3) {
        throw std::invalid_argument("Face must have exactly three vertices.");
    }

    // Get the vectors representing two edges of the triangle
    Vector3d edge1 = faceVertices[1] - faceVertices[0];
    Vector3d edge2 = faceVertices[2] - faceVertices[0];

    // The cross product of two edge vectors gives a vector perpendicular to the plane of the triangle
    // with a magnitude equal to twice the area of the triangle
    Vector3d crossProduct = edge1.cross(edge2);

    // The area of the triangle is half the magnitude of the cross product vector
    double area = 0.5 * crossProduct.norm();

    return area;
}

//----------------------- Collision detection -----------------------//
void Simulation::collisionDetection(Vertex& simVertex) {
    Vector3d groundNormal(0, 1, 0); // ground plane normal is (0, 1, 0)
    double groundFriction = 0.5; // Friction coefficient for the ground, adjust as needed
    double restitutionCoefficient = 1.0; // Restitution coefficient for bounciness, adjust as needed
    double groundHeight = 0.0; // y value of the floor

    if (simVertex.position.y() < groundHeight) {
        simVertex.position.y() = groundHeight; // Reset the position to be on the ground
        Vector3d vNormal = groundNormal * groundNormal.dot(simVertex.velocity); // Normal component of velocity
        Vector3d vTangent = simVertex.velocity - vNormal; // Tangential component of velocity

        // Reflect the normal component of the velocity and apply restitution
        vNormal = -restitutionCoefficient * vNormal;

        // Apply friction to the tangential component of the velocity
        vTangent = groundFriction * vTangent;

        // Combine the adjusted normal and tangential components
        simVertex.velocity = vNormal + vTangent;
    }
}

void Simulation::collisionDetection(Vertex& simVertex, const SimShape& mesh) {
    // Ground collision detection (existing code)
    Vector3d groundNormal(0, 1, 0);
    double groundFriction = 0.5;
    double restitutionCoefficient = 1.0;
    double groundHeight = 0.0;

    if (simVertex.position.y() < groundHeight
        && simVertex.position.x() <= 5 && simVertex.position.x() >= -5
        && simVertex.position.z() <= 5 && simVertex.position.z() >= -5) {
        simVertex.position.y() = groundHeight + 0.000001;
        Vector3d vNormal = groundNormal * groundNormal.dot(simVertex.velocity);
        Vector3d vTangent = simVertex.velocity - vNormal;
        vNormal = -restitutionCoefficient * vNormal;
        vTangent = groundFriction * vTangent;
        simVertex.velocity = vNormal + vTangent;
    }

    // Mesh collision detection - revised
    for (const auto& face : mesh.faces) {
        Vector3d worldVertices[3];
        for (int i = 0; i < 3; i++) {
            worldVertices[i] = mesh.simVertices[face[i]].position;
        }

        Vector3d intersectionPoint;
        if (checkLineTriangleIntersection(simVertex.stepStartPosition, simVertex.position, worldVertices[0], worldVertices[1], worldVertices[2], intersectionPoint)) {

            Vector3d faceNormal = (worldVertices[1] - worldVertices[0]).cross(worldVertices[2] - worldVertices[0]).normalized();
            simVertex.position = intersectionPoint + faceNormal * 0.001;
            Vector3d vNormal = faceNormal * faceNormal.dot(simVertex.velocity);
            Vector3d vTangent = simVertex.velocity - vNormal;
            vNormal = -restitutionCoefficient * vNormal;
            vTangent = groundFriction * vTangent;
            simVertex.velocity = vNormal + vTangent;
        }
    }
}

bool Simulation::isPointInTriangle(const Vector3d& point, const Vector3d& a, const Vector3d& b, const Vector3d& c) {
    Vector3d v0 = b - a, v1 = c - a, v2 = point - a;

    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;

    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    return (v >= 0) && (w >= 0) && (u >= 0);
}

bool Simulation::checkLineTriangleIntersection(
    const Vector3d& lineStart, const Vector3d& lineEnd,
    const Vector3d& v0, const Vector3d& v1, const Vector3d& v2,
    Vector3d& intersectionPoint) {

    Vector3d dir = lineEnd - lineStart; // Direction of the line
    Vector3d edge1 = v1 - v0;
    Vector3d edge2 = v2 - v0;
    Vector3d h = dir.cross(edge2);
    double a = edge1.dot(h);

    if (a > -1e-10 && a < 1e-10)
        return false; // This means that the line is parallel to the triangle.

    double f = 1.0 / a;
    Vector3d s = lineStart - v0;
    double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0)
        return false; // The intersection is outside of the triangle.

    Vector3d q = s.cross(edge1);
    double v = f * dir.dot(q);

    if (v < 0.0 || u + v > 1.0)
        return false; // The intersection is outside of the triangle.

    // At this stage, we can compute t to find out where the intersection point is on the line.
    double t = f * edge2.dot(q);

    if (t > 1e-10 && t < 1.0) { // ray intersection
        intersectionPoint = lineStart + dir * t;
        return true;
    } else {
        return false; // No intersection
    }
}

//----------------------- Solver -----------------------//
void Simulation::eulerIntegrate(double deltaT, SimShape& simShape) {
    applyGravity(simShape);
    applyInternalForces(simShape);

    for (auto& simVertex : simShape.simVertices) {
        simVertex.stepStartPosition = simVertex.position;

        // Integrate acceleration to get new velocity
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;
        simVertex.velocity += deltaT * simVertex.acceleration;

        // Integrate velocity to get new position
        simVertex.position += deltaT * simVertex.velocity;

        // Resolve collision
        collisionDetection(simVertex, staticShape);

        // Reset the force and acceleration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }
}

void Simulation::eulerIntegrateParallel(double deltaT, SimShape& simShape) {
    applyGravity(simShape);
    applyInternalForces(simShape);

    #pragma omp parallel for
    for (auto& simVertex : simShape.simVertices) {
        simVertex.stepStartPosition = simVertex.position;

        // Integrate acceleration to get new velocity
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;
        simVertex.velocity += deltaT * simVertex.acceleration;

        // Integrate velocity to get new position
        simVertex.position += deltaT * simVertex.velocity;

        // Resolve collision
        collisionDetection(simVertex, staticShape);

        // Reset the force and acceleration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }
}

void Simulation::midPointIntegrate(double deltaT, SimShape& simShape) {
    applyGravity(simShape);
    applyInternalForces(simShape);

    for (auto& simVertex : simShape.simVertices) {
        simVertex.stepStartPosition = simVertex.position;
        simVertex.stepStartVelocity = simVertex.velocity;

        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;
        simVertex.velocity += deltaT * simVertex.acceleration;
        simVertex.position += deltaT * simVertex.velocity / 2;
        simVertex.velocity = simVertex.stepStartVelocity + 0.5 * deltaT * simVertex.acceleration;

        // Reset the force and acceleration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }

    applyGravity(simShape);
    applyInternalForces(simShape);

    for (auto& simVertex : simShape.simVertices) {
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;
        simVertex.velocity += 0.5 * deltaT * simVertex.acceleration;
        simVertex.position = simVertex.stepStartPosition + deltaT * simVertex.velocity;

        // Resolve collision
        collisionDetection(simVertex, staticShape);

        // Reset the force and acceleration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }
}

void Simulation::midPointIntegrateParallel(double deltaT, SimShape& simShape) {
    applyGravity(simShape);
    applyInternalForces(simShape);

    const int numVertices = simShape.simVertices.size();
    const int numTasks = QThread::idealThreadCount();
    QVector<QFuture<void>> futures;

    // Lambda to process a chunk of vertices for the first half of the integration
    auto firstHalfIntegration = [deltaT, &simShape](int start, int end) {
        for (int i = start; i < end; ++i) {
            auto& simVertex = simShape.simVertices[i];
            simVertex.stepStartPosition = simVertex.position;
            simVertex.stepStartVelocity = simVertex.velocity;

            simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;
            simVertex.velocity += deltaT * simVertex.acceleration;
            simVertex.position += deltaT * simVertex.velocity / 2;
            simVertex.velocity = simVertex.stepStartVelocity + 0.5 * deltaT * simVertex.acceleration;

            simVertex.force = Vector3d(0.0, 0.0, 0.0);
            simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
        }
    };

    // Divide work for the first half of the integration and submit tasks
    int chunkSize = (numVertices + numTasks - 1) / numTasks; // Ensure even distribution
    for (int i = 0; i < numVertices; i += chunkSize) {
        int end = std::min(i + chunkSize, numVertices);
        futures.push_back(QtConcurrent::run(firstHalfIntegration, i, end));
    }

    // Wait for all tasks to complete before proceeding
    for (auto &future : futures) {
        future.waitForFinished();
    }

    applyGravity(simShape);
    applyInternalForces(simShape);

    // Clear futures and prepare for the second half
    futures.clear();

    // Lambda for the second half of the integration
    auto secondHalfIntegration = [deltaT, &simShape, this](int start, int end) {
        for (int i = start; i < end; ++i) {
            auto& simVertex = simShape.simVertices[i];
            simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;
            simVertex.velocity += 0.5 * deltaT * simVertex.acceleration;
            simVertex.position = simVertex.stepStartPosition + deltaT * simVertex.velocity;

            // Resolve collision
            collisionDetection(simVertex, staticShape);

            simVertex.force = Vector3d(0.0, 0.0, 0.0);
            simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
        }
    };

    // Divide work for the second half of the integration and submit tasks
    for (int i = 0; i < numVertices; i += chunkSize) {
        int end = std::min(i + chunkSize, numVertices);
        futures.push_back(QtConcurrent::run(secondHalfIntegration, i, end));
    }

    // Wait for all tasks to complete
    for (auto &future : futures) {
        future.waitForFinished();
    }
}

void Simulation::rk4Integrate(double deltaT, SimShape& simShape) {
    // First, calculate the initial force and acceleration on each vertex
    applyGravity(simShape);
    applyInternalForces(simShape);

    std::vector<Vector3d> k1Velocity, k1Position, k2Velocity, k2Position, k3Velocity, k3Position, k4Velocity, k4Position;

//    #pragma omp parallel for
    for (auto& simVertex : simShape.simVertices) {
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;

        // k1 is the initial slope
        k1Velocity.push_back(deltaT * simVertex.acceleration);
        k1Position.push_back(deltaT * simVertex.velocity);

        // Store the position and velocity at the start of the timestep
        simVertex.stepStartPosition = simVertex.position;
        simVertex.stepStartVelocity = simVertex.velocity;

        // Prepare for k2 calculation
        simVertex.velocity += 0.5 * k1Velocity.back();
        simVertex.position += 0.5 * k1Position.back();

        // Reset the force and acceleration for the next iteration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }

    // Re-calculate forces for k2
    applyGravity(simShape);
    applyInternalForces(simShape);

//    #pragma omp parallel for
    for (size_t i = 0; i < simShape.simVertices.size(); ++i) {
        auto& simVertex = simShape.simVertices[i];
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;

        // k2 is the slope at the midpoint using k1 adjustments
        k2Velocity.push_back(deltaT * simVertex.acceleration);
        k2Position.push_back(deltaT * simVertex.velocity);

        // Prepare for k3 calculation
        simVertex.velocity = simShape.simVertices[i].stepStartVelocity + 0.5 * k2Velocity.back();
        simVertex.position = simShape.simVertices[i].stepStartPosition + 0.5 * k2Position.back();

        // Reset the force and acceleration for the next iteration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }

    // Re-calculate forces for k3
    applyGravity(simShape);
    applyInternalForces(simShape);

//    #pragma omp parallel for
    for (size_t i = 0; i < simShape.simVertices.size(); ++i) {
        auto& simVertex = simShape.simVertices[i];
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;

        // k3 is the slope at the midpoint using k2 adjustments
        k3Velocity.push_back(deltaT * simVertex.acceleration);
        k3Position.push_back(deltaT * simVertex.velocity);

        // Prepare for k4 calculation
        simVertex.velocity = simShape.simVertices[i].velocity + k3Velocity.back();
        simVertex.position = simShape.simVertices[i].position + k3Position.back();

        // Reset the force and acceleration for the next iteration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }

    // Final re-calculation of forces for k4
    applyGravity(simShape);
    applyInternalForces(simShape);

//    #pragma omp parallel for
    for (size_t i = 0; i < simShape.simVertices.size(); ++i) {
        auto& simVertex = simShape.simVertices[i];
        simVertex.acceleration = simVertex.mass.inverse() * simVertex.force;

        // k4 is the slope at the endpoint using k3 adjustments
        k4Velocity.push_back(deltaT * simVertex.acceleration);
        k4Position.push_back(deltaT * simVertex.velocity);

        // Reset the force and acceleration for the next iteration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }

//    #pragma omp parallel for
    // Update velocities and positions with a weighted sum of k1, k2, k3, k4
    for (size_t i = 0; i < simShape.simVertices.size(); ++i) {
        auto& simVertex = simShape.simVertices[i];
        simVertex.velocity = simShape.simVertices[i].stepStartVelocity + (k1Velocity[i] + 2*k2Velocity[i] + 2*k3Velocity[i] + k4Velocity[i]) / 6;
        simVertex.position = simShape.simVertices[i].stepStartPosition + (k1Position[i] + 2*k2Position[i] + 2*k3Position[i] + k4Position[i]) / 6;

        // Resolve collision after the final update
        collisionDetection(simVertex, staticShape);

        // Reset the force and acceleration for the next iteration
        simVertex.force = Vector3d(0.0, 0.0, 0.0);
        simVertex.acceleration = Vector3d(0.0, 0.0, 0.0);
    }
}

double Simulation::calculateMaxPositionError(const SimShape& simShape1, const SimShape& simShape2) {
    double maxError = 0.0;

    for (size_t i = 0; i < simShape1.simVertices.size(); ++i) {
        // Calculate the difference in positions
        Vector3d diff = simShape1.simVertices[i].position - simShape2.simVertices[i].position;

        // Compute the Euclidean norm of the difference
        double error = diff.norm();

        // Update the maximum error
        maxError = std::max(maxError, error);
    }

    return maxError;
}

double Simulation::calculateMaxPositionErrorParallel(const SimShape& simShape1, const SimShape& simShape2) {
    const int numVertices = simShape1.simVertices.size();
    const int numTasks = QThread::idealThreadCount();
    QVector<QFuture<double>> futures;

    // Lambda to process a chunk of vertices and return the maximum error in that chunk
    auto processChunk = [&simShape1, &simShape2](int start, int end) -> double {
        double localMaxError = 0.0;
        for (int i = start; i < end; ++i) {
            Vector3d diff = simShape1.simVertices[i].position - simShape2.simVertices[i].position;
            double error = diff.norm();
            localMaxError = std::max(localMaxError, error);
        }
        return localMaxError;
    };

    // Divide the work into chunks and submit them as tasks
    int chunkSize = numVertices / numTasks;
    for (int i = 0; i < numTasks; ++i) {
        int start = i * chunkSize;
        int end = (i == numTasks - 1) ? numVertices : start + chunkSize; // Ensure the last chunk covers the rest
        futures.push_back(QtConcurrent::run(processChunk, start, end));
    }

    // Wait for all tasks to finish and find the maximum error across all chunks
    double maxError = 0.0;
    for (auto &future : futures) {
        future.waitForFinished();
        maxError = std::max(maxError, future.result());
    }

    return maxError;
}

//----------------------- Other stencil code -----------------------//
void Simulation::draw(Shader *shader)
{
    m_shape.draw(shader);
    m_shape_static.draw(shader);
    m_ground.draw(shader);
}

void Simulation::toggleWire()
{
    m_shape.toggleWireframe();
    m_shape_static.toggleWireframe();
}

void Simulation::initGround()
{
    std::vector<Vector3d> groundVerts;
    std::vector<Vector3i> groundFaces;
    groundVerts.emplace_back(-5, 0, -5);
    groundVerts.emplace_back(-5, 0, 5);
    groundVerts.emplace_back(5, 0, 5);
    groundVerts.emplace_back(5, 0, -5);
    groundFaces.emplace_back(0, 1, 2);
    groundFaces.emplace_back(0, 2, 3);
    m_ground.init(groundVerts, groundFaces);
}
