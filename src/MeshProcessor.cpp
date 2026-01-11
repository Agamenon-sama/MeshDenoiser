#include "MeshProcessor.h"

#include <random>
#include <chrono>

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/boost/graph/properties_Surface_mesh.h>
#include <CGAL/subdivision_method_3.h>

#include <Necrosis/Window.h>

namespace Denoiser {

using namespace std::chrono;

using Kernel = CGAL::Simple_cartesian<double>;
using Point_3 = CGAL::Simple_cartesian<double>::Point_3;
using Vector_3 = CGAL::Simple_cartesian<double>::Vector_3;
namespace Subdivision = CGAL::Subdivision_method_3;

std::vector<float> createFlatBuffer(SurfaceMesh mesh) {
    auto vnormalsOpt = mesh.property_map<SurfaceMesh::Vertex_index, Kernel::Vector_3>("v:normal");

    // if we found normals in the optional, we use them, otherwise we generate new ones
    auto vnormals = vnormalsOpt.has_value()
     ? vnormalsOpt.value()
        : mesh.add_property_map<SurfaceMesh::Vertex_index, Kernel::Vector_3>(
            "v:normal", CGAL::NULL_VECTOR
        ).first;

    CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh, vnormals);

    std::vector<float> vertices;

    for (auto v : mesh.vertices()) {
        const auto &p = mesh.point(v);
        const auto &n = vnormals[v];

        vertices.push_back(static_cast<float>(p.x()));
        vertices.push_back(static_cast<float>(p.y()));
        vertices.push_back(static_cast<float>(p.z()));

        vertices.push_back(static_cast<float>(n.x()));
        vertices.push_back(static_cast<float>(n.y()));
        vertices.push_back(static_cast<float>(n.z()));
    }

    return vertices;
}

NoiseResult createNoisyMesh(SurfaceMesh mesh, float sigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.f, sigma);
    auto rng = std::bind(dist, gen);

    for (auto v: mesh.vertices()) {
        auto &p = mesh.point(v);

        p = Kernel::Point_3(p.x() + rng(), p.y() + rng(), p.z() + rng());
    }

    return {mesh, createFlatBuffer(mesh)};
}

LaplacianSmoothingResult smoothLaplacian(SurfaceMesh mesh, float lambda, int numIterations, std::string name = "") {
    try {
        std::vector<Point_3> nextPositions;
        nextPositions.reserve(mesh.number_of_vertices());

        for (int i = 0; i < numIterations; i++) {
            for (auto v : mesh.vertices()) {
                bool boundary = false;
                for (auto h : halfedges_around_source(v, mesh)) {
                    if (mesh.is_border(h)) {
                        boundary = true;
                        break;
                    }
                }
                if (boundary) {
                    // vertices on the boundary should apparently be left unchanged
                    nextPositions.push_back(mesh.point(v));
                    continue;
                }

                Vector_3 laplacian{0, 0, 0};
                int numNeighbours = 0;

                // sum of displacements from the vertex to the neighbours
                for (auto neighbour : CGAL::vertices_around_target(v, mesh)) {
                    laplacian += mesh.point(neighbour) - mesh.point(v);
                    numNeighbours++;
                }

                if (numNeighbours > 0) {
                    // displace the vertex by lambda * average_displacement
                    nextPositions.push_back(mesh.point(v) + lambda * (laplacian / numNeighbours));
                }
                else {
                    nextPositions.push_back(mesh.point(v));
                }
            }
        }

        int i = 0;
        for (auto v : mesh.vertices()) {
            mesh.point(v) = nextPositions[i];
            i++;
        }
    }
    catch (const CGAL::Assertion_exception &e) {
        slog::error("Failed to apply Laplacian smoothing: {}", e.what());
        Necrosis::Window::showWarningMessageBox(
            std::format(
                "Failed to apply Laplacian smoothing on '{}'.\nThe mesh is probably broken.",
                name
            )
        );
    }

    return {mesh, createFlatBuffer(mesh)};
}

SubdivisionResult loopSubdivision(SurfaceMesh mesh, std::string name = "") {
    try {
        Subdivision::Loop_subdivision(mesh);
    }
    catch (CGAL::Assertion_exception &e) {
        slog::warning("Failed to apply Loop Subdivision: {}", e.what());
        Necrosis::Window::showWarningMessageBox(
            std::format(
                "Failed to apply Loop subdivision on '{}'.\nThe mesh is probably broken.",
                name
            )
        );
    }

    return {mesh, createFlatBuffer(mesh)};
}

bool MeshProcessor::isProcessing(Model &model) const {
    return _noiseTasks.contains(&model)
            || _laplacianSmoothingTasks.contains(&model)
            || _flattenTasks.contains(&model)
            || _subdivisionTasks.contains(&model);
    ;
}

void MeshProcessor::applyNoise(Model &model, float sigma) {
    if (_noiseTasks.contains(&model)) { return; }

    _noiseTasks[&model] = std::async(std::launch::async, createNoisyMesh, model.getCurrentMesh(), sigma);
}

void MeshProcessor::applySubdivision(Model &model) {
    if (_subdivisionTasks.contains(&model)) { return; }

    _subdivisionTasks[&model] = std::async(
        std::launch::async, loopSubdivision, model.getCurrentMesh(), model.getName()
    );
}

void MeshProcessor::flattenMeshToGLBuffer(Model &model, SurfaceMesh &mesh) {
    if (_flattenTasks.contains(&model)) { return; }

    _flattenTasks[&model] = std::async(std::launch::async, createFlatBuffer, mesh);
}

void MeshProcessor::applyLaplacianSmoothing(Model &model, float lambda, int numIterations) {
    if (_laplacianSmoothingTasks.contains(&model)) { return; }

    _laplacianSmoothingTasks[&model] = std::async(
        std::launch::async, smoothLaplacian, model.getCurrentMesh(), lambda,
        numIterations, model.getName()
    );
}


void MeshProcessor::update() {
    for (auto it = _noiseTasks.begin(); it != _noiseTasks.end();) {
        auto &[model, task] = *it;
        if (task.valid() && task.wait_for(0s) == std::future_status::ready) {
            auto [mesh, vertices] = task.get();

            model->pushMesh(std::move(mesh));
            model->updateGLBuffers(vertices);

            it = _noiseTasks.erase(it);
        }
        else {
            ++it;
        }
    }

    for (auto it = _laplacianSmoothingTasks.begin(); it != _laplacianSmoothingTasks.end();) {
        auto &[model, task] = *it;
        if (task.valid() && task.wait_for(0s) == std::future_status::ready) {
            auto [mesh, vertices] = task.get();

            model->pushMesh(std::move(mesh));
            model->updateGLBuffers(vertices);

            it = _laplacianSmoothingTasks.erase(it);
        }
        else {
            ++it;
        }
    }

    for (auto it = _subdivisionTasks.begin(); it != _subdivisionTasks.end();) {
        auto &[model, task] = *it;
        if (task.valid() && task.wait_for(0s) == std::future_status::ready) {
            auto [mesh, vertices] = task.get();

            model->pushMesh(std::move(mesh));
            model->updateGLBuffers(vertices);

            it = _subdivisionTasks.erase(it);
        }
        else {
            ++it;
        }
    }

    for (auto it = _flattenTasks.begin(); it != _flattenTasks.end();) {
        auto &[model, task] = *it;
        if (task.valid() && task.wait_for(0s) == std::future_status::ready) {
            auto vertices = task.get();

            // model->pushMesh(std::move(mesh));
            model->updateGLBuffers(vertices);

            it = _flattenTasks.erase(it);
        }
        else {
            ++it;
        }
    }
}


}
