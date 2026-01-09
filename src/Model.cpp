#include "Model.h"

#include <random>
#include <algorithm>

#include <glad/glad.h>
#include <imgui/imgui.h>
#include <CGAL/boost/graph/IO/OBJ.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>

#include "MeshProcessor.h"

namespace Denoiser {

using namespace std::chrono_literals;

using Kernel = CGAL::Simple_cartesian<double>;
namespace PMP = CGAL::Polygon_mesh_processing;

Model::Model()
    : _currentMesh(-1), _listSize(-1), _sigma(0.001f)
    , _isApplyingNoise(false), _vao(nullptr), _vbo(nullptr), _ibo(nullptr) {}
Model::~Model() {}

std::expected<std::shared_ptr<Model>, std::string> Model::loadFromFile(const std::filesystem::path &path) {
    auto model = std::make_shared<Model>();
    if (!CGAL::IO::read_OBJ(path.string(), model->_meshes[0])) {
        return std::unexpected(std::format("Error reading OBJ file {}", path.string()));
    }

    model->_currentMesh = 0;
    model->_listSize = 0;

    CGAL::Polygon_mesh_processing::triangulate_faces(model->_meshes[0]);

    auto vnormals = model->_meshes[0].add_property_map<SurfaceMesh::Vertex_index, Kernel::Vector_3>(
        "v:normal", CGAL::NULL_VECTOR
    ).first;

    CGAL::Polygon_mesh_processing::compute_vertex_normals(model->_meshes[0], vnormals);

    std::vector<float> vertices;
    std::vector<u32> indices;

    for (auto v : model->_meshes[0].vertices()) {
        const auto &p = model->_meshes[0].point(v);
        const auto &n = vnormals[v];

        vertices.push_back(static_cast<float>(p.x()));
        vertices.push_back(static_cast<float>(p.y()));
        vertices.push_back(static_cast<float>(p.z()));

        vertices.push_back(static_cast<float>(n.x()));
        vertices.push_back(static_cast<float>(n.y()));
        vertices.push_back(static_cast<float>(n.z()));
    }

    for (auto f : model->_meshes[0].faces()) {
        for (auto v : CGAL::vertices_around_face(model->_meshes[0].halfedge(f), model->_meshes[0])) {
            indices.push_back(v);
        }
    }

    model->_vao = std::make_shared<Necrosis::VertexArray>();
    model->_vao->bind();

    model->_vbo = std::make_shared<Necrosis::VertexBuffer>(&vertices[0], vertices.size() * sizeof(float));
    model->_vbo->bind();

    model->_ibo = std::make_shared<Necrosis::IndexBuffer>(&indices[0], indices.size());
    model->_ibo->bind();

    Necrosis::VertexBufferLayout layout;
    layout.push(GL_FLOAT, 3); // pos
    layout.push(GL_FLOAT, 3); // norm
    // layout.push(GL_FLOAT, 2); // tex
    model->_vao->addBuffer(*(model->_vbo), layout);

    return model;
}

void Model::render() const {
    _vao->bind();
    _vbo->bind();
    _ibo->bind();
    glDrawElements(GL_TRIANGLES, _ibo->getCount(), GL_UNSIGNED_INT, nullptr);
}

void Model::uiRender(MeshProcessor &processor) {
    ImGui::Begin("Model");
        ImGui::Text("Vertices: %d", _currentMesh >= 0 ? _meshes[_currentMesh].num_vertices() : 0);
        ImGui::Text("Faces: %d", _currentMesh >= 0 ? _meshes[_currentMesh].num_faces() : 0);

        auto currentMesh = _currentMesh;

        // ImGui::Text("current mesh index: %d", _currentMesh);
        // ImGui::Text("Redo max index: %d", _listSize);
        if (currentMesh <= 0) { ImGui::BeginDisabled(); }
        if (ImGui::Button("Undo")) {
            _currentMesh--;
            processor.flattenMeshToGLBuffer(*this, _meshes[_currentMesh]);
        }
        if (currentMesh <= 0) { ImGui::EndDisabled(); }
        ImGui::SameLine();
        if (currentMesh >= _listSize) { ImGui::BeginDisabled(); }
        if (ImGui::Button("Redo")) {
            _currentMesh++;
            processor.flattenMeshToGLBuffer(*this, _meshes[_currentMesh]);
        }
        if (currentMesh >= _listSize) { ImGui::EndDisabled(); }

        ImGui::Separator();

        ImGui::SeparatorText("Gaussian Noise");
        ImGui::InputFloat("Noise std deviation", &_sigma, 0.001);

        bool isProcessing = processor.isProcessing(*this);

        if (isProcessing) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Apply Noise")) {
            processor.applyNoise(*this, _sigma);
        }
        if (isProcessing) {
            ImGui::EndDisabled();
            ImGui::Text("Currently processing...");
        }

        static int numIterations = 5;
        static float lambda = 0.5f;
        ImGui::SeparatorText("Laplacian Smoothing");
        ImGui::InputInt("number of iterations", &numIterations);
        ImGui::SliderFloat("lambda", &lambda, 0.f, 1.f);

        if (isProcessing) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Smooth")) {
            auto mesh = _meshes[_currentMesh];
            processor.applyLaplacianSmoothing(*this, lambda, numIterations);
        }

        if (isProcessing) {
            ImGui::EndDisabled();
            ImGui::Text("Currently processing...");
        }
    ImGui::End();
}

void Model::pushMesh(SurfaceMesh mesh) {
    assert(
        _currentMesh >= 0
        && static_cast<size_t>(_currentMesh) < _meshes.size()
        && "There is an error with the current Mesh index"
    );

    if (static_cast<size_t>(_currentMesh) == _meshes.size() - 1) {
        std::shift_left(_meshes.begin(), _meshes.end(), 1);
    }
    else {
        _currentMesh++;
        _listSize = _currentMesh;
    }
    _meshes[_currentMesh] = std::move(mesh);
}

SurfaceMesh Model::getCurrentMesh() {
    return _currentMesh >= 0 ? _meshes[_currentMesh] : SurfaceMesh();
}

void Model::updateGLBuffers(std::vector<float> vertices) {
    _vao->bind();
    _vbo->bind();
    // glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
}

}
