#pragma once

#include <array>
#include <expected>
#include <filesystem>
#include <memory>

#include <Necrosis/renderer/Renderer.h>
#include <Necrosis/renderer/Buffer.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

namespace Denoiser {

using SurfaceMesh = CGAL::Surface_mesh<CGAL::Simple_cartesian<double>::Point_3>;

class MeshProcessor;

class Model : public Necrosis::Renderable {
public:
    static std::expected<std::shared_ptr<Model>, std::string> loadFromFile(const std::filesystem::path& path);

    Model();
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;
    virtual ~Model();

    SurfaceMesh getCurrentMesh();
    void pushMesh(SurfaceMesh mesh);
    void updateGLBuffers(std::vector<float> vertices);

    void uiRender(MeshProcessor &processor);
    void render() const override;

private:
    // mesh stuff
    i8 _currentMesh;
    std::array<SurfaceMesh, 16> _meshes;
    float _sigma;
    bool _isApplyingNoise;

    // rendering stuff
    std::shared_ptr<Necrosis::VertexArray> _vao;
    std::shared_ptr<Necrosis::VertexBuffer> _vbo;
    std::shared_ptr<Necrosis::IndexBuffer> _ibo;
};
}
