#pragma once

#include <unordered_map>
#include <future>

#include "Model.h"

namespace Denoiser {
struct NoiseResult {
    SurfaceMesh mesh;
    std::vector<float> vertexData;
};

struct LaplacianSmoothingResult {
    SurfaceMesh mesh;
    std::vector<float> vertexData;
};

class MeshProcessor {
public:
    MeshProcessor() = default;

    void update();

    void applyNoise(Model &model, float sigma);
    void applyLaplacianSmoothing(Model &model, float lambda, int numIterations);
    void flattenMeshToGLBuffer(Model &model, SurfaceMesh &mesh);

    bool isProcessing(Model &model) const;


private:
    // WARNING: dangling pointer risk
    std::unordered_map<Model*, std::future<NoiseResult>> _noiseTasks;
    std::unordered_map<Model*, std::future<LaplacianSmoothingResult>> _laplacianSmoothingTasks;
    std::unordered_map<Model*, std::future<std::vector<float>>> _flattenTasks;
};
}
