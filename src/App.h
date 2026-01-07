#pragma once

#include <memory>

#include <Necrosis/Engine.h>
#include <Necrosis/renderer/Renderer.h>
#include <Necrosis/InputManager.h>
#include <Necrosis/scene/Camera.h>
#include <Necrosis/renderer/Texture.h>
#include <Necrosis/renderer/FrameBuffer.h>

#include "Model.h"
#include "MeshProcessor.h"

namespace Denoiser {

class App : public Necrosis::Engine {
public:
    App();
    ~App() = default;

    void run();

private:
    std::unique_ptr<Necrosis::Renderer> _renderer;
    std::unique_ptr<Necrosis::EventManager> _eventManager;
    Necrosis::Camera _camera;
    Necrosis::InputManager _input;
    bool _isFramebufferHovered;

    std::shared_ptr<Necrosis::Shader> _shader = nullptr;
    std::shared_ptr<Model> _model = nullptr;
    MeshProcessor _processor;

    Necrosis::TextureID _texture;
    std::unique_ptr<Necrosis::Framebuffer> _framebuffer;

    void _setupMouseEventListeners();
};

}
