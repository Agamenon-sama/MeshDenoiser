#include <print>
#include <filesystem>

#include "App.h"

int main(int argc, char *argv[]) {
    std::filesystem::current_path("../..");

    std::println("MeshDenoiser");

    Denoiser::App app;
    app.run();

    return 0;
}
