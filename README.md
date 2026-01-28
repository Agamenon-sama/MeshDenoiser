# Mesh Denoiser



# Dependencies

- SDL3
- OpenGL
- glm
- Assimp
- CGAL
- Eigen

You also need a C++23 capable compiler and CMake to build the program.

# Build

```bash
mkdir build
mkdir build/release
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cd build/release
make -j
```

# License

GNU General Public License v3.0