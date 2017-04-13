#define STB_IMAGE_WRITE_IMPLEMENTATION

/* Dependency: image writing to disk: PNG, TGA, BMP
*  https://github.com/nothings/stb
*/
#include "stb_image_write.h"

#include <experimental/filesystem>
#include <iostream>
#include <string>
#include <chrono>

#include "options.h"
#include "pray.h"
#include "io.h"
#include "path_trace.cu.h"
#include "scene_preprocessor.h"

namespace fs = std::experimental::filesystem;

using Image = std::vector<uint8_t>;

int main(int argc, char *argv[]) {


    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <input.json> <output.bmp>\n";
        return 1;
    }

    fs::path inputFileName(argv[1]);
    fs::path outputFileName(argv[2]);

    auto totalStart = std::chrono::steady_clock::now();

    std::cout << "--------------------------------------------------------\nINPUT FILE: " << argv[1] << "\n\n";
    std::cout << "Reading scene description..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    pray::Scene scene;
    pray::readSceneDescription(inputFileName, scene);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Finished reading scene description. (" << std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count() << " ms)" << std::endl;

    std::cout << "\n";
    std::cout << "Scene information: Resolution: " << scene.width << "x" << scene.height
              << ", # Lights: " << scene.lights.size()
              << ", # Triangles: " << scene.triangles.size();
    if (scene.mode == pray::Mode::PATHTRACING) {
        std::cout << ", PATH TRACING: "
                  << "max depth: " << scene.maxDepth
                  << ", # samples: " << scene.numSamples;
    }
    std::cout << "\n" << std::endl;

    std::cout << "Preprocessing scene..." << std::endl;
    start = std::chrono::steady_clock::now();
    pray::preprocessScene(scene);
    end = std::chrono::steady_clock::now();
    std::cout << "Preprocessing time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    Image image = std::vector<uint8_t>(3 * scene.width * scene.height);

    if (scene.mode == pray::Mode::RAYTRACING) {
        std::cout << "\nRendering scene with ray tracing with up to " << omp_get_max_threads() << " threads ";
#if defined(ENABLE_AVX)
        std::cout << "and avx";
#elif defined(ENABLE_SSE)
        std::cout << "and sse";
#endif
        std::cout << "...";
    } else {
        std::cout << "Rendering scene with path tracing ";
        if (scene.useGPU) {
            std::cout << "on GPU...";
        } else {
            std::cout << "with up to " << omp_get_max_threads() << " threads ";
            std::cout << "...";
        }
    }

    std::cout << std::endl;

    start = std::chrono::steady_clock::now();
    pray::render(image, scene);
    end = std::chrono::steady_clock::now();
    std::cout << "Finished rendering. (" << std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count() << " ms)" << std::endl;

    std::cout << "Writing result image..." << std::endl;
    start = std::chrono::steady_clock::now();
    int write_error = stbi_write_bmp(outputFileName.c_str(), scene.width,
                                     scene.height, 3 , image.data()); // TODO: move to io?
    if (write_error == 0) {
        std::cerr << "stbi_write_bmp failed\n";
        return 1;
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Finished writing. (" << std::chrono::duration_cast<std::chrono::milliseconds>
            (end - start).count() << " ms)" << std::endl;

    auto totalEnd = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>
            (totalEnd - totalStart).count() << " ms\n" << std::endl;

    return 0;
}
