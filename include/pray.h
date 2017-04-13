#pragma once

#include <vector>
#include <cstdint>
#include <memory>

#include "options.h"
#include "geometry.h"
#include "spatial.h"

using std::uint64_t;
using std::uint8_t;

using Image = std::vector<uint8_t>;

namespace pray {
    struct LightSource {
        Vec3 position;
        Vec3 color;

        LightSource() = default;

        LightSource(Vec3 _position, Vec3 _color) : position(_position), color(_color) {}
    };

    struct Scene {
        Mode mode;
        PtMode ptMode;
        bool useGPU = false;

        size_t maxDepth;
        size_t numSamples;

        uint64_t width;
        uint64_t height;

        uint64_t field_of_view;
        Vec3 camera_position;
        Vec3 camera_look;

        std::vector<LightSource> lights;
        std::vector<Triangle> triangles;
        std::unique_ptr<SpatialDatabase> triangleDB;

        Vec3 backgroundColor;

        std::vector<Ray> primRays;
    };

    void render(Image &image, const Scene &scene);

    void pathTrace(Image &image, const Scene &scene);
    void rayTrace(Image &image, const Scene &scene);

    Vec3 evaluateRay(size_t rayIdx, const Scene &scene);

#if defined(ENABLE_AVX)
    std::vector<Vec3> evaluateRay(const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                               const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ, const Scene &scene);

#elif defined(ENABLE_SSE)
    std::vector<Vec3> evaluateRay(const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                               const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ, const Scene &scene);
#endif


}