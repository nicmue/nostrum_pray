#pragma once

#include <vector>

#include "options.h"
#include "geometry.h"

namespace pray
{
    class SpatialDatabase {
    public:
        virtual void build(const std::vector<Triangle> &triangles) = 0;
        virtual void build(std::vector<Triangle> &&triangles) = 0;


        virtual bool intersect(const Ray &ray, Triangle &triangle, float &dist) const = 0;

        size_t maxdepth; // DEBUG ONLY
        size_t nnodes; // DEBUG ONLY
        size_t leafNodes;
        size_t emptyLeafNodes; // DEBUG ONLY
        size_t triangleCount;

        SpatialDatabase() : maxdepth(0), nnodes(0), emptyLeafNodes(0), leafNodes(0) {}

#if defined(ENABLE_AVX)
        virtual __m256 intersect(const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                 const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ, const __m256 &rayMask,
                                 std::array<Triangle, 8> &trs, __m256 &distance) const = 0;

#elif defined(ENABLE_SSE)
        virtual __m128 intersect(const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                 const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ, const __m128 &rayMask,
                                 std::vector<Triangle> &trs, __m128 &distance) const = 0;
#endif

        virtual bool isEmpty() const = 0;
        virtual size_t size() const = 0;
    };
}