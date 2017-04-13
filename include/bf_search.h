#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

using size_t = std::size_t;
using uint_8 = std::uint8_t;

#include "spatial.h"

namespace pray {
    class BFSearch : public SpatialDatabase {
    public:

        virtual void build(const std::vector<Triangle> &triangles) override;
        virtual void build(std::vector<Triangle> &&triangles) override;
        virtual bool intersect(const Ray &ray, Triangle &triangle, float &dist) const override;
#if defined(ENABLE_AVX)
        virtual __m256 intersect(const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                                 const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ, const __m256 &rayMask,
                                 std::array<Triangle, 8> &trs, __m256 &distance) const override;
#elif defined(ENABLE_SSE)
        virtual __m128 intersect(const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                                 const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ, const __m128 &rayMask,
                                 std::vector<Triangle> &trs, __m128 &distance) const override;
#endif

        inline bool isEmpty() const override {
            return triangles.empty();
        }

        inline size_t size() const override {
            return triangles.size();
        }

    private:
        std::vector<Triangle> triangles;
    };
}

