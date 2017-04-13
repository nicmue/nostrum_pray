#pragma once

#include "options.h"
#include "geometry.h"

namespace pray {

#if defined(ENABLE_AVX)
    inline void cross_avx(const __m256 &aX, const __m256 &aY, const __m256 &aZ,
                          const __m256 &bX, const __m256 &bY, const __m256 &bZ,
                          __m256 &resultX, __m256 &resultY, __m256 &resultZ) {
        resultX = _mm256_sub_ps(_mm256_mul_ps(aY, bZ), _mm256_mul_ps(aZ, bY));
        resultY = _mm256_sub_ps(_mm256_mul_ps(aZ, bX), _mm256_mul_ps(aX, bZ));
        resultZ = _mm256_sub_ps(_mm256_mul_ps(aX, bY), _mm256_mul_ps(aY, bX));
    }

    inline __m256 dot_avx(const __m256 &aX, const __m256 &aY, const __m256 &aZ,
                          const __m256 &bX, const __m256 &bY, const __m256 &bZ) {
        return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(aX, bX), _mm256_mul_ps(aY, bY)), _mm256_mul_ps(aZ, bZ));
    }

    inline __m256 norm_avx(const __m256 &aX, const __m256 &aY, const __m256 &aZ) {
        return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(aX, aX), _mm256_mul_ps(aY, aY)), _mm256_mul_ps(aZ, aZ));
    }

    inline void normalize_avx(__m256 &aX, __m256 &aY, __m256 &aZ) {

        __m256 invNorm = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_sqrt_ps(norm_avx(aX, aY, aZ)));

        aX = _mm256_mul_ps(aX, invNorm);
        aY = _mm256_mul_ps(aY, invNorm);
        aZ = _mm256_mul_ps(aZ, invNorm);
    }

    __m256 intersectBB_avx(const BoundingBox &bb, const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                           const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ);

    __m256 intersectTriangle_avx(const Triangle &tri, const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                            const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ, const __m256 &rayMask,
                            __m256 &distance);


#elif defined(ENABLE_SSE)
    inline void cross_sse(const __m128 &aX, const __m128 &aY, const __m128 &aZ,
                          const __m128 &bX, const __m128 &bY, const __m128 &bZ,
                          __m128 &resultX, __m128 &resultY, __m128 &resultZ) {
        resultX = _mm_sub_ps(_mm_mul_ps(aY, bZ), _mm_mul_ps(aZ, bY));
        resultY = _mm_sub_ps(_mm_mul_ps(aZ, bX), _mm_mul_ps(aX, bZ));
        resultZ = _mm_sub_ps(_mm_mul_ps(aX, bY), _mm_mul_ps(aY, bX));
    }

    inline __m128 dot_sse(const __m128 &aX, const __m128 &aY, const __m128 &aZ,
                          const __m128 &bX, const __m128 &bY, const __m128 &bZ) {
        return _mm_add_ps(_mm_add_ps(_mm_mul_ps(aX, bX), _mm_mul_ps(aY, bY)), _mm_mul_ps(aZ, bZ));
    }

    inline __m128 norm_sse(const __m128 &aX, const __m128 &aY, const __m128 &aZ) {
        return _mm_add_ps(_mm_add_ps(_mm_mul_ps(aX, aX), _mm_mul_ps(aY, aY)), _mm_mul_ps(aZ, aZ));
    }

    inline void normalize_sse(__m128 &aX, __m128 &aY, __m128 &aZ) {

        __m128 invNorm = _mm_div_ps(_mm_set_ps1(1.f), _mm_sqrt_ps(norm_sse(aX, aY, aZ)));

        aX = _mm_mul_ps(aX, invNorm);
        aY = _mm_mul_ps(aY, invNorm);
        aZ = _mm_mul_ps(aZ, invNorm);
    }

    __m128 intersectBB_sse(const BoundingBox &bb, const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                           const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ);

    __m128 intersectTriangle_sse(const Triangle &tri, const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                            const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ, const __m128 &rayMask,
                            __m128 &distance);
#endif

}

