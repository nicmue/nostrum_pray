#include <limits>
#include <iostream>

#include "bf_search.h"
#include "geometry_simd.h"

static constexpr float INFTY = std::numeric_limits<float>::max();

namespace pray {
    void BFSearch::build(const std::vector<Triangle> &triangles) {
        std::cout << "\tBuilding bf search..." << std::endl;
        this->triangles = triangles;
        this->triangleCount = this->triangles.size();
        this->maxdepth = 0;
        this->nnodes = this->triangles.size();
    }

    void BFSearch::build(std::vector<Triangle> &&triangles) {
        std::cout << "\tBuilding bf search..." << std::endl;
        this->triangles = std::move(triangles);
        this->triangleCount = this->triangles.size();
        this->maxdepth = 0;
        this->nnodes = this->triangles.size();
    }

    bool BFSearch::intersect(const Ray &ray, Triangle &triangle, float &dist) const {
        bool intersected = false;
        dist = INFTY;

        for (auto &tri : triangles) {
            float currentDist;
            if (tri.intersect(ray, currentDist) && currentDist > 0 && currentDist < dist) {
                triangle = tri;
                intersected = true;
                dist = currentDist;
            }
        }

        return intersected;
    }

#ifdef ENABLE_AVX
    __m256 BFSearch::intersect(const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ, const __m256 &rayDirX,
                               const __m256 &rayDirY, const __m256 &rayDirZ, const __m256 &rayMask, std::array<Triangle, 8> &trs,
                               __m256 &distance) const {
        __m256 hit = _mm256_setzero_ps();
        __m256 minDist = _mm256_set1_ps(INFTY);

        for (size_t i = 0; i < triangles.size(); i++) {
            __m256 tmpDist;
            __m256 trIntersect = intersectTriangle_avx(triangles[i], rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, rayMask, tmpDist);
            __m256 check = _mm256_and_ps(trIntersect,
                                         _mm256_and_ps(
                                                 _mm256_cmp_ps(tmpDist, minDist, _CMP_LT_OQ),
                                                 _mm256_cmp_ps(tmpDist, _mm256_setzero_ps(), _CMP_GT_OQ)
                                         )
            );

            int hitMask = _mm256_movemask_ps(check);
            if (hitMask == 0)
            {
                continue;
            }

            minDist = _mm256_blendv_ps(minDist, tmpDist, check);

            if ((hitMask >> 0) & 0x1) trs[0] = triangles[i];
            if ((hitMask >> 1) & 0x1) trs[1] = triangles[i];
            if ((hitMask >> 2) & 0x1) trs[2] = triangles[i];
            if ((hitMask >> 3) & 0x1) trs[3] = triangles[i];
            if ((hitMask >> 4) & 0x1) trs[4] = triangles[i];
            if ((hitMask >> 5) & 0x1) trs[5] = triangles[i];
            if ((hitMask >> 6) & 0x1) trs[6] = triangles[i];
            if ((hitMask >> 7) & 0x1) trs[7] = triangles[i];

            hit = _mm256_or_ps(hit, check);
        }

        distance = minDist;
        return hit;
    }
#elif defined(ENABLE_SSE)
    __m128 BFSearch::intersect(const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ, const __m128 &rayDirX,
                               const __m128 &rayDirY, const __m128 &rayDirZ, const __m128 &rayMask, std::vector<Triangle> &trs,
                               __m128 &distance) const {
        __m128 hit = _mm_setzero_ps();
        __m128 minDist = _mm_set_ps1(INFTY);

        for (size_t i = 0; i < triangles.size(); i++) {
            __m128 tmpDist;
            __m128 trIntersect = intersectTriangle_sse(triangles[i], rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, rayMask, tmpDist);
            __m128 check = _mm_and_ps(trIntersect, _mm_and_ps(_mm_cmplt_ps(tmpDist, minDist), _mm_cmpgt_ps(tmpDist, _mm_setzero_ps())));

            int hitMask = _mm_movemask_ps(check);
            if (hitMask == 0)
            {
                continue;
            }

            __m128 d1 = _mm_andnot_ps(check, minDist);
            __m128 d2 = _mm_and_ps(check, tmpDist);
            minDist = _mm_add_ps(d1, d2);

            if ((hitMask >> 0) & 0x1) trs[0] = triangles[i];
            if ((hitMask >> 1) & 0x1) trs[1] = triangles[i];
            if ((hitMask >> 2) & 0x1) trs[2] = triangles[i];
            if ((hitMask >> 3) & 0x1) trs[3] = triangles[i];

            hit = _mm_or_ps(hit, check);
        }

        distance = minDist;
        return hit;
    }
#endif
}