#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

using size_t = std::size_t;
using uint_8 = std::uint8_t;

#include "spatial.h"

namespace pray {
    class KDTreeElement; // forward declaration

    typedef enum { LEFT=-1, RIGHT=1, UNKNOWN=0 } PlaneSide;

    typedef enum { LEFT_ONLY=-1, RIGHT_ONLY=1, BOTH=0} TriangleSide;

    class KDTree : public SpatialDatabase {
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

        std::unique_ptr<KDTreeElement> root;
        std::vector<Triangle> triangles;
    protected:
        static const size_t TASK_DEPTH = 5;
        static constexpr size_t THRESHOLD = 32;

        static constexpr size_t MAX_DEPTH = 20;

        void createRopes(std::unique_ptr<KDTreeElement> &current, std::array<KDTreeElement*, 6> currentRopes);
        void optimizeRope(KDTreeElement* &rope, Face f, BoundingBox &bb);

    private:
        void build();

        std::unique_ptr<KDTreeElement> buildRecursive(std::vector<size_t> &triIndices, size_t depth);

        bool intersectRecursive(const std::unique_ptr<KDTreeElement> &current, const Ray &ray, size_t &triIdx, float &dist) const;


#if defined(ENABLE_AVX)
        __m256 intersectRecursive(const std::unique_ptr<KDTreeElement> &current,
                                  const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                                  const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ,
                                  __m256 rayMask, std::array<size_t, 8> &trsIdx, __m256 &distance) const;

#elif defined(ENABLE_SSE)
        __m128 intersectRecursive(const std::unique_ptr<KDTreeElement> &current,
                           const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                           const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ,
                           __m128 rayMask, std::vector<size_t> &trsIdx, __m128 &distance) const;
#endif

    };

    class KDTreeElement {
    public:

        BoundingBox boundingBox;

        std::vector<size_t> triangleIndices;

        std::unique_ptr<KDTreeElement> left;
        std::unique_ptr<KDTreeElement> right;

        SplitPlane splitPlane;
        PlaneSide pSide;

        std::array<KDTreeElement*, 6> ropes = {{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};

        size_t nSubNodes = 0;
        size_t maxDepthToLeaf = 0;
        size_t nSubLeafs = 0;
        size_t nSubEmptyLeafs = 0;

        KDTreeElement()
                : boundingBox({0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}),  triangleIndices(0), left(nullptr),
                  right(nullptr) {
        }

        KDTreeElement(BoundingBox boundingBox, std::unique_ptr<KDTreeElement> &left,
                      std::unique_ptr<KDTreeElement> &right)
                : boundingBox(boundingBox), triangleIndices(0), left(std::move(left)), right(std::move(right)) {}

        KDTreeElement(BoundingBox boundingBox, std::vector<size_t> triangleIndices)
                : boundingBox(boundingBox), triangleIndices(triangleIndices), left(nullptr), right(nullptr) {}

        KDTreeElement(BoundingBox boundingBox, std::unique_ptr<KDTreeElement> &left,
                      std::unique_ptr<KDTreeElement> &right, SplitPlane splitPlane, PlaneSide pSide)
                : boundingBox(boundingBox), triangleIndices(0), left(std::move(left)), right(std::move(right)),
                  splitPlane(splitPlane), pSide(pSide) {}

        KDTreeElement(BoundingBox boundingBox, std::vector<size_t> triangleIndices, SplitPlane splitPlane)
                : boundingBox(boundingBox), triangleIndices(triangleIndices), left(nullptr), right(nullptr), splitPlane(splitPlane) {}

        inline bool isLeaf() const {
            return !triangleIndices.empty() || (!left && !right);
        };

        inline void log() {
            assert(left.get() != nullptr && right.get() != nullptr);

            nSubNodes = left->nSubNodes + right->nSubNodes + 2;
            maxDepthToLeaf = std::max(left->maxDepthToLeaf, right->maxDepthToLeaf) + 1;
            nSubLeafs += left->isLeaf() ? 1 : left->nSubLeafs;
            nSubLeafs += right->isLeaf() ? 1 : right->nSubLeafs;
            nSubEmptyLeafs += left->isLeaf() ? (left->triangleIndices.empty() ? 1 : 0) : left->nSubEmptyLeafs;
            nSubEmptyLeafs += right->isLeaf() ? (right->triangleIndices.empty() ? 1 : 0) : right->nSubEmptyLeafs;
        }
    };
}
