#include <limits>
#include <iostream>

#include "kd_tree.h"
#include "geometry_simd.h"

namespace pray {

    static constexpr float INFTY = std::numeric_limits<float>::max();

    void KDTree::build(const std::vector<Triangle> &triangles) {
        std::cout << "\tBuilding kd tree..." << std::endl;
        this->triangles = triangles;
        this->triangleCount = this->triangles.size();
        build();
    }

    void KDTree::build(std::vector<Triangle> &&triangles) {
        std::cout << "\tBuilding kd tree..." << std::endl;
        this->triangles = std::move(triangles);
        this->triangleCount = this->triangles.size();
        build();
    }

    void KDTree::build() {
        if (triangles.empty()) {
            root = std::make_unique<KDTreeElement>();
            return;
        }

        std::vector<size_t> triangleIndices(triangles.size());
        for (size_t i = 0; i < triangleIndices.size(); i++) {
            triangleIndices[i] = i;
        }

        #pragma omp parallel
        #pragma omp single nowait
        root = buildRecursive(triangleIndices, 0);

        nnodes = root->nSubNodes + 1;
        leafNodes = root->nSubLeafs;
        emptyLeafNodes = root->nSubEmptyLeafs;
        maxdepth = root->maxDepthToLeaf;

        //createRopes(root, {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    std::unique_ptr<KDTreeElement> KDTree::buildRecursive(std::vector<size_t> &triIndices, size_t depth) {
        assert(triIndices.size() > 0);

        BoundingBox nodeBB = triangles[triIndices[0]].boundingBox;
        for (size_t i = 1; i < triIndices.size(); i++) {
            nodeBB = nodeBB.expand(triangles[triIndices[i]].boundingBox);
        }

        uint8_t axis = nodeBB.getLongestAxis();

        if (triIndices.size() <= KDTree::THRESHOLD || depth >= MAX_DEPTH) {
            return std::make_unique<KDTreeElement>(nodeBB, triIndices);
        }

        std::nth_element(triIndices.begin(), triIndices.begin() + triIndices.size() / 2, triIndices.end(),
                         [&](size_t lhs, size_t rhs) {
                             return triangles[lhs].midPoint[axis] < triangles[rhs].midPoint[axis];
                         });

        std::vector<size_t> leftIndices(triIndices.begin(), triIndices.begin() + triIndices.size() / 2);
        std::vector<size_t> rightIndices(triIndices.begin() + triIndices.size() / 2, triIndices.end());

        std::unique_ptr<KDTreeElement> left, right;
        #pragma omp task shared(left) if (depth < TASK_DEPTH)
        left = buildRecursive(leftIndices, depth + 1);
        #pragma omp task shared(right) if (depth < TASK_DEPTH)
        right = buildRecursive(rightIndices, depth + 1);
        #pragma omp taskwait
        auto resultNode = std::make_unique<KDTreeElement>(nodeBB, left, right);

        resultNode->splitPlane.axis = axis;
        resultNode->splitPlane.distance = triangles[triIndices[triIndices.size() / 2]].midPoint[axis];

        resultNode->log();

        return resultNode;
    }

    void KDTree::optimizeRope(KDTreeElement* &rope, Face f, BoundingBox &bb) {
        if (rope == nullptr) {
            return;
        }

        while(!rope->isLeaf()) {
            uint8_t parallel = isFaceParallelToAxis(f, rope->splitPlane.axis);
            if (parallel) {
                if (parallel == 1) {
                    rope = rope->left.get();
                } else {
                    rope = rope->right.get();
                }
            } else if (rope->splitPlane.distance > bb.min[rope->splitPlane.axis]) {
                rope = rope->right.get();
            } else if (rope->splitPlane.distance < bb.max[rope->splitPlane.axis]) {
                rope = rope->left.get();
            } else {
                break;
            }
        }
    }

    void KDTree::createRopes(std::unique_ptr<KDTreeElement> &current, std::array<KDTreeElement*, 6> currentRopes) {
        if(current->isLeaf()) {
            current->ropes = currentRopes;
            return;
        }

        for (Face f : allFaces) {
            optimizeRope(currentRopes[f], f, current->boundingBox);
        }

        Face leftFace;
        Face rightFace;

        if(current->splitPlane.axis == 0) {
            leftFace = Face::left;
            rightFace = Face::right;
        } else if (current->splitPlane.axis == 1) {
            leftFace = Face::front;
            rightFace = Face::back;
        } else {
            leftFace = Face::top;
            rightFace = Face::bottom;
        }

        std::array<KDTreeElement*, 6> ropesLeft = currentRopes;
        ropesLeft[rightFace] = current->right.get();
        createRopes(current->left, ropesLeft);

        std::array<KDTreeElement*, 6> ropesRight = currentRopes;
        ropesRight[leftFace] = current->left.get();
        createRopes(current->right, ropesRight);
    }

    bool KDTree::intersect(const Ray &ray, Triangle &triangle, float &dist) const {
//        if (triangles.empty())
//        {
//            return false;
//        }
//
//        bool hit = false;
//        size_t minTriangleIdx = 0;
//
//        KDTreeElement *current = root.get();
//
//        float lambdaEntry, lambdaExit;
//        if (!current->boundingBox.intersect(ray, lambdaEntry, lambdaExit)) {
//            return false;
//        }
//
//        size_t lastRopeFace = 7;
//        while (lambdaEntry <= lambdaExit) {
//            Vec3 pEntry = ray.position + ray.direction * lambdaEntry;
//
//            while (!current->isLeaf()) {
//                    if ((current->pSide == PlaneSide::LEFT && pEntry[current->splitPlane.axis] <= current->splitPlane.distance)
//                        || (current->pSide == PlaneSide::RIGHT && pEntry[current->splitPlane.axis] < current->splitPlane.distance)) {
//                        current = current->left.get();
//                    } else {
//                        current = current->right.get();
//                    }
//            }
//
//            for (int i = 0; i < current->triangleIndices.size(); i++) {
//                auto curTriIdx = current->triangleIndices[i];
//                float tmpDist;
//                if (triangles[curTriIdx].intersect(ray, tmpDist)) {
//                    if (tmpDist <= lambdaExit && tmpDist >= lambdaEntry) {
//                        hit = true;
//                        lambdaExit = tmpDist;
//                        minTriangleIdx = curTriIdx;
//                    }
//                }
//            }
//
//            // Exit the leaf
//            float tmpEntry; // Not needed
//            current->boundingBox.intersect(ray, tmpEntry, lambdaEntry);
//
//            Vec3 outPoint = ray.position + ray.direction * lambdaEntry;
//
//            float minDist = INFTY;
//            size_t minDistFace = 7;
//            for (size_t i = 0; i < 3; i++) {
//                if (current->boundingBox.isPlanar(i) && lastRopeFace != 7) {
//                    minDistFace = lastRopeFace;
//                    break;
//                } else {
//                    float tmpDist = outPoint[i] - current->boundingBox.min[i];
//                    if (tmpDist < minDist) {
//                        minDist = tmpDist;
//                        minDistFace = i * 2;
//                    }
//
//                    tmpDist = current->boundingBox.max[i] - outPoint[i];
//                    if (tmpDist < minDist) {
//                        minDist = tmpDist;
//                        minDistFace = (i * 2) + 1;
//                    }
//                }
//            }
//
//            current = current->ropes[minDistFace];
//            lastRopeFace = minDistFace;
//            if (current == nullptr) {
//                break;
//            }
//        }
//
//        if (!hit) {
//            return false;
//        }
//
//        triangle = triangles[minTriangleIdx];
//        dist = lambdaExit;
//        return true;

//        if (triangles.empty())
//        {
//            return false;
//        }
//
        dist = INFTY;
        size_t triangleIdx;
        auto hit = intersectRecursive(root, ray, triangleIdx, dist);
        if (hit) triangle = triangles[triangleIdx];
        return hit;
    }

    bool KDTree::intersectRecursive(const std::unique_ptr<KDTreeElement> &current, const Ray &ray, size_t &triIdx,
                                    float &dist) const {
        if (!current->boundingBox.intersect(ray)) {
            return false;
        }

        if (current->left || current->right) {
            bool hitLeft = false;
            bool hitRight = false;
            if (current->left) {
                hitLeft = intersectRecursive(current->left, ray, triIdx, dist);
            }

            if (current->right) {
                hitRight = intersectRecursive(current->right, ray, triIdx, dist);
            }

            return hitLeft || hitRight;
        }

        bool hit = false;
        float minDist = INFTY;
        size_t minTriangleIdx = 0;
        for (size_t i = 0; i < current->triangleIndices.size(); i++) {
            auto curTriIdx = current->triangleIndices[i];
            float tmpDist;
            if (triangles[curTriIdx].intersect(ray, tmpDist)) {
                if (tmpDist < minDist && tmpDist > 0) {
                    hit = true;
                    minDist = tmpDist;
                    minTriangleIdx = curTriIdx;
                }
            }
        }

        if (hit && minDist < dist)
        {
            triIdx = minTriangleIdx;
            dist = minDist;
            return true;
        }
        return false;
    }

#if defined(ENABLE_AVX)
    __m256 KDTree::intersect(const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                             const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ,
                             const __m256 &rayMask, std::array<Triangle, 8> &trs, __m256 &distance) const {
        if (triangles.empty()) {
            return _mm256_setzero_ps();
        }

        distance = _mm256_set1_ps(INFTY);

        std::array<size_t, 8> trsIdx;
        auto hit = intersectRecursive(root, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, rayMask, trsIdx, distance);
        int mask = _mm256_movemask_ps(hit);

        for (size_t i = 0; i < 8; i++) {
            if ((mask >> i) & 0x1) trs[i] = triangles[trsIdx[i]];
        }
        return hit;
    }

    __m256 KDTree::intersectRecursive(const std::unique_ptr<KDTreeElement> &current,
                                      const __m256 &rayPosX, const __m256 &rayPosY, const __m256 &rayPosZ,
                                      const __m256 &rayDirX, const __m256 &rayDirY, const __m256 &rayDirZ,
                                      __m256 rayMask, std::array<size_t, 8> &trsIdx, __m256 &distance) const {

        // if no MSB are set there was no intersection -> return
        __m256 bbIntersect = intersectBB_avx(current->boundingBox, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ);
        if (_mm256_movemask_ps(bbIntersect) == 0x0) {
            // None of the rays hit the BB
            return bbIntersect;
        }

        // At least one ray hit the bb
        rayMask = _mm256_and_ps(rayMask, bbIntersect);
        if (current->left || current->right) {
            __m256 hitLeft = _mm256_setzero_ps();
            __m256 hitRight = _mm256_setzero_ps();

            if (current->left) {
                hitLeft = intersectRecursive(current->left, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ,
                                             rayMask, trsIdx, distance);
            }

            if (current->right) {
                hitRight = intersectRecursive(current->right, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ,
                                              rayMask, trsIdx, distance);
            }

            return _mm256_or_ps(hitLeft, hitRight);
        }

        __m256 hit = _mm256_setzero_ps();
        __m256 minDist = _mm256_set1_ps(INFTY);
        std::array<size_t, 8> minTriangleIdx;

        for (size_t i = 0; i < current->triangleIndices.size(); i++) {
            auto triIdx = current->triangleIndices[i];
            __m256 tmpDist;
            __m256 trIntersect = intersectTriangle_avx(triangles[triIdx], rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, rayMask, tmpDist);
            __m256 check = _mm256_and_ps(trIntersect, _mm256_and_ps(_mm256_cmp_ps(tmpDist, minDist, _CMP_LT_OQ), _mm256_cmp_ps(tmpDist, _mm256_setzero_ps(), _CMP_GT_OQ)));

            int hitMask = _mm256_movemask_ps(check);
            if (hitMask == 0)
            {
                continue;
            }

            minDist = _mm256_blendv_ps(minDist, tmpDist, check);

            for (size_t j = 0; j < 8; j++) {
                if ((hitMask >> j) & 0x1) minTriangleIdx[j] = triIdx;
            }

            hit = _mm256_or_ps(hit, check);
        }

        __m256 endCheck = _mm256_and_ps(hit, _mm256_cmp_ps(minDist, distance, _CMP_LT_OQ));

        int endCheckMask = _mm256_movemask_ps(endCheck);
        for (size_t i = 0; i < 8; i++) {
            if ((endCheckMask >> i) & 0x1) trsIdx[i] = minTriangleIdx[i];
        }

        distance = _mm256_blendv_ps(distance, minDist, endCheck);

        return endCheck;
    }

#elif defined(ENABLE_SSE)
    __m128 KDTree::intersect(const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                             const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ,
                             const __m128 &rayMask, std::vector<Triangle> &trs, __m128 &distance) const {
        if (triangles.empty()) {
            return _mm_setzero_ps();
        }

        distance = _mm_set_ps1(INFTY);

        std::vector<size_t> trsIdx(4);

        auto hit = intersectRecursive(root, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, rayMask, trsIdx, distance);
        int mask = _mm_movemask_ps(hit);

        for (size_t i = 0; i < 4; i++) {
            if ((mask >> i) & 0x1) trs[i] = triangles[trsIdx[i]];
        }

        return hit;
    }

    __m128 KDTree::intersectRecursive(const std::unique_ptr<KDTreeElement> &current,
                                         const __m128 &rayPosX, const __m128 &rayPosY, const __m128 &rayPosZ,
                                         const __m128 &rayDirX, const __m128 &rayDirY, const __m128 &rayDirZ,
                                         __m128 rayMask, std::vector<size_t> &trsIdx, __m128 &distance) const {

        // if no MSB are set there was no intersection -> return
        __m128 bbIntersect = intersectBB_sse(current->boundingBox, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ);
        if (_mm_movemask_ps(bbIntersect) == 0x0) {
            // None of the rays hit the BB
            return bbIntersect;
        }

        // At least one ray hit the bb
        rayMask = _mm_and_ps(rayMask, bbIntersect);
        if (current->left || current->right) {
            __m128 hitLeft = _mm_setzero_ps();
            __m128 hitRight = _mm_setzero_ps();

            if (current->left) {
                hitLeft = intersectRecursive(current->left, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ,
                                             rayMask, trsIdx, distance);
            }

            if (current->right) {
                hitRight = intersectRecursive(current->right, rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ,
                                              rayMask, trsIdx, distance);
            }

            return _mm_or_ps(hitLeft, hitRight);
        }

        __m128 hit = _mm_setzero_ps();
        __m128 minDist = _mm_set_ps1(INFTY);
        std::vector<size_t> minTriangleIdx(4);

        for (size_t i = 0; i < current->triangleIndices.size(); i++) {
            auto triIdx = current->triangleIndices[i];
            __m128 tmpDist;
            __m128 trIntersect = intersectTriangle_sse(triangles[triIdx], rayPosX, rayPosY, rayPosZ, rayDirX, rayDirY, rayDirZ, rayMask, tmpDist);
            __m128 check = _mm_and_ps(trIntersect, _mm_and_ps(_mm_cmplt_ps(tmpDist, minDist), _mm_cmpgt_ps(tmpDist, _mm_setzero_ps())));

            int hitMask = _mm_movemask_ps(check);
            if (hitMask == 0)
            {
                continue;
            }

            __m128 d1 = _mm_andnot_ps(check, minDist);
            __m128 d2 = _mm_and_ps(check, tmpDist);
            minDist = _mm_add_ps(d1, d2);

            for (size_t j = 0; j < 4; j++) {
                if ((hitMask >> j) & 0x1) minTriangleIdx[j] = triIdx;
            }

            hit = _mm_or_ps(hit, check);
        }

        __m128 endCheck = _mm_and_ps(hit, _mm_cmplt_ps(minDist, distance));

        int endCheckMask = _mm_movemask_ps(endCheck);
        for (size_t i = 0; i < 4; i++) {
            if ((endCheckMask >> i) & 0x1) trsIdx[i] = minTriangleIdx[i];
        }

        __m128 d1 = _mm_andnot_ps(endCheck, distance);
        __m128 d2 = _mm_and_ps(endCheck, minDist);
        distance = _mm_add_ps(d1, d2);
        return endCheck;
    }
#endif
    
}
