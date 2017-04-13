#include <limits>
#include <iostream>
#include <chrono>
#include <random>

#ifdef __GNUG__
#include <parallel/algorithm>
#endif

#include "kd_tree_hybrid.h"

namespace pray {

    void KDTreeHybrid::build(const std::vector<Triangle> &triangles) {
        std::cout << "\tBuilding kd tree (Hybrid)..." << std::endl;
        this->triangles = triangles;
        this->triangleCount = this->triangles.size();
        build();
    }

    void KDTreeHybrid::build(std::vector<Triangle> &&triangles) {
        std::cout << "\tBuilding kd tree (Hybrid)..." << std::endl;
        this->triangles = std::move(triangles);
        this->triangleCount = this->triangles.size();
        build();
    }

    void KDTreeHybrid::build() {
        if (triangles.empty())
        {
            root = std::make_unique<KDTreeElement>();
            return;
        }


        std::vector<size_t> triangleIndices(triangles.size());
        for (size_t i = 0; i < triangleIndices.size(); i++) {
            triangleIndices[i] = i;
        }
        #pragma omp single nowait
        root = buildNoSAHRecursive(triangleIndices, 0);

        nnodes = root->nSubNodes + 1;
        leafNodes = root->nSubLeafs;
        emptyLeafNodes = root->nSubEmptyLeafs;
        maxdepth = root->maxDepthToLeaf;

    }

    std::unique_ptr<KDTreeElement> KDTreeHybrid::buildNoSAHRecursive(std::vector<size_t> &triIndices, size_t depth) {
        assert(triIndices.size() > 0);
        BoundingBox nodeBB = triangles[triIndices[0]].boundingBox;
        for (size_t i = 1; i < triIndices.size(); i++) {
            nodeBB = nodeBB.expand(triangles[triIndices[i]].boundingBox);
        }

        if (depth >= 3) {
            std::vector<Event> events;
            events.reserve(triIndices.size() * 2 * 3);
            for (auto triIdx : triIndices) {
                generateEvents(triIdx, nodeBB, events);
            }

            #ifdef __GNUG__
                __gnu_parallel::sort(events.begin(), events.end());
            #else
                std::sort(events.begin(), events.end());
            #endif

            return buildRecursive(triIndices, nodeBB, events, SplitPlane(), 0);

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
        left = buildNoSAHRecursive(leftIndices, depth + 1);
        #pragma omp task shared(right) if (depth < TASK_DEPTH)
        right = buildNoSAHRecursive(rightIndices, depth + 1);
        #pragma omp taskwait

        auto resultNode = std::make_unique<KDTreeElement>(nodeBB, left, right);

        resultNode->splitPlane.axis = axis;
        resultNode->splitPlane.distance = triangles[triIndices[triIndices.size() / 2]].midPoint[axis];

        resultNode->log();

        return resultNode;
    }
}
