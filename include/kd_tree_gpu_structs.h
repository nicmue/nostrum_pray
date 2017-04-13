#pragma once

#include "geometry.h"

namespace pray {

    struct KDTreeGPUNode {

        int leftChildIndex;
        int rightChildIndex;

        bool isLeaf;

        size_t firstTriangleIdxIdx;
        size_t numTriangles;

        // Bounding Box
        BoundingBox bb;
        SplitPlane splitPlane;

        int ropes[6];
    };
}
