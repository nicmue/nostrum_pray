#include "gpu_util.h"

#include <queue>
#include <algorithm>

namespace pray {
    KDTreeGPU buildGPUTreeFromKDTree(const std::unique_ptr<SpatialDatabase> &spatial) {

        KDTree& kdTree = dynamic_cast<KDTree&>(*spatial);

        KDTreeGPU gpu;
        gpu.nodes.resize(kdTree.nnodes, KDTreeGPUNode());
        gpu.triangles = std::move(kdTree.triangles);
        gpu.triangleIndices.reserve(kdTree.size());

        int indexCounter = 0;
        int currentElement = 0;
        size_t currentTriIndex = 0;

        std::queue<KDTreeElement*> queue;
        queue.push(kdTree.root.get());
        while(!queue.empty()) {
            KDTreeElement* current = queue.front();
            queue.pop();

            gpu.nodes[currentElement].bb = current->boundingBox;
            gpu.nodes[currentElement].splitPlane = current->splitPlane;

            if (!current->isLeaf()) {
                if (current->left) {
                    gpu.nodes[currentElement].leftChildIndex = ++indexCounter;
                    queue.push(current->left.get());
                } else {
                    gpu.nodes[currentElement].leftChildIndex = -1;
                }
                if (current->right) {
                    gpu.nodes[currentElement].rightChildIndex = ++indexCounter;
                    queue.push(current->right.get());
                } else {
                    gpu.nodes[currentElement].rightChildIndex = -1;
                }
            } else {
                gpu.nodes[currentElement].isLeaf = true;
                gpu.nodes[currentElement].numTriangles = current->triangleIndices.size();
                gpu.nodes[currentElement].firstTriangleIdxIdx = currentTriIndex;

                for (int i = 0; i < current->triangleIndices.size(); i++) {
                    gpu.triangleIndices.push_back(current->triangleIndices[i]);
                    currentTriIndex++;
                }
            }
            currentElement++;
        }

        //gpuCreateRopes(gpu, 0, {-1,-1,-1,-1,-1,-1});
        return gpu;
    }

    void gpuOptimizeRope(KDTreeGPU &kdTree, int &ropeIdx, Face f, BoundingBox &bb) {
        if (ropeIdx == -1) {
            return;
        }
        while(!kdTree.nodes[ropeIdx].isLeaf) {
            uint8_t parallel = isFaceParallelToAxis(f, kdTree.nodes[ropeIdx].splitPlane.axis);
            if (parallel) {
                if (parallel == 1) {
                    ropeIdx = kdTree.nodes[ropeIdx].leftChildIndex;
                } else {
                    ropeIdx = kdTree.nodes[ropeIdx].rightChildIndex;
                }
            } else if (kdTree.nodes[ropeIdx].splitPlane.distance > bb.min[kdTree.nodes[ropeIdx].splitPlane.axis]) {
                ropeIdx = kdTree.nodes[ropeIdx].rightChildIndex;
            } else if (kdTree.nodes[ropeIdx].splitPlane.distance < bb.max[kdTree.nodes[ropeIdx].splitPlane.axis]) {
                ropeIdx = kdTree.nodes[ropeIdx].leftChildIndex;
            } else {
                break;
            }
        }
    }

    void gpuCreateRopes(KDTreeGPU &kdTree, int curNodeIdx, std::array<int, 6> currentRopes) {
        KDTreeGPUNode &currentNode = kdTree.nodes[curNodeIdx];
        if(currentNode.isLeaf) {
            std::copy(currentRopes.begin(), currentRopes.end(), currentNode.ropes);
            return;
        }
        for (Face f : allFaces) {
            gpuOptimizeRope(kdTree, currentRopes[f], f, currentNode.bb);
        }

        Face leftFace;
        Face rightFace;

        if(currentNode.splitPlane.axis == 0) {
            leftFace = Face::left;
            rightFace = Face::right;
        } else if (currentNode.splitPlane.axis == 1) {
            leftFace = Face::front;
            rightFace = Face::back;
        } else {
            leftFace = Face::top;
            rightFace = Face::bottom;
        }

        std::array<int, 6> ropesLeft = currentRopes;
        ropesLeft[rightFace] = currentNode.rightChildIndex;
        gpuCreateRopes(kdTree, currentNode.leftChildIndex, ropesLeft);

        std::array<int, 6> ropesRight = currentRopes;
        ropesRight[leftFace] = currentNode.leftChildIndex;
        gpuCreateRopes(kdTree, currentNode.rightChildIndex, ropesRight);
    }

    Vec3 colorDFS(const ColorTreeNode &node) {
        if (node.childs.empty()) {
            return node.color;
        }

        Vec3 colorSum(0.f, 0.f, 0.f);
        for (const ColorTreeNode &child : node.childs) {
            colorSum += colorDFS(child);
        }
        return (node.color * colorSum) / node.childs.size();
    }
}
