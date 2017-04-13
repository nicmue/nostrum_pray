#include <limits>
#include <iostream>
#include <chrono>
#include <random>

#ifdef __GNUG__
#include <parallel/algorithm>
#endif


#include "kd_tree_sah.h"
#include "geometry_simd.h"

namespace pray {

    static constexpr float INFTY = std::numeric_limits<float>::max();

    void KDTreeSAH::build(const std::vector<Triangle> &triangles) {
        std::cout << "\tBuilding kd tree (SAH)..." << std::endl;
        this->triangles = triangles;
        this->triangleCount = this->triangles.size();
        build();
    }

    void KDTreeSAH::build(std::vector<Triangle> &&triangles) {
        std::cout << "\tBuilding kd tree (SAH)..." << std::endl;
        this->triangles = std::move(triangles);
        this->triangleCount = this->triangles.size();
        build();
    }

    void KDTreeSAH::build() {
        if (triangles.empty())
        {
            root = std::make_unique<KDTreeElement>();
            return;
        }

        auto start = std::chrono::steady_clock::now();
        std::vector<size_t> triangleIndices(triangles.size());
        BoundingBox initialBB = triangles[0].boundingBox;

        for (size_t i = 0; i < triangleIndices.size(); i++) {
            triangleIndices[i] = i;
            initialBB = initialBB.expand(triangles[i].boundingBox);
        }

        std::cout << "\tGenerating events..." << std::endl;
        std::vector<Event> events;
        events.reserve(triangleIndices.size() * 2 * 3);

        if (triangleIndices.size() <= SAMPLE_EVENTS_THRES) {
            for (auto triIdx : triangleIndices) {
                generateEvents(triIdx, initialBB, events);
            }
        } else {
            std::random_device r;
            std::default_random_engine e1(r());
            std::uniform_real_distribution<double> uniform_dist(0, 1);
            double prob = (1.0 * NUM_TRIANGLES_SAMPLING) / triangleIndices.size();
            std::cout << "\t\tSampling with " << prob << "%..." << std::endl;
            for (auto triIdx : triangleIndices) {
                if (prob < uniform_dist(e1)) {
                    generateEvents(triIdx, initialBB, events);
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        std::cout << "\tDone: " << events.size() << " Events, "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

#ifdef __GNUG__
        __gnu_parallel::sort(events.begin(), events.end());
#else
        std::sort(events.begin(), events.end());
#endif

        #pragma omp parallel
        #pragma omp single nowait
        root = buildRecursive(triangleIndices, initialBB, events, SplitPlane(), 0);

        nnodes = root->nSubNodes + 1;
        leafNodes = root->nSubLeafs;
        emptyLeafNodes = root->nSubEmptyLeafs;
        maxdepth = root->maxDepthToLeaf;

        //std::array<KDTreeElement*, 6> ropes{{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};
        //createRopes(root, ropes);
    }

    std::unique_ptr<KDTreeElement> KDTreeSAH::buildRecursive(std::vector<size_t> &triIndices, const BoundingBox &b, const std::vector<Event> &events,
                                                             const SplitPlane &prevp, size_t depth) {
        assert(depth < 100); // just as a protection for when the stopping criterion fails

        SplitPlane p;
        float Cp;
        PlaneSide pside;

        findPlane(triIndices.size(), events, b, p, Cp, pside);

        if (triIndices.size() <= KDTree::THRESHOLD || depth >= MAX_DEPTH || p == prevp|| terminate(triIndices.size(), Cp)) // NOT IN PAPER
        {
            return std::make_unique<KDTreeElement>(b, triIndices);
        }

        std::map<size_t, TriangleSide> triSides;
        classifyLeftRightBoth(triIndices, events, p, pside, triSides);

        std::vector<Event> lo, ro;
        lo.resize(events.size());
        ro.resize(events.size());
        splitEvents(events, triSides, lo, ro);

        BoundingBox VL, VR;
        b.split(p, VL, VR); // TODO: avoid doing this step twice

        std::vector<Event> bl, br;
        std::vector<size_t> tl, tr;
        splitAndGenerate(triIndices, triSides, VL, VR, p, tl, tr, bl, br);

        std::vector<Event> leftEvents;
        std::vector<Event> rightEvents;
        mergeStrains(lo, ro, bl, br, leftEvents, rightEvents);

        std::unique_ptr<KDTreeElement> left, right;
        #pragma omp task shared(left) if (depth < TASK_DEPTH)
        left = buildRecursive(tl, VL, leftEvents, p, depth+1);
        #pragma omp task shared(right) if (depth < TASK_DEPTH)
        right =  buildRecursive(tr, VR, rightEvents, p, depth+1);
        #pragma omp taskwait

        auto resultNode = std::make_unique<KDTreeElement>(b, left, right, p, pside);

        resultNode->log();

        return resultNode;
    }

    // SAH heuristic for computing the cost of splitting a BoundingBox b using a plane p
    void KDTreeSAH::sah(const SplitPlane& p, const BoundingBox& b, size_t nl, size_t nr, size_t np, float& costP, PlaneSide& pside) const {
        costP = INFTY;

        BoundingBox bl, br;
        b.split(p, bl, br);

        float pl, pr;
        pl = probBSubGivenB(bl, b);
        pr = probBSubGivenB(br, b);

        if (pl == 0 || pr == 0) return; // NOT IN PAPER
        if (b.max[p.axis] - b.min[p.axis] == 0) return; // NOT IN PAPER


        float costPl, costPr;
        costPl = cost(pl, pr, nl + np, nr);
        costPr = cost(pl, pr, nl, np + nr);

        if (costPl < costPr) {
            costP = costPl;
            pside = LEFT;
        } else {
            costP = costPr;
            pside = RIGHT;
        }
    }

    // best spliting plane using SAH heuristic
    void KDTreeSAH::findPlane(size_t numTriangles, const std::vector<Event> &events, const BoundingBox& b,
                              SplitPlane& pest, float& cest, PlaneSide& psideest) const {
        cest = INFTY;

        std::array<size_t, 3> Nlk{0, 0, 0}, Npk{0, 0, 0}, Nrk{numTriangles, numTriangles, numTriangles};

        for(size_t i = 0; i < events.size(); i++) {
            const SplitPlane &p = events[i].p;
            size_t pLyingOnPlane = 0, pStartingOnPlane = 0, pEndingOnPlane = 0;
            while(i < events.size() && events[i].p.axis == p.axis && events[i].p.distance == p.distance && events[i].type == Event::EventType::endingOnPlane) {
                pEndingOnPlane++;
                i++;
            }
            while(i < events.size() && events[i].p.axis  == p.axis && events[i].p.distance == p.distance && events[i].type == Event::EventType::lyingOnPlane) {
                pLyingOnPlane++;
                i++;
            }
            while(i < events.size() && events[i].p.axis  == p.axis && events[i].p.distance == p.distance && events[i].type == Event::EventType::startingOnPlane) {
                pStartingOnPlane++;
                i++;
            }
            Npk[p.axis] = pLyingOnPlane;
            Nrk[p.axis] -= pLyingOnPlane;
            Nrk[p.axis] -= pEndingOnPlane;

            float C;
            PlaneSide pside = UNKNOWN;
            sah(p, b, Nlk[p.axis], Nrk[p.axis], Npk[p.axis], C, pside);

            if(C < cest) {
                cest = C;
                pest = p;
                psideest = pside;
            }
            Nlk[p.axis] += pStartingOnPlane;
            Nlk[p.axis] += pLyingOnPlane;
            Npk[p.axis] = 0;
        }
    }

    void KDTreeSAH::classifyLeftRightBoth(const std::vector<size_t> &triIndices, const std::vector<Event> &events,
                                          const SplitPlane& pest, const PlaneSide& psideest, std::map<size_t, TriangleSide> &triSides)
    {
        for (auto &idx : triIndices)
        {
            triSides[idx] = TriangleSide::BOTH;
        }
        //triSides = std::vector<TriangleSide>(triangles.size(), TriangleSide::BOTH);

        for(size_t i = 0; i < events.size(); i++) {
            if (events[i].type == Event::EventType::endingOnPlane && events[i].p.axis == pest.axis && events[i].p.distance <= pest.distance) {
                triSides[events[i].triIdx] = TriangleSide::LEFT_ONLY;
            } else if (events[i].type == Event::EventType::startingOnPlane && events[i].p.axis == pest.axis && events[i].p.distance >= pest.distance) {
                triSides[events[i].triIdx] = TriangleSide::RIGHT_ONLY;
            } else if (events[i].type == Event::EventType::lyingOnPlane && events[i].p.axis == pest.axis) {
                if (events[i].p.distance < pest.distance || (events[i].p.distance == pest.distance && psideest == PlaneSide::LEFT)) {
                    triSides[events[i].triIdx] = TriangleSide::LEFT_ONLY;
                } else if (events[i].p.distance > pest.distance || (events[i].p.distance == pest.distance && psideest == PlaneSide::RIGHT)) {
                    triSides[events[i].triIdx] = TriangleSide::RIGHT_ONLY;
                } else {
                    assert(false);
                }
            }
        }
    }

    void KDTreeSAH::splitEvents(const std::vector<Event> &events, std::map<size_t, TriangleSide> &triSides, std::vector<Event> &lo, std::vector<Event> &ro) {
        size_t leftCtr = 0, rightCtr = 0;
        for (size_t i = 0; i < events.size(); i++) {
            if (triSides[events[i].triIdx] == TriangleSide::LEFT_ONLY) {
                lo[leftCtr] = events[i];
                leftCtr++;
            } else if (triSides[events[i].triIdx] == TriangleSide::RIGHT_ONLY) {
                ro[rightCtr] = events[i];
                rightCtr++;
            }
        }
        lo.resize(leftCtr);
        ro.resize(rightCtr);
    }

    void KDTreeSAH::splitAndGenerate(const std::vector<size_t> &triIndices, std::map<size_t, TriangleSide> &triSides,
                                   const BoundingBox &vl, const BoundingBox &vr, const SplitPlane& pest,
                                   std::vector<size_t> &tl, std::vector<size_t> &tr,
                                   std::vector<Event> &bl, std::vector<Event> &br) {
        for (size_t i = 0; i < triIndices.size(); i++)
        {
            if (triSides[triIndices[i]] == TriangleSide::LEFT_ONLY) {
                tl.push_back(triIndices[i]);
            } else if (triSides[triIndices[i]] == TriangleSide::RIGHT_ONLY) {
                tr.push_back(triIndices[i]);
            } else {
                // triSides[triIndices[i]] == TriangleSide::BOTH)

                generateEvents(triIndices[i], vl, bl);
                generateEvents(triIndices[i], vr, br);

                tl.push_back(triIndices[i]);
                tr.push_back(triIndices[i]);
            }
        }
    }

    void KDTreeSAH::generateEvents(size_t triIdx, const BoundingBox &box, std::vector<Event> &events) {
        const Triangle &tri = triangles[triIdx];
        BoundingBox eventBox = tri.boundingBox.clip(box);

        for (uint8_t k = 0; k < 3; k++) {
            if (eventBox.isPlanar(k)) {
                events.push_back(Event(triIdx, k, eventBox.min[k], Event::EventType::lyingOnPlane));
            } else {
                events.push_back(Event(triIdx, k, eventBox.min[k], Event::EventType::startingOnPlane));
                events.push_back(Event(triIdx, k, eventBox.max[k], Event::EventType::endingOnPlane));
            }
        }
    }

    void KDTreeSAH::mergeStrains(std::vector<Event> &lo, std::vector<Event> &ro, std::vector<Event> &bl, std::vector<Event> &br,
                                 std::vector<Event> &resultl, std::vector<Event> &resultr) {
        std::sort(bl.begin(), bl.end());
        std::sort(br.begin(), br.end());

        resultl.resize(bl.size() + lo.size());
        std::merge(lo.begin(), lo.end(), bl.begin(), bl.end(), resultl.begin());

        resultr.resize(br.size() + ro.size());
        std::merge(ro.begin(), ro.end(), br.begin(), br.end(), resultr.begin());
    }
}
