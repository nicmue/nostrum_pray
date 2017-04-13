#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>
#include <map>

using size_t = std::size_t;
using uint_8 = std::uint8_t;

#include "kd_tree.h"

namespace pray {
    class Event {
    public:
        enum class EventType {
            endingOnPlane = 0, lyingOnPlane = 1, startingOnPlane = 2
        };

        size_t triIdx; // triangle
        SplitPlane p;
        EventType type;

        Event() = default;
        ~Event() = default;

        Event(size_t et0, uint8_t k, float ee0, EventType type0) :
                triIdx(et0), type(type0) {
            p = SplitPlane(k, ee0);
        }

        inline bool operator<(const Event &e) const {
            return ((p.distance < e.p.distance) || (p.distance == e.p.distance && p.axis < e.p.axis) ||
                    (p.distance == e.p.distance && p.axis == e.p.axis && type < e.type));
        }
    };

    class KDTreeSAH : public KDTree {
    public:

        virtual void build(const std::vector<Triangle> &triangles) override;
        virtual void build(std::vector<Triangle> &&triangles) override;

    protected:
        void generateEvents(size_t triIdx, const BoundingBox &box, std::vector<Event> &events);
        std::unique_ptr<KDTreeElement> buildRecursive(std::vector<size_t> &triIndices, const BoundingBox &b, const std::vector<Event> &events,
                                                      const SplitPlane &prevp, size_t depth);
    private:
        float traversalCost = 5.f;
        float triangleIntersectionCost = 1.5f;

        // probability of hitting the subbox bSub given that the box b was hit
        inline float probBSubGivenB(const BoundingBox& bSub, const BoundingBox& b) const {
            float saBSub = bSub.surfaceArea();
            float saB = b.surfaceArea();
            return saBSub / saB;
        }

        // bias for the cost function s.t. it is reduced if NL or NR becomes zero
        inline float lambda(size_t nl, size_t nr, float pl, float pr) const {
            if((nl == 0 || nr == 0) && !(pl == 1 || pr == 1)) {
                return 0.8f;
            }

            return 1.0f;
        }

        // cost of a complete tree approximated using the cost cb of subdividing the BoundingBox b with a plane p
        inline float cost(float pl, float pr, size_t nl, size_t nr) const {
            return(lambda(nl, nr, pl, pr) * (traversalCost + triangleIntersectionCost * (pl * nl + pr * nr)));
        }

        void build();

        inline bool terminate(size_t N, float minCv) const {
            return minCv > triangleIntersectionCost * N;
        }

        void sah(const SplitPlane& p, const BoundingBox& b, size_t nl, size_t nr, size_t np, float& cp, PlaneSide& pside) const;
        void findPlane(size_t numTriangles, const std::vector<Event> &events, const BoundingBox& b,
                       SplitPlane& pest, float& cest, PlaneSide& psideest) const;
        void splitEvents(const std::vector<Event> &events, std::map<size_t, TriangleSide> &triSides, std::vector<Event> &lo, std::vector<Event> &ro);
        void classifyLeftRightBoth(const std::vector<size_t> &triIndices, const std::vector<Event> &events,
                                   const SplitPlane& pest, const PlaneSide& psideest, std::map<size_t, TriangleSide> &triSides);
        void splitAndGenerate(const std::vector<size_t> &triIndices, std::map<size_t, TriangleSide> &triSides,
                         const BoundingBox &vl, const BoundingBox &vr, const SplitPlane& pest,
                         std::vector<size_t> &tl, std::vector<size_t> &tr,
                         std::vector<Event> &bl, std::vector<Event> &br);
        void mergeStrains(std::vector<Event> &lo, std::vector<Event> &ro, std::vector<Event> &bl, std::vector<Event> &br,
                          std::vector<Event> &resultl, std::vector<Event> &resultr);
    };
}
