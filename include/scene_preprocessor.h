#pragma once

namespace pray
{
    struct Scene;
    void preprocessScene(Scene &scene);

    void initialiseTriangles(Scene &scene);
    void calculatePrimRays(Scene &scene);

}


