#pragma once

#include "pray.h"

#include <experimental/filesystem>

namespace pray
{
    void readSceneDescription(const std::experimental::filesystem::path &inputFileName, Scene &scene);
}