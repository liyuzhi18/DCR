#pragma once
#include <string>
#include "ConfigStructs.hpp"

namespace dcr::io {

    class ConfigLoader {
    public:
        static Config load(const std::string& filepath);
    };

}
