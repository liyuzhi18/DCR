#pragma once

#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense> 

namespace dcr::base {

    using Real = double;
    using Index = int;

    using RealVec = std::vector<Real>;
    using IndexVec = std::vector<Index>;
    using String = std::string;

    using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

} 