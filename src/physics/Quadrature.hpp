#pragma once

#include <vector>

struct QuadraturePoint {
    double point = 0.0;  // node
    double weight = 0.0; // weight
};

struct Quadrature {
    std::vector<QuadraturePoint> points;
};
