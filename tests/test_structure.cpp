#include <iostream>
#include <cmath>
#include <cassert>

// 1. Test Internal Include (Base Module)
// If this fails, dcr_base include paths are wrong in CMake.
#include "../src/base/Types.hpp"

// 2. Test External Dependency (Eigen via Base)
// If this fails, dcr_base is not exporting Eigen correctly.
#include <Eigen/Dense>

int main() {
    std::cout << "--- DCR Solver Structure Test ---\n";

    // A. Check Base Type Definitions
    dcr::base::Real val = 42.0;
    
    // rigorous check: ensure Real is actually double (8 bytes)
    if (sizeof(dcr::base::Real) != sizeof(double)) {
        std::cerr << "[FAIL] dcr::base::Real size mismatch! Expected 8 bytes.\n";
        return 1;
    }
    std::cout << "[PASS] Base Types: dcr::base::Real is valid (size=" << sizeof(val) << ").\n";

    // B. Check Eigen Linkage
    // If the linker setup is wrong, this line will fail to compile.
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    double norm_sq = v.squaredNorm(); // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14

    if (std::abs(norm_sq - 14.0) < 1e-9) {
        std::cout << "[PASS] Eigen Linkage: Vector math works (NormSq=" << norm_sq << ")\n";
    } else {
        std::cerr << "[FAIL] Eigen Math Check failed! Expected 14.0, got " << norm_sq << "\n";
        return 1;
    }

    std::cout << "--- All Checks Passed ---\n";
    return 0;
}