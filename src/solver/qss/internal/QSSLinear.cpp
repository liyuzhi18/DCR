#include "QSSLinear.hpp"

#include <Eigen/LU>
#include <Eigen/QR>

namespace dcr::solver {

dcr::base::Vector solve_linear(const dcr::base::Matrix& A,
                               const dcr::base::Vector& b) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) return lu.solve(b);
    }
    return A.colPivHouseholderQr().solve(b);
}

} // namespace dcr::solver
