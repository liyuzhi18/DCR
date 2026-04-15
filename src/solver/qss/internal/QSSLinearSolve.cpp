#include "QSSLinearSolve.hpp"

#include <Eigen/LU>
#include <Eigen/QR>

namespace dcr::solver::qss_model_detail {

dcr::base::Vector solve_linear(const dcr::base::Matrix& A,
                               const dcr::base::Vector& b) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) return lu.solve(b);
    }
    return A.colPivHouseholderQr().solve(b);
}

dcr::base::Matrix solve_linear_matrix(const dcr::base::Matrix& A,
                                      const dcr::base::Matrix& B) {
    if (A.rows() == A.cols()) {
        Eigen::FullPivLU<dcr::base::Matrix> lu(A);
        if (lu.rank() == A.rows()) return lu.solve(B);
    }
    return A.colPivHouseholderQr().solve(B);
}

} // namespace dcr::solver::qss_model_detail
