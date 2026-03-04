#include <cassert>
#include <iostream>
#include <vector>

#include "../src/physics/EEDF.hpp"
#include "../src/processes/AtomicProcess.hpp"
#include "../src/state/PlasmaState.hpp"
#include "../src/base/Types.hpp"

int main() {
    std::cout << "--- Testing AtomicExcitationProcess ---\n";

    dcr::state::PlasmaState plasma(1, 3);
    plasma.init_ne().setConstant(1.0e12);

    std::vector<double> energies;
    std::vector<double> weights;
    for (int i = 1; i <= 50; ++i) {
        energies.push_back(static_cast<double>(i));
        weights.push_back(1.0);
    }

    EEDFConfig cfg;
    EEDF eedf(10.0, cfg);
    eedf.normalize_on_grid(energies, weights);
    EEDFGridView grid(energies, weights, &eedf);

    dcr::base::Vector population = dcr::base::Vector::Zero(3);
    dcr::base::Matrix R = dcr::base::Matrix::Zero(3, 3);

    std::array<double, 8> params{};
    params[0] = 0.1;
    params[1] = 1.0;
    params[2] = 0.1;
    params[3] = 0.1;
    params[4] = 0.5;

    crm_detail::AtomicExcitationProcess proc(
        0, 1, 5.0, 2.0, 4.0, 0.0, 10.0, 0.5, 99, params
    );

    const double ne = plasma.electron_density_cm3();
    dcr::base::Matrix before = R;
    proc.apply(plasma, grid, population, R, nullptr);

    const double r_ex = R(1, 0) - before(1, 0);
    const double k_ex = (ne > 0.0) ? r_ex / ne : 0.0;
    std::cout << "Atomic excitation: r_up=" << r_ex << "  k_up=" << k_ex << "\n";
    assert(r_ex > 0.0);

    // Atomic ionization (f) and three-body recombination
    std::array<double, 4> f_params{{1.0, 2.0, 0.1, 0.1}};
    crm_detail::AtomicIonizationProcess ion_proc(1, 2, 10.0, 99, f_params, 2.0, 1.0);
    before = R;
    ion_proc.apply(plasma, grid, population, R, nullptr);
    const double r_ion = R(2, 1) - before(2, 1);
    const double k_ion = (ne > 0.0) ? r_ion / ne : 0.0;
    const double r_tbr = R(1, 2) - before(1, 2);
    const double k_tbr = (ne > 0.0) ? r_tbr / (ne * ne) : 0.0;
    std::cout << "Atomic ionization: r=" << r_ion << "  k_eii=" << k_ion << "\n";
    std::cout << "Three-body recomb: r=" << r_tbr << "  k_tbr=" << k_tbr << "\n";
    assert(r_ion > 0.0);

    // Atomic photo-recombination (p): should move population from ion to neutral.
    std::array<double, 7> p_params{{1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 20.0}};
    crm_detail::AtomicPhotoProcess photo_proc(0, 2, 5.0, 99, p_params, 2.0, 1.0);
    before = R;
    photo_proc.apply(plasma, grid, population, R, nullptr);
    const double r_rr = R(0, 2) - before(0, 2);
    const double k_rr = (ne > 0.0) ? r_rr / ne : 0.0;
    std::cout << "Photo-recombination: r=" << r_rr << "  k_rr=" << k_rr << "\n";
    assert(r_rr > 0.0);

    std::cout << "[PASS] AtomicExcitationProcess checks.\n";
    return 0;
}
