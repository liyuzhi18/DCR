#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../src/physics/EEDF.hpp"
#include "../src/processes/MolecularProcess.hpp"
#include "../src/state/PlasmaState.hpp"
#include "../src/base/Types.hpp"

int main() {
    std::cout << "--- Testing Molecular Processes ---\n";

    dcr::state::PlasmaState plasma(1, 6);
    plasma.init_ne().setConstant(1.0e12);
    plasma.init_Te().setConstant(5.0);
    // MCX integrates over ion temperature, so set a non-zero Ti.
    plasma.init_Ti().setConstant(5.0);

    std::vector<double> energies;
    std::vector<double> weights;
    for (int i = 1; i <= 100; ++i) {
        energies.push_back(static_cast<double>(i));
        weights.push_back(1.0);
    }

    EEDFConfig cfg;
    EEDF eedf(10.0, cfg);
    eedf.normalize_on_grid(energies, weights);
    EEDFGridView grid(energies, weights, &eedf);

    dcr::base::Vector population = dcr::base::Vector::Zero(6);
    // Provide reactant densities for two-body molecular processes (e.g., MCX).
    population(0) = 1.0e12;
    population(1) = 5.0e11;
    population(2) = 2.0e11;
    population(3) = 1.0e11;
    population(4) = 5.0e10;
    dcr::base::Matrix R = dcr::base::Matrix::Zero(6, 6);

    // Molecular excitation (ev)
    std::array<double, 6> ev_params{};
    ev_params[0] = 1.0;
    ev_params[1] = 0.5;
    ev_params[2] = 0.1;
    ev_params[3] = 0.05;
    ev_params[4] = 0.01;
    ev_params[5] = 0.005;

    crm_detail::MolecularExcitationProcess ev_proc(0, 1, 5.0, ev_params);
    const double ne = plasma.electron_density_cm3();
    dcr::base::Matrix before = R;
    ev_proc.apply(plasma, grid, population, R, nullptr);

    const double r_ev = R(1, 0) - before(1, 0);
    const double k_ev = (ne > 0.0) ? r_ev / ne : 0.0;
    std::cout << "Molecular ev: r=" << r_ev << "  k=" << k_ev << "\n";
    assert(r_ev > 0.0);

    // Molecular radiative attachment (ra)
    crm_detail::MolecularRAProcess ra_proc(0, 1, 1.0);
    before = R;
    ra_proc.apply(plasma, grid, population, R, nullptr);
    const double r_ra = R(1, 0) - before(1, 0);
    const double k_ra = (ne > 0.0) ? r_ra / ne : 0.0;
    std::cout << "Molecular ra: r=" << r_ra << "  k=" << k_ra << "\n";
    assert(r_ra > 0.0);

    // Molecular dissociative attachment (da)
    crm_detail::MolecularDAProcess da_proc(1, 2, 3, 5.0, 1.0);
    before = R;
    da_proc.apply(plasma, grid, population, R, nullptr);
    const double r_da = R(2, 1) - before(2, 1);
    std::cout << "Molecular da: r=" << r_da << "\n";
    assert(r_da > 0.0);

    // Molecular dissociative recombination (dr)
    std::array<double, 6> dr_params{};
    dr_params[0] = 1.0;
    dr_params[1] = 0.1;
    dr_params[2] = 0.5;
    crm_detail::MolecularDRProcess dr_proc(2, 0, 3, dr_params);
    before = R;
    dr_proc.apply(plasma, grid, population, R, nullptr);
    const double r_dr = R(0, 2) - before(0, 2);
    std::cout << "Molecular dr: r=" << r_dr << "\n";
    assert(r_dr > 0.0);

    // Molecular ionization (mi)
    std::array<double, 7> mi_params{};
    mi_params[0] = 1.0;
    mi_params[1] = 0.5;
    mi_params[2] = 0.2;
    mi_params[3] = 0.1;
    mi_params[4] = 0.05;
    mi_params[5] = 0.02;
    mi_params[6] = 1.0;
    crm_detail::MolecularMIProcess mi_proc(3, 4, 15.0, mi_params);
    before = R;
    mi_proc.apply(plasma, grid, population, R, nullptr);

    const double r_mi = R(4, 3) - before(4, 3);
    const double r_mi_branch = R(5, 3) - before(5, 3);
    const double k_mi = (ne > 0.0) ? r_mi / ne : 0.0;
    std::cout << "Molecular mi: r(v0)=" << r_mi
              << " r(v1)=" << r_mi_branch
              << "  k=" << k_mi << "\n";
    assert(r_mi > 0.0);
    assert(r_mi_branch > 0.0);

    // Molecular dissociation (ed) - electron impact (flag 99)
    std::array<double, 11> ed_params{};
    ed_params[0] = 1.0;
    ed_params[1] = 1.0;
    ed_params[2] = 1.0;
    ed_params[3] = 1.0;
    crm_detail::MolecularEDProcess ed_e_proc(3, -1, {4}, 99, ed_params, nullptr);
    before = R;
    ed_e_proc.apply(plasma, grid, population, R, nullptr);
    const double r_ed_e = R(4, 3) - before(4, 3);
    std::cout << "Molecular ed (e-): r=" << r_ed_e << "\n";
    assert(r_ed_e > 0.0);

    // Molecular charge exchange (mcx)
    std::vector<QuadraturePoint> quad;
    quad.push_back({0.5, 0.5});
    quad.push_back({1.5, 0.5});

    std::array<double, 11> mcx_params{};
    mcx_params[0] = 1.0;
    mcx_params[1] = 0.5;
    mcx_params[2] = 0.5;
    mcx_params[3] = 1.0;
    mcx_params[4] = 0.1;
    mcx_params[5] = 0.0;

    // Molecular dissociation (ed) - ion impact (flag 98)
    std::array<double, 11> ed_i_params{};
    crm_detail::MolecularEDProcess ed_i_proc(0, 1, {2, 3}, 98, ed_i_params, &quad);
    before = R;
    ed_i_proc.apply(plasma, grid, population, R, nullptr);
    const double r_ed_i = R(2, 0) - before(2, 0);
    std::cout << "Molecular ed (ion): r=" << r_ed_i << "\n";
    assert(r_ed_i > 0.0);

    crm_detail::MolecularMCXProcess mcx_proc(0, 1, 1, 2, 99, mcx_params, 1.0, &quad, 0.5, 1);
    before = R;
    mcx_proc.apply(plasma, grid, population, R, nullptr);

    const double r_mcx = R(1, 0) - before(1, 0);
    std::cout << "Molecular mcx: r=" << r_mcx << "\n";
    assert(r_mcx > 0.0);

    // Molecular dissociation (de) - temperature fit
    std::array<double, 6> de_params{};
    crm_detail::MolecularDEProcess de_proc(4, 0, 1, 0.0, de_params);
    before = R;
    de_proc.apply(plasma, grid, population, R, nullptr);
    const double r_de = R(0, 4) - before(0, 4);
    std::cout << "Molecular de: r=" << r_de << "\n";
    assert(r_de > 0.0);

    // Molecular MIDE
    auto pool = std::make_shared<crm_detail::MolecularMIDEPool>();
    std::array<double, 3> mide_params{{1.0, 0.5, 1.0}};
    crm_detail::MolecularMIDEProcess mide_proc(0, 1, 2, 5.0, mide_params, pool);
    before = R;
    mide_proc.apply(plasma, grid, population, R, nullptr);
    const double r_mide = R(1, 0) - before(1, 0);
    std::cout << "Molecular mide: r=" << r_mide << "\n";
    assert(r_mide > 0.0);

    std::cout << "[PASS] Molecular process checks.\n";
    return 0;
}
