#include "../src/atomic/AtomicData.hpp"
#include "../src/io/ConfigLoader.hpp"
#include "../src/physics/EEDF.hpp"
#include "../src/physics/EnergyGrid.hpp"
#include "../src/state/PlasmaState.hpp"

#include <H5Cpp.h>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

dcr::base::Vector read_last_vector(const H5::H5File& file, const std::string& name) {
    const H5::DataSet ds = file.openDataSet(name);
    const H5::DataSpace space = ds.getSpace();
    hsize_t dims[2] = {0, 0};
    space.getSimpleExtentDims(dims);
    std::vector<double> all(static_cast<size_t>(dims[0] * dims[1]), 0.0);
    ds.read(all.data(), H5::PredType::NATIVE_DOUBLE);
    dcr::base::Vector out(static_cast<int>(dims[1]));
    const size_t row = static_cast<size_t>(dims[0] - 1);
    for (hsize_t j = 0; j < dims[1]; ++j) {
        out(static_cast<int>(j)) = all[row * static_cast<size_t>(dims[1]) + static_cast<size_t>(j)];
    }
    return out;
}

dcr::base::Matrix read_matrix(const H5::H5File& file, const std::string& name) {
    const H5::DataSet ds = file.openDataSet(name);
    const H5::DataSpace space = ds.getSpace();
    hsize_t dims[2] = {0, 0};
    space.getSimpleExtentDims(dims);
    std::vector<double> all(static_cast<size_t>(dims[0] * dims[1]), 0.0);
    ds.read(all.data(), H5::PredType::NATIVE_DOUBLE);
    dcr::base::Matrix out(static_cast<int>(dims[0]), static_cast<int>(dims[1]));
    for (hsize_t i = 0; i < dims[0]; ++i) {
        for (hsize_t j = 0; j < dims[1]; ++j) {
            out(static_cast<int>(i), static_cast<int>(j)) =
                all[static_cast<size_t>(i * dims[1] + j)];
        }
    }
    return out;
}

dcr::base::Vector read_vector(const H5::H5File& file, const std::string& name) {
    const H5::DataSet ds = file.openDataSet(name);
    const H5::DataSpace space = ds.getSpace();
    hsize_t dims[1] = {0};
    space.getSimpleExtentDims(dims);
    std::vector<double> all(static_cast<size_t>(dims[0]), 0.0);
    ds.read(all.data(), H5::PredType::NATIVE_DOUBLE);
    dcr::base::Vector out(static_cast<int>(dims[0]));
    for (hsize_t i = 0; i < dims[0]; ++i) out(static_cast<int>(i)) = all[static_cast<size_t>(i)];
    return out;
}

double read_last_scalar(const H5::H5File& file, const std::string& name) {
    const H5::DataSet ds = file.openDataSet(name);
    const H5::DataSpace space = ds.getSpace();
    hsize_t dims[1] = {0};
    space.getSimpleExtentDims(dims);
    std::vector<double> all(static_cast<size_t>(dims[0]), 0.0);
    ds.read(all.data(), H5::PredType::NATIVE_DOUBLE);
    return all.empty() ? 0.0 : all.back();
}

double quasineutral_ne(const dcr::base::Vector& n,
                       const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double ne = 0.0;
    const int count = std::min<int>(n.size(), static_cast<int>(levels.size()));
    for (int i = 0; i < count; ++i) {
        const int z = levels[static_cast<size_t>(i)].charge;
        if (z != 0) ne += static_cast<double>(z) * std::max(n(i), 0.0);
    }
    return std::max(0.0, ne);
}

double nuclei_total(const dcr::base::Vector& n,
                    const std::vector<dcr::atomic::EnergyLevel>& levels) {
    double total = 0.0;
    const int count = std::min<int>(n.size(), static_cast<int>(levels.size()));
    for (int i = 0; i < count; ++i) {
        total += static_cast<double>(std::max(1, levels[static_cast<size_t>(i)].atomicity)) *
                 std::max(n(i), 0.0);
    }
    return total;
}

double group_sum(const dcr::base::Vector& n,
                 const std::vector<dcr::atomic::EnergyLevel>& levels,
                 dcr::atomic::SpeciesType type,
                 int charge,
                 int min_atomicity = 1,
                 int max_atomicity = 100) {
    double out = 0.0;
    const int count = std::min<int>(n.size(), static_cast<int>(levels.size()));
    for (int i = 0; i < count; ++i) {
        const auto& level = levels[static_cast<size_t>(i)];
        if (level.type == type && level.charge == charge &&
            level.atomicity >= min_atomicity && level.atomicity <= max_atomicity) {
            out += std::max(n(i), 0.0);
        }
    }
    return out;
}

void redirect_h2plus_dr_products_to_ground(const dcr::io::Config& config,
                                           const std::vector<dcr::atomic::EnergyLevel>& levels,
                                           dcr::base::Matrix& rates) {
    if (!config.numerics.disable_h2plus_dr) return;
    if (rates.rows() != rates.cols()) return;
    const int total_states = static_cast<int>(levels.size());
    if (rates.rows() != total_states) return;

    int ground_row = -1;
    for (int i = 0; i < total_states; ++i) {
        const auto& level = levels[static_cast<size_t>(i)];
        if (level.type == dcr::atomic::SpeciesType::Atom &&
            level.charge == 0 &&
            level.internal_id == 1) {
            ground_row = i;
            break;
        }
    }
    if (ground_row < 0) return;

    for (int col = 0; col < total_states; ++col) {
        const auto& source = levels[static_cast<size_t>(col)];
        if (source.atomicity != 2 || source.charge != 1 || !source.is_background) continue;
        for (int row = 0; row < total_states; ++row) {
            const auto& target = levels[static_cast<size_t>(row)];
            if (target.type != dcr::atomic::SpeciesType::Atom) continue;
            if (target.charge != 0 || target.internal_id <= 1) continue;
            rates(ground_row, col) += rates(row, col);
            rates(row, col) = 0.0;
        }
    }
}

dcr::base::Matrix build_rate_matrix(const dcr::io::Config& config,
                                    const dcr::atomic::AtomicData& atomic_data,
                                    const dcr::base::Vector& n,
                                    double Te_eV,
                                    double Ti_eV) {
    const int total_states = atomic_data.get_total_states();
    dcr::state::PlasmaState plasma(1, total_states);
    plasma.init_Te().setConstant(Te_eV);
    plasma.init_Ti().setConstant(Ti_eV);
    plasma.init_ne().setConstant(quasineutral_ne(n, atomic_data.get_levels()));

    std::vector<double> energies;
    std::vector<double> weights;
    const auto energy_grid = dcr::physics::make_default_energy_grid();
    energies.reserve(energy_grid.size());
    weights.reserve(energy_grid.size());
    for (const auto& p : energy_grid) {
        energies.push_back(p.energy_eV);
        weights.push_back(p.width_eV);
    }
    EEDFConfig eedf_cfg;
    eedf_cfg.type = config.plasma.eedf.type;
    eedf_cfg.power_p = config.plasma.eedf.power_p;
    eedf_cfg.hot_fraction = config.plasma.eedf.hot_fraction;
    eedf_cfg.hot_temperature_factor = config.plasma.eedf.hot_temperature_factor;
    EEDF eedf(Te_eV, eedf_cfg);
    eedf.normalize_on_grid(energies, weights);
    EEDFGridView grid(energies, weights, &eedf);

    dcr::base::Matrix R = dcr::base::Matrix::Zero(total_states, total_states);
    for (const auto& proc : atomic_data.get_processes()) {
        if (!proc) continue;
        proc->apply(plasma, grid, n, R, nullptr);
    }
    redirect_h2plus_dr_products_to_ground(config, atomic_data.get_levels(), R);
    return R;
}

dcr::base::Vector solve_standalone_cr(const dcr::io::Config& config,
                                      const dcr::atomic::AtomicData& atomic_data,
                                      const dcr::base::Vector& seed,
                                      double Te_eV,
                                      double Ti_eV,
                                      double target_nuclei) {
    const auto& levels = atomic_data.get_levels();
    dcr::base::Vector n = seed;
    const double initial_nuclei = nuclei_total(n, levels);
    if (initial_nuclei > 0.0) n *= target_nuclei / initial_nuclei;

    for (int iter = 0; iter < 1000; ++iter) {
        dcr::base::Matrix C = build_rate_matrix(config, atomic_data, n, Te_eV, Ti_eV);
        dcr::base::Vector rhs = dcr::base::Vector::Zero(C.rows());
        const int closure_row = 0;
        C.row(closure_row).setZero();
        for (int j = 0; j < C.cols() && j < static_cast<int>(levels.size()); ++j) {
            C(closure_row, j) = static_cast<double>(std::max(1, levels[static_cast<size_t>(j)].atomicity));
        }
        rhs(closure_row) = target_nuclei;

        dcr::base::Vector next = C.fullPivLu().solve(rhs);
        for (int i = 0; i < next.size(); ++i) {
            if (!std::isfinite(next(i)) || next(i) < 0.0) next(i) = 0.0;
        }
        const double next_nuclei = nuclei_total(next, levels);
        if (next_nuclei > 0.0) next *= target_nuclei / next_nuclei;

        const double denom = std::max({1.0, n.norm(), next.norm()});
        const double rel = (next - n).norm() / denom;
        n = next;
        if (rel < 1.0e-12) break;
    }
    return n;
}

void print_compare(const char* name, double full, double standalone) {
    const double rel = (std::abs(full) > 0.0) ? (standalone - full) / full : 0.0;
    std::cout << name << "," << full << "," << standalone << "," << rel << "\n";
}

void write_profile_csv(const dcr::io::Config& config,
                       const dcr::atomic::AtomicData& atomic_data,
                       const std::string& h5_path,
                       const std::string& csv_path,
                       double last_cm) {
    H5::H5File file(h5_path, H5F_ACC_RDONLY);
    const auto x = read_vector(file, "/grid/x_cm");
    const auto Te = read_vector(file, "/rates/Te_eV");
    const auto Ti = read_vector(file, "/rates/Ti_eV");
    const auto target_nuclei = read_vector(file, "/variable_nuclei/total_nuclei_cm3");
    const auto background = read_matrix(file, "/population/background_full");
    const auto& levels = atomic_data.get_levels();

    const double x_min = x.size() > 0 ? x(x.size() - 1) - std::max(0.0, last_cm) : 0.0;
    std::ofstream out(csv_path);
    out.precision(12);
    out << "x_cm,full_Hplus,standalone_Hplus,full_H,standalone_H,full_H2,standalone_H2,"
           "full_Hminus,standalone_Hminus,total_nuclei_cm3\n";
    for (int i = 0; i < x.size(); ++i) {
        if (x(i) < x_min) continue;
        dcr::base::Vector full = background.row(i).transpose();
        const dcr::base::Vector standalone = solve_standalone_cr(
            config, atomic_data, full, Te(i), Ti(i), target_nuclei(i)
        );
        out << x(i) << ","
            << group_sum(full, levels, dcr::atomic::SpeciesType::Ion, 1, 1, 1) << ","
            << group_sum(standalone, levels, dcr::atomic::SpeciesType::Ion, 1, 1, 1) << ","
            << group_sum(full, levels, dcr::atomic::SpeciesType::Atom, 0) << ","
            << group_sum(standalone, levels, dcr::atomic::SpeciesType::Atom, 0) << ","
            << group_sum(full, levels, dcr::atomic::SpeciesType::Molecule, 0) << ","
            << group_sum(standalone, levels, dcr::atomic::SpeciesType::Molecule, 0) << ","
            << group_sum(full, levels, dcr::atomic::SpeciesType::Ion, -1) << ","
            << group_sum(standalone, levels, dcr::atomic::SpeciesType::Ion, -1) << ","
            << target_nuclei(i) << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string config_path = (argc > 1)
        ? argv[1]
        : "config/codex_full_T2_nnuc1e15_w100_variable_L25_n200_tol3e-6_ufloor001_TiBohm_localTi.yaml";
    const std::string h5_path = (argc > 2)
        ? argv[2]
        : "output/full_T2_nnuc1e15_w100_variable_L25_n200_tol3e-6_ufloor001_TiBohm_localTi/dcr_results.h5";

    dcr::io::Config config = dcr::io::ConfigLoader::load(config_path);
    auto* cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(std::cerr.rdbuf());
    dcr::atomic::AtomicData atomic_data(config);
    std::cout.rdbuf(cout_buf);
    const auto& levels = atomic_data.get_levels();

    if (argc > 3) {
        const std::string profile_csv = argv[3];
        const double last_cm = (argc > 4) ? std::stod(argv[4]) : 5.0;
        write_profile_csv(config, atomic_data, h5_path, profile_csv, last_cm);
        return 0;
    }

    H5::H5File file(h5_path, H5F_ACC_RDONLY);
    const dcr::base::Vector endpoint = read_last_vector(file, "/population/background_full");
    const double Te = read_last_scalar(file, "/rates/Te_eV");
    const double Ti = read_last_scalar(file, "/rates/Ti_eV");
    const double target_nuclei = read_last_scalar(file, "/variable_nuclei/total_nuclei_cm3");

    const dcr::base::Vector standalone = solve_standalone_cr(
        config, atomic_data, endpoint, Te, Ti, target_nuclei
    );
    const dcr::base::Matrix R_endpoint = build_rate_matrix(config, atomic_data, endpoint, Te, Ti);
    const dcr::base::Matrix R_standalone = build_rate_matrix(config, atomic_data, standalone, Te, Ti);

    std::cout.precision(12);
    std::cout << "quantity,full_endpoint,standalone_cr,rel_diff_vs_full\n";
    print_compare("nuclei_total", nuclei_total(endpoint, levels), nuclei_total(standalone, levels));
    print_compare("ne", quasineutral_ne(endpoint, levels), quasineutral_ne(standalone, levels));
    print_compare("ne_over_nnuc", quasineutral_ne(endpoint, levels) / target_nuclei,
                  quasineutral_ne(standalone, levels) / target_nuclei);
    print_compare("H+", endpoint(0), standalone(0));
    print_compare("H", group_sum(endpoint, levels, dcr::atomic::SpeciesType::Atom, 0),
                  group_sum(standalone, levels, dcr::atomic::SpeciesType::Atom, 0));
    print_compare("H2+", group_sum(endpoint, levels, dcr::atomic::SpeciesType::Ion, 1, 2),
                  group_sum(standalone, levels, dcr::atomic::SpeciesType::Ion, 1, 2));
    print_compare("H-", group_sum(endpoint, levels, dcr::atomic::SpeciesType::Ion, -1),
                  group_sum(standalone, levels, dcr::atomic::SpeciesType::Ion, -1));
    print_compare("H2", group_sum(endpoint, levels, dcr::atomic::SpeciesType::Molecule, 0),
                  group_sum(standalone, levels, dcr::atomic::SpeciesType::Molecule, 0));

    const double endpoint_resid = (R_endpoint * endpoint).norm() / std::max(1.0, endpoint.norm());
    const double standalone_resid = (R_standalone * standalone).norm() / std::max(1.0, standalone.norm());
    std::cout << "endpoint_CR_residual_norm_per_density," << endpoint_resid << ",,\n";
    std::cout << "standalone_CR_residual_norm_per_density," << standalone_resid << ",,\n";
    return 0;
}
