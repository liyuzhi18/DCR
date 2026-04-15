#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/core/ConfigPaths.hpp"
#include "../src/solver/core/RateAnalysis.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "TestDCRSetup.hpp"

namespace fs = std::filesystem;

namespace {

struct ComparisonRow {
    std::string label;
    double hot_fraction = 0.0;
    double hot_temperature_factor = 2.5;
    double scd_cm3_s = 0.0;
    double acd_cm3_s = 0.0;
    double h_cm3 = 0.0;
    double hplus_cm3 = 0.0;
};

bool is_atomic_neutral(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Atom &&
           level.atomicity == 1 &&
           level.charge == 0;
}

bool is_atomic_ion(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Ion &&
           level.atomicity == 1 &&
           level.charge > 0;
}

double sum_population_if(const dcr::base::Vector& population,
                         const std::vector<dcr::atomic::EnergyLevel>& levels,
                         bool (*predicate)(const dcr::atomic::EnergyLevel&)) {
    const int n = std::min<int>(population.size(), static_cast<int>(levels.size()));
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!predicate(levels[static_cast<size_t>(i)])) continue;
        total += std::max(population(i), 0.0);
    }
    return total;
}

ComparisonRow run_case(const dcr::io::Config& cfg,
                       const dcr::atomic::AtomicData& atomic_data,
                       double hot_fraction,
                       double hot_temperature_factor,
                       const std::string& label) {
    auto cfg_run = cfg;
    cfg_run.plasma.eedf.type = (label == "maxwellian") ? "maxwellian" : "bi_maxwellian";
    cfg_run.plasma.eedf.hot_fraction = hot_fraction;
    cfg_run.plasma.eedf.hot_temperature_factor = hot_temperature_factor;

    auto plasma = test_dcr::make_plasma_state(cfg_run, atomic_data.get_total_states());
    const auto boundary_temperatures = dcr::solver::evaluate_plasma_temperatures(cfg_run, 0.0);
    const double ion_mass_amu = test_dcr::estimate_ion_mass_amu(cfg_run);
    const auto wall = dcr::physics::compute_wall_recycling(
        cfg_run.wall.material,
        boundary_temperatures.electron_eV,
        boundary_temperatures.ion_eV,
        ion_mass_amu,
        cfg_run.wall.sheath_potential_drop
    );
    test_dcr::EEDFContext eedf(boundary_temperatures.electron_eV, test_dcr::make_eedf_config(cfg_run));

    const auto boundary = dcr::solver::run_boundary_phase(
        cfg_run, atomic_data, plasma, eedf.grid, ion_mass_amu, wall
    );
    const dcr::base::Vector bg_compact = test_dcr::compact_background_from_boundary(boundary);
    const dcr::base::Vector bg_full = test_dcr::full_background_from_compact(
        bg_compact, boundary, atomic_data.get_total_states()
    );
    const auto local = dcr::solver::assemble_local_system(
        cfg_run,
        atomic_data,
        plasma,
        eedf.grid,
        boundary,
        bg_full,
        test_dcr::flowA_from_boundary(boundary),
        test_dcr::flowM_from_boundary(boundary),
        0.0
    );

    dcr::solver::AtomicRateCalculator calculator(atomic_data);
    const auto rates = calculator.evaluate(cfg_run, boundary, local, bg_full, 0.0);
    if (!rates.atomic_effective.valid) {
        throw std::runtime_error("Atomic effective rates are invalid for case: " + label);
    }

    ComparisonRow row;
    row.label = label;
    row.hot_fraction = hot_fraction;
    row.hot_temperature_factor = hot_temperature_factor;
    row.scd_cm3_s = rates.atomic_effective.scd_cm3_s;
    row.acd_cm3_s = rates.atomic_effective.acd_cm3_s;
    row.h_cm3 = sum_population_if(bg_full, atomic_data.get_levels(), is_atomic_neutral);
    row.hplus_cm3 = sum_population_if(bg_full, atomic_data.get_levels(), is_atomic_ion);
    return row;
}

void write_csv(const fs::path& output_path,
               const std::vector<ComparisonRow>& rows) {
    fs::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    out << "label,hot_fraction,hot_temperature_factor,scd_cm3_s,acd_cm3_s,scd_ratio_to_maxwellian,acd_ratio_to_maxwellian,h_cm3,hplus_cm3\n";
    const double scd_ref = rows.front().scd_cm3_s;
    const double acd_ref = rows.front().acd_cm3_s;
    for (const auto& row : rows) {
        out << row.label << ','
            << row.hot_fraction << ','
            << row.hot_temperature_factor << ','
            << row.scd_cm3_s << ','
            << row.acd_cm3_s << ','
            << (scd_ref > 0.0 ? row.scd_cm3_s / scd_ref : 0.0) << ','
            << (acd_ref > 0.0 ? row.acd_cm3_s / acd_ref : 0.0) << ','
            << row.h_cm3 << ','
            << row.hplus_cm3 << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    fs::path config_path = test_dcr::source_root() / "config" / "hydro_am.yaml";
    fs::path output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " [--config <path>] [--output <csv_path>]\n";
            return 1;
        }
    }

    dcr::io::Config cfg = dcr::io::ConfigLoader::load(config_path.string());
    dcr::solver::normalize_input_roots(cfg, config_path.string());
    cfg.io.verbose_logging = false;

    dcr::atomic::AtomicData atomic_data(cfg);
    const std::vector<ComparisonRow> rows{
        run_case(cfg, atomic_data, 0.0, 2.5, "maxwellian"),
        run_case(cfg, atomic_data, 0.05, 2.5, "bi_maxwellian_f0.05"),
        run_case(cfg, atomic_data, 0.10, 2.5, "bi_maxwellian_f0.10")
    };

    const double scd_ref = rows.front().scd_cm3_s;
    const double acd_ref = rows.front().acd_cm3_s;

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "label,hot_fraction,hot_temperature_factor,scd_cm3_s,acd_cm3_s,scd_ratio_to_maxwellian,acd_ratio_to_maxwellian,h_cm3,hplus_cm3\n";
    for (const auto& row : rows) {
        std::cout << row.label << ','
                  << row.hot_fraction << ','
                  << row.hot_temperature_factor << ','
                  << row.scd_cm3_s << ','
                  << row.acd_cm3_s << ','
                  << (scd_ref > 0.0 ? row.scd_cm3_s / scd_ref : 0.0) << ','
                  << (acd_ref > 0.0 ? row.acd_cm3_s / acd_ref : 0.0) << ','
                  << row.h_cm3 << ','
                  << row.hplus_cm3 << '\n';
    }

    if (!output_path.empty()) {
        write_csv(output_path, rows);
    }

    return 0;
}
