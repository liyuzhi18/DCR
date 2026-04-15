#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../src/atomic/AtomicData.hpp"
#include "../src/atomic/SpeciesFileReader.hpp"
#include "../src/physics/WallBoundary.hpp"
#include "../src/solver/boundary/BoundaryPhase.hpp"
#include "../src/solver/core/ConfigPaths.hpp"
#include "../src/solver/core/RateAnalysis.hpp"
#include "../src/solver/core/TemperatureProfile.hpp"
#include "../src/solver/marching/LocalSystemAssembler.hpp"
#include "TestDCRSetup.hpp"

namespace fs = std::filesystem;

namespace {

constexpr double EV_TO_J = 1.602176634e-19;
constexpr double SE_RATE_CONST = 4.34327e7;

struct AtomicLineTransition {
    int lower_global = -1;
    int upper_global = -1;
    int lower_n = -1;
    int upper_n = -1;
    double spontaneous_rate_s = 0.0;
    double delta_e_ev = 0.0;
};

struct CaseSpec {
    std::string label;
    std::string eedf_type;
    double power_p = 1.0;
};

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool is_atomic_neutral(const dcr::atomic::EnergyLevel& level) {
    return level.type == dcr::atomic::SpeciesType::Atom &&
           level.atomicity == 1 &&
           level.charge == 0;
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

const dcr::io::SpeciesConfig& find_atomic_species_config(const dcr::io::Config& cfg) {
    for (const auto& sp : cfg.species) {
        if (!sp.is_molecule) return sp;
    }
    throw std::runtime_error("No atomic species entry found in config");
}

std::vector<AtomicLineTransition> load_atomic_line_transitions(
    const dcr::io::Config& cfg,
    const dcr::atomic::AtomicData& atomic_data) {

    const auto& atomic_sp = find_atomic_species_config(cfg);
    const fs::path atomic_file = fs::path(cfg.io.atomic_data_root) / atomic_sp.filename;
    const int Z = std::max(1, static_cast<int>(atomic_sp.mass_amu > 0.0 ? atomic_sp.mass_amu : 1.0));
    const auto raw = dcr::atomic::SpeciesFileReader::load_atomic(atomic_file.string(), Z);
    const auto& levels = atomic_data.get_levels();

    std::map<std::pair<int, int>, int> global_by_charge_and_local;
    const std::string prefix = atomic_sp.name + " ";
    for (const auto& level : levels) {
        if (!starts_with(level.label, prefix)) continue;
        global_by_charge_and_local[{level.charge, level.internal_id}] = level.global_index;
    }

    std::vector<AtomicLineTransition> lines;
    for (const auto& tr : raw.transitions) {
        if (tr.type != "b") continue;
        const int q_from = raw.Z - tr.state_params[0];
        const int q_to = raw.Z - tr.state_params[2];
        const int local_from = tr.state_params[1];
        const int local_to = tr.state_params[3];
        const auto it_from = global_by_charge_and_local.find({q_from, local_from});
        const auto it_to = global_by_charge_and_local.find({q_to, local_to});
        if (it_from == global_by_charge_and_local.end() || it_to == global_by_charge_and_local.end()) continue;

        const auto& lower = levels[static_cast<size_t>(it_from->second)];
        const auto& upper = levels[static_cast<size_t>(it_to->second)];
        if (!is_atomic_neutral(lower) || !is_atomic_neutral(upper)) continue;
        if (upper.energy_eV <= lower.energy_eV) continue;
        if (tr.oscillator_strength <= 0.0 || lower.degeneracy <= 0.0 || upper.degeneracy <= 0.0) continue;

        const double delta_e_ev = std::max({1e-6, upper.energy_eV - lower.energy_eV, tr.threshold_ev});
        const double rate_se =
            tr.oscillator_strength * SE_RATE_CONST * delta_e_ev * delta_e_ev *
            (lower.degeneracy / upper.degeneracy);
        if (rate_se <= 0.0) continue;
        lines.push_back({it_from->second, it_to->second, lower.internal_id, upper.internal_id, rate_se, delta_e_ev});
    }
    return lines;
}

dcr::base::Vector add_atomic_flow_to_population(const dcr::base::Vector& bg_full,
                                                const dcr::base::Vector& flowA,
                                                const dcr::solver::BoundaryPhaseResult& boundary,
                                                const std::vector<dcr::atomic::EnergyLevel>& levels) {
    dcr::base::Vector total = bg_full;
    const int n = std::min<int>(flowA.size(), static_cast<int>(boundary.A_indices.size()));
    for (int j = 0; j < n; ++j) {
        const int gi = boundary.A_indices[static_cast<size_t>(j)];
        if (gi < 0 || gi >= total.size() || gi >= static_cast<int>(levels.size())) continue;
        if (!is_atomic_neutral(levels[static_cast<size_t>(gi)])) continue;
        total(gi) += std::max(flowA(j), 0.0);
    }
    return total;
}

void inspect_case(const dcr::io::Config& cfg,
                  const dcr::atomic::AtomicData& atomic_data,
                  const std::vector<AtomicLineTransition>& lines,
                  const CaseSpec& spec) {
    auto cfg_run = cfg;
    cfg_run.plasma.eedf.type = spec.eedf_type;
    cfg_run.plasma.eedf.power_p = spec.power_p;

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
    const dcr::base::Vector total_atomic_population =
        add_atomic_flow_to_population(bg_full, test_dcr::flowA_from_boundary(boundary), boundary, atomic_data.get_levels());

    const auto local = dcr::solver::assemble_local_system(
        cfg_run, atomic_data, plasma, eedf.grid, boundary, bg_full,
        test_dcr::flowA_from_boundary(boundary), test_dcr::flowM_from_boundary(boundary), 0.0
    );
    dcr::solver::AtomicRateCalculator calculator(atomic_data);
    const auto rates = calculator.evaluate(cfg_run, boundary, local, bg_full, 0.0);

    struct Contribution {
        int upper_n = -1;
        int lower_n = -1;
        double upper_population = 0.0;
        double power = 0.0;
    };
    std::vector<Contribution> top;
    double total_rpl = 0.0;
    for (const auto& line : lines) {
        if (line.upper_global < 0 || line.upper_global >= total_atomic_population.size()) continue;
        const double n_upper = std::max(total_atomic_population(line.upper_global), 0.0);
        const double power = n_upper * line.spontaneous_rate_s * line.delta_e_ev * EV_TO_J;
        total_rpl += power;
        top.push_back({line.upper_n, line.lower_n, n_upper, power});
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.power > b.power; });

    std::cout << "\nCASE " << spec.label << "\n";
    std::cout << "  SCD=" << rates.atomic_effective.scd_cm3_s
              << " ACD=" << rates.atomic_effective.acd_cm3_s
              << " total_atomic_RPL=" << total_rpl << "\n";
    std::cout << "  H=" << sum_population_if(bg_full, atomic_data.get_levels(),
                    [](const dcr::atomic::EnergyLevel& lvl){ return is_atomic_neutral(lvl); })
              << " H+=" << sum_population_if(bg_full, atomic_data.get_levels(),
                    [](const dcr::atomic::EnergyLevel& lvl){ return lvl.type==dcr::atomic::SpeciesType::Ion && lvl.atomicity==1 && lvl.charge>0; })
              << "\n";

    std::map<int, double> by_n;
    for (const auto& level : atomic_data.get_levels()) {
        if (!is_atomic_neutral(level) || level.internal_id < 2) continue;
        if (level.global_index < total_atomic_population.size()) {
            by_n[level.internal_id] += std::max(total_atomic_population(level.global_index), 0.0);
        }
    }
    std::cout << "  excited populations:\n";
    for (const auto& [n, val] : by_n) {
        std::cout << "    n=" << n << " : " << val << "\n";
    }

    std::cout << "  top line contributions:\n";
    const int nprint = std::min<int>(10, static_cast<int>(top.size()));
    for (int i = 0; i < nprint; ++i) {
        std::cout << "    " << top[static_cast<size_t>(i)].upper_n
                  << "->" << top[static_cast<size_t>(i)].lower_n
                  << "  n_upper=" << top[static_cast<size_t>(i)].upper_population
                  << "  power=" << top[static_cast<size_t>(i)].power << "\n";
    }
}

} // namespace

int main() {
    fs::path config_path = test_dcr::source_root() / "config" / "hydro_am.yaml";
    dcr::io::Config cfg = dcr::io::ConfigLoader::load(config_path.string());
    dcr::solver::normalize_input_roots(cfg, config_path.string());
    cfg.io.verbose_logging = false;

    dcr::atomic::AtomicData atomic_data(cfg);
    const auto lines = load_atomic_line_transitions(cfg, atomic_data);
    inspect_case(cfg, atomic_data, lines, {"maxwellian", "maxwellian", 1.0});
    inspect_case(cfg, atomic_data, lines, {"generalized_power_p0.7", "generalized_power", 0.7});
    inspect_case(cfg, atomic_data, lines, {"generalized_power_p2.0", "generalized_power", 2.0});
    return 0;
}
