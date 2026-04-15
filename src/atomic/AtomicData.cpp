#include "AtomicData.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "../processes/AtomicProcess.hpp"    
#include "../processes/MolecularProcess.hpp" 

namespace dcr::atomic {

    namespace {
        struct MCCCDissociationRateFit {
            double threshold_ev = 0.0;
            std::vector<double> coeffs;

            bool valid() const { return !coeffs.empty(); }
        };

        struct MCCCDissociationRateTable {
            std::vector<MCCCDissociationRateFit> fits_by_vi;

            const MCCCDissociationRateFit* fit_for_vi(int vi) const {
                if (vi < 0 || vi >= static_cast<int>(fits_by_vi.size())) return nullptr;
                return fits_by_vi[vi].valid() ? &fits_by_vi[vi] : nullptr;
            }
        };

        std::vector<std::string> split_csv_line(const std::string& line) {
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string field;
            while (std::getline(ss, field, ',')) {
                fields.push_back(field);
            }
            return fields;
        }

        MCCCDissociationRateTable load_mccc_dissociation_rate_table(const std::filesystem::path& path) {
            MCCCDissociationRateTable table;
            std::ifstream in(path);
            if (!in) {
                std::cout << "[AtomicData] MCCC H2 dissociation fit not found: " << path
                          << " (using legacy de fits)\n";
                return table;
            }

            std::string header_line;
            if (!std::getline(in, header_line)) return table;
            const auto header = split_csv_line(header_line);

            size_t vi_col = header.size();
            size_t threshold_col = header.size();
            std::vector<size_t> coeff_cols;
            for (size_t i = 0; i < header.size(); ++i) {
                if (header[i] == "vi") vi_col = i;
                else if (header[i] == "threshold_eV") threshold_col = i;
                else if (!header[i].empty() && header[i][0] == 'a') coeff_cols.push_back(i);
            }
            if (vi_col >= header.size() || threshold_col >= header.size() || coeff_cols.empty()) {
                std::cout << "[AtomicData] Invalid MCCC H2 dissociation fit header: " << path << "\n";
                return table;
            }

            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                const auto fields = split_csv_line(line);
                if (fields.size() <= std::max(threshold_col, coeff_cols.back())) continue;

                const int vi = std::stoi(fields[vi_col]);
                if (vi < 0) continue;
                if (vi >= static_cast<int>(table.fits_by_vi.size())) {
                    table.fits_by_vi.resize(static_cast<size_t>(vi) + 1);
                }

                auto& fit = table.fits_by_vi[vi];
                fit.threshold_ev = std::stod(fields[threshold_col]);
                fit.coeffs.clear();
                fit.coeffs.reserve(coeff_cols.size());
                for (size_t col : coeff_cols) {
                    fit.coeffs.push_back(std::stod(fields[col]));
                }
            }

            size_t count = 0;
            for (const auto& fit : table.fits_by_vi) {
                if (fit.valid()) ++count;
            }
            std::cout << "[AtomicData] Loaded MCCC H2 dissociation fits: " << count
                      << " levels from " << path << "\n";
            return table;
        }

        const MCCCDissociationRateTable& get_mccc_dissociation_rate_table(
            const std::filesystem::path& path) {
            static std::unordered_map<std::string, MCCCDissociationRateTable> cache;
            const std::string key = path.lexically_normal().string();
            auto it = cache.find(key);
            if (it == cache.end()) {
                it = cache.emplace(key, load_mccc_dissociation_rate_table(path)).first;
            }
            return it->second;
        }

        std::vector<QuadraturePoint> load_quadrature_table(const std::filesystem::path& path) {
            std::vector<QuadraturePoint> points;
            std::ifstream in(path);
            if (!in) {
                std::cout << "[AtomicData] Quadrature file not found: " << path << "\n";
                return points;
            }

            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#' || line[0] == '!') continue;
                std::istringstream iss(line);
                int idx = 0;
                double point = 0.0;
                double weight = 0.0;
                if (!(iss >> idx >> point >> weight)) continue;
                points.push_back({point, weight});
            }
            if (points.empty()) {
                std::cout << "[AtomicData] Quadrature file empty: " << path << "\n";
            }
            return points;
        }

        int extract_vibrational_quantum(const std::string& label) {
            size_t pos = 0;
            while ((pos = label.find('v', pos)) != std::string::npos) {
                size_t idx = pos + 1;
                while (idx < label.size() &&
                       !std::isdigit(static_cast<unsigned char>(label[idx]))) {
                    ++idx;
                }
                const size_t start = idx;
                while (idx < label.size() &&
                       std::isdigit(static_cast<unsigned char>(label[idx]))) {
                    ++idx;
                }
                if (idx > start) {
                    try {
                        return std::stoi(label.substr(start, idx - start));
                    } catch (...) {
                        return -1;
                    }
                }
                ++pos;
            }
            return -1;
        }

        double compute_high_vibrational_factor(int v) {
            if (v < 9) return 0.0;
            if (v == 9) return 1.0;
            return 1.97 / std::pow(static_cast<double>(v - 8), 1.23);
        }

        bool is_h2p_label(const std::string& label) {
            for (size_t i = 0; i + 2 < label.size(); ++i) {
                const unsigned char c0 = static_cast<unsigned char>(label[i]);
                if (std::tolower(c0) != 'h') continue;
                const unsigned char c1 = static_cast<unsigned char>(label[i + 1]);
                const unsigned char c2 = static_cast<unsigned char>(label[i + 2]);
                if (c1 == '2' && std::tolower(c2) == 'p') {
                    return true;
                }
            }
            return false;
        }

        bool is_neutral_h2_level(const MolecularLevel& level) {
            return level.charge_number == 0 && level.proton_count == 2;
        }
    }

    AtomicData::AtomicData(const dcr::io::Config& config) {
        std::string root_dir = config.io.atomic_data_root;
        if (!root_dir.empty() && root_dir.back() != '/') root_dir += "/";
        use_mccc_h2_dissociation_ = config.numerics.use_mccc_h2_dissociation;

        // Load MCX quadrature table from data_tables (used for ion-temperature integrals).
        std::filesystem::path data_root = config.io.data_tables_root;
        if (data_root.empty()) {
            if (!config.io.atomic_data_root.empty()) {
                data_root = std::filesystem::path(config.io.atomic_data_root).parent_path() / "data_tables";
            } else {
                data_root = "data_tables";
            }
        }
        mcx_quadrature_ = load_quadrature_table(data_root / "quad_max.in");
        if (!mcx_quadrature_.empty()) {
            std::cout << "[AtomicData] Loaded MCX quadrature points: " << mcx_quadrature_.size() << "\n";
        }

        for (const auto& sp : config.species) {
            process_species_config(sp, root_dir);
        }

        // Build Recycling Indices (Neutrals Only)
        for (const auto& lvl : levels_) {
            if (lvl.is_recycling) {
                recycling_indices_.push_back(lvl.global_index);
            }
        }
        
        std::cout << "[AtomicData] Loaded " << levels_.size() << " states (" 
                  << recycling_indices_.size() << " recycling).\n";
    }

    void AtomicData::process_species_config(const dcr::io::SpeciesConfig& sp, const std::string& root_dir) {
        std::string full_path = root_dir + sp.filename;
        
        if (!sp.is_molecule) {
            atomic_species_configs_.push_back(sp);
            int Z = (int)sp.mass_amu; if (Z < 1) Z = 1; 
            auto raw = get_atomic_file(full_path, Z);
            
            // Register Levels
            for (const auto& rlvl : raw->levels) {
                if (sp.charge_set && rlvl.charge_number != (int)sp.charge) continue;
                EnergyLevel lvl;
                lvl.global_index = (base::Index)levels_.size();
                lvl.type = (rlvl.charge_number == 0) ? SpeciesType::Atom : SpeciesType::Ion;
                lvl.label = sp.name + " " + rlvl.label;
                lvl.energy_eV = rlvl.excitation_energy_ev;
                lvl.degeneracy = rlvl.statistical_weight;
                lvl.charge = rlvl.charge_number;
                lvl.internal_id = rlvl.local_index;
                lvl.block_index = rlvl.file_block_index; // Needed for atomic transitions
                lvl.atomicity = 1;
                lvl.mass_amu = (sp.mass_amu > 0.0) ? sp.mass_amu : 1.0;

                // Recycling assignment comes from config; only neutrals can be recycling.
                lvl.is_recycling = sp.is_recycling && (rlvl.charge_number == 0);
                lvl.is_background = !lvl.is_recycling;

                levels_.push_back(lvl);
                global_index_map_[sp.name][lvl.charge][lvl.internal_id] = lvl.global_index;
            }
            // Build Processes
            build_atomic_processes(raw, sp.name, sp.charge);
        } else {
            // Molecular
            molecular_species_by_file_[full_path].push_back(sp);
            auto raw = get_molecular_file(full_path, 1); // ID=1 for H2
            
            for (const auto& rlvl : raw->levels) {
                if (sp.charge_set && rlvl.charge_number != (int)sp.charge) continue;
                EnergyLevel lvl;
                lvl.global_index = (base::Index)levels_.size();
                lvl.type = (rlvl.charge_number == 0) ? SpeciesType::Molecule : SpeciesType::Ion;
                lvl.label = sp.name + " " + rlvl.label;
                lvl.energy_eV = rlvl.excitation_energy_ev;
                lvl.degeneracy = 1.0;
                lvl.charge = rlvl.charge_number;
                lvl.internal_id = rlvl.file_local_index; // 'v' number
                // Use proton count from the file block (e.g., H- has atomicity 1, H2/H2+ have 2).
                lvl.atomicity = std::max(1, rlvl.proton_count);
                // If mass is not explicitly provided in YAML, avoid underestimating molecular masses.
                lvl.mass_amu = std::max<base::Real>(
                    (sp.mass_amu > 0.0) ? sp.mass_amu : 1.0,
                    static_cast<base::Real>(lvl.atomicity)
                );

                // Recycling assignment comes from config; only neutrals can be recycling.
                lvl.is_recycling = sp.is_recycling && (rlvl.charge_number == 0);
                lvl.is_background = !lvl.is_recycling;

                levels_.push_back(lvl);
                // For molecules, we use the local_index from the file which is sequential
                global_index_map_[sp.name][lvl.charge][rlvl.local_index] = lvl.global_index;
            }
            build_molecular_processes(raw, sp.name, sp.charge, full_path);
        }
    }

    base::Index AtomicData::find_global_index(const std::string& name, int charge, int local) const {
        auto it_sp = global_index_map_.find(name);
        if (it_sp == global_index_map_.end()) return -1;
        auto it_ch = it_sp->second.find(charge);
        if (it_ch == it_sp->second.end()) return -1;
        auto it_idx = it_ch->second.find(local);
        if (it_idx == it_ch->second.end()) return -1;
        return it_idx->second;
    }

    // --- Process Builders ---

    void AtomicData::build_atomic_processes(const std::shared_ptr<AtomicSpecies>& raw, const std::string& name, int charge) {
        using namespace crm_detail;
        for (const auto& tr : raw->transitions) {
            // Map Atomic: state_params = {from_block, from_loc, to_block, to_loc}
            int q_from = raw->Z - tr.state_params[0];
            int q_to   = raw->Z - tr.state_params[2];
            
            base::Index i_from = find_global_index(name, q_from, tr.state_params[1]);
            base::Index i_to   = find_global_index(name, q_to,   tr.state_params[3]);

            if (i_from == -1 || i_to == -1) continue;

            const auto& lf = levels_[i_from];
            const auto& lt = levels_[i_to];

            if (tr.type == "b") { // Excitation
                std::array<double, 8> p; 
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                processes_.push_back(std::make_shared<AtomicExcitationProcess>(
                    i_from, i_to, tr.threshold_ev, lf.degeneracy, lt.degeneracy,
                    lf.energy_eV, lt.energy_eV, tr.oscillator_strength, tr.flag, p
                ));
            }
            else if (tr.type == "f") { // Electron-impact ionization
                std::array<double, 4> p{{tr.params[0], tr.params[1], tr.params[2], tr.params[3]}};
                processes_.push_back(std::make_shared<AtomicIonizationProcess>(
                    i_from, i_to, tr.threshold_ev, tr.flag, p, lf.degeneracy, lt.degeneracy
                ));
            }
            else if (tr.type == "p") { // Photo-ionization / radiative recombination
                std::array<double, 7> p{{tr.params[0], tr.params[1], tr.params[2], tr.params[3], tr.params[4], tr.params[5], tr.params[6]}};
                processes_.push_back(std::make_shared<AtomicPhotoProcess>(
                    i_from, i_to, tr.threshold_ev, tr.flag, p, lf.degeneracy, lt.degeneracy
                ));
            }
            else if (tr.type == "a") { // Autoionization (constant rate)
                processes_.push_back(std::make_shared<AtomicAutoionizationProcess>(
                    i_from, i_to, tr.params[0]
                ));
            }
        }
    }

    void AtomicData::build_molecular_processes(const std::shared_ptr<MolecularSpecies>& raw, const std::string& name, int charge, const std::string& file_path) {
        using namespace crm_detail;
        auto mide_pool = std::make_shared<MolecularMIDEPool>();
        const std::filesystem::path mccc_de_dir =
            std::filesystem::path(file_path).parent_path() / "mccc-dissociation";
        const std::filesystem::path reconstructed_de_fit_path =
            std::filesystem::path(file_path).parent_path() / "reconstructed_dissociation_rate_fit.csv";
        const MCCCDissociationRateTable* reconstructed_de_fit_table = nullptr;
        if (!use_mccc_h2_dissociation_) {
            reconstructed_de_fit_table = &get_mccc_dissociation_rate_table(reconstructed_de_fit_path);
        }
        std::unordered_set<int> built_mccc_de_v;
        std::unordered_set<int> built_reconstructed_fit_v;
        std::unordered_map<int, const MolecularLevel*> level_by_local;
        level_by_local.reserve(raw->levels.size());
        for (const auto& lvl : raw->levels) {
            level_by_local[lvl.local_index] = &lvl;
        }

        auto resolve_molecular_state = [&](const MolecularStateRef& ref) -> base::Index {
            if (ref.local_index > 0) {
                auto it = level_by_local.find(ref.local_index);
                if (it == level_by_local.end()) return -1;
                const int lvl_charge = it->second->charge_number;
                auto it_cfg = molecular_species_by_file_.find(file_path);
                if (it_cfg != molecular_species_by_file_.end()) {
                    for (const auto& cfg : it_cfg->second) {
                        if (!cfg.charge_set || cfg.charge == lvl_charge) {
                            return find_global_index(cfg.name, lvl_charge, ref.local_index);
                        }
                    }
                }
                return -1;
            }

            // MCX files reference atomic partners by species index (0/1) that live in atomic files.
            if (ref.species_index == 0 || ref.species_index == 1) {
                const int desired_charge = (ref.species_index == 0) ? 1 : 0;
                for (const auto& cfg : atomic_species_configs_) {
                    if (!cfg.charge_set || cfg.charge == desired_charge) {
                        return find_global_index(cfg.name, desired_charge, ref.file_local_index);
                    }
                }
            }
            return -1;
        };

        // Build hydrogen DR branching targets (H(n=2..5), then n>=6 tail).
        HydrogenDRBranchingSpec hydrogen_dr_template;
        base::Index atomic_ground_index = -1;
        for (const auto& cfg : atomic_species_configs_) {
            if (cfg.charge_set && cfg.charge != 0) continue;

            if (atomic_ground_index < 0) {
                atomic_ground_index = find_global_index(cfg.name, 0, 1);
            }

            std::array<int, 4> discrete{{-1, -1, -1, -1}};
            bool have_discrete = true;
            for (int n = 2; n <= 5; ++n) {
                const base::Index idx = find_global_index(cfg.name, 0, n);
                if (idx < 0) {
                    have_discrete = false;
                    break;
                }
                discrete[n - 2] = static_cast<int>(idx);
            }
            if (!have_discrete) continue;

            std::vector<int> high_n;
            for (int n = 6; n <= 40; ++n) {
                const base::Index idx = find_global_index(cfg.name, 0, n);
                if (idx < 0) break;
                high_n.push_back(static_cast<int>(idx));
            }
            if (high_n.empty()) continue;

            hydrogen_dr_template.enabled = true;
            hydrogen_dr_template.discrete_state_indices = discrete;
            hydrogen_dr_template.high_n_state_indices = std::move(high_n);
            break;
        }

        for (const auto& tr : raw->transitions) {
            // Resolve primary reactant/product from file-local indices.
            base::Index i_from = resolve_molecular_state(tr.reactants[0]);
            if (i_from == -1) continue;

            base::Index i_to = -1;
            if (tr.product_count > 0) {
                i_to = resolve_molecular_state(tr.products[0]);
            }

            if (tr.type == "ve") {
                // Explicitly discard VE transitions (not used in current DCR setup).
                continue;
            }
            else if (tr.type == "ev") {
                std::array<double, 6> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                processes_.push_back(std::make_shared<MolecularExcitationProcess>(
                    i_from, i_to, tr.threshold_ev, p
                ));
            }
            else if (tr.type == "mi") {
                std::array<double, 7> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                processes_.push_back(std::make_shared<MolecularMIProcess>(
                    i_from, i_to, tr.threshold_ev, p
                ));
            }
            else if (tr.type == "dr") {
                const base::Index i_to2 = (tr.product_count > 1)
                    ? resolve_molecular_state(tr.products[1]) : -1;
                if (i_to == -1 || i_to2 == -1) continue;
                std::array<double, 6> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                HydrogenDRBranchingSpec branching = hydrogen_dr_template;
                if (branching.enabled) {
                    auto lvl_it = level_by_local.find(tr.reactants[0].local_index);
                    if (lvl_it != level_by_local.end() && is_h2p_label(lvl_it->second->label)) {
                        const int v = extract_vibrational_quantum(lvl_it->second->label);
                        if (v >= 0) {
                            branching.vibrational_quantum = v;
                        } else {
                            branching.enabled = false;
                        }
                    } else {
                        branching.enabled = false;
                    }
                }
                processes_.push_back(std::make_shared<MolecularDRProcess>(
                    i_from, i_to, i_to2, p, branching
                ));
            }
            else if (tr.type == "ra") {
                if (i_to == -1) continue;
                processes_.push_back(std::make_shared<MolecularRAProcess>(
                    i_from, i_to, tr.params[0]
                ));
            }
            else if (tr.type == "da") {
                const base::Index i_to2 = (tr.product_count > 1)
                    ? resolve_molecular_state(tr.products[1]) : -1;
                if (i_to == -1 || i_to2 == -1) continue;
                processes_.push_back(std::make_shared<MolecularDAProcess>(
                    i_from, i_to, i_to2, tr.params[0], tr.params[1]
                ));
            }
            else if (tr.type == "de") {
                const base::Index i_to2 = (tr.product_count > 1)
                    ? resolve_molecular_state(tr.products[1]) : -1;
                if (i_to == -1 || i_to2 == -1) continue;
                std::array<double, 6> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                auto lvl_it = level_by_local.find(tr.reactants[0].local_index);
                if (use_mccc_h2_dissociation_ &&
                    lvl_it != level_by_local.end() &&
                    is_neutral_h2_level(*lvl_it->second)) {
                    const int v = extract_vibrational_quantum(lvl_it->second->label);
                    const std::filesystem::path mccc_path =
                        mccc_de_dir / ("MCCC-el-H2-DISS.X1Sg_vi=" + std::to_string(v) + ".txt");
                    if (std::filesystem::exists(mccc_path)) {
                        if (built_mccc_de_v.find(v) != built_mccc_de_v.end()) {
                            continue;
                        }
                        processes_.push_back(std::make_shared<MolecularDEProcess>(
                            i_from, i_to, i_to2, tr.threshold_ev, p, mccc_path.string()
                        ));
                        built_mccc_de_v.insert(v);
                        continue;
                    }
                }
                if (!use_mccc_h2_dissociation_ &&
                    reconstructed_de_fit_table != nullptr &&
                    lvl_it != level_by_local.end() &&
                    is_neutral_h2_level(*lvl_it->second)) {
                    const int v = extract_vibrational_quantum(lvl_it->second->label);
                    if (const auto* fit = reconstructed_de_fit_table->fit_for_vi(v); fit != nullptr) {
                        processes_.push_back(std::make_shared<MolecularDEProcess>(
                            i_from, i_to, i_to2, fit->threshold_ev, p, std::string{}, fit->coeffs
                        ));
                        built_reconstructed_fit_v.insert(v);
                        continue;
                    }
                }
                processes_.push_back(std::make_shared<MolecularDEProcess>(
                    i_from, i_to, i_to2, tr.threshold_ev, p
                ));
            }
            else if (tr.type == "ed") {
                base::Index i_partner = -1;
                if (tr.reactant_count > 1) {
                    i_partner = resolve_molecular_state(tr.reactants[1]);
                    if (i_partner == -1) continue;
                }
                std::vector<base::Index> products;
                products.reserve(tr.product_count);
                for (int j = 0; j < tr.product_count; ++j) {
                    const base::Index idx = resolve_molecular_state(tr.products[j]);
                    if (idx >= 0) products.push_back(idx);
                }
                if (products.empty()) continue;
                std::array<double, 11> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                const std::vector<QuadraturePoint>* quad = mcx_quadrature_.empty() ? nullptr : &mcx_quadrature_;
                processes_.push_back(std::make_shared<MolecularEDProcess>(
                    i_from, i_partner, std::move(products), tr.flag, p, quad
                ));
            }
            else if (tr.type == "mcx") {
                std::array<double, 11> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());

                // Resolve cross-species reactants/products using file species indices.
                const base::Index i_primary = resolve_molecular_state(tr.reactants[0]);
                const base::Index i_partner = resolve_molecular_state(tr.reactants[1]);
                const base::Index i_prod_primary = resolve_molecular_state(tr.products[0]);
                const base::Index i_prod_partner = resolve_molecular_state(tr.products[1]);
                if (i_primary == -1) continue;

                double high_v_factor = 0.0;
                if (tr.flag == 90) {
                    auto lvl_it = level_by_local.find(tr.reactants[0].local_index);
                    if (lvl_it != level_by_local.end()) {
                        const int v = extract_vibrational_quantum(lvl_it->second->label);
                        high_v_factor = compute_high_vibrational_factor(v);
                    }
                }

                double reduced_mass = MCX_REDUCED_MASS;
                if (i_primary >= 0 && i_primary < static_cast<base::Index>(levels_.size()) &&
                    i_partner >= 0 && i_partner < static_cast<base::Index>(levels_.size())) {
                    const double m1 = std::max(levels_[i_primary].mass_amu, 1.0e-12);
                    const double m2 = std::max(levels_[i_partner].mass_amu, 1.0e-12);
                    const double mu = (m1 * m2) / (m1 + m2);
                    if (mu > 0.0) reduced_mass = mu;
                }

                const std::vector<QuadraturePoint>* quad = mcx_quadrature_.empty() ? nullptr : &mcx_quadrature_;
                processes_.push_back(std::make_shared<MolecularMCXProcess>(
                    i_primary, i_partner, i_prod_primary, i_prod_partner,
                    tr.flag, p, tr.threshold_ev, quad,
                    reduced_mass, 1, high_v_factor // Hint Z=1
                ));
            }
            else if (tr.type == "mide") {
                const base::Index i_to2 = (tr.product_count > 1)
                    ? resolve_molecular_state(tr.products[1]) : -1;
                if (i_to == -1 || i_to2 == -1) continue;
                std::array<double, 3> p;
                std::copy_n(tr.params.begin(), p.size(), p.begin());
                double threshold = tr.threshold_ev;
                auto lvl_it = level_by_local.find(tr.reactants[0].local_index);
                if (lvl_it != level_by_local.end()) {
                    const double diss = lvl_it->second->dissociation_energy_ev;
                    if (diss > 0.0) threshold = diss;
                }
                processes_.push_back(std::make_shared<MolecularMIDEProcess>(
                    i_from, i_to, i_to2, threshold, p, mide_pool
                ));
            }
        }

        if (use_mccc_h2_dissociation_ && atomic_ground_index >= 0) {
            int synthetic_mccc_de = 0;
            for (const auto& lvl : raw->levels) {
                if (!is_neutral_h2_level(lvl)) continue;
                const int v = extract_vibrational_quantum(lvl.label);
                if (built_mccc_de_v.find(v) != built_mccc_de_v.end()) continue;
                const std::filesystem::path mccc_path =
                    mccc_de_dir / ("MCCC-el-H2-DISS.X1Sg_vi=" + std::to_string(v) + ".txt");
                if (!std::filesystem::exists(mccc_path)) continue;

                const base::Index i_from = find_global_index(name, 0, lvl.local_index);
                if (i_from < 0) continue;

                processes_.push_back(std::make_shared<MolecularDEProcess>(
                    i_from, atomic_ground_index, atomic_ground_index,
                    lvl.dissociation_energy_ev, std::array<double, 6>{}, mccc_path.string()
                ));
                built_mccc_de_v.insert(v);
                ++synthetic_mccc_de;
            }

            if (!built_mccc_de_v.empty()) {
                std::cout << "[AtomicData] Using MCCC H2 dissociation cross sections: "
                          << built_mccc_de_v.size() << " levels from " << mccc_de_dir << "\n";
            }
            if (synthetic_mccc_de > 0) {
                std::cout << "[AtomicData] Added synthetic MCCC H2 dissociation channels: "
                          << synthetic_mccc_de << "\n";
            }
        } else if (!use_mccc_h2_dissociation_) {
            if (!built_reconstructed_fit_v.empty()) {
                std::cout << "[AtomicData] Using reconstructed H2 dissociation fits: "
                          << built_reconstructed_fit_v.size() << " levels from "
                          << reconstructed_de_fit_path << "\n";
            } else {
                std::cout << "[AtomicData] Using legacy fitted H2 dissociation rates from species file\n";
            }
        }
    }

    // --- Caching ---
    std::shared_ptr<AtomicSpecies> AtomicData::get_atomic_file(const std::string& path, int Z) {
        std::string key = path + "_" + std::to_string(Z);
        if (atomic_cache_.find(key) == atomic_cache_.end()) {
            atomic_cache_[key] = std::make_shared<AtomicSpecies>(SpeciesFileReader::load_atomic(path, Z));
        }
        return atomic_cache_[key];
    }

    std::shared_ptr<MolecularSpecies> AtomicData::get_molecular_file(const std::string& path, int id) {
        std::string key = path + "_" + std::to_string(id);
        if (molecular_cache_.find(key) == molecular_cache_.end()) {
            molecular_cache_[key] = std::make_shared<MolecularSpecies>(SpeciesFileReader::load_molecular(path, id));
        }
        return molecular_cache_[key];
    }
}
