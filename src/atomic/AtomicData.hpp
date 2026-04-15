#pragma once
#include "../base/Types.hpp"
#include "../base/ProcessBase.hpp" 
#include "SpeciesFileReader.hpp"
#include "../io/ConfigStructs.hpp"
#include "../physics/Quadrature.hpp"
#include "EnergyLevel.hpp"
#include <vector>
#include <memory>
#include <map>

namespace dcr::atomic {

    class AtomicData {
    public:
        AtomicData(const dcr::io::Config& config);

        base::Index get_total_states() const { return levels_.size(); }
        const std::vector<EnergyLevel>& get_levels() const { return levels_; }

        const base::IndexVec& get_recycling_indices() const { return recycling_indices_; }
    
        const std::vector<std::shared_ptr<dcr::ProcessBase>>& get_processes() const { return processes_; }
    private:
        void process_species_config(const dcr::io::SpeciesConfig& sp, const std::string& root_dir);

        // Factory Methods
        void build_atomic_processes(const std::shared_ptr<AtomicSpecies>& raw, const std::string& name, int charge);
        void build_molecular_processes(const std::shared_ptr<MolecularSpecies>& raw, const std::string& name, int charge, const std::string& file_path);

        base::Index find_global_index(const std::string& name, int charge, int local) const;

        // Data Storage
        std::vector<EnergyLevel> levels_;
        std::vector<std::shared_ptr<dcr::ProcessBase>> processes_;
        base::IndexVec recycling_indices_;
        // Ion-temperature quadrature table for MCX (loaded once per AtomicData).
        std::vector<QuadraturePoint> mcx_quadrature_;
        bool use_mccc_h2_dissociation_ = true;

        // Global Index Map
        std::map<std::string, std::map<int, std::map<int, base::Index>>> global_index_map_;

        // File Cache
        std::map<std::string, std::shared_ptr<AtomicSpecies>> atomic_cache_;
        std::map<std::string, std::shared_ptr<MolecularSpecies>> molecular_cache_;

        // Species config helpers for cross-species molecular reactions (e.g., MCX).
        std::vector<dcr::io::SpeciesConfig> atomic_species_configs_;
        std::map<std::string, std::vector<dcr::io::SpeciesConfig>> molecular_species_by_file_;
        
        std::shared_ptr<AtomicSpecies> get_atomic_file(const std::string& path, int Z);
        std::shared_ptr<MolecularSpecies> get_molecular_file(const std::string& path, int id);
    };
}
