#pragma once

#include <cmath>
#include <vector>

namespace dcr::physics {

struct EnergyGridPoint {
    double energy_eV = 0.0;
    double width_eV = 0.0;
};

// Legacy flychk-flow-new electron-energy grid used for EEDF integrations.
inline std::vector<EnergyGridPoint> make_default_energy_grid() {
    struct Segment { int steps; double dE; };

    std::vector<double> energies;
    energies.reserve(1400);

    const int segment1 = 300;
    const double base_dE = 1.0e-6;
    for (int i = 0; i < segment1; ++i) {
        const double E = base_dE * std::pow(1.03, static_cast<double>(i));
        energies.push_back(E);
    }

    const Segment segments[] = {
        {100, 1.0e-2},
        {100, 5.0e-2},
        {100, 1.0e-1},
        {100, 1.0},
        {100, 10.0},
        {100, 100.0},
        {100, 1000.0},
        {200, 10000.0},
        {100, 100000.0}
    };

    for (const auto& seg : segments) {
        double current = energies.empty() ? 0.0 : energies.back();
        for (int i = 0; i < seg.steps; ++i) {
            current += seg.dE;
            energies.push_back(current);
        }
    }

    std::vector<EnergyGridPoint> grid;
    grid.reserve(energies.size());
    for (size_t i = 0; i < energies.size(); ++i) {
        double dE = 0.0;
        if (i == 0) {
            dE = energies[1] - energies[0];
        } else if (i == energies.size() - 1) {
            dE = energies[i] - energies[i - 1];
        } else {
            dE = 0.5 * (energies[i + 1] - energies[i - 1]);
        }
        grid.push_back({energies[i], dE});
    }

    return grid;
}

} // namespace dcr::physics

