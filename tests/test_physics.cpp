#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>

#include "../src/physics/Sheath.hpp"
#include "../src/physics/WallReflection.hpp"
#include "../src/physics/WallBoundary.hpp"

bool is_close(double a, double b, double rel_tol = 1e-3) {
    double diff = std::abs(a - b);
    double avg = (std::abs(a) + std::abs(b)) / 2.0;
    return diff < (rel_tol * avg);
}

int main() {
    std::cout << "--- Testing Physics Module (CGS Units) ---\n";
    std::cout << std::scientific << std::setprecision(4);

    double Te = 10.0;
    double Ti = 0.0;
    double mass_amu = 2.0; 

    // Manual Calculation Check:
    // 10 eV = 1.60218e-11 erg
    // Mass = 2.0 * 1.66054e-24 g = 3.32108e-24 g
    // c_s = sqrt( 1.60218e-11 / 3.32108e-24 ) 
    //     = sqrt( 4.8242e12 ) 
    //     ≈ 2.1964e6 cm/s
    double expected_cs = 2.1964e6;

    double cs = dcr::physics::calculate_Bohm_speed(Te, Ti, mass_amu);
    
    std::cout << "  Calculated c_s (D, 10eV): " << cs << " cm/s\n";
    std::cout << "  Expected   c_s          : " << expected_cs << " cm/s\n";

    if (is_close(cs, expected_cs, 0.01)) { // Allow 1% tolerance for constant variations
        std::cout << "[PASS] Bohm Speed logic is correct.\n";
    } else {
        std::cerr << "[FAIL] Bohm Speed mismatch!\n";
        return 1;
    }

    // --- Case 2: Neutral Thermal Speed ---
    // Parameters: D atom (A=2.0), T = 0.1 eV (Room temp-ish / Wall temp)
    double T_neutral = 0.1;
    
    // Manual Calculation:
    // v_th = sqrt( 8 * k * T / pi * m )
    //      = sqrt( 8 * (0.1 * 1.602e-12) / (3.14159 * 3.321e-24) )
    //      ≈ 3.504e5 cm/s
    double expected_vth = 3.504e5;

    double vth = dcr::physics::calculate_thermal_speed(T_neutral, mass_amu);

    std::cout << "\n  Calculated v_th (D, 0.1eV): " << vth << " cm/s\n";
    std::cout << "  Expected   v_th           : " << expected_vth << " cm/s\n";

    if (is_close(vth, expected_vth, 0.01)) {
        std::cout << "[PASS] Thermal Speed logic is correct.\n";
    } else {
        std::cerr << "[FAIL] Thermal Speed mismatch!\n";
        return 1;
    }

    // --- Case 3: Wall reflection coefficients (H on W) ---
    double incident_energy_ev = 100.0;
    auto coeffs = dcr::physics::reflection_coefficients("W", incident_energy_ev);
    std::cout << "\n  Reflection (W, 100 eV): alpha=" << coeffs.alpha
              << "  R_E=" << coeffs.R_E
              << "  gamma_E=" << coeffs.gamma_E << "\n";
    if (coeffs.alpha >= 0.0 && coeffs.alpha <= 1.0 &&
        coeffs.R_E >= 0.0 && coeffs.R_E <= 1.0) {
        std::cout << "[PASS] Wall reflection coefficients are bounded.\n";
    } else {
        std::cerr << "[FAIL] Wall reflection coefficients out of bounds!\n";
        return 1;
    }

    // --- Case 4: Ion impact energy at sheath entrance ---
    double sheath_drop = 3.0; // ~3*Te
    double expected_E0 = 0.5 * Te + sheath_drop * Te; // Ti=0, gamma_i=0
    double E0 = dcr::physics::calculate_ion_impact_energy_ev(Te, Ti, mass_amu, sheath_drop);
    std::cout << "\n  Ion impact energy (Te=10eV, drop=3Te): " << E0 << " eV\n";
    if (is_close(E0, expected_E0, 1e-3)) {
        std::cout << "[PASS] Ion impact energy logic is correct.\n";
    } else {
        std::cerr << "[FAIL] Ion impact energy mismatch!\n";
        return 1;
    }

    // --- Case 5: Wall recycling wrapper ---
    auto wall = dcr::physics::compute_wall_recycling("W", Te, Ti, mass_amu, sheath_drop);
    std::cout << "\n  Wall recycling (W): E0=" << wall.ion_impact_energy_ev
              << " alpha_atom=" << wall.alpha_atom
              << " alpha_mol=" << wall.alpha_molecule << "\n";
    if (wall.alpha_atom >= 0.0 && wall.alpha_atom <= 1.0 &&
        wall.alpha_molecule >= 0.0 && wall.alpha_molecule <= 1.0) {
        std::cout << "[PASS] Wall recycling fractions are bounded.\n";
    } else {
        std::cerr << "[FAIL] Wall recycling fractions out of bounds!\n";
        return 1;
    }

    return 0;

}
