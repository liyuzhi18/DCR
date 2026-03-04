#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace dcr::physics {

enum class WallMaterial {
    Carbon,
    Tungsten,
    Unknown
};

struct ReflectionCoefficients {
    double R_N = 0.0;      // particle reflection coefficient (alpha)
    double R_E = 0.0;      // energy reflection coefficient
    double alpha = 0.0;    // same as R_N
    double gamma_E = 0.0;  // R_E / alpha
};

inline WallMaterial parse_wall_material(const std::string& material) {
    std::string upper;
    upper.reserve(material.size());
    for (char c : material) upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));

    if (upper == "C" || upper == "CARBON" || upper == "GRAPHITE") return WallMaterial::Carbon;
    if (upper == "W" || upper == "TUNGSTEN") return WallMaterial::Tungsten;
    return WallMaterial::Unknown;
}

struct ReflectionFit {
    double a1 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double a4 = 0.0;
    double eps_L = 1.0; // scaling factor (eV)
};

inline double evaluate_reflection(const ReflectionFit& fit, double incident_energy_ev) {
    if (incident_energy_ev <= 0.0 || fit.eps_L <= 0.0) return 0.0;
    const double eps = incident_energy_ev / fit.eps_L;
    if (eps <= 0.0) return 0.0;

    const double num = fit.a1 * std::pow(eps, fit.a2);
    const double denom = 1.0 + fit.a3 * std::pow(eps, fit.a4);
    if (!(denom > 0.0) || !std::isfinite(num)) return 0.0;

    double R = num / denom;
    if (!std::isfinite(R)) return 0.0;
    return std::clamp(R, 0.0, 1.0);
}

inline ReflectionCoefficients reflection_coefficients(WallMaterial material, double incident_energy_ev) {
    // Fit constants for H on C and H on W at normal incidence (from your table).
    // R_N parameters
    const ReflectionFit C_RN{0.2119e0, -0.2217e0, 0.2461e0, 0.1316e1, 4.14659e2};
    const ReflectionFit W_RN{0.4116e0, -0.9122e-1, 0.6956e0, 0.1357e1, 9.86986e3};
    // R_E parameters
    const ReflectionFit C_RE{0.8723e-1, -0.2768e0, 0.3988e0, 0.1282e1, 4.14659e2};
    const ReflectionFit W_RE{0.2039e0, -0.1501e0, 0.1064e1, 0.1359e1, 9.86986e3};

    ReflectionCoefficients out{};
    switch (material) {
        case WallMaterial::Carbon:
            out.R_N = evaluate_reflection(C_RN, incident_energy_ev);
            out.R_E = evaluate_reflection(C_RE, incident_energy_ev);
            break;
        case WallMaterial::Tungsten:
            out.R_N = evaluate_reflection(W_RN, incident_energy_ev);
            out.R_E = evaluate_reflection(W_RE, incident_energy_ev);
            break;
        default:
            return out;
    }

    out.alpha = out.R_N;
    out.gamma_E = (out.alpha > 0.0) ? (out.R_E / out.alpha) : 0.0;
    return out;
}

inline ReflectionCoefficients reflection_coefficients(const std::string& material,
                                                      double incident_energy_ev) {
    return reflection_coefficients(parse_wall_material(material), incident_energy_ev);
}

} // namespace dcr::physics
