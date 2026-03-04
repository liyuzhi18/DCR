#include "ConfigPaths.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace dcr::solver {

namespace {

std::string resolve_root_path(const std::string& root_str, const std::filesystem::path& config_path) {
    if (root_str.empty()) return root_str;
    std::filesystem::path root = root_str;
    if (!root.is_relative()) {
        return root.lexically_normal().string();
    }

    // Prefer paths relative to the config file.
    std::filesystem::path candidate = config_path.parent_path() / root;
    if (std::filesystem::exists(candidate)) {
        return candidate.lexically_normal().string();
    }

#ifdef DCR_SOURCE_DIR
    std::filesystem::path fallback = std::filesystem::path(DCR_SOURCE_DIR) / root;
    if (std::filesystem::exists(fallback)) {
        return fallback.lexically_normal().string();
    }
#endif

    // Last resort: current working directory.
    candidate = std::filesystem::current_path() / root;
    if (std::filesystem::exists(candidate)) {
        return candidate.lexically_normal().string();
    }

    // If nothing exists, keep config-relative (better error message).
    return (config_path.parent_path() / root).lexically_normal().string();
}

void normalize_atomic_root(dcr::io::Config& config, const std::filesystem::path& config_path) {
    if (config.io.atomic_data_root.empty()) return;
    config.io.atomic_data_root = resolve_root_path(config.io.atomic_data_root, config_path);
    if (!config.io.atomic_data_root.empty() && config.io.atomic_data_root.back() != '/') {
        config.io.atomic_data_root.push_back('/');
    }
}

void normalize_data_tables_root(dcr::io::Config& config, const std::filesystem::path& config_path) {
    if (config.io.data_tables_root.empty()) return;
    config.io.data_tables_root = resolve_root_path(config.io.data_tables_root, config_path);
    if (!config.io.data_tables_root.empty() && config.io.data_tables_root.back() != '/') {
        config.io.data_tables_root.push_back('/');
    }
}

} // namespace

std::string resolve_config_path() {
    const char* env = std::getenv("DCR_CONFIG");
    if (env && *env) return std::string(env);

    std::filesystem::path candidate = "config/hydro_am.yaml";
    if (std::filesystem::exists(candidate)) return candidate.string();

#ifdef DCR_SOURCE_DIR
    std::filesystem::path fallback = std::filesystem::path(DCR_SOURCE_DIR) / "config" / "hydro_am.yaml";
    if (std::filesystem::exists(fallback)) return fallback.string();
#endif

    throw std::runtime_error("Unable to locate config file; set DCR_CONFIG or run from repo root.");
}

void normalize_input_roots(dcr::io::Config& config, const std::string& config_path) {
    const std::filesystem::path p(config_path);
    normalize_atomic_root(config, p);
    normalize_data_tables_root(config, p);
}

} // namespace dcr::solver

