cmake_minimum_required(VERSION 3.15)

# --- 1. Policy Settings ---
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(DCR_Solver_Pro VERSION 1.0 LANGUAGES CXX)

option(DCR_ENABLE_SLOW_TESTS
    "Register expensive reduced global-BVP integration tests" OFF)
option(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP
    "Compile the experimental target-conditioned global BVP" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =========================================================
# 2. Dependencies
# =========================================================
include(FetchContent)

# --- (A) yaml-cpp (Standard Build) ---
message(STATUS "--- FETCHING YAML-CPP ---")
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0
  GIT_SHALLOW    TRUE
  OVERRIDE_FIND_PACKAGE
)
# Attempt to disable install to be safe
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# --- (B) Eigen3 (Header-Only Fix) ---
message(STATUS "--- FETCHING EIGEN3 (Headers Only) ---")

FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# CRITICAL FIX: Manually populate to avoid "uninstall" target conflict
FetchContent_GetProperties(Eigen3)
if(NOT Eigen3_POPULATED)
  FetchContent_Populate(Eigen3)
  
  # Manually create the library target. 
  # We bypass Eigen's own CMakeLists.txt entirely.
  add_library(eigen_headers INTERFACE)
  target_include_directories(eigen_headers INTERFACE "${eigen3_SOURCE_DIR}")
  
  # Create the alias expected by your code (Eigen3::Eigen)
  add_library(Eigen3::Eigen ALIAS eigen_headers)
endif()

# --- (C) HDF5 (Output) ---
find_package(HDF5 REQUIRED COMPONENTS CXX)

# =========================================================
# 3. Modular Libraries
# =========================================================

# --- BASE ---
add_library(dcr_base INTERFACE)
target_include_directories(dcr_base INTERFACE src/base)
target_link_libraries(dcr_base INTERFACE Eigen3::Eigen)

# --- PHYSICS ---
add_library(dcr_physics INTERFACE)
target_include_directories(dcr_physics INTERFACE src/physics)
target_link_libraries(dcr_physics INTERFACE dcr_base)

# --- IO ---
add_library(dcr_io 
    src/io/ConfigLoader.cpp
    src/io/DatabaseLoader.cpp
)
target_include_directories(dcr_io PUBLIC src/io)
target_link_libraries(dcr_io PUBLIC dcr_base yaml-cpp)

# --- STATE ---
add_library(dcr_state 
    src/state/Mesh.cpp
    src/state/PlasmaState.cpp
)
target_include_directories(dcr_state PUBLIC src/state)
target_link_libraries(dcr_state PUBLIC dcr_base dcr_physics)

# --- ATOMIC ---
add_library(dcr_atomic
    src/atomic/AtomicData.cpp
    src/atomic/SpeciesFileReader.cpp
)
target_include_directories(dcr_atomic PUBLIC src/atomic src/processes)
target_link_libraries(dcr_atomic PUBLIC dcr_base dcr_state)

# --- SOLVER ---
add_library(dcr_solver 
    src/solver/DCR_Solver.cpp
    src/solver/boundary/BoundaryPhase.cpp
    src/solver/core/ConfigPaths.cpp
    src/solver/core/RateAnalysis.cpp
    src/solver/marching/MarchingDriver.cpp
    src/solver/marching/AdaptiveRecycling.cpp
    src/solver/marching/CellSolve.cpp
    src/solver/marching/LogNewtonCellSolve.cpp
    src/solver/marching/LocalSystemAssembler.cpp
    src/solver/marching/CellAdvance.cpp
    src/solver/marching/MarchingDiagnostics.cpp
    src/solver/output/HDF5Output.cpp
    src/solver/qss/internal/QSSAtomicDonorReduction.cpp
    src/solver/qss/internal/QSSBoundaryTemplate.cpp
    src/solver/qss/internal/QSSLayout.cpp
    src/solver/qss/internal/QSSLayoutSupport.cpp
    src/solver/qss/internal/QSSLinearSolve.cpp
    src/solver/qss/internal/QSSLocalBlocks.cpp
    src/solver/qss/internal/QSSMatrixBlocks.cpp
    src/solver/qss/internal/QSSPopulationLift.cpp
    src/solver/qss/internal/QSSReduction.cpp
    src/solver/qss/internal/QSSBackgroundSummary.cpp
    src/solver/qss/internal/QSSFlowSeed.cpp
    src/solver/qss/internal/QSSGrid.cpp
    src/solver/qss/internal/QSSLinear.cpp
    src/solver/qss/internal/QSSLogBoundary.cpp
    src/solver/qss/internal/QSSLogLayout.cpp
    src/solver/qss/internal/QSSLogRates.cpp
    src/solver/qss/internal/QSSNormalization.cpp
    src/solver/qss/internal/QSSPopulationOps.cpp
    src/solver/qss/internal/QSSRecyclingModel.cpp
    src/solver/qss/internal/QSSSourceAssembly.cpp
    src/solver/qss/internal/QSSTransport.cpp
    src/solver/qss/QSSBoundary.cpp
    src/solver/qss/QSSMarching.cpp
    src/solver/qss/QSSSolver.cpp
    src/solver/WallBoundary.cpp
)
target_include_directories(dcr_solver PUBLIC src/solver)
target_include_directories(dcr_solver PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(dcr_solver PUBLIC dcr_state dcr_io dcr_physics dcr_atomic ${HDF5_CXX_LIBRARIES})
target_compile_definitions(dcr_solver PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP)
    target_sources(dcr_solver PRIVATE src/solver/marching/GlobalBVP.cpp)
    target_compile_definitions(dcr_solver PUBLIC DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP=1)
endif()

# =========================================================
# 4. Main Executable
# =========================================================
add_executable(DCR_Main src/main.cpp)
target_link_libraries(DCR_Main PRIVATE dcr_solver)

# =========================================================
# 5. Unit Testing
# =========================================================
enable_testing()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_structure.cpp")
    add_executable(check_structure tests/test_structure.cpp)
    target_link_libraries(check_structure PRIVATE dcr_base)
    add_test(NAME StructureCheck COMMAND check_structure)
else()
    message(WARNING "Skipping tests: 'tests/test_structure.cpp' not found.")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_physics.cpp")
    add_executable(check_physics tests/test_physics.cpp)
    target_link_libraries(check_physics PRIVATE dcr_physics)
    add_test(NAME PhysicsCheck COMMAND check_physics)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_state.cpp")
    add_executable(check_state tests/test_state.cpp)
    target_link_libraries(check_state PRIVATE dcr_state)
    add_test(NAME StateCheck COMMAND check_state)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_io.cpp")
    add_executable(check_io tests/test_io.cpp)
    target_link_libraries(check_io PRIVATE dcr_io)
    add_test(NAME IOCheck COMMAND check_io)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_temperature_profile.cpp")
    add_executable(check_temperature_profile tests/test_temperature_profile.cpp)
    target_link_libraries(check_temperature_profile PRIVATE dcr_state)
    add_test(NAME TemperatureProfileCheck COMMAND check_temperature_profile)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_eedf_generalized_power.cpp")
    add_executable(check_eedf_generalized_power tests/test_eedf_generalized_power.cpp)
    target_link_libraries(check_eedf_generalized_power PRIVATE dcr_physics)
    add_test(NAME EEDFGeneralizedPowerCheck COMMAND check_eedf_generalized_power)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_atomic_reader.cpp")
    add_executable(check_atomic_reader
        tests/test_atomic_reader.cpp
        src/atomic/SpeciesFileReader.cpp
    )
    target_compile_definitions(check_atomic_reader PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(check_atomic_reader PRIVATE src/atomic)
    add_test(NAME AtomicReaderCheck COMMAND check_atomic_reader)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_atomic_data.cpp")
    add_executable(check_atomic_data tests/test_atomic_data.cpp)
    target_compile_definitions(check_atomic_data PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_atomic_data PRIVATE dcr_atomic)
    add_test(NAME AtomicDataCheck COMMAND check_atomic_data)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_atomic_process.cpp")
    add_executable(check_atomic_process tests/test_atomic_process.cpp)
    target_link_libraries(check_atomic_process PRIVATE dcr_state dcr_base)
    add_test(NAME AtomicProcessCheck COMMAND check_atomic_process)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_molecular_process.cpp")
    add_executable(check_molecular_process tests/test_molecular_process.cpp)
    target_link_libraries(check_molecular_process PRIVATE dcr_state dcr_base)
    add_test(NAME MolecularProcessCheck COMMAND check_molecular_process)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_rate_matrix.cpp")
    add_executable(check_rate_matrix tests/test_rate_matrix.cpp)
    target_compile_definitions(check_rate_matrix PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_rate_matrix PRIVATE dcr_atomic dcr_state dcr_base)
    add_test(NAME RateMatrixCheck COMMAND check_rate_matrix)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_rate_analysis.cpp")
    add_executable(check_rate_analysis tests/test_rate_analysis.cpp)
    target_compile_definitions(check_rate_analysis PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_rate_analysis PRIVATE dcr_solver)
    add_test(NAME RateAnalysisCheck COMMAND check_rate_analysis)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_mcx_integration.cpp")
    add_executable(check_mcx_integration tests/test_mcx_integration.cpp)
    target_compile_definitions(check_mcx_integration PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_mcx_integration PRIVATE dcr_atomic dcr_state dcr_base)
    add_test(NAME MCXIntegrationCheck COMMAND check_mcx_integration)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_boundary_phase.cpp")
    add_executable(check_boundary_phase tests/test_boundary_phase.cpp)
    target_compile_definitions(check_boundary_phase PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_boundary_phase PRIVATE dcr_solver)
    add_test(NAME BoundaryPhaseCheck COMMAND check_boundary_phase)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_eedf_boundary_regression.cpp")
    add_executable(check_eedf_boundary_regression tests/test_eedf_boundary_regression.cpp)
    target_compile_definitions(check_eedf_boundary_regression PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_eedf_boundary_regression PRIVATE dcr_solver)
    add_test(NAME EEDFBoundaryRegressionCheck COMMAND check_eedf_boundary_regression)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/inspect_boundary_h_balance.cpp")
    add_executable(inspect_boundary_h_balance tests/inspect_boundary_h_balance.cpp)
    target_compile_definitions(inspect_boundary_h_balance PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(inspect_boundary_h_balance PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/compare_gcr_power_eedf.cpp")
    add_executable(compare_gcr_power_eedf tests/compare_gcr_power_eedf.cpp)
    target_compile_definitions(compare_gcr_power_eedf PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(compare_gcr_power_eedf PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/compare_gcr_bimaxwell_eedf.cpp")
    add_executable(compare_gcr_bimaxwell_eedf tests/compare_gcr_bimaxwell_eedf.cpp)
    target_compile_definitions(compare_gcr_bimaxwell_eedf PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(compare_gcr_bimaxwell_eedf PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/compare_gcr_rpl_eedf.cpp")
    add_executable(compare_gcr_rpl_eedf tests/compare_gcr_rpl_eedf.cpp)
    target_compile_definitions(compare_gcr_rpl_eedf PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(compare_gcr_rpl_eedf PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/inspect_gcr_rpl_case.cpp")
    add_executable(inspect_gcr_rpl_case tests/inspect_gcr_rpl_case.cpp)
    target_compile_definitions(inspect_gcr_rpl_case PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(inspect_gcr_rpl_case PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_marching_implicit.cpp")
    add_executable(check_marching_implicit tests/test_marching_implicit.cpp)
    target_compile_definitions(check_marching_implicit PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_marching_implicit PRIVATE dcr_solver)
    add_test(NAME MarchingImplicitCheck COMMAND check_marching_implicit)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_transport.cpp")
    add_executable(check_global_bvp_transport tests/test_global_bvp_transport.cpp)
    target_compile_definitions(check_global_bvp_transport PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_transport PRIVATE dcr_solver)
    add_test(NAME check_global_bvp_transport COMMAND check_global_bvp_transport)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_fixed_velocity_scale.cpp")
    add_executable(check_global_bvp_fixed_velocity_scale tests/test_global_bvp_fixed_velocity_scale.cpp)
    target_compile_definitions(check_global_bvp_fixed_velocity_scale PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_fixed_velocity_scale PRIVATE dcr_solver)
    add_test(NAME check_global_bvp_fixed_velocity_scale COMMAND check_global_bvp_fixed_velocity_scale)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_width_separation.cpp")
    add_executable(check_boundary_spatial_width_separation tests/test_global_bvp_width_separation.cpp)
    target_compile_definitions(check_boundary_spatial_width_separation PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_boundary_spatial_width_separation PRIVATE dcr_solver)
    add_test(NAME BoundarySpatialWidthSeparationCheck COMMAND check_boundary_spatial_width_separation)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_local_chemistry_conservation.cpp")
    add_executable(check_local_chemistry_conservation tests/test_local_chemistry_conservation.cpp)
    target_compile_definitions(check_local_chemistry_conservation PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_local_chemistry_conservation PRIVATE dcr_solver)
    add_test(NAME LocalChemistryConservationCheck COMMAND check_local_chemistry_conservation)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_jacobian.cpp")
    add_executable(check_global_bvp_jacobian tests/test_global_bvp_jacobian.cpp)
    target_compile_definitions(check_global_bvp_jacobian PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_jacobian PRIVATE dcr_solver)
    add_test(NAME check_global_bvp_jacobian COMMAND check_global_bvp_jacobian)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP)
foreach(global_bvp_reduced_test IN ITEMS
        forward_M_transport
        no_M_global_unknowns
        fd_M_coupling)
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_${global_bvp_reduced_test}.cpp")
        add_executable(check_global_bvp_${global_bvp_reduced_test}
            tests/test_global_bvp_${global_bvp_reduced_test}.cpp)
        target_compile_definitions(check_global_bvp_${global_bvp_reduced_test}
            PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        target_link_libraries(check_global_bvp_${global_bvp_reduced_test} PRIVATE dcr_solver)
        add_test(NAME check_global_bvp_${global_bvp_reduced_test}
            COMMAND check_global_bvp_${global_bvp_reduced_test})
    endif()
endforeach()
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_boundary_count.cpp")
    add_executable(check_global_bvp_boundary_count tests/test_global_bvp_boundary_count.cpp)
    target_compile_definitions(check_global_bvp_boundary_count PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_boundary_count PRIVATE dcr_solver)
    add_test(NAME check_global_bvp_boundary_count COMMAND check_global_bvp_boundary_count)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_two_node.cpp")
    add_executable(check_global_bvp_two_node tests/test_global_bvp_two_node.cpp)
    target_compile_definitions(check_global_bvp_two_node PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_two_node PRIVATE dcr_solver)
    add_test(NAME check_global_bvp_two_node COMMAND check_global_bvp_two_node)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_small_grid.cpp")
    add_executable(check_global_bvp_small_grid tests/test_global_bvp_small_grid.cpp)
    target_compile_definitions(check_global_bvp_small_grid PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_small_grid PRIVATE dcr_solver)
    add_test(NAME check_global_bvp_small_grid COMMAND check_global_bvp_small_grid)
    add_test(NAME check_global_bvp_nuclei_balance COMMAND check_global_bvp_small_grid)
    add_test(NAME check_global_bvp_flux_chemistry_balance COMMAND check_global_bvp_small_grid)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_global_bvp_grid_refinement.cpp")
    add_executable(check_global_bvp_reduced_refinement
        tests/test_global_bvp_grid_refinement.cpp)
    target_compile_definitions(check_global_bvp_reduced_refinement PRIVATE
        DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_global_bvp_reduced_refinement PRIVATE dcr_solver)
    set(global_bvp_refinement_checkpoint_dir
        "${CMAKE_CURRENT_BINARY_DIR}/Testing/global_bvp_refinement_checkpoints")
    add_test(NAME check_global_bvp_reduced_refinement_fast
        COMMAND check_global_bvp_reduced_refinement --fast
            "${global_bvp_refinement_checkpoint_dir}")
    set_tests_properties(check_global_bvp_reduced_refinement_fast PROPERTIES
        TIMEOUT 3600
        LABELS "short;global-bvp;refinement"
        FIXTURES_SETUP global_bvp_refinement_checkpoint)
    if(DCR_ENABLE_SLOW_TESTS)
        add_test(NAME check_global_bvp_reduced_refinement_slow
            COMMAND check_global_bvp_reduced_refinement --slow
                "${global_bvp_refinement_checkpoint_dir}")
        set_tests_properties(check_global_bvp_reduced_refinement_slow PROPERTIES
            TIMEOUT 10800
            LABELS "slow;global-bvp;refinement"
            FIXTURES_REQUIRED global_bvp_refinement_checkpoint)
    endif()
    add_custom_target(run_global_bvp_reduced_refinement_slow
        COMMAND check_global_bvp_reduced_refinement --slow
            "${global_bvp_refinement_checkpoint_dir}"
        DEPENDS check_global_bvp_reduced_refinement
        USES_TERMINAL)
endif()

if(DCR_ENABLE_EXPERIMENTAL_GLOBAL_BVP AND EXISTS "${CMAKE_SOURCE_DIR}/tests/run_global_bvp_physical.cpp")
    add_executable(run_global_bvp_physical tests/run_global_bvp_physical.cpp)
    target_compile_definitions(run_global_bvp_physical PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(run_global_bvp_physical PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_adaptive_recycling.cpp")
    add_executable(check_adaptive_recycling tests/test_adaptive_recycling.cpp)
    target_link_libraries(check_adaptive_recycling PRIVATE dcr_solver)
    add_test(NAME AdaptiveRecyclingCheck COMMAND check_adaptive_recycling)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_marching_driver.cpp")
    add_executable(check_marching_driver tests/test_marching_driver.cpp)
    target_compile_definitions(check_marching_driver PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_marching_driver PRIVATE dcr_solver)
    add_test(NAME MarchingDriverCheck COMMAND check_marching_driver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_hdf5_output.cpp")
    add_executable(check_hdf5_output tests/test_hdf5_output.cpp)
    target_compile_definitions(check_hdf5_output PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(check_hdf5_output PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(check_hdf5_output PRIVATE dcr_solver ${HDF5_CXX_LIBRARIES})
    add_test(NAME HDF5OutputCheck COMMAND check_hdf5_output)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_regression_reference_case.cpp")
    add_executable(check_regression_reference tests/test_regression_reference_case.cpp)
    target_compile_definitions(check_regression_reference PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_regression_reference PRIVATE dcr_solver)
    add_test(NAME RegressionReferenceCheck COMMAND check_regression_reference)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/inspect_boundary_h2_balance.cpp")
    add_executable(inspect_boundary_h2_balance tests/inspect_boundary_h2_balance.cpp)
    target_compile_definitions(inspect_boundary_h2_balance PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(inspect_boundary_h2_balance PRIVATE dcr_solver)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/verify_endpoint_standalone_cr.cpp")
    add_executable(verify_endpoint_standalone_cr tests/verify_endpoint_standalone_cr.cpp)
    target_compile_definitions(verify_endpoint_standalone_cr PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_include_directories(verify_endpoint_standalone_cr PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(verify_endpoint_standalone_cr PRIVATE dcr_solver ${HDF5_CXX_LIBRARIES})
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_postprocess_smoke.py")
        add_test(
            NAME PostprocessSmokeCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_postprocess_smoke.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_postprocess_labels.py")
        add_test(
            NAME PostprocessLabelCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_postprocess_labels.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_parser.py")
        add_test(
            NAME DissociationChannelParserCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_parser.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_smoke.py")
        add_test(
            NAME DissociationChannelSmokeCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_smoke.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_validation.py")
        add_test(
            NAME DissociationChannelValidationCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_validation.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_summary_table.py")
        add_test(
            NAME DissociationChannelSummaryTableCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_summary_table.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_nonadiabatic.py")
        add_test(
            NAME DissociationChannelNonadiabaticCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_dissociation_channel_nonadiabatic.py
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_ocr_reference_table.py")
        add_test(
            NAME OCRReferenceTableCheck
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_ocr_reference_table.py
        )
    endif()
endif()
