cmake_minimum_required(VERSION 3.15)

# --- 1. Policy Settings ---
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(DCR_Solver_Pro VERSION 1.0 LANGUAGES CXX)

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
    src/solver/marching/CellSolve.cpp
    src/solver/marching/LocalSystemAssembler.cpp
    src/solver/marching/CellAdvance.cpp
    src/solver/marching/MarchingDiagnostics.cpp
    src/solver/output/HDF5Output.cpp
    src/solver/WallBoundary.cpp
)
target_include_directories(dcr_solver PUBLIC src/solver)
target_include_directories(dcr_solver PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(dcr_solver PUBLIC dcr_state dcr_io dcr_physics dcr_atomic ${HDF5_CXX_LIBRARIES})
target_compile_definitions(dcr_solver PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

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

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_marching_implicit.cpp")
    add_executable(check_marching_implicit tests/test_marching_implicit.cpp)
    target_compile_definitions(check_marching_implicit PRIVATE DCR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(check_marching_implicit PRIVATE dcr_solver)
    add_test(NAME MarchingImplicitCheck COMMAND check_marching_implicit)
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
endif()
