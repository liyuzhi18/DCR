# CMake generated Testfile for 
# Source directory: /Users/asimida/Desktop/DCR_code
# Build directory: /Users/asimida/Desktop/DCR_code/src
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(StructureCheck "/Users/asimida/Desktop/DCR_code/src/bin/check_structure")
set_tests_properties(StructureCheck PROPERTIES  _BACKTRACE_TRIPLES "/Users/asimida/Desktop/DCR_code/CMakeLists.txt;109;add_test;/Users/asimida/Desktop/DCR_code/CMakeLists.txt;0;")
add_test(PhysicsCheck "/Users/asimida/Desktop/DCR_code/src/bin/check_physics")
set_tests_properties(PhysicsCheck PROPERTIES  _BACKTRACE_TRIPLES "/Users/asimida/Desktop/DCR_code/CMakeLists.txt;117;add_test;/Users/asimida/Desktop/DCR_code/CMakeLists.txt;0;")
subdirs("_deps/yaml-cpp-build")
