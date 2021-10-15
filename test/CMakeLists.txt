set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)
set (CTEST_BINARY_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

include_directories("${PROJECT_SOURCE_DIR}/src")

file (GLOB test_files "test_*.cpp")

foreach (test_src ${test_files})
    string (REGEX REPLACE "(^.*/|\\.cpp$)" "" ex_name ${test_src})
    add_executable (${ex_name} ${test_src})

    target_link_libraries (${ex_name} mg_gcn_lib)
    add_test (NAME ${ex_name} COMMAND
        sh -c "
        ${CTEST_BINARY_DIRECTORY}/${ex_name}
        exit $?
        " --
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties (${ex_name} PROPERTIES TIMEOUT 120)
endforeach()

file (GLOB testers "*_tester.cpp")

foreach (tester_src ${testers})
    string (REGEX REPLACE "(^.*/|\\.cpp$)" "" ex_name ${tester_src})
    add_executable (${ex_name} ${tester_src})

    target_link_libraries (${ex_name} mg_gcn_lib)
endforeach()

# add_test (NAME basic_mpi COMMAND
#     bash -c "
#     mpirun -n 4 ${CTEST_BINARY_DIRECTORY}/basic_mpi_tester 4 | md5sum | grep ^7b583a790d173ebf03f5b1a3e72f4bf7 || exit 1
#     mpirun -n 15 ${CTEST_BINARY_DIRECTORY}/basic_mpi_tester 15 | md5sum | grep ^9306ec0709cc72558045559ada26573b || exit 1
#     exit 0
#     " --
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})