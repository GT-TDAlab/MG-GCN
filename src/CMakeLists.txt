
set (mg_gcn_src
    matrix.hpp
    cuda_utils.hpp
    gcn.hpp
    mg_gcn.hpp
    pagerank.hpp
    cuda_utils.cu
)

OPTION(LOG "Enable logging" OFF)
IF(LOG)
    ADD_DEFINITIONS(-DLOG)
ENDIF(LOG)

add_library (mg_gcn_lib STATIC ${mg_gcn_src})

add_library(__caffe2_nccl INTERFACE)
target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})

find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

target_link_libraries(mg_gcn_lib
                    CUDA::cusparse
                    CUDA::cublas
                    __caffe2_nccl
                    pigo
                    Threads::Threads
                    TBB::tbb)

add_executable (mg_gcn main.cpp)

target_link_libraries(mg_gcn mg_gcn_lib TBB::tbb)

install (TARGETS mg_gcn DESTINATION ${CMAKE_INSTALL_BINDIR})
