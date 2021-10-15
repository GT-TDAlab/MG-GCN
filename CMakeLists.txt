cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 OLD)

# ----------------------------------------------------------------------------
# Project
# ----------------------------------------------------------------------------
project (MG_GCN
         LANGUAGES CUDA CXX)

include (GNUInstallDirs)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pedantic -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DDEBUG -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
# C/CXX Flags and Standards
# ----------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------
# Other flags
# -------------------------------------------------------------

# set(CUDA_FTZ "--ftz=false" CACHE STRING "Set flush to zero for GPU")

# set(CUDA_MATH --use_fast_math CACHE STRING "Enable fast-math" )
# option(CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
# option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
# option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

# if(CUDA_SHOW_CODELINES)
#   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -Xcompiler -rdynamic -lineinfo)
#   set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
# endif(CUDA_SHOW_CODELINES)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
# if(CUDA_SHOW_REGISTER)
#   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
# endif(CUDA_SHOW_REGISTER)

# if(CUDA_KEEP_FILES)
#   make_directory("${PROJECT_BINARY_DIR}/nvcc_tmp")
#   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
# endif(CUDA_KEEP_FILES)

# save the current configuration to be used with the linker.
# any further additions to CUDA_NVCC_FLAGS will NOT be passed to
# the linker. See the modified FindCUDA.cmake module
# set(CUDA_NVCC_LINKER_FLAGS ${CUDA_NVCC_FLAGS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -Xcompiler -Wall -Xcompiler -Wextra -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -Xcompiler -march=native")

# ----------------------------------------------------------------------------
# Depedencies
# ----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(CUDAToolkit REQUIRED)
find_package(NCCL REQUIRED)

# ----------------------------------------------------------------------------
# Src Files and Targets
# ----------------------------------------------------------------------------
add_subdirectory (PIGO)
add_subdirectory (src)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
enable_testing ()

add_subdirectory (test)
