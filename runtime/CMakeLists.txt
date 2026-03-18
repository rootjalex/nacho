cmake_minimum_required(VERSION 3.15...3.26)

message("Building for Redwood")
set(CMAKE_CUDA_ARCHITECTURES 89)
project(nacho_runtime LANGUAGES CUDA CXX)

# -----------------------------------------------------------------------------
# nacho codegen integration
# -----------------------------------------------------------------------------
# Runtime kernels for nacho-generated ops are produced by the compiler binary.
# Keep these generated .cu files in the build directory so the source tree
# stays clean and reproducible.
get_filename_component(NACHO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(NACHO_BUILD_DIR "${NACHO_REPO_ROOT}/build")
set(NACHO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/nacho_generated")

set(NACHO_GENERATED_OPS
  sparse_vec_mul
  sparse_vec_add
  sparse_vec_apb_c
  sparse_vec_ab_pc
  dcsr_mul
  dcsr_add
  csr_add
  tcsf_add
)

# Allow callers to provide a prebuilt compiler path. If omitted, default to
# the top-level build output and bootstrap the compiler if it does not exist.
set(NACHO_COMPILER "" CACHE FILEPATH "Path to the nacho compiler executable.")
if (NACHO_COMPILER)
  set(NACHO_COMPILER_EXE "${NACHO_COMPILER}")
else ()
  set(NACHO_COMPILER_EXE "${NACHO_BUILD_DIR}/compiler")
endif ()

if (NOT EXISTS "${NACHO_COMPILER_EXE}")
  message(STATUS
          "nacho compiler not found at '${NACHO_COMPILER_EXE}'. "
          "Bootstrapping compiler build in '${NACHO_BUILD_DIR}'.")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${NACHO_REPO_ROOT}" -B "${NACHO_BUILD_DIR}"
    RESULT_VARIABLE NACHO_COMPILER_CONFIG_RESULT
  )
  if (NOT NACHO_COMPILER_CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Failed to configure nacho compiler in '${NACHO_BUILD_DIR}'. "
            "Provide -DNACHO_COMPILER=/abs/path/to/compiler to override.")
  endif ()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${NACHO_BUILD_DIR}" --target nacho_compiler
    RESULT_VARIABLE NACHO_COMPILER_BUILD_RESULT
  )
  if (NOT NACHO_COMPILER_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Failed to build target 'nacho_compiler'. "
            "Provide -DNACHO_COMPILER=/abs/path/to/compiler to override.")
  endif ()
endif ()

if (NOT EXISTS "${NACHO_COMPILER_EXE}")
  message(FATAL_ERROR
          "nacho compiler executable does not exist at '${NACHO_COMPILER_EXE}'.")
endif ()

message(STATUS "Using nacho compiler: ${NACHO_COMPILER_EXE}")

# Generate one .cu file per op into the build tree.
set(NACHO_GENERATED_CUDA_SRC)
foreach (NACHO_OP IN LISTS NACHO_GENERATED_OPS)
  set(NACHO_OP_CU "${NACHO_GENERATED_DIR}/${NACHO_OP}.cu")
  add_custom_command(
    OUTPUT "${NACHO_OP_CU}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NACHO_GENERATED_DIR}"
    COMMAND "${NACHO_COMPILER_EXE}" --emit "${NACHO_GENERATED_DIR}" --name "${NACHO_OP}"
    DEPENDS "${NACHO_COMPILER_EXE}"
    COMMENT "Generating nacho kernel ${NACHO_OP}.cu"
    VERBATIM
  )
  list(APPEND NACHO_GENERATED_CUDA_SRC "${NACHO_OP_CU}")
endforeach ()
set_source_files_properties(${NACHO_GENERATED_CUDA_SRC} PROPERTIES GENERATED TRUE)

# Generate runtime wrapper files (nacho_ops.h, .hpp, .cpp)
set(NACHO_OPS_H "${NACHO_GENERATED_DIR}/nacho_ops.h")
set(NACHO_OPS_HPP "${NACHO_GENERATED_DIR}/nacho_ops.hpp")
set(NACHO_OPS_CPP "${NACHO_GENERATED_DIR}/nacho_ops.cpp")
add_custom_command(
  OUTPUT "${NACHO_OPS_H}" "${NACHO_OPS_HPP}" "${NACHO_OPS_CPP}"
  COMMAND "${NACHO_COMPILER_EXE}" --emit-wrappers "${NACHO_GENERATED_DIR}"
  DEPENDS "${NACHO_COMPILER_EXE}"
  COMMENT "Generating nacho runtime wrapper code"
  VERBATIM
)
set_source_files_properties("${NACHO_OPS_CPP}" PROPERTIES GENERATED TRUE)

add_custom_target(generate_nacho_kernels DEPENDS
  ${NACHO_GENERATED_CUDA_SRC}
  "${NACHO_OPS_H}" "${NACHO_OPS_HPP}" "${NACHO_OPS_CPP}"
)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject] torch
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
# message(CMAKE_CUDA_ARCHITECTURES)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--use_fast_math")


set(nacho_runtime_CUDA_INCLUDES
    # Include folders go here
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(nacho_runtime_CUDA_SRC
  src/cuda_utils/cuda_utils.cu
  src/csr_add/csr_add.cu
  src/coo_add/coo_add.cu
  src/mergepath_utils/mergepath_utils.cu
  src/sparse_vector/sparse_vector.cu
  src/sparse_vector/sp_ab_c.cu
  src/broadcasts/broadcasts.cu
  src/spgemm/spgemm.cu
)
list(APPEND nacho_runtime_CUDA_SRC ${NACHO_GENERATED_CUDA_SRC})

set(CPM_DOWNLOAD_VERSION 0.39.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CPM_DOWNLOAD_LOCATION}"
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)


set(nacho_runtime_CUDA_DEPENDENCIES
    CUDA::cusparse
    cuco
)

add_library(nacho_runtime_CUDA_KERNELS STATIC ${nacho_runtime_CUDA_SRC})
add_dependencies(nacho_runtime_CUDA_KERNELS generate_nacho_kernels)
set_target_properties(nacho_runtime_CUDA_KERNELS
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(nacho_runtime_CUDA_KERNELS PRIVATE ${nacho_runtime_CUDA_DEPENDENCIES} ${Python_LIBRARIES})
target_include_directories(nacho_runtime_CUDA_KERNELS PRIVATE ${nacho_runtime_CUDA_INCLUDES})

# All source files
set(nacho_runtime_CPP_SRC
    src/nacho_runtime_ext.cpp
    src/csr_add/csr_add.cpp  # Add this if it exists
    src/coo_add/coo_add.cpp
    src/sparse_vector/sparse_vector.cpp
    src/broadcasts/broadcasts.cpp
    src/spgemm/spgemm.cpp
    ${NACHO_OPS_CPP}
)

# Add the CUDA flags (Redwood GPU arch)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -G -g -lineinfo -arch=sm_89 -dc")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  nacho_runtime_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  # src/nacho_runtime_ext.cpp
  ${nacho_runtime_CPP_SRC}
  ${nacho_runtime_CUDA_SRC}
)
add_dependencies(nacho_runtime_ext generate_nacho_kernels)

# Include paths for generated wrapper files: the generated nacho_ops.cpp needs
# nb_utils.hpp and cuda_utils/cuda_utils.h from src/, and nacho_runtime_ext.cpp
# needs nacho_generated/nacho_ops.hpp from the build tree.
target_include_directories(nacho_runtime_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Link the CUDA libraries (optional, only needed if you require specific CUDA libraries)
# target_link_libraries(nacho_runtime_ext PRIVATE CUDA::cudart)
target_link_libraries(nacho_runtime_ext PRIVATE ${nacho_runtime_CUDA_DEPENDENCIES})

# Install directive for scikit-build-core
install(TARGETS nacho_runtime_ext LIBRARY DESTINATION nacho_runtime)
