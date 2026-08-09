cmake_minimum_required(VERSION 3.15...3.30)

# Builds the `nacho` Python extension from whatever the nacho compiler last emitted into
# generated/, plus the hand-written runtime and baselines. Separate from the top-level
# project, which builds the compiler itself and must not require CUDA.
#
# Driven by scikit-build-core (see ../pyproject.toml). The usual loop is:
#   cmake -S . -B build && cmake --build build -j    # build the compiler
#   ./build/compiler --kernels csr_mul               # emit generated/
#   pip install --no-build-isolation -ve .           # build this

set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
project(nacho_ext LANGUAGES CUDA CXX)

# C++20 for std::atomic_ref, which the host reduction kernels use.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NACHO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(NACHO_GENERATED_DIR "${NACHO_ROOT}/generated")
set(NACHO_RUNTIME_DIR "${NACHO_ROOT}/runtime")
set(NACHO_BASELINES_DIR "${NACHO_ROOT}/baselines")

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# cuCollections, used by the hand-written comparison baselines.
set(CPM_DOWNLOAD_VERSION 0.39.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
if (NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CPM_DOWNLOAD_LOCATION}")
endif ()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

# Only generated/ is globbed. The compiler wipes that directory on every run, so the glob
# is exactly the kernels that were asked for — nothing stale survives to be compiled.
file(GLOB NACHO_GENERATED_CPP CONFIGURE_DEPENDS "${NACHO_GENERATED_DIR}/*.cpp")
file(GLOB NACHO_GENERATED_CU CONFIGURE_DEPENDS "${NACHO_GENERATED_DIR}/*.cu")

if (NOT NACHO_GENERATED_CPP)
    message(FATAL_ERROR
        "No generated sources in ${NACHO_GENERATED_DIR}. "
        "Build the compiler and run it first, e.g. ./build/compiler --kernels csr_mul")
endif ()

# Hand-written sources are listed explicitly: they are stable, and an accidental stray
# file in these directories should not silently join the build.
set(NACHO_RUNTIME_SRC
    "${NACHO_RUNTIME_DIR}/register_runtime.cpp"
    "${NACHO_RUNTIME_DIR}/cuda_utils/cuda_utils.cu"
)

# Hand-written comparison kernels. Independent of generated/, so they build whatever
# subset of kernels the compiler last emitted.
set(NACHO_BASELINE_SRC
    "${NACHO_BASELINES_DIR}/csr_add/csr_add_baselines_nb.cpp"
    "${NACHO_BASELINES_DIR}/csr_add/csr_add_taco.cu"
    "${NACHO_BASELINES_DIR}/csr_add/csr_add_cusparse.cu"
)

nanobind_add_module(
  nacho_ext
  STABLE_ABI
  NB_STATIC
  ${NACHO_GENERATED_CPP}
  ${NACHO_GENERATED_CU}
  ${NACHO_RUNTIME_SRC}
  ${NACHO_BASELINE_SRC}
)

target_include_directories(nacho_ext PRIVATE
    ${NACHO_GENERATED_DIR}
    ${NACHO_RUNTIME_DIR}
    ${NACHO_BASELINES_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(nacho_ext PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(nacho_ext PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g -O0>
)

target_link_libraries(nacho_ext PRIVATE CUDA::cusparse cuco TBB::tbb)

install(TARGETS nacho_ext LIBRARY DESTINATION nacho)
