cmake_minimum_required(VERSION 3.30)
project(nacho VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CMakeDependentOption)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif ()

##
# Custom CMake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

##
# Project options

cmake_dependent_option(
    nacho_BUILD_TESTING
    "Compile nacho with common compiler warnings"
    "${PROJECT_IS_TOP_LEVEL}"
    BUILD_TESTING
    OFF
)

option(
    nacho_ENABLE_WARNINGS
    "Compile nacho with common compiler warnings"
    "${PROJECT_IS_TOP_LEVEL}"
)

option(
    nacho_SKIP_INSTALL_RULES
    "Exclude nacho's install rules from the build"
    OFF
)

##
# Process options
if (nacho_SKIP_INSTALL_RULES)
    set(CMAKE_SKIP_INSTALL_RULES ON)
endif ()

##
# Dependencies

##
# nacho compiler library

add_library(nacho_library STATIC)
add_library(nacho::library ALIAS nacho_library)

target_compile_features(nacho_library PUBLIC cxx_std_20)

target_sources(
    nacho_library
    PRIVATE
    src/CIN.cpp
    src/Compile.cpp
    src/Equality.cpp
    src/Format.cpp
    src/Frontend.cpp
    src/GeneratePartition.cpp
    src/Lattice.cpp
    src/Mutator.cpp
    src/Printer.cpp
    src/CPUPrinter.cpp
    src/Seq.cpp
    src/Simplify.cpp
    src/Visitor.cpp
    src/llir/Function.cpp
    src/llir/LLIR.cpp
    src/backend/compile.cpp
    src/backend/partition.cpp
    src/backend/tensor.cpp
    src/backend/compute.cpp
    src/backend/base_lowerer.cpp
)

target_sources(
    nacho_library
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
    include/CIN.h
    include/Compile.h
    include/Equality.h
    include/Format.h
    include/Frontend.h
    include/GeneratePartition.h
    include/IntrusivePtr.h
    include/IRFwdDecl.h
    include/IRHandle.h
    include/IRNode.h
    include/Lattice.h
    include/Mutator.h
    include/Nacho.h
    include/Printer.h
    include/CPUPrinter.h
    include/cpu_runtime.h
    include/RefCount.h
    include/Scope.h
    include/Seq.h
    include/Simplify.h
    include/Type.h
    include/Visitor.h
    include/llir/Function.h
    include/llir/LLIR.h
    include/backend/partition.h
    include/backend/compile.h
    include/backend/tensor.h
    include/backend/compute.h
    include/backend/base_lowerer.h
)

if (nacho_ENABLE_WARNINGS)
    target_compile_options(
        nacho_library
        PUBLIC
        "$<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-Wall>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W3>"
    )
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=undefined -fno-omit-frame-pointer")

##
# nacho compiler

add_executable(nacho_compiler compiler.cpp)
add_executable(nacho::compiler ALIAS nacho_compiler)
set_target_properties(
    nacho_compiler
    PROPERTIES
    EXPORT_NAME compiler
    OUTPUT_NAME compiler
)

target_link_libraries(nacho_compiler PRIVATE nacho::library)

##
# Testing

if (nacho_BUILD_TESTING)
    set(NACHO_TESTS
        sparse_vec_mul sparse_vec_add sparse_vec_apb_c sparse_vec_ab_pc
        dcsr_mul dcsr_add csr_add tcsf_add
        format_inference lattice locator
    )
    foreach (t IN LISTS NACHO_TESTS)
        add_test(NAME ${t} COMMAND nacho_compiler --test ${t})
    endforeach ()

    # CPU codegen tests: generate CPU code, compile, and run
    set(NACHO_CPU_EXPRESSIONS
        sparse_vec_mul sparse_vec_add sparse_vec_apb_c sparse_vec_ab_pc
        csr_add dcsr_mul dcsr_add tcsf_add coo_add coo_mul
    )

    set(NACHO_CPU_GEN_DIR "${CMAKE_BINARY_DIR}/cpu_gen")
    file(MAKE_DIRECTORY "${NACHO_CPU_GEN_DIR}")

    foreach(expr IN LISTS NACHO_CPU_EXPRESSIONS)
        add_custom_command(
            OUTPUT "${NACHO_CPU_GEN_DIR}/${expr}.cpp"
            COMMAND nacho_compiler --emit "${NACHO_CPU_GEN_DIR}" --name ${expr} --target cpu
            DEPENDS nacho_compiler
            COMMENT "Generating CPU code for ${expr}"
        )
        list(APPEND NACHO_CPU_GEN_SRC "${NACHO_CPU_GEN_DIR}/${expr}.cpp")
    endforeach()

    find_package(TBB REQUIRED)
    add_executable(cpu_codegen_test tests/cpu_codegen_test.cpp ${NACHO_CPU_GEN_SRC})
    target_include_directories(cpu_codegen_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${NACHO_CPU_GEN_DIR}"
    )
    target_link_libraries(cpu_codegen_test PRIVATE TBB::tbb)
    add_test(NAME cpu_codegen COMMAND cpu_codegen_test)

    # Shared library for Python ctypes access to all CPU-generated operations
    add_library(cpu_codegen SHARED tests/cpu_runtime_wrappers.cpp ${NACHO_CPU_GEN_SRC})
    target_include_directories(cpu_codegen PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${NACHO_CPU_GEN_DIR}"
    )
    target_link_libraries(cpu_codegen PRIVATE TBB::tbb)
    set_target_properties(cpu_codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_test(NAME cpu_runtime_pytest
        COMMAND python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_generated.py -v --forked
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif ()

##
# Installation rules

if (NOT CMAKE_SKIP_INSTALL_RULES)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(nacho_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/cmake/nacho"
        CACHE STRING "Install destination for nacho CMake package files")

    install(
        TARGETS nacho_compiler
        EXPORT nacho-targets
        COMPONENT nacho-runtime
    )

    install(
        EXPORT nacho-targets
        NAMESPACE nacho::
        DESTINATION "${nacho_INSTALL_CMAKEDIR}"
        COMPONENT nacho-development
    )

    configure_package_config_file(
        cmake/nacho-config.cmake.in
        cmake/nacho-config.cmake
        INSTALL_DESTINATION "${nacho_INSTALL_CMAKEDIR}"
    )

    write_basic_package_version_file(
        cmake/nacho-config-version.cmake
        COMPATIBILITY SameMajorVersion
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cmake/"
        DESTINATION "${nacho_INSTALL_CMAKEDIR}"
        COMPONENT nacho-development
    )
endif ()