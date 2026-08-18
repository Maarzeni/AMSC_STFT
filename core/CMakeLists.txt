cmake_minimum_required(VERSION 3.20)
project(AMSC_STFT VERSION 1.0.0 LANGUAGES CXX)

# Require at least C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMAKE_CXX_STANDARD only asks the compiler to enable C++20; it does not check
# that the compiler is new enough to actually support the subset this project
# uses (<concepts>, <numbers>, CRTP with concept-constrained members). Below
# the versions here, that shows up as a wall of unrelated-looking errors deep
# inside a standard header instead of one readable message at configure time.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR
        "GCC >= 11 is required for the C++20 features this project uses "
        "(concepts, <numbers>); found GCC ${CMAKE_CXX_COMPILER_VERSION}. "
        "Ubuntu 22.04 and later ship a new enough GCC by default.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR
        "Clang >= 14 is required for the C++20 features this project uses "
        "(concepts, <numbers>); found Clang ${CMAKE_CXX_COMPILER_VERSION}.")
endif()

# Generate compile_commands.json for IDE support and autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compilation flags:
# enable strict warnings and high optimization level (-O3)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Core library containing the main implementation files
add_library(amsc_stft_core 
    src/audio/WavReader.cpp
    src/fft/IterativeFFT.cpp
    src/fft/RecursiveFFT.cpp
    src/fft/ParallelFFT.cpp
    src/output/ImageExporter.cpp
    # MPI module will be enabled once the MPI toolchain is fully configured
    # src/mpi/MPIContext.cpp
)

# Header search path for the library
target_include_directories(amsc_stft_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Enable OpenMP automatically if available on the system
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(amsc_stft_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── MPI distributed layer ─────────────────────────────────────────────────────
# Kept separate from amsc_stft_core so that non-MPI consumers (tests, serial
# benchmarks) do not need an MPI toolchain.
find_package(MPI REQUIRED)

add_library(amsc_stft_mpi src/mpi/MPIContext.cpp)

target_include_directories(amsc_stft_mpi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(amsc_stft_mpi PUBLIC
    amsc_stft_core
    MPI::MPI_CXX)

# Enable testing support
enable_testing()

# Project subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)