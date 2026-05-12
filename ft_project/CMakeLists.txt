cmake_minimum_required(VERSION 3.20)
project(AMSC_STFT VERSION 1.0.0 LANGUAGES CXX)

# Require at least C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


# Enable testing support
enable_testing()

# Project subdirectories
add_subdirectory(tests)
# add_subdirectory(examples)    # Enable when example applications are added
# add_subdirectory(benchmarks)  # Enable when benchmarking/HPC support is ready