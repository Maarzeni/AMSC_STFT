cmake_minimum_required(VERSION 3.20)
project(AMSC_STFT VERSION 1.0.0 LANGUAGES CXX)

# Richiedi C++17 come minimo (essenziale per template avanzati e constexpr)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Utile per l'autocompletamento in VSCode/CLion

# Flag di compilazione: massimi warning per il debug, massima velocità (-O3) per l'esecuzione
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Creiamo la libreria "core" del tuo progetto raccogliendo i file .cpp
# (Nota: i file .hpp verranno trovati in automatico tramite l'include_directories qui sotto)
add_library(amsc_stft_core 
    src/audio/WavReader.cpp
    src/fft/IterativeFFT.cpp
    src/fft/RecursiveFFT.cpp
    src/fft/ParallelFFT.cpp
    src/output/ImageExporter.cpp
    # src/mpi/MPIContext.cpp  <-- Tienilo commentato finché non scrivi il codice MPI
)

# Diciamo alla libreria dove trovare i file header (.hpp)
target_include_directories(amsc_stft_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# (Opzionale per ora) Predisposizione per OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(amsc_stft_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Abilitiamo i test
enable_testing()

# Includiamo le sotto-cartelle (se commentate, significa che le attiveremo dopo)
add_subdirectory(tests)
# add_subdirectory(examples)   # Decommenta quando crei main.cpp
# add_subdirectory(benchmarks) # Decommenta quando passi all'HPC