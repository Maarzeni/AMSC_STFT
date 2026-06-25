# ===================================================
# Benchmarks for AMSC_STFT Project
# ===================================================
# Performance comparison tools, separate from the unit tests.
# Use only std::chrono + STL + project code (no external benchmark framework).

# ---------------------------------------------------
# Serial / OpenMP benchmarks (FFT engines + STFT)
# ---------------------------------------------------
# Links amsc_stft_core only (no MPI). OpenMP is inherited transitively from
# amsc_stft_core, but we link it explicitly for the omp_* calls in the driver.
add_executable(benchmark_Main benchmark_Main.cpp)
target_link_libraries(benchmark_Main PRIVATE amsc_stft_core OpenMP::OpenMP_CXX)
set_target_properties(benchmark_Main PROPERTIES COMPILE_FLAGS "-O3")

# ---------------------------------------------------
# Distributed benchmarks (MPI + hybrid MPI/OpenMP STFT)
# ---------------------------------------------------
# Links amsc_stft_mpi (which pulls in MPI + amsc_stft_core + OpenMP).
add_executable(benchmark_MPI_Main benchmark_MPI_Main.cpp)
target_link_libraries(benchmark_MPI_Main PRIVATE amsc_stft_mpi OpenMP::OpenMP_CXX)
set_target_properties(benchmark_MPI_Main PROPERTIES COMPILE_FLAGS "-O3")
