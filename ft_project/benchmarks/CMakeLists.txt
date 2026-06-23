# ===================================================
# Benchmarks for AMSC_STFT Project
# ===================================================
# Benchmarks are performance comparison tools
# They are separate from unit tests and focus on metrics

# ===================================================
# Benchmark 1: ParallelFFT Performance Analysis
# ===================================================
# This benchmark compares ParallelFFT performance with different thread counts
# and array sizes to measure speedup and efficiency
add_executable(bench_ParallelFFT bench_ParallelFFT.cpp)
target_link_libraries(bench_ParallelFFT PRIVATE amsc_stft_core OpenMP::OpenMP_CXX)

# Mark it as a benchmark (not a test)
set_target_properties(bench_ParallelFFT PROPERTIES COMPILE_FLAGS "-O3")
