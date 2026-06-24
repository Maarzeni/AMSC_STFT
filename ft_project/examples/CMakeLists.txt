# ── examples/CMakeLists.txt ───────────────────────────────────────────────────

# Serial STFT demo: WAV in, PNG spectrogram out.
# Links to amsc_stft_core only — no MPI dependency.
# OpenMP is inherited transitively (amsc_stft_core links it PUBLIC).
add_executable(main main.cpp)
target_link_libraries(main PRIVATE amsc_stft_core)

# MPI distributed STFT demo.
add_executable(mpi_main mpi_main.cpp)
target_link_libraries(mpi_main PRIVATE amsc_stft_mpi)
