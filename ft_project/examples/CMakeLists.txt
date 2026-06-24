# ── examples/CMakeLists.txt ───────────────────────────────────────────────────
# Build the MPI end-to-end demonstration executable.
# The serial main.cpp is left as a placeholder; mpi_main is the primary example.

add_executable(mpi_main mpi_main.cpp)
target_link_libraries(mpi_main PRIVATE amsc_stft_mpi)
