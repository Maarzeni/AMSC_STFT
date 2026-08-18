# ── examples/CMakeLists.txt ───────────────────────────────────────────────────

# The two examples take the same two paths, baked in as absolute so they
# resolve from whatever directory the binaries are launched from — the build
# tree, a job script, or a container.
#   EXAMPLES_DATA_DIR    where a bare input file name is looked up
#   EXAMPLES_OUTPUT_DIR  where the PNG spectrogram is written
# CMAKE_SOURCE_DIR is core/, so its parent is the repository root.
set(EXAMPLES_DEFS
    EXAMPLES_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    EXAMPLES_OUTPUT_DIR="${CMAKE_SOURCE_DIR}/../results/results_examples"
)

# Shared-memory STFT demo: WAV in, PNG spectrogram out.
# Links to amsc_stft_core only — no MPI dependency.
# OpenMP is inherited transitively (amsc_stft_core links it PUBLIC).
add_executable(main main.cpp)
target_link_libraries(main PRIVATE amsc_stft_core)
target_compile_definitions(main PRIVATE ${EXAMPLES_DEFS})

# MPI distributed STFT demo: the same pipeline, frames spread over ranks.
add_executable(mpi_main mpi_main.cpp)
target_link_libraries(mpi_main PRIVATE amsc_stft_mpi)
target_compile_definitions(mpi_main PRIVATE ${EXAMPLES_DEFS})
