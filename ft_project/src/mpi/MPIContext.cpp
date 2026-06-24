/**
 * @file MPIContext.cpp
 * @brief Implementation of MPIContext (RAII MPI lifecycle wrapper).
 */

#include "mpi/MPIContext.hpp"
#include <mpi.h>
#include <stdexcept>

namespace stft {

MPIContext::MPIContext(int& argc, char**& argv) {
    int already = 0;
    MPI_Initialized(&already);

    if (!already) {
        if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
            throw std::runtime_error("MPIContext: MPI_Init failed.");
    }

    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank_) != MPI_SUCCESS)
        throw std::runtime_error("MPIContext: MPI_Comm_rank failed.");

    if (MPI_Comm_size(MPI_COMM_WORLD, &size_) != MPI_SUCCESS)
        throw std::runtime_error("MPIContext: MPI_Comm_size failed.");
}

MPIContext::~MPIContext() noexcept {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
}

void MPIContext::barrier() const {
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
        throw std::runtime_error("MPIContext::barrier: MPI_Barrier failed.");
}

} // namespace stft
