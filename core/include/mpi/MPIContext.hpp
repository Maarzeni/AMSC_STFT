/**
 * @file MPIContext.hpp
 * @brief RAII wrapper around the MPI process lifecycle.
 *
 * @details
 * MPIContext calls MPI_Init in its constructor and MPI_Finalize in its
 * destructor.  Create exactly one instance at the very top of main() and
 * keep it alive for the entire run.
 *
 * The class is non-copyable and non-movable because there can only ever be
 * one MPI environment.
 *
 * Usage:
 * @code
 *   int main(int argc, char** argv) {
 *       stft::MPIContext ctx(argc, argv);
 *       // ... use ctx.rank(), ctx.size(), ctx.isRoot() ...
 *       return 0;
 *   }
 * @endcode
 */

#pragma once

#include <stdexcept>

namespace stft {

class MPIContext {
public:
    static constexpr int root = 0;  ///< Rank of the root process

    /**
     * @brief Initialises the MPI environment.
     *
     * Calls MPI_Init(&argc, &argv) if MPI has not already been initialised,
     * then queries and caches the rank and size of MPI_COMM_WORLD.
     *
     * @throws std::runtime_error if MPI_Init or rank/size queries fail.
     */
    MPIContext(int& argc, char**& argv);

    /**
     * @brief Finalises the MPI environment.
     *
     * Calls MPI_Finalize() if it has not already been called.
     * Destructors must not throw, so errors are silently swallowed.
     */
    ~MPIContext() noexcept;

    MPIContext(const MPIContext&)            = delete;
    MPIContext& operator=(const MPIContext&) = delete;
    MPIContext(MPIContext&&)                 = delete;
    MPIContext& operator=(MPIContext&&)      = delete;

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] int  rank()   const noexcept { return rank_; }
    [[nodiscard]] int  size()   const noexcept { return size_; }
    [[nodiscard]] bool isRoot() const noexcept { return rank_ == root; }

    /**
     * @brief Blocks until all ranks reach this call (MPI_Barrier).
     */
    void barrier() const;

private:
    int rank_ = 0;
    int size_ = 1;
};

} // namespace stft
