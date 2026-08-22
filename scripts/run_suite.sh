#!/usr/bin/env bash
# =============================================================================
# AMSC_STFT — full benchmark suite, one recipe for every environment.
#
# This script is the SINGLE SOURCE OF TRUTH for what the project measures. The
# GitHub workflow, the SLURM job and a human at a terminal all call it, so the
# numbers they produce are comparable by construction: same repetitions, same
# frame geometry, same rank/thread layout, same output file names.
#
# ─── How to run it ──────────────────────────────────────────────────────────
#
#   Laptop or cluster, inside the container (recommended — no toolchain needed):
#       apptainer exec --bind "$PWD:$PWD" amsc_stft.sif bash scripts/run_suite.sh
#
#   Anywhere the project is already built natively (this is what CI does):
#       bash scripts/run_suite.sh
#
#   Under SLURM: see job.sh, which is just an sbatch header around the line above.
#
# ─── Knobs (all optional, all environment variables) ────────────────────────
#
#   BUILD_DIR    where the compiled binaries live. Auto-detected: the in-image
#                path when running inside the container, else core/build.
#   RESULTS_DIR      where benchmark CSVs and the environment log are written.
#                    default ./results/results_benchmark
#   TEST_RESULTS_DIR where the ctest log and its working copy are written.
#                    default ./results/results_test
#   PREFIX       file name prefix, i.e. which machine.  auto: cluster/github/local
#   RANKS        MPI ranks.                default SLURM_NTASKS, else 2
#   THREADS      OpenMP threads per rank.   default SLURM_CPUS_PER_TASK, else 2
#   BENCH_ARGS   "reps warmup frame hop".   default "7 2 1024 512"
#   MEM_DURATION seconds of audio for the memory comparison.  default 30
#
#   MPI_LAUNCHER   how to start MPI ranks, up to and including the flag that
#                  takes their number. Default "mpirun --bind-to none -np".
#                  Multi-node runs under SLURM want "srun --mpi=pmix -n"
#                  instead: the batch scheduler, not mpirun, is what knows how
#                  to place ranks on nodes other than this one.
#
#   TESTS_ONLY     set to 1 to run section 1 (ctest) and stop. The CI/CD
#                  pipeline uses this: it exists to prove that the code builds,
#                  passes its tests and deploys, and a pipeline that runs on
#                  every push must not also spend half an hour measuring.
#                  Benchmarks are launched by hand when numbers are wanted.
#
#   SCALING        set to 0 to skip the strong-scaling sweep.  default on
#   MAX_WORKERS    largest rank count in the sweep. default: the SLURM
#                  allocation, or the machine's physical cores
#   SCALING_RANKS  explicit rank list, e.g. "1 2 4 8", overriding MAX_WORKERS
#   SCALING_DURATIONS seconds of audio to sweep at, one curve each.
#                  default "5 15 60"; SCALING_DURATION (singular) forces one
#   SCALING_THREADS  threads/rank for the hybrid row of the rank sweep. default 2
#
#   WEAK_SCALING   set to 0 to skip the weak-scaling sweep.   default on
#   HYBRID_SPLIT   set to 0 to skip the rank/thread split sweep (section 8).
#   HYBRID_SPLIT_THREADS      thread counts to try there. default "1 2 4 8 12 24 48"
#   HYBRID_SPLIT_DURATION     seconds of audio for it. default: last SCALING_DURATIONS
#   WEAK_BASE      seconds of audio PER RANK in that sweep.    default 5
#
#   THREAD_SCALING set to 0 to skip the OpenMP thread sweep.  default on
#   THREAD_LIST    explicit thread list, e.g. "1 2 4 8"; default: powers of two
#                  up to the physical cores, plus one SMT point if there is one
#
#   e.g.  RANKS=4 THREADS=1 PREFIX=laptop bash scripts/run_suite.sh
#         SCALING_RANKS="1 2 4 8 16" bash scripts/run_suite.sh
#         SCALING=0 bash scripts/run_suite.sh          # benchmarks only
# =============================================================================

# No `set -e`: a failing benchmark must not prevent the remaining sections from
# running, otherwise one broken step costs us every other result of the run.
set -uo pipefail

# ── Where are the binaries? ─────────────────────────────────────────────────
# Inside the container the project lives at a fixed absolute path baked in at
# build time; outside, it sits next to this script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_IMAGE_BUILD=/app/AMSC_STFT/core/build

if [ -z "${BUILD_DIR:-}" ]; then
    if [ -d "${IN_IMAGE_BUILD}" ]; then
        BUILD_DIR="${IN_IMAGE_BUILD}"
    else
        BUILD_DIR="${REPO_ROOT}/core/build"
    fi
fi

# ── Preflight ───────────────────────────────────────────────────────────────
# Checked up front and all at once: a half-equipped machine otherwise produces a
# full set of result files containing nothing but "command not found", which is
# far worse than refusing to start.
container_hint() {
    echo "" >&2
    echo "  The container carries the whole toolchain, so this always works:" >&2
    echo "    apptainer exec --bind \"\$PWD:\$PWD\" amsc_stft.sif bash scripts/run_suite.sh" >&2
    echo "" >&2
    echo "  Get amsc_stft.sif from the CI artifact 'amsc-stft-sif', or build it:" >&2
    echo "    apptainer build --fakeroot amsc_stft.sif Singularity.def" >&2
}

# Settled here rather than further down because the preflight below has to know
# which launcher to look for: a multi-node run uses srun and may not need
# mpirun on PATH at all.
# No binding policy here on purpose: one is chosen per launch by bind_flags()
# below, which needs to know the rank and thread counts of that particular
# launch to pick it. A policy baked in here would collide with that one —
# OpenMPI rejects two conflicting --bind-to directives outright — so if you
# override MPI_LAUNCHER and put a --bind-to of your own in it, bind_flags()
# stands down and yours is used everywhere.
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun -np}"
LAUNCHER_CMD="${MPI_LAUNCHER%% *}"

missing=""
[ -x "${BUILD_DIR}/benchmarks/benchmark_Main" ]     || missing="${missing} benchmark_Main"
[ -x "${BUILD_DIR}/benchmarks/benchmark_MPI_Main" ] || missing="${missing} benchmark_MPI_Main"
command -v ctest  >/dev/null 2>&1                   || missing="${missing} ctest"
command -v "${LAUNCHER_CMD}" >/dev/null 2>&1        || missing="${missing} ${LAUNCHER_CMD}"

if [ -n "${missing}" ]; then
    echo "ERROR: missing:${missing}" >&2
    echo "       (looked for binaries under ${BUILD_DIR})" >&2
    container_hint
    exit 1
fi

# An executable bit proves nothing about a binary built elsewhere: one compiled
# against a newer glibc dies at load time. ldd resolves the same libraries the
# loader would, and reports "not found" without running anything, so the problem
# surfaces here instead of as four files full of loader errors.
# Only an unresolved library counts as a failure: ldd also exits non-zero for a
# statically linked binary or a wrapper script, and neither is a problem.
ldd_out=$(ldd "${BUILD_DIR}/benchmarks/benchmark_Main" 2>&1)
if echo "${ldd_out}" | grep -qi 'not found'; then
    echo "ERROR: ${BUILD_DIR}/benchmarks/benchmark_Main cannot be loaded here:" >&2
    echo "${ldd_out}" | grep -i 'not found' | sed 's/^/       /' >&2
    echo "       (typically a build produced on a different distribution)" >&2
    container_hint
    exit 1
fi

# Sections record their own outcome so the run ends with a verdict instead of an
# unconditional "Done".
FAILED=""
note_status() {  # note_status <exit code> <section name>
    [ "$1" -eq 0 ] || FAILED="${FAILED} $2"
}

# Runs a benchmark binary, appending its CSV rows to FILE. Several sections
# call this many times per file (once per rank/thread/duration in a sweep),
# and each invocation prints its own header — so every call after the first
# has it stripped before appending, leaving exactly one header per file.
# PIPESTATUS[0] (what note_status reads right after calling this) still
# reflects the benchmark binary's own exit code either way, since it is
# always the first stage of whichever pipeline runs.
csv_append() {  # csv_append <file> <cmd...>
    local file=$1; shift
    local want_header=1
    [ -s "${file}" ] && want_header=0

    # Only CSV reaches the file; everything reaches the console. The benchmarks
    # print CSV and nothing else, but the ENVIRONMENT injects on the same
    # stream — mpirun's "No protocol specified" X11 complaint is one that
    # keeps turning up, and unsetting DISPLAY does not stop it. Letting a
    # stray line through is not cosmetic: it shifts every column of the rows
    # after it and can push a whole sweep into the wrong output file.
    #
    # Data rows always start with a quoted "kind" field, the header with
    # `kind,` — those two shapes are the whitelist. Errors still print to the
    # job log, and PIPESTATUS[0] is still the benchmark's own exit code,
    # because it remains the first stage of the pipeline.
    "$@" 2>&1 | awk -v f="${file}" -v hdr="${want_header}" '
        { print }
        /^"/     { print >> f; next }
        /^kind,/ { if (hdr) print >> f; next }
    '
}

# Powers of two up to <max>, with <max> appended when it is not itself one, so a
# 6-core machine sweeps 1 2 4 6 instead of stopping at 4 and wasting two cores.
sweep_list() {  # sweep_list <max>
    local max=$1 p=1 out=""
    while [ "${p}" -le "${max}" ]; do
        out="${out} ${p}"
        p=$(( p * 2 ))
    done
    case " ${out} " in
        *" ${max} "*) ;;
        *) out="${out} ${max}" ;;
    esac
    echo "${out}"
}

# ── Which machine are we on? ────────────────────────────────────────────────
# Only used to label the output files, so a laptop run never overwrites — or
# gets mistaken for — a cluster run.
if [ -z "${PREFIX:-}" ]; then
    if   [ -n "${SLURM_JOB_ID:-}" ];    then PREFIX=cluster
    elif [ -n "${PBS_JOBID:-}" ];       then PREFIX=cluster
    elif [ -n "${GITHUB_ACTIONS:-}" ];  then PREFIX=github
    else                                     PREFIX=local
    fi
fi

# A batch job is the only place where lscpu lies about what we may use: it
# describes the whole node while the scheduler handed us a slice of it. Both
# schedulers are asked what they actually reserved.
BATCH_CORES=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    BATCH_CORES=$(( ${SLURM_NTASKS:-1} * ${SLURM_CPUS_PER_TASK:-1} ))
elif [ -n "${PBS_JOBID:-}" ]; then
    # PBS exports NCPUS for the chunk; PBS_NODEFILE has one line per MPI slot,
    # which is the better answer when both are present and disagree.
    if [ -r "${PBS_NODEFILE:-/nonexistent}" ]; then
        BATCH_CORES=$(wc -l < "${PBS_NODEFILE}")
    fi
    [ -n "${NCPUS:-}" ] && [ "${NCPUS}" -gt "${BATCH_CORES:-0}" ] && BATCH_CORES=${NCPUS}
fi

RESULTS_DIR=${RESULTS_DIR:-${PWD}/results/results_benchmark}
TEST_RESULTS_DIR=${TEST_RESULTS_DIR:-${PWD}/results/results_test}
RANKS=${RANKS:-${SLURM_NTASKS:-2}}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-2}}
BENCH_ARGS=${BENCH_ARGS:-"7 2 1024 512"}
MEM_DURATION=${MEM_DURATION:-30}
# One curve per duration in the scaling figures: how the parallel gain grows with
# the problem is the behaviour Amdahl's law predicts, and a single duration
# cannot show it. SCALING_DURATION (singular) still forces a single curve.
SCALING_DURATIONS=${SCALING_DURATIONS:-${SCALING_DURATION:-"5 15 60"}}
TOTAL_CORES=$(( RANKS * THREADS ))

mkdir -p "${RESULTS_DIR}" "${TEST_RESULTS_DIR}"

# ── MPI: make it survive both a container and an oversubscribed laptop ──────
# The launcher is a variable because a container's own mpirun cannot reach
# another node: only the batch scheduler can. Everything below that tunes
# mpirun is therefore applied only when mpirun is what we are using.
case "${MPI_LAUNCHER}" in
    mpirun*) USING_MPIRUN=1 ;;
    *)       USING_MPIRUN=0 ;;
esac
MPIRUN_EXTRA=""

# OpenMPI needs a writable TMPDIR for its ORTE session directory. On the CINECA
# login nodes the default points at a read-only filesystem, which aborts
# mpirun before any rank starts, so fall back to a directory we know we can
# write. Tested rather than assumed, since it is writable almost everywhere.
if ! touch "${TMPDIR:-/tmp}/.amsc_write_test" 2>/dev/null; then
    export TMPDIR="${RESULTS_DIR}/../tmp"
    mkdir -p "${TMPDIR}"
    echo "note: default TMPDIR not writable, using ${TMPDIR}"
    # Only passed when the fallback was actually needed: the parameter is an
    # OpenMPI 4 name (OpenMPI 5 renamed ORTE to PRRTE), so handing it to a
    # newer mpirun for no reason invites a complaint we do not need.
    [ "${USING_MPIRUN}" = "1" ] && \
        MPIRUN_EXTRA="${MPIRUN_EXTRA} --mca orte_tmpdir_base ${TMPDIR}"
else
    rm -f "${TMPDIR:-/tmp}/.amsc_write_test"
fi

# Inside a container, OpenMPI's shared-memory path tries Cross Memory Attach
# (process_vm_readv), which needs ptrace privileges the container lacks: every
# large transfer then fails with "Read -1 ... errno = 1" and is retried on a
# slower path, polluting the timings. Skipping single-copy avoids the detour.
if [ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ] \
        && [ "${USING_MPIRUN}" = "1" ]; then
    MPIRUN_EXTRA="${MPIRUN_EXTRA} --mca btl_vader_single_copy_mechanism none"
fi

# OpenMPI counts a "slot" per PHYSICAL core, not per hardware thread: on a
# 2-core laptop with SMT, nproc says 4 but only 2 slots exist, and anything
# asking for more is refused outright with "not enough slots available".
# The unit tests hit this before the benchmarks do — they launch np=1..4
# regardless of RANKS — so the permission is granted unconditionally, exactly
# as the CI workflow and the container image already do.
PHYSICAL_CORES=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "${PHYSICAL_CORES}" -gt 0 ] 2>/dev/null || PHYSICAL_CORES=$(nproc)

# Captured once, here, because coreutils nproc honours OMP_NUM_THREADS: once a
# section sets it to 2, every later nproc answers 2 and the machine appears to
# shrink. `nproc --all` would ignore it but would also ignore CPU affinity, and
# under a SLURM cgroup that reports the whole node instead of the allocation.
LOGICAL_CPUS=$(nproc)

# How many cores the sweeps may actually use, and the rank list they walk.
# Under a scheduler this is the ALLOCATION, not what lscpu reports: on a login
# node lscpu sees every core of the machine while the job holds a handful, and
# a sweep sized on the former would trample the other users. Computed once
# here because both the memory comparison (section 4) and the scaling sweeps
# (sections 5-7) need it.
if [ -n "${BATCH_CORES}" ] && [ "${BATCH_CORES}" -gt 0 ]; then
    AVAILABLE_CORES=${BATCH_CORES}
else
    AVAILABLE_CORES=${PHYSICAL_CORES}
fi
MAX_WORKERS=${MAX_WORKERS:-${AVAILABLE_CORES}}
SCALING_RANKS=${SCALING_RANKS:-$(sweep_list "${MAX_WORKERS}")}

# Threads for the SHARED-MEMORY sections (unit tests, section 2). Those run as a
# single process on ONE node, so their thread count must be bounded by that
# node, not by the whole allocation. On a multi-node job the two differ sharply:
# 4 x 48 makes TOTAL_CORES 192 while the process still has 48 cores under it,
# and 192 OpenMP threads on 48 cores is the oversubscription that drives
# ParallelFFT into the pathological regime which once ran a job out of walltime.
# SLURM_CPUS_ON_NODE is the allocation on THIS node; nproc is the fallback.
NODE_CORES=${SLURM_CPUS_ON_NODE:-${LOGICAL_CPUS}}
[ "${NODE_CORES}" -gt 0 ] 2>/dev/null || NODE_CORES=${LOGICAL_CPUS}
SHARED_MEM_THREADS=${TOTAL_CORES}
[ "${SHARED_MEM_THREADS}" -gt "${NODE_CORES}" ] && SHARED_MEM_THREADS=${NODE_CORES}

# ── Process binding ─────────────────────────────────────────────────────────
# Until now every launch ran with --bind-to none, which lets the OS move ranks
# and their OpenMP threads across cores freely. Two things follow on a machine
# like a 2-socket compute node, and the second is the expensive one:
#
#   NUMA. Linux places a page on the memory of the socket that first touches
#   it. A rank that migrates afterwards reads its own data across the socket
#   interconnect for the rest of its life. The gather phase runs at memory
#   bandwidth, so it pays this directly.
#
#   Collective jitter. With P unbound processes on P cores the scheduler
#   occasionally stacks two on one core and idles another. For a single process
#   that is noise that averages out. For a collective it does not: every rank
#   waits for the slowest, so one descheduled rank stalls all the others, and
#   the cost is paid once per collective rather than once per rank. This is the
#   most likely source of the run-to-run spread seen at high rank counts —
#   stddev around 40% of the mean, on an algorithm that is deterministic.
#
# Binding is therefore the default here, but only where it is safe:
#
#   - not when the caller supplied its own --bind-to (theirs wins);
#   - not when mpirun is not the launcher (srun binds by its own rules);
#   - not when the probe below says this mpirun does not understand PE=;
#   - not when P x T exceeds the cores available, where binding would either be
#     refused outright or pin several ranks onto one core. That case still runs
#     unbound, exactly as before — the unit tests launch np=1..4 regardless of
#     machine size and must keep working on a 2-core laptop.
BIND_PROBE_OK=0
case "${MPI_LAUNCHER}" in
    *--bind-to*) BIND_USER_POLICY=1 ;;
    *)           BIND_USER_POLICY=0 ;;
esac

if [ "${USING_MPIRUN}" = "1" ] && [ "${BIND_USER_POLICY}" = "0" ]; then
    # PE= is the part worth probing: it is what maps T cores to one rank, and
    # an mpirun that does not know it fails the whole launch rather than
    # ignoring the flag. One np=1 launch of /bin/true settles it.
    if mpirun --map-by slot:PE=1 --bind-to core ${MPIRUN_EXTRA}               -np 1 /bin/true >/dev/null 2>&1; then
        BIND_PROBE_OK=1
    else
        echo "note: this mpirun rejected --map-by slot:PE=; running unbound"
    fi
fi

# Binding flags for a launch of P ranks x T threads, or --bind-to none when
# binding is not applicable. Always emits exactly one binding directive, so the
# caller can append it unconditionally.
bind_flags() {
    _bf_p=$1
    _bf_t=$2

    # OpenMPI binds to PHYSICAL cores, and it refuses the whole launch — it does
    # not fall back — when a mapping asks for more of them than a node has. So
    # the check has to be per node and in physical cores, not in whatever
    # AVAILABLE_CORES happens to count. On a machine with SMT the two differ by
    # a factor of two, and getting this wrong would not degrade a sweep point,
    # it would delete it.
    _bf_nodes=${SLURM_JOB_NUM_NODES:-1}
    _bf_per_node=$(( (_bf_p * _bf_t + _bf_nodes - 1) / _bf_nodes ))

    if [ "${BIND_USER_POLICY}" = "1" ]; then
        echo ""                                   # caller already said how
    elif [ "${BIND_PROBE_OK}" != "1" ] \
      || [ $(( _bf_p * _bf_t )) -gt "${AVAILABLE_CORES}" ] \
      || [ "${_bf_per_node}" -gt "${PHYSICAL_CORES}" ]; then
        echo "--bind-to none"
    else
        echo "--map-by slot:PE=${_bf_t} --bind-to core"
    fi
}

# Largest T that keeps P x T inside the machine, capped at a requested maximum.
# Used by the sweeps so that no point of a curve is oversubscribed: a point
# measured on more workers than there are cores reports the scheduler's
# behaviour, not the algorithm's, and it lands in the same CSV column as the
# points that do not.
fit_threads() {
    _ft_p=$1
    _ft_max=$2
    _ft_t=$(( AVAILABLE_CORES / _ft_p ))
    [ "${_ft_t}" -lt 1 ] && _ft_t=1
    [ "${_ft_t}" -gt "${_ft_max}" ] && _ft_t=${_ft_max}
    [ "${_ft_t}" -gt "${NODE_CORES}" ] && _ft_t=${NODE_CORES}
    echo "${_ft_t}"
}

export OMPI_MCA_rmaps_base_oversubscribe=1

# mpirun still needs to be told explicitly when we knowingly ask for more
# workers than there are cores to put them on.
# PHYSICAL_CORES describes ONE node, so on a multi-node allocation it is the
# wrong yardstick: 192 ranks across four 48-core nodes is a perfect fit, not a
# four-fold oversubscription. AVAILABLE_CORES is the allocation.
if [ "${TOTAL_CORES}" -gt "${AVAILABLE_CORES}" ] && [ "${USING_MPIRUN}" = "1" ]; then
    MPIRUN_EXTRA="${MPIRUN_EXTRA} --oversubscribe"
fi

# The login nodes export DISPLAY without a reachable X server, which makes the
# container print "No protocol specified" over the results. Purely cosmetic.
unset DISPLAY XAUTHORITY

# ── Section 0: what machine produced these numbers ──────────────────────────
# PHYSICAL_CORES was computed with the MPI settings above; it is printed here
# because nproc counts hardware threads, and a 2-core machine with SMT reports
# 4. Reading a 2x speedup as poor scaling "on 4 cores" is the single easiest
# way to misread these tables.
{
    echo "environment  : ${PREFIX}${SLURM_JOB_ID:+ (SLURM job ${SLURM_JOB_ID})}${PBS_JOBID:+ (PBS job ${PBS_JOBID})}"
    echo "node         : $(hostname)"
    echo "partition    : ${SLURM_JOB_PARTITION:-${PBS_QUEUE:-n/a (not a batch job)}}"
    echo "ranks x thr  : ${RANKS} x ${THREADS} = ${TOTAL_CORES} workers"
    # Three numbers, three scopes, kept apart on purpose. lscpu describes the
    # whole NODE; nproc respects the scheduler's cgroup and so describes the
    # ALLOCATION. Printing one as "physical" and the other as "logical" would
    # invite the misreading "56 physical, 28 logical", which is impossible —
    # the labels below name what each number actually describes instead.
    echo "cores        : ${AVAILABLE_CORES} usable by this run"
    echo "node total   : ${PHYSICAL_CORES} physical / $(nproc --all 2>/dev/null || echo '?') logical"
    echo "bench args   : ${BENCH_ARGS}  (reps warmup frame hop)"
    echo "container    : ${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-none (native build)}}"
    echo "date (UTC)   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|CPU MHz|L3 cache'
} | tee "${RESULTS_DIR}/${PREFIX}_env.txt"

# ── Section 1: unit tests ───────────────────────────────────────────────────
# ctest insists on writing Testing/Temporary/LastTest.log inside the build
# directory, which is read-only when it lives in a .sif image. Copying just the
# generated CTestTestfile.cmake files elsewhere sidesteps that: each one
# registers its tests by ABSOLUTE path, so the very same binaries still run.
echo ""
echo "==================== [1/8] Unit tests (ctest) ============================"
CTEST_MIRROR="${TEST_RESULTS_DIR}"
rm -rf "${CTEST_MIRROR}"
mkdir -p "${CTEST_MIRROR}"
( cd "${BUILD_DIR}" && find . -name CTestTestfile.cmake -exec cp --parents -t "${CTEST_MIRROR}" {} + )

export OMP_NUM_THREADS=${SHARED_MEM_THREADS}
( cd "${CTEST_MIRROR}" && ctest --output-on-failure ) \
    2>&1 | tee "${TEST_RESULTS_DIR}/${PREFIX}_ctest.txt"
note_status "${PIPESTATUS[0]}" "unit-tests"

if [ "${TESTS_ONLY:-0}" = "1" ]; then
    echo ""
    echo "TESTS_ONLY=1 — benchmarks skipped, stopping after the unit tests."
    if [ -n "${FAILED}" ]; then
        echo "FAILED sections:${FAILED}"
        exit 1
    fi
    exit 0
fi

# ── Section 2: serial / OpenMP benchmark ────────────────────────────────────
echo ""
echo "============== [2/8] Benchmark: serial / OpenMP =========================="
# Nothing in this section involves MPI: it is the FFT-engine comparison and the
# shared-memory STFT, both of which are properties of one node. A distributed
# run that already has good single-node numbers can set OPENMP_BENCH=0 and keep
# them rather than pay to re-measure what will not have changed.
if [ "${OPENMP_BENCH:-1}" != "0" ]; then
    export OMP_NUM_THREADS=${SHARED_MEM_THREADS}
    csv_append "${RESULTS_DIR}/${PREFIX}_benchmark_openmp.csv" \
        "${BUILD_DIR}/benchmarks/benchmark_Main" ${BENCH_ARGS}
    note_status "${PIPESTATUS[0]}" "benchmark-openmp"
else
    echo "OPENMP_BENCH=0 — skipped (keeping the existing single-node numbers)."
fi

# ── Section 3: distributed / hybrid benchmark ───────────────────────────────
echo ""
echo "============ [3/8] Benchmark: MPI / hybrid ==============================="
# The binary reports both a 1-thread row and a THREADS-thread row from this one
# launch, so the thread count has to leave room for the wider of the two.
# The scaling sweeps below cover the same ground at every rank count; this
# section is the single-configuration summary. MPI_BENCH=0 drops it.
if [ "${MPI_BENCH:-1}" = "0" ]; then
    echo "MPI_BENCH=0 — skipped (the scaling sweeps cover this ground)."
else
MPI_BENCH_THREADS=$(fit_threads "${RANKS}" "${THREADS}")
[ "${MPI_BENCH_THREADS}" != "${THREADS}" ] && \
    echo "  note: ${RANKS} x ${THREADS} exceeds ${AVAILABLE_CORES} cores;" \
         "hybrid row measured at ${MPI_BENCH_THREADS} thr/rank"
export OMP_NUM_THREADS=${MPI_BENCH_THREADS}
csv_append "${RESULTS_DIR}/${PREFIX}_benchmark_mpi.csv" \
    ${MPI_LAUNCHER} "${RANKS}" ${MPIRUN_EXTRA} \
    $(bind_flags "${RANKS}" "${MPI_BENCH_THREADS}") \
    "${BUILD_DIR}/benchmarks/benchmark_MPI_Main" ${BENCH_ARGS}
note_status "${PIPESTATUS[0]}" "benchmark-mpi"
fi

# ── Section 4: memory, broadcast vs scatter ─────────────────────────────────
# VmHWM is a high-water mark of the whole process, so one process can only
# report one strategy: running both in sequence would attribute the broadcast's
# peak to the scatter too. Hence two launches, identical but for the strategy.
echo ""
echo "======== [4/8] Memory: broadcast vs scatter (peak RSS, paired runs) ======"
MEMFILE="${RESULTS_DIR}/${PREFIX}_memory_bcast_vs_scatter.csv"
: > "${MEMFILE}"

# Both strategies at every rank count of the sweep. Measuring the broadcast at a
# single rank count would leave its curve as one point, and the whole claim —
# that the scatter's per-rank footprint falls as 1/P while the broadcast's does
# not — is a statement about how the two behave AS P GROWS.
MEM_RANKS=${MEM_RANKS:-${SCALING_RANKS:-${RANKS}}}
for P in ${MEM_RANKS}; do
    MEM_THREADS=$(fit_threads "${P}" "${THREADS}")
    export OMP_NUM_THREADS=${MEM_THREADS}
    for STRATEGY in bcast scatter; do
        echo "  -np ${P} x ${MEM_THREADS} thr · distribution: ${STRATEGY}"

        csv_append "${MEMFILE}" \
            ${MPI_LAUNCHER} "${P}" ${MPIRUN_EXTRA} \
            $(bind_flags "${P}" "${MEM_THREADS}") \
            "${BUILD_DIR}/benchmarks/benchmark_MPI_Main" \
            ${BENCH_ARGS} "${MEM_DURATION}" "${STRATEGY}"
        note_status "${PIPESTATUS[0]}" "memory-np${P}-${STRATEGY}"
    done
done

# ── Section 5: strong scaling ───────────────────────────────────────────────
# Sections 2-4 deliberately pin the same RANKS x THREADS everywhere, so tables
# from different machines can be compared row by row. That is the wrong shape
# for a scaling study, which needs the opposite: one machine, a fixed problem,
# an increasing number of workers.
#
# The sweep therefore sizes itself to the machine — the SLURM allocation when
# there is one, the physical core count otherwise — and walks the powers of two
# up to it, adding the exact core count when that is not itself a power of two
# (a 6-core machine gets 1 2 4 6). One thread per rank throughout, so the only
# variable is the number of ranks.
#
# Reading it: speedup(P) = time(-np 1) / time(-np P). Every run also prints its
# own "serial (1 rank, 1 thr)" row, which should stay roughly constant across
# the sweep — if it drifts, the machine was busy and the run is suspect.
if [ "${SCALING:-1}" != "0" ]; then
    # Each run of the binary reports BOTH decompositions at the rank count it was
    # given: "MPI pure" always uses one thread per rank, "hybrid" uses
    # OMP_NUM_THREADS of them. Leaving that at 1 makes the two rows identical and
    # hides OpenMP entirely, so the sweep runs with 2 threads per rank: at every
    # P the file then shows P x 1 against P x 2, i.e. what the second thread buys.
    SCALING_THREADS=${SCALING_THREADS:-2}
    export OMP_NUM_THREADS=${SCALING_THREADS}

    # The hybrid row uses P x SCALING_THREADS workers, so it outgrows the
    # machine one step earlier than the pure row does. That used to be left to
    # the reader, on the grounds that every row carries its own ranks and
    # threads; in practice it meant the top third of the sweep measured the
    # scheduler rather than the algorithm, and it did so in the same column as
    # the points that did not. SCALING_THREADS is now a CEILING: each P gets
    # the largest thread count that still fits in the machine, which at the wide
    # end is 1, where the hybrid row simply coincides with the pure one. Nothing
    # is lost — a point that does not fit was never a measurement of scaling —
    # and the CSV still records what each row actually ran with.
    echo ""
    echo "======== [5/8] Strong scaling: ranks =${SCALING_RANKS} ==================="
    SCALEFILE="${RESULTS_DIR}/${PREFIX}_scaling.csv"
    : > "${SCALEFILE}"

    for DURATION in ${SCALING_DURATIONS}; do
    for P in ${SCALING_RANKS}; do
        T=$(fit_threads "${P}" "${SCALING_THREADS}")
        export OMP_NUM_THREADS=${T}
        echo "  ${DURATION}s · -np ${P} x ${T} thr"

        # Fixed problem size across the sweep — that is what makes it STRONG
        # scaling — and the same repetitions and geometry as every other section.
        csv_append "${SCALEFILE}" \
            ${MPI_LAUNCHER} "${P}" ${MPIRUN_EXTRA} \
            $(bind_flags "${P}" "${T}") \
            "${BUILD_DIR}/benchmarks/benchmark_MPI_Main" \
            ${BENCH_ARGS} "${DURATION}" scatter
        note_status "${PIPESTATUS[0]}" "scaling-np${P}-${DURATION}s"
    done
    done
fi

# ── Section 6: OpenMP thread scaling (one process, no MPI) ──────────────────
# The counterpart of section 5: there the ranks grew, here the threads do, and
# MPI is out of the picture entirely. Section 2 only ever contrasts 1 thread
# with all of them, which is two points — not a curve.
#
# benchmark_Main re-reads OMP_NUM_THREADS at startup and prints, in the same
# table, its own single-thread baseline next to the threaded run. Every point of
# the sweep is therefore self-baselined: no need to divide by a number measured
# minutes earlier, when the machine may have been under different load.
#
# T=1 makes the two rows the same computation, so its "speedup" should read
# 1.00x. Whatever it reads instead is this machine's noise floor, measured for
# free — a number worth knowing before trusting any other row in the file.
if [ "${THREAD_SCALING:-1}" != "0" ]; then
    if [ -z "${THREAD_LIST:-}" ]; then
        THREAD_LIST=$(sweep_list "${AVAILABLE_CORES}")

        # One point past the physical cores quantifies what SMT contributes,
        # which is otherwise easy to mistake for poor scaling. Skipped inside a
        # SLURM allocation, where the budget is expressed in cores and the extra
        # threads would land on cores belonging to somebody else.
        if [ -z "${BATCH_CORES}" ] && [ "${LOGICAL_CPUS}" -gt "${AVAILABLE_CORES}" ]; then
            THREAD_LIST="${THREAD_LIST} ${LOGICAL_CPUS}"
        fi
    fi

    echo ""
    echo "======== [6/8] OpenMP thread scaling: threads =${THREAD_LIST} ==========="
    THREADFILE="${RESULTS_DIR}/${PREFIX}_scaling_threads.csv"
    : > "${THREADFILE}"

    for DURATION in ${SCALING_DURATIONS}; do
    for T in ${THREAD_LIST}; do
        echo "  ${DURATION}s · OMP_NUM_THREADS=${T}"

        # benchmark_Main re-reads OMP_NUM_THREADS at startup and writes its own
        # 1-thread baseline in the same row set, so every point of the sweep is
        # self-baselined: `speedup` never depends on a number measured minutes
        # earlier under different load. At T=1 the two rows are the same
        # computation, so speedup should read 1.00 — whatever it reads instead
        # is this machine's noise floor, measured for free.
        csv_append "${THREADFILE}" \
            env AMSC_SKIP_GRANULARITY=1 OMP_NUM_THREADS=${T} \
            "${BUILD_DIR}/benchmarks/benchmark_Main" \
            ${BENCH_ARGS} "${DURATION}"
        note_status "${PIPESTATUS[0]}" "threads-${T}-${DURATION}s"
    done
    done
fi

# ── Section 7: weak scaling ─────────────────────────────────────────────────
# Strong scaling asks "does a FIXED problem finish sooner on more workers".
# Weak scaling asks the complementary question — "can a PROPORTIONALLY larger
# problem be solved in the same time" — by growing the audio with the rank
# count, so the work per rank stays constant. A perfect implementation draws a
# flat line; whatever slope appears is the communication and the serial part.
#
# Both matter for a report: strong scaling is bounded by Amdahl, weak scaling by
# Gustafson, and an implementation can look good under one and poor under the
# other.
if [ "${WEAK_SCALING:-1}" != "0" ]; then
    WEAK_BASE=${WEAK_BASE:-5}          # seconds of audio per rank
    echo ""
    echo "======== [7/8] Weak scaling: ${WEAK_BASE}s of audio per rank ============="
    WEAKFILE="${RESULTS_DIR}/${PREFIX}_weak_scaling.csv"
    : > "${WEAKFILE}"
    export OMP_NUM_THREADS=1

    for P in ${SCALING_RANKS}; do
        DUR=$(( WEAK_BASE * P ))
        echo "  -np ${P} · ${DUR}s of audio (${WEAK_BASE}s per rank)"

        csv_append "${WEAKFILE}" \
            ${MPI_LAUNCHER} "${P}" ${MPIRUN_EXTRA} \
            $(bind_flags "${P}" 1) \
            "${BUILD_DIR}/benchmarks/benchmark_MPI_Main" \
            ${BENCH_ARGS} "${DUR}" scatter
        note_status "${PIPESTATUS[0]}" "weak-np${P}"
    done
fi

# ── Section 8: how to split a fixed machine between ranks and threads ───────
# Every other sweep varies the amount of hardware. This one holds it fixed at
# AVAILABLE_CORES and varies only how it is DIVIDED: 192x1, 96x2, 48x4, ...,
# 4x48. Same cores, same problem, same binary — the only thing that changes is
# where the boundary between process parallelism and thread parallelism falls.
#
# It exists because "should several ranks share a node, or one rank per node
# with OpenMP inside?" cannot be answered from the strong-scaling curve, where
# ranks and cores move together and the two effects are confounded. Here they
# are separated, and the answer is not obvious in either direction:
#
#   more ranks  — no shared mutable state, no OpenMP barrier, each rank's
#                 output buffer first-touched by the one thread that fills it;
#                 but one more message into root at every gather.
#   more threads — fewer, larger messages into root; but one output buffer per
#                 rank shared by T threads, and past a socket those threads are
#                 writing across the NUMA interconnect.
#
# The expected optimum is therefore neither end but one rank per NUMA domain —
# on a 2-socket node, 2 ranks x (cores/2) threads. Whether that is where it
# actually lands is the point of measuring.
#
# Every point uses exactly AVAILABLE_CORES workers, so no point is
# oversubscribed and the times are directly comparable: read the "hybrid" row of
# each launch, which is the one that ran with T threads per rank.
#
# Compare those rows on min_ms, not on speedup. The workload is identical across
# the splits, so the wall times are directly comparable — but each launch
# re-measures its own single-rank serial baseline, at a different moment and on
# a differently loaded machine, so the speedup column of this file is divided by
# a slightly different number in every row. That is the right convention for the
# scaling sweeps, where each point is its own experiment; here it adds noise to
# the one comparison the file exists to make.
# Below four cores there is no split to speak of, so the section stands down.
if [ "${HYBRID_SPLIT:-1}" != "0" ] && [ "${AVAILABLE_CORES}" -ge 4 ]; then
    # Thread counts worth trying: powers of two plus the divisors that land on
    # a socket or a node. Filtered below to those that divide the machine.
    HYBRID_SPLIT_THREADS=${HYBRID_SPLIT_THREADS:-"1 2 4 8 12 24 48"}
    # One workload, big enough that the communication matters (a short one would
    # compare startup costs) but not so big that the serial baseline each launch
    # re-measures dominates the job. Defaults to the LAST entry of
    # SCALING_DURATIONS; set HYBRID_SPLIT_DURATION to override.
    if [ -z "${HYBRID_SPLIT_DURATION:-}" ]; then
        for _hsd in ${SCALING_DURATIONS}; do HYBRID_SPLIT_DURATION=${_hsd}; done
        HYBRID_SPLIT_DURATION=${HYBRID_SPLIT_DURATION:-60}
    fi

    echo ""
    echo "======== [8/8] Rank/thread split at ${AVAILABLE_CORES} fixed cores ======="
    SPLITFILE="${RESULTS_DIR}/${PREFIX}_hybrid_split.csv"
    : > "${SPLITFILE}"

    for T in ${HYBRID_SPLIT_THREADS}; do
        # A rank cannot straddle nodes, so T is capped by one node's cores; and
        # a split that does not divide the machine evenly would leave cores idle
        # and make the point incomparable with the others.
        [ "${T}" -le "${NODE_CORES}" ]              || continue
        [ $(( AVAILABLE_CORES % T )) -eq 0 ]        || continue
        P=$(( AVAILABLE_CORES / T ))
        [ "${P}" -ge 1 ]                            || continue

        export OMP_NUM_THREADS=${T}
        echo "  -np ${P} x ${T} thr = ${AVAILABLE_CORES} workers · ${HYBRID_SPLIT_DURATION}s"

        csv_append "${SPLITFILE}" \
            ${MPI_LAUNCHER} "${P}" ${MPIRUN_EXTRA} \
            $(bind_flags "${P}" "${T}") \
            "${BUILD_DIR}/benchmarks/benchmark_MPI_Main" \
            ${BENCH_ARGS} "${HYBRID_SPLIT_DURATION}" scatter
        note_status "${PIPESTATUS[0]}" "split-${P}x${T}"
    done
fi

echo ""
echo "=========================================================================="
if [ -n "${FAILED}" ]; then
    echo "FAILED sections:${FAILED}"
    echo "Results in ${RESULTS_DIR} (incomplete):"
    ls -1 "${RESULTS_DIR}" | sed 's/^/  /'
    echo "Test results in ${TEST_RESULTS_DIR}:"
    ls -1 "${TEST_RESULTS_DIR}" | sed 's/^/  /'
    exit 1
fi

echo "All sections completed. Results in ${RESULTS_DIR}:"
ls -1 "${RESULTS_DIR}" | sed 's/^/  /'
echo "Test results in ${TEST_RESULTS_DIR}:"
ls -1 "${TEST_RESULTS_DIR}" | sed 's/^/  /'
