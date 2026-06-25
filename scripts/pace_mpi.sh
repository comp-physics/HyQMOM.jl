#!/usr/bin/env bash
# Turnkey multi-rank launcher for HyQMOM.jl on GT PACE.
# See RUNNING.md for the full explanation of each step.
#
# Usage:
#   scripts/pace_mpi.sh <NRANKS> <julia-script.jl> [args...]
#   scripts/pace_mpi.sh 16 examples/run_3d_jets_timeseries.jl --no-viz
#   REPRO_MA=2.0 scripts/pace_mpi.sh 16 test/repro/run_crossing.jl
#
# Env overrides:
#   JULIA_MODULE   (default julia/1.11.3)
#   OPENMPI_MODULE (default openmpi/4.1.5)   # must match LocalPreferences OpenMPI ABI
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <NRANKS> <julia-script.jl> [args...]" >&2
  exit 2
fi
NR="$1"; shift
SCRIPT="$1"; shift

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# 1. modules (must be loaded in this shell)
source /etc/profile >/dev/null 2>&1 || true
module load "${JULIA_MODULE:-julia/1.11.3}" "${OPENMPI_MODULE:-openmpi/4.1.5}"

# 2. headless + single-node MPI transport
export HYQMOM_SKIP_PLOTTING=true
export CI=true
export UCX_TLS="${UCX_TLS:-sm,self}"   # avoids UCX IB wireup failure on shared nodes

# 3. precompile once (serial) so the N ranks don't contend on the cache lock
echo "[pace_mpi] precompiling HyQMOM (serial)..."
julia --project=. -e 'using HyQMOM' >/dev/null

# 4. launch
echo "[pace_mpi] launching: mpiexec -n $NR julia --project=. $SCRIPT $*"
exec mpiexec -n "$NR" julia --project=. "$SCRIPT" "$@"
