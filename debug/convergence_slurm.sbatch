#!/usr/bin/env bash
#SBATCH --job-name=hyqmom-conv
#SBATCH --nodes=4                 # 1024^3 needs multi-node: ~300 GB/field-copy x several
#SBATCH --ntasks-per-node=24      # tune to cores; total ranks = nodes * ntasks-per-node
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                    # all node memory
#SBATCH --time=08:00:00
#SBATCH --output=conv_%j.log
#
# Grid-convergence reference runs for the crossing problem (Rodney: first-order
# reference on fine grids, convergence judged on density M000).
#
# Submit one resolution per job, e.g.:
#   sbatch --nodes=1 --export=ALL,CONV_NP=256,CONV_ORDER=1 debug/convergence_slurm.sbatch
#   sbatch --nodes=2 --export=ALL,CONV_NP=512,CONV_ORDER=1 debug/convergence_slurm.sbatch
#   sbatch --nodes=8 --export=ALL,CONV_NP=1024,CONV_ORDER=1 debug/convergence_slurm.sbatch
#
# Memory guide (per field copy = Np^3 * 35 * 8 B; solver holds ~5-10 copies live):
#   256^3  ~ 4.6 GB/copy   -> 1 node
#   512^3  ~ 37 GB/copy    -> 1 node (fits ~1.5 TB), 2 for headroom
#   1024^3 ~ 300 GB/copy   -> >= 4-8 nodes (multi-node REQUIRED)
# Output is density-only (~8 GB at 1024^3), not the full state.
set -euo pipefail

REPO_DIR="${HYQMOM_DIR:-/storage/project/r-sbryngelson3-0/sbryngelson3/HyQMOM.jl}"
cd "$REPO_DIR"
source /etc/profile >/dev/null 2>&1 || true
module load "${JULIA_MODULE:-julia/1.11.3}" "${OPENMPI_MODULE:-openmpi/4.1.5}"

export HYQMOM_SKIP_PLOTTING=true CI=true
# IMPORTANT: multi-node needs InfiniBand transport, NOT the single-node sm,self.
# Let UCX auto-select (rc/ud over IB) unless overridden. Verify wireup on first run.
unset UCX_TLS || true

# precompile once (serial) to avoid rank cache-lock contention
julia --project=. -e 'using HyQMOM' >/dev/null

NR=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[conv] Np=${CONV_NP:-?} order=${CONV_ORDER:-1} ranks=$NR nodes=$SLURM_NNODES"
# srun integrates with Slurm's PMIx; if OMPI lacks PMIx use: mpirun -np $NR --map-by ppr:${SLURM_NTASKS_PER_NODE}:node
srun --mpi=pmix -n "$NR" julia --project=. \
    /storage/project/r-sbryngelson3-0/sbryngelson3/debug/convergence_run.jl
