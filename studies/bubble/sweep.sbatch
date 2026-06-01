#!/bin/bash
#SBATCH -J bubble_sweep
#SBATCH -o studies/bubble/sweep-%j.out
#SBATCH -A gts-sbryngelson3
#SBATCH -p cpu-gnr
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=120GB
#SBATCH --constraint=graniterapids
#SBATCH --qos=embers

set -e
cd "$SLURM_SUBMIT_DIR"
[ -f Project.toml ] || { echo "must submit from HyQMOM.jl root"; exit 1; }
[ -f LocalPreferences.toml ] || { echo "MPI not configured (run setup_mpi.jl)"; exit 1; }

module load julia/1.11.3 openmpi/4.1.5
export HYQMOM_SKIP_PLOTTING=true CI=true
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
export OMPI_MCA_btl=^openib
export UCX_WARN_UNUSED_ENV_VARS=n

echo "host=$(hostname) ntasks=$SLURM_NTASKS date=$(date)"
cat LocalPreferences.toml

TMAX=0.05
NLADDER="128 256 512 1024"
KNS="0.01 0.1 1.0"

# --- Bubble grid-refinement + correction/conservation diagnostics, per Kn ---
for Kn in $KNS; do
  echo "===== BUBBLE SWEEP Kn=$Kn ====="
  srun --mpi=pmix julia --project=. studies/bubble/run_refinement.jl $Kn $TMAX $NLADDER
done

# --- Mach-number sweep (correction activity vs Ma) on crossing jets ---
echo "===== MA SWEEP (jets) ====="
srun --mpi=pmix julia --project=. studies/bubble/run_ma_jets.jl 256 1.0 $TMAX 0.0 2.0 4.0

# --- Analysis (serial) ---
for Kn in $KNS; do
  echo "===== ANALYSIS Kn=$Kn ====="
  julia --project=. studies/bubble/analyze_convergence.jl $Kn $NLADDER || true
  julia --project=. studies/bubble/analyze_rotation.jl    $Kn $NLADDER || true
done

echo "=== sweep done $(date) ==="
echo "--- correction summaries ---"
cat studies/bubble/out/corrections_*.txt 2>/dev/null || true
