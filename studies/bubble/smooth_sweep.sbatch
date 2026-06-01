#!/bin/bash
#SBATCH -J bubble_smooth
#SBATCH -o studies/bubble/smooth-%j.out
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
[ -f LocalPreferences.toml ] || { echo "MPI not configured"; exit 1; }

module load julia/1.11.3 openmpi/4.1.5
export HYQMOM_SKIP_PLOTTING=true CI=true
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
export OMPI_MCA_btl=^openib UCX_WARN_UNUSED_ENV_VARS=n

export BUBBLE_PROFILE=smooth     # Gaussian density bump -> smooth solution
export BUBBLE_RHO_IN=1.5         # gentle amplitude so it stays smooth
export BUBBLE_WIDTH=0.1

echo "host=$(hostname) ntasks=$SLURM_NTASKS profile=$BUBBLE_PROFILE rho_in=$BUBBLE_RHO_IN date=$(date)"
TMAX=0.05
NLADDER="128 256 512 1024"
KNS="0.01 0.1 1.0"

for Kn in $KNS; do
  echo "===== SMOOTH BUBBLE Kn=$Kn ====="
  srun --mpi=pmix julia --project=. studies/bubble/run_refinement.jl $Kn $TMAX $NLADDER
done

for Kn in $KNS; do
  echo "===== ANALYSIS (smooth) Kn=$Kn ====="
  julia --project=. studies/bubble/analyze_convergence.jl $Kn $NLADDER || true
  julia --project=. studies/bubble/analyze_rotation.jl    $Kn $NLADDER || true
done

echo "=== smooth sweep done $(date) ==="
