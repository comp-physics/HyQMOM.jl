#!/bin/bash
#SBATCH -J bubble_pilot
#SBATCH -o studies/bubble/pilot-%j.out
#SBATCH -A gts-sbryngelson3
#SBATCH -p cpu-gnr
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --constraint=graniterapids
#SBATCH --qos=embers

set -e
cd "$SLURM_SUBMIT_DIR"
[ -f Project.toml ] || { echo "must submit from HyQMOM.jl root"; exit 1; }

module load julia/1.11.3
export HYQMOM_SKIP_PLOTTING=true
export CI=true
export JULIA_NUM_THREADS=1

echo "host=$(hostname) cores=$(nproc) date=$(date)"
echo "=== refinement: Kn=1, tmax=0.06, N=64,128,256 ==="
julia --project=. studies/bubble/run_refinement.jl 1.0 0.06 64 128 256

echo "=== convergence analysis ==="
julia --project=. studies/bubble/analyze_convergence.jl 1.0 64 128 256

echo "=== done $(date) ==="
