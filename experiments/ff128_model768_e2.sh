#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=01-23:20:00
#SBATCH --job-name=ff128_model768_e2
#SBATCH --output=/scratch/jam1657/rebase/slurm/slurm_%j.out
#SBATCH --signal=SIGUSR1@90

module purge

singularity exec --nv \
	    --overlay /scratch/jam1657/conda/rebase-torch/overlay-7.5GB-300K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python modeling.py model=both esm=vanilla model.gpu=-1"