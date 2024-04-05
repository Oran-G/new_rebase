#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=96:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=og2114@nyu.edu
#SBATCH --output=/vast/og2114/output_home/logs/slurm_$SLURM_JOB_ID/job.out
#SBATCH --ntasks-per-node=1
#SBATCH --job-name='RFdiffusion Complexes and fold conditioning model training script'
#SBATCH --signal=SIGUSR1@90
module purge
mkdir -p /vast/og2114/output_home/logs/slurm_$SLURM_JOB_ID
export WANDB_CACHE_DIR=/vast/og2114/output_home/logs/wandb_cache
singularity exec --nv --overlay /vast/og2114/rebase/inverse_fold.ext3:ro /vast/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c 'source /ext3/env.sh; python3 /vast/og2114/new_rebase/full_diffusor.py'
