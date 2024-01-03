#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=og2114@nyu.edu
#SBATCH --output=/scratch/og2114/rebase/logs/slurm_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --job-name='esm inverse folding model binding site validation prediction'
#SBATCH --signal=SIGUSR1@90
module purge
singularity exec --nv --overlay /scratch/og2114/rebase/inverse_fold.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c 'source /ext3/env.sh; python3 /scratch/og2114/rebase/focus/pred.py model=lightning esm=grad model.gpu=1 model.lr=1e-3'
