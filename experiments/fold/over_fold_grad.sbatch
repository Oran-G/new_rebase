#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=og2114@nyu.edu
#SBATCH --output=./
#SBATCH --ntasks-per-node=1
#SBATCH --job-name='preproccess embeddings for esm2'
#SBATCH --signal=SIGUSR1@90

singularity exec --nv --overlay /vast/og2114/rebase/inverse_fold.ext3:ro /vast/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c 'source /ext3/env.sh; python3 /vast/og2114/new_rebase/folding.py esm=grad model.gpu=1 model.lr=3e-4'
