#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=og2114@nyu.edu
#SBATCH --output=/scratch/og2114/rebase/logs/slurm_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=esm34_both_nograde3
#SBATCH --signal=SIGUSR1@90
source $SCRATCH/venv/bin/activate
python modeling.py model=both esm=esm34 model.gpu=4 model.d_model=1280 model.lr=1e-3 model.name=esm34_both_grade3
