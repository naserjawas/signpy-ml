#!/bin/bash
#SBATCH --job-name=signpy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=08:00:00

module purge
module load GCCcore/11.2.0 Python/3.9.6

source ./signpy/bin/activate
python -W ignore phoenix_testpoints_hpc.py
deactivate
