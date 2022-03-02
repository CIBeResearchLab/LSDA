#!/bin/bash -l
#BATCH -o std_out
#SBATCH -e std_err
#SBATCH -p CiBeR
#SBATCH -w GPU43
source activate nlp
python hlt.py
