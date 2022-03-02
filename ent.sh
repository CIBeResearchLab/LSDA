#!/bin/bash -l
#BATCH -o std_out
#SBATCH -e std_err
source activate nlp
python ent.py
