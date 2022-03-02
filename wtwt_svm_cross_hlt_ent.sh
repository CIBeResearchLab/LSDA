#!/bin/bash -l
#SBATCH -o outsvm_cross_hlt_ent
#SBATCH -e errsvm
#SBATCH -w GPU43
#SBATCH -p CiBeR
#SBATCH --time=7-0
source activate nlp
python wtwt_svm_cross_hlt_ent.py
