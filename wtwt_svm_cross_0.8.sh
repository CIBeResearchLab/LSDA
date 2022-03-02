#!/bin/bash -l
#SBATCH -o outsvm
#SBATCH -e errsvm
%#SBATCH -w GPU43
#SBATCH --time=7-0
%#SBATCH -p CiBeR
source activate nlp
python wtwt_svm_cross_0.8.py
