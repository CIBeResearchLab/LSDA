#!/bin/bash -l
#BATCH -o out
#SBATCH -e err
#SBATCH -p CiBeR
#SBATCH -w GPU43
#SBATCH --time=7-0
#SBATCH --mem=100GB
source activate nlp
jupyter lab --ip='0.0.0.0' --port=8888 --NotebookApp.token='' --NotebookApp.password=''
