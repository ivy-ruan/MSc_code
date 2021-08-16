#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N non_linear_daphne            
#$ -cwd
#$ -pe sharedmem 16                  
 
#$ -l h_vmem=8G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  Initialise the environment modules
#  memory limit of 1 Gbyte: -l h_vme

# Initialise the environment modules
. /etc/profile.d/modules.sh

module unload anaconda/5.3.1
# Run the program
module load anaconda/5.3.1

source activate new_env

python PCMCI_non_linear.py
