#!/bin/bash

# Time format is D-HH:MM:SS (see "man sbatch")
# Here we allow the job to run at most ten minutes
#SBATCH --time=0-00:10:00

# Run one task on one node
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Make 48 cores available to our task (otherwise defaults to 1)
#SBATCH --cpus-per-task=48

# Use any of the compute nodes in the 'all' partition
#SBATCH --partition=ci
#SBATCH --nodelist=ant13

# Redirect output and error output
#SBATCH --output=job.out
#SBATCH --error=job.err



. /opt/spack/main/env.sh
module load python

#!/bin/bash
source ./env/bin/activate

srun --cpus-per-task=48 --mem-per-cpu=2048 --job-name='Pathpave_Pymoo' --partition='members' python3 cluster.py>&1 & 
