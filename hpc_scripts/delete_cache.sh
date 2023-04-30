
#!/bin/bash
# Job name:
#SBATCH --job-name=delete_cache
#
# Partition:
#SBATCH --partition=gablab
#
# Nodes:
#SBATCH -N 1 # one node
#
# Tasks:
#SBATCH -c 8 # 8 hyperthreaded cores 
#
# Memory:
#SBATCH --mem=1G
#
# Wall clock limit:
#SBATCH --time=2-00:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

# username
username=maedbh

# delete pydra-ml cache from openmind (takes up to omuch space)
python3 delete_cache.py --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/