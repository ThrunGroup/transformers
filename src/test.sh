#!/bin/bash
#SBATCH --partition=thrun --account=thrun --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:4

#SBATCH --job-name="test"
#SBATCH --output=test-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=lukeleeai@gmail.com
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

python experiments/create_models.py
# done
echo "Done"