#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --exclude=lac-142
#SBATCH --time=2:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes    (same as -N)
#SBATCH -c 8           # number of CPUs (or cores) per task (same as -c
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name real-none # you can give your job a name for easier identification (same as -J)

source ~/.bashrc
# source ~/anaconda3/bin/activate torch18
source ~/anaconda3/bin/activate a100
srun -n 1 -o aug28/EERM_arxiv_%j.out  --unbuffered --exclusive python main.py --method eerm --gnn gcn --lr 0.005 --K 5 --T 5 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device 0 --runs 1
scontrol show job $SLURM_JOB_ID
# salloc c 8 --nodes=1 --time=8:00:00 --mem=32G --gres=gpu:k80:1
