#!/bin/bash
#SBATCH --job-name=spmv
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=results/results_%j.txt
module load cuda/11.3
cd $SLURM_SUBMIT_DIR
for m in data/*.mtx; do
         echo "===== $m"
	./bin/bench "$m"
done
