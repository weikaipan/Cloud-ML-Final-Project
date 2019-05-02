#!/bin/sh
#
#BATCH --verbose
#SBATCH --job-name=training
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:k80:1

#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd /home/wkp219/Cloud-ML-Final-Project/train/

module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2
# module load pytorch/python3.6/0.3.0_4

source /home/wkp219/Cloud-ML-Final-Project/train/py3.6.3/bin/activate
python3 -u train.py -topology GRU -packed True
