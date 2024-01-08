#!/bin/bash -l
#SBATCH --job-name=soundstream_training_100_resume
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=LOG_%x.%j.out
#SBATCH --error=LOG_%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV # These 2 commands give var from env

source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Conda
source activate encodec_copy
echo "Job_bash: Activated conda env: ${CONDA_DEFAULT_ENV}"

python -c "import torch; print('Cuda is_available: ',torch.cuda.is_available()); print('Cuda version: ',torch.version.cuda);"

if ! python -c "import torch; torch.cuda.is_available()"; then
    echo "CUDA is not available. Exiting the batch job."
    exit 1
fi

# Go to project directory
# Logges will be saves here too
cd $HOME/SoundStream

echo "Job_bash: Begin training"

python train.py

echo "Job_bash: Finished"

