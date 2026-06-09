#!/bin/bash -l
#SBATCH --job-name=fedssp
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-eemcs-bsc-ti

echo "=============================="
echo " JOB START"
echo "=============================="

echo "Hostname: $(hostname)"
echo "Date: $(date)"

# -----------------------------
# Conda setup
# -----------------------------
module load 2024r1
module load cuda/11.7

module load miniforge3/25.11.0
source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh

conda activate /scratch/$USER/conda_envs/fedgraph

# -----------------------------
# Move to project
# -----------------------------
cd /home/$USER/rssls

export PYTHONPATH=$(pwd)

# -----------------------------
# CUDA diagnostics
# -----------------------------
echo "=============================="
echo " CUDA CHECK"
echo "=============================="

python - <<EOF
import torch
import torch_geometric

print(torch.__version__)
print(torch.cuda.is_available())
print(torch_geometric.__version__)
EOF

python - <<EOF
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.rand(1000,1000).cuda()
    y = torch.mm(x, x)
    print("CUDA tensor test successful")
else:
    raise RuntimeError("CUDA NOT AVAILABLE")
EOF

# -----------------------------
# Run experiment
# -----------------------------
echo "=============================="
echo " START TRAINING"
echo "=============================="

for seed in 4 5 6 7 8
do
    echo "=============================="
    echo " RUNNING SEED $seed"
    echo "=============================="

    srun python main_multiDS.py \
        --alg fedSSP \
        --data_group biosncv \
        --spectral_mode topk \
        --spectral_k 4 \
        --seed $seed \
        --repeat $seed
done

echo "=============================="
echo " JOB FINISHED"
echo "=============================="
