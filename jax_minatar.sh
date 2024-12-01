#!/bin/bash

#SBATCH --job-name=jax_PPO_minatar
#SBATCH --output=out/jax_ppo_minatar_%j.out
#SBATCH --error=err/jax_error_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
 #SBATCH --gpus-per-task=rtx8000:1
 #SBATCH --cpus-per-task=6
 #SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
seed=$1
env_name=$2
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Load any modules and activate your Python environment here
module load python/3.10 

Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0 #for cuda device error

cd /home/mila/q/qingchen.hu/test_PPO/test/PPO_meta-minatar

# install or activate requirements
if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r jax_requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# log into WandB
export WANDB_API_KEY=""
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# "MinAtar/Asterix-v0" "MinAtar/Freeway-v0" "MinAtar/Seaquest-v0" "MinAtar/SpaceInvaders-v0"
# Run existing Python script in repo for tuning
python PPO_jax.py seed=$seed env_name=$env_name
#sbatch jax_minatar.sh 10 "minatar-asterix"