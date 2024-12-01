#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=DQN_onlineEWC
#SBATCH --output=out/DQN_onlineEWC_%j.out
#SBATCH --time=95:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=30G

seed=$1
# env_id=$2
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "seed used: $seed"
# echo "envs used: $env_id"

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
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# log into WandB
export WANDB_API_KEY="5602093e351ccd9235bbc1d17997cc8c7dcacd43"
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# "MinAtar/Asterix-v0" "MinAtar/Freeway-v0" "MinAtar/Seaquest-v0" "MinAtar/SpaceInvaders-v0"
# Run existing Python script in repo for tuning
# sbatch dqn_job.sh 11 freeway
python DQN_minatar.py --seed=$seed 
# --env=$env_id