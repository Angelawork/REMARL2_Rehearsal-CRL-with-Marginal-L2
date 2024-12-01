#!/bin/bash

#SBATCH --job-name=PPO_minatar_baseline
#SBATCH --output=out/ppo_minatar_baseline_%j.out
#SBATCH --error=err/ppo_minatar_baseline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
 #SBATCH --gpus-per-task=rtx8000:1
 #SBATCH --cpus-per-task=6
 #SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
seed=$1
atar_env=$2
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "seed used: $seed"
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
export WANDB_API_KEY=""
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# "MinAtar/Asterix-v0" "MinAtar/Freeway-v0" "MinAtar/Seaquest-v0" "MinAtar/SpaceInvaders-v0" "MinAtar/Breakout-v0"
# Run existing Python script in repo for tuning
python PPO_baseline.py --exp-type="ppo_minatar" \
        --exp-name="ppo_minatar" \
        --env-ids $atar_env \
        --wandb-project-name="pytorch-minatar-ppo-baseline" \
        --seed=$seed \
        --torch-deterministic=True \
        --cuda=True \
        --track=True \
        --rolling-window=1000 \
        --eval-interval=5000 \
        --capture-video=False \
        --total-timesteps=20000000 \
        --learning-rate=2.5e-4 \
        --num-envs=128 \
        --num-steps=128 \
        --num-minibatches=4 \

# sbatch minatar_baseline_job.sh 3 "MinAtar/Asterix-v0"