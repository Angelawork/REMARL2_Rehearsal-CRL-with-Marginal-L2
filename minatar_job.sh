#!/bin/bash

#SBATCH --job-name=0.01_new_rehearsal
#SBATCH --output=out/0.01_new_rehearsal__%j.out
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G

seed=$1
l2_loss=$2
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
export WANDB_API_KEY=""
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# "MinAtar/Asterix-v0" "MinAtar/Freeway-v0" "MinAtar/Seaquest-v0" "MinAtar/SpaceInvaders-v0"
# Run existing Python script in repo for tuning
# args.value_norm and not args.global_value_norm for local value-norm/rew rescale!!
# args.value_norm and args.global_value_norm for global value-norm/rew rescale!!

# MinAtar/Breakout-v0 MinAtar/Freeway-v0 MinAtar/Asterix-v0 MinAtar/Seaquest-v0 MinAtar/SpaceInvaders-v0
#candidate l2 need use-l2-loss to be True
python PPO_Experiment.py --exp-type="ppo_minatar" \
        --exp-name="0.01_rehearsal_uniform_16batch" \
        --env-ids "MinAtar/Breakout-v0" "MinAtar/Freeway-v0" "MinAtar/Asterix-v0" "MinAtar/Seaquest-v0" "MinAtar/SpaceInvaders-v0" \
        --wandb-project-name="PPO_minatar" \
        --seed=$seed \
        --use-vcl=False \
        --use-rehearsal=True \
        --rehearsal-coef=0.01 \
        --rehearsal-batch=16 \
        --use-packnet=False \
        --use-tderror=False \
        --global-tderror=False \
        --use-clip-l2=False \
        --use-l2-loss=$l2_loss \
        --use-l2-0-loss=False \
        --l2-coef=10 \
        --periodic-l2=False \
        --candidate-l2=False \
        --candidate-metric="cosine" \
        --use-ewc=False \
        --ewc-coef=0.01 \
        --use-parseval-reg=False \
        --parseval-coef=0.1 \
        --use-inputScaling=False \
        --use-DiagonalLayer=False \
        --use-weight-clip=False \
        --weight-clipping=2.0 \
        --use-crelu=False \
        --value-norm=False \
        --global-value-norm=False \
        --reward-rescale=False \
        --global-reward-rescale=False \
        --wandb-log-off=True \
        --torch-deterministic=True \
        --cuda=True \
        --track=True \
        --rolling-window=1000 \
        --eval-interval=10000 \
        --capture-video=False \
        --total-timesteps=10000000 \
        --learning-rate=2.5e-4 \
        --num-envs=128 \
        --num-steps=128 \
        --num-minibatches=4 \
# python DQN_minatar.py --seed=$seed --env=$env_id