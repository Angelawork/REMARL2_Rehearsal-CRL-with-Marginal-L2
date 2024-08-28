#!/bin/bash

#SBATCH --job-name=PPO_minatar
#SBATCH --output=out/ppo_minatar_%j.out
#SBATCH --error=err/ppo_minatar_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
 #SBATCH --gpus-per-task=rtx8000:1
 #SBATCH --cpus-per-task=6
 #SBATCH --ntasks-per-node=1
#SBATCH --mem=30G

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Load any modules and activate your Python environment here
module load python/3.10 

Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0 #for cuda device error

cd /home/mila/q/qingchen.hu/test_PPO/CL_PPOexpr/

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
export WANDB_API_KEY="c2bec171ac9b0c5f49bc0a2efcbbe5ffba60da23"
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# Run existing Python script in repo for tuning
python PPO_Experiment.py --exp_type="ppo_minatar" \
        --exp_name="ppo_minatar" \
        --env_ids "MinAtar/Breakout-v0","MinAtar/Asterix-v0", "MinAtar/Freeway-v0" \
        --seed=42 \
        --torch_deterministic=True \
        --cuda=True \
        --track=True \
        --capture_video=False \
        --total_timesteps=10000000 \
        --learning_rate=2.5e-4 \
        --num_envs=8 \
        --num_steps=128 \