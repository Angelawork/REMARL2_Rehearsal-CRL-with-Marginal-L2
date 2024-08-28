#!/bin/bash

#SBATCH --job-name=PPO_meta
#SBATCH --output=out/ppo_metaworld_%j.out
#SBATCH --error=err/ppo_metaworld_%j.err
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
    pip install -r meta_requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# log into WandB
export WANDB_API_KEY="..."
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# Run existing Python script in repo for tuning
python PPO_Experiment.py --exp_type="ppo_metaworld" \
        --exp_name="ppo_metaworld" \
        --env_ids 'push-back-v2' 'sweep-into-v2' 'window-close-v2' \
        --seed=42 \
        --torch_deterministic=True \
        --cuda=True \
        --track=True \
        --capture_video=False \
        --total_timesteps=1000000 \
        --learning_rate=2.5e-4 \
        --num_envs=8 \
        --num_steps=128 \