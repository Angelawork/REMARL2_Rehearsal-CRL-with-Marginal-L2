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

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module load python/3.10 

Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
export CUDA_VISIBLE_DEVICES=0 #for cuda device error

cd ~

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

python PPO_Experiment.py --exp-type="ppo_metaworld" \
        --exp-name="ppo_metaworld" \
        --env-ids 'push-back-v2' 'sweep-into-v2' 'window-close-v2' \
        --seed=42 \
        --torch-deterministic=True \
        --cuda=True \
        --track=True \
        --capture-video=False \
        --total-timesteps=1000000 \
        --learning-rate=2.5e-4 \
        --num-envs=8 \
        --num-steps=128 \