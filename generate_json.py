import json

original_data = [
    {
        "cmd": "--exp-type=ppo_minatar --exp-name=0.01_rehearsal_uniform_32batch --env-ids MinAtar/Breakout-v0 MinAtar/Freeway-v0 MinAtar/Asterix-v0 MinAtar/Seaquest-v0 MinAtar/SpaceInvaders-v0 --wandb-project-name=PPO_minatar --use-vcl=False --use-rehearsal=True --rehearsal-coef=0.01 --rehearsal-batch=32 --use-packnet=False --use-tderror=False --global-tderror=False --use-clip-l2=False --use-l2-loss=False --use-l2-0-loss=False --l2-coef=10 --periodic-l2=False --candidate-l2=False --candidate-metric=cosine --use-ewc=False --ewc-coef=0.01 --use-parseval-reg=False --parseval-coef=0.1 --use-inputScaling=False --use-DiagonalLayer=False --use-weight-clip=False --weight-clipping=2.0 --use-crelu=False --value-norm=False --global-value-norm=False --reward-rescale=False --global-reward-rescale=False --wandb-log-off=True --torch-deterministic=True --cuda=True --track=True --rolling-window=1000 --eval-interval=10000 --capture-video=False --total-timesteps=10000000 --learning-rate=2.5e-4 --num-envs=128 --num-steps=128 --num-minibatches=4"
    }
]

seeds = [13, 14, 15]

new_data = []
for entry in original_data:
    for seed in seeds:
        new_entry = entry.copy()
        new_entry["cmd"] += f" --seed={seed}"
        new_data.append(new_entry)

output_path = "param_minatar.json"
with open(output_path, "w") as f:
    json.dump(new_data, f, indent=4)