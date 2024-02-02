from stable_baselines3 import PPO
import os
from snakeenv import SnekEnv
import time

import wandb

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

wandb.init(project="GAME-AI-PA-1", sync_tensorboard=True, tensorboard=logdir)

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SnekEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100
iters = 0
max_iters = 10
while iters < max_iters:
	print("current iteration ", iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	print(f"Saved model to {models_dir}/{TIMESTEPS*iters}")
	
wandb.finish()