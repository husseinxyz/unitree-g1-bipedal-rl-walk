import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train(args):
    # Initialize wandb with more comprehensive config
    wandb_config = {
        "task": args.task,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "headless": args.headless,
        "curriculum": args.curriculum,
        "domain_rand": args.domain_rand,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
    }
    
    # Initialize wandb with TensorBoard integration
    wandb.init(
        project="unitree_rl",
        name=args.experiment_name,
        sync_tensorboard=True,
        monitor_gym=False,  # Enable gym monitoring
        config=wandb_config,
        tags=[args.task, "g1", "bipedal"],
        notes=f"Training run with {args.num_envs} environments, {args.max_iterations} iterations"
    )
    
    try:
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        
        # Log environment and training configurations
        wandb.config.update({
            "env_config": env_cfg.__dict__,
            "train_config": train_cfg.__dict__
        })
        
        # Start training with progress tracking
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations, 
            init_at_random_ep_len=True
        )
        
        # Log final metrics
        wandb.log({
            "final_mean_reward": ppo_runner.logger.get_last("train/episode_reward"),
            "final_mean_episode_length": ppo_runner.logger.get_last("train/episode_length"),
            "training_time": ppo_runner.logger.get_last("time/total")
        })
        
    except Exception as e:
        # Log any errors that occur during training
        wandb.log({"error": str(e)})
        raise e
    finally:
        # Ensure proper cleanup
        wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
