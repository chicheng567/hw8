import os
import time
import argparse
from typing import List

import numpy as np
import gymnasium as gym
import retro

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecTransposeImage,
    VecVideoRecorder,
)
from control import (
    COMBOS,
    DiscreteActionWrapper,
    LifeTerminationWrapper,
    ExtraInfoWrapper,
    RewardOverrideWrapper,
    InfoLogger,
)
def make_base_env(game: str, state: str):
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = DiscreteActionWrapper(env, COMBOS)
    env = ExtraInfoWrapper(env)
    env = LifeTerminationWrapper(env, death_penalty=0)
    env = RewardOverrideWrapper(env)
    env = InfoLogger(env)
    return env


def _make_env_thunk(game: str, state: str):
    def _thunk():
        return make_base_env(game, state)

    return _thunk


def make_vec_env(game: str, state: str, n_envs: int, use_subproc: bool = True):
    env_fns = [_make_env_thunk(game, state) for _ in range(n_envs)]
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecTransposeImage(vec_env)
    return vec_env

def evaluate_policy(model: PPO, game: str, state: str, n_episodes: int, max_steps: int):
    env = make_base_env(game, state)

    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            steps += 1

        returns.append(ep_ret)

    env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    best_ret = float(np.max(returns)) if returns else 0.0
    return mean_ret, best_ret

def record_video(model: PPO, game: str, state: str, out_dir: str, video_len: int, prefix: str):
    os.makedirs(out_dir, exist_ok=True)

    vec_env = make_vec_env(game, state, n_envs=1)
    vec_env = VecVideoRecorder(
        vec_env,
        out_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=video_len,
        name_prefix=prefix,
    )

    obs = vec_env.reset()
    for _ in range(video_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        if dones[0]:
            obs = vec_env.reset()

    vec_env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="SuperMarioWorld-Snes")
    parser.add_argument("--state", type=str, default="YoshiIsland1")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--train-chunk", type=int, default=50_000)   # 每段訓練多久就 eval + record
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=4000)
    parser.add_argument("--record-steps", type=int, default=3000)
    parser.add_argument("--logdir", type=str, default="./runs_smw")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    video_dir = os.path.join(args.logdir, "videos")
    ckpt_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_env = make_vec_env(args.game, args.state, n_envs=args.n_envs)

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        tensorboard_log=os.path.join(args.logdir, "tb"),
    )

    best_mean = -1e18
    trained = 0
    round_idx = 0

    while trained < args.total_steps:
        round_idx += 1
        chunk = min(args.train_chunk, args.total_steps - trained)

        print(f"\n=== Round {round_idx} | learn {chunk} steps (trained={trained}) ===")
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained += chunk

        ckpt_path = os.path.join(ckpt_dir, f"ppo_step_{trained}.zip")
        model.save(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        mean_ret, best_ret = evaluate_policy(
            model,
            args.game,
            args.state,
            n_episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
        )
        print(f"[EVAL] mean_return={mean_ret:.3f}, best_return={best_ret:.3f}")

        if mean_ret > best_mean:
            best_mean = mean_ret
            best_path = os.path.join(args.logdir, "best_model.zip")
            model.save(best_path)
            print(f"New best mean_return={best_mean:.3f} -> saved {best_path}")

            prefix = f"best_step_{trained}_mean_{best_mean:.2f}"
            print(f"Recording video: {prefix} ({args.record_steps} steps)")
            record_video(
                model,
                args.game,
                args.state,
                video_dir,
                video_len=args.record_steps,
                prefix=prefix,
            )
        else:
            print("No improvement, skip recording this round.")

    train_env.close()


if __name__ == "__main__":
    main()
