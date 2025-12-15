import os
import time
import argparse
from typing import List

import numpy as np
import gymnasium as gym
import retro
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
)
from control import (
    COMBOS,
    DiscreteActionWrapper,
    LifeTerminationWrapper,
    ExtraInfoWrapper,
    AuxObservationWrapper,
    RewardOverrideWrapper,
    InfoLogger,
    PreprocessObsWrapper,
)
from custom_policy import VisionBackbonePolicy, CustomPPO
def make_base_env(game: str, state: str):
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = PreprocessObsWrapper(env)
    env = DiscreteActionWrapper(env, COMBOS)
    env = ExtraInfoWrapper(env)
    env = LifeTerminationWrapper(env)
    env = RewardOverrideWrapper(env)
    env = AuxObservationWrapper(env)
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

    return vec_env

def evaluate_policy(model: CustomPPO, game: str, state: str, n_episodes: int, max_steps: int):
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

def _format_info(info: dict, max_len: int = 120) -> str:
    if not isinstance(info, dict) or not info:
        return "{}"
    parts = []
    total_len = 0
    for key, value in info.items():
        fragment = f"{key}={value}"
        if total_len + len(fragment) > max_len:
            parts.append("...")
            break
        parts.append(fragment)
        total_len += len(fragment) + 2
    return "{" + ", ".join(parts) + "}"


def _annotate_frame(frame: np.ndarray, cumulative_reward: float, last_reward: float, info: dict, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    info_str = _format_info(info)
    lines = [
        f"reward={last_reward:.3f}",
        f"cum_reward={cumulative_reward:.3f}",
        f"info: {info_str}",
    ]
    padding = 4
    bbox_sample = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox_sample[3] - bbox_sample[1]
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
    box_width = max(line_widths) + padding * 2
    box_height = line_height * len(lines) + padding * (len(lines) + 1)
    draw.rectangle([0, 0, box_width, box_height], fill=(0, 0, 0, 200))
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height + padding
    return np.array(img)


def record_video(model: CustomPPO, game: str, state: str, out_dir: str, video_len: int, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.mp4")

    env = make_base_env(game, state)
    fps = env.metadata.get("render_fps", 60)
    writer = imageio.get_writer(out_path, fps=fps)
    font = ImageFont.load_default()

    obs, info = env.reset()
    cumulative_reward = 0.0
    for _ in range(video_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is None:
            continue
        cumulative_reward += float(reward)
        annotated = _annotate_frame(frame, cumulative_reward, float(reward), info, font)
        writer.append_data(annotated)
        if terminated or truncated:
            obs, info = env.reset()
            cumulative_reward = 0.0

    writer.close()
    env.close()
    print(f"Saved video to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="SuperMarioWorld-Snes")
    parser.add_argument("--state", type=str, default="YoshiIsland1")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--train-chunk", type=int, default=50_000)   # 每段訓練多久就 eval + record
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=18000)
    parser.add_argument("--record-steps", type=int, default=18000)
    parser.add_argument("--logdir", type=str, default="./runs_smw")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    video_dir = os.path.join(args.logdir, "videos")
    ckpt_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_env = make_vec_env(args.game, args.state, n_envs=args.n_envs)

    model = CustomPPO(
        VisionBackbonePolicy,
        train_env,
        policy_kwargs=dict(normalize_images=False),
        n_epochs=10,
        n_steps=512,
        batch_size=256,
        learning_rate=1e-4,
        verbose=1,
        gamma=0.99,
        kl_coef=1,
        clip_range=0.5,
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

        ckpt_path = os.path.join(ckpt_dir, f"CustomPPO_step_{trained}.zip")
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


        record_video(
            model,
            args.game,
            args.state,
            video_dir,
            video_len=args.record_steps,
            prefix=f"step_{trained}_mean_{mean_ret:.2f}",
        )

    train_env.close()


if __name__ == "__main__":
    main()
