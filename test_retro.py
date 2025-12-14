import os
import retro
import gymnasium as gym

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from control import DiscreteActionWrapper

GAME = "SuperMarioWorld-Snes"
VIDEO_DIR = "./videos"
TRAIN_STEPS = 5_000
RECORD_STEPS = 1_000

import retro

COMBOS = [
    [],                  # 0: NOOP
    ["RIGHT"],           # 1: 走右
    ["LEFT"],            # 2: 走左（可選）
    ["DOWN"],            # 3: 下蹲
    ["A"],               # 4: 跳
    ["B"],               # 5: 跑
    ["RIGHT", "A"],      # 6: 右 + 跳
    ["RIGHT", "B"],      # 7: 右 + 跑
    ["RIGHT", "A", "B"], # 8: 右 + 跳 + 跑
    ["A","B"],           # 9: 跳 + 跑（原地）
    ["LEFT", "A"],       # 10: 左 + 跳（可選）
    ["LEFT", "B"],       # 11: 左 + 跑（可選）
    ["LEFT", "A", "B"],  # 12: 左 + 跳 + 跑（可選）
]

def make_env(record=False):
    env = retro.make(game=GAME, render_mode="rgb_array")
    env = DiscreteActionWrapper(env, COMBOS)
    if record:
        env = RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            episode_trigger=lambda ep: True,
            name_prefix="eval",
        )
    return env



def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: make_env(record=False)])
    env = VecTransposeImage(env)
    print("Buttons:", env.unwrapped.buttons)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
    )

    print(f"Training PPO for {TRAIN_STEPS} steps...")
    model.learn(total_timesteps=TRAIN_STEPS)

    obs = env.reset()
    for _ in range(RECORD_STEPS):
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(action)
        if dones[0]:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
