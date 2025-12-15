import os
import argparse
import retro
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
)
from eval import evaluate_policy, record_video
from wrappers import (
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
