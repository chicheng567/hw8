import numpy as np
import gymnasium as gym

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    change action space from MultiBinary to Discrete with predefined button combos
    """
    def __init__(self, env, combos):
        super().__init__(env)


        if not hasattr(env.unwrapped, "buttons"):
            raise ValueError("unsupported env, must have 'buttons' attribute")

        self.buttons = list(env.unwrapped.buttons)  # e.g. ['B','Y','SELECT',...]
        self.button_to_idx = {b: i for i, b in enumerate(self.buttons)}

        # Get combos
        self.combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))

        self._mapped = []
        n = env.action_space.n  # MultiBinary(n)
        for keys in combos:
            a = np.zeros(n, dtype=np.int8)
            for k in keys:
                if k not in self.button_to_idx:
                    raise ValueError(f"unsupported buttons in this env.buttons: {self.buttons}")
                a[self.button_to_idx[k]] = 1
            self._mapped.append(a)

    def action(self, act):
        return self._mapped[int(act)].copy()
    

class LifeTerminationWrapper(gym.Wrapper):
    def __init__(self, env, death_penalty=0):
        super().__init__(env)
        self.death_penalty = float(death_penalty)
        self._prev_lives = None

    def _get_lives(self, info):
        if not isinstance(info, dict):
            return None
        if "lives" in info:
            return int(info["lives"])
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lives = self._get_lives(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        lives = self._get_lives(info)

        died = False
        if lives is not None and self._prev_lives is not None:
            if lives < self._prev_lives:
                died = True
        self._prev_lives = lives

        if died:
            reward = float(reward) + self.death_penalty
            terminated = True
            if isinstance(info, dict):
                info = dict(info)
                info["death"] = True

        return obs, reward, terminated, truncated, info


class ExtraInfoWrapper(gym.Wrapper):
    """
    Attach extra RAM-derived signals (HUD timer, x-position) to info.
    """

    TIMER_HUNDREDS = 0x0F31
    TIMER_TENS = 0x0F32
    TIMER_ONES = 0x0F33
    X_POS_HIGH = 0x0094
    X_POS_LOW = 0x0095

    def __init__(self, env):
        super().__init__(env)
        self._episode_start_x = None

    def _get_ram(self):
        base_env = self.env.unwrapped
        if not hasattr(base_env, "get_ram"):
            return None
        return base_env.get_ram()

    def _read_time_left(self, ram):
        if ram is None:
            return None
        hundreds = int(ram[self.TIMER_HUNDREDS]) & 0x0F
        tens = int(ram[self.TIMER_TENS]) & 0x0F
        ones = int(ram[self.TIMER_ONES]) & 0x0F
        return hundreds * 100 + tens * 10 + ones

    def _read_x_pos(self, ram):
        if ram is None:
            return None
        high = int(ram[self.X_POS_HIGH])
        low = int(ram[self.X_POS_LOW])
        return (high << 8) | low

    def _inject_extra(self, info):
        ram = self._get_ram()
        time_left = self._read_time_left(ram)
        x_pos = self._read_x_pos(ram)
        if time_left is None and x_pos is None:
            return info
        if not isinstance(info, dict):
            info = {}
        # copy to avoid mutating shared dict instances
        info = dict(info)
        if time_left is not None:
            info["time_left"] = time_left
        if x_pos is not None:
            if self._episode_start_x is None:
                self._episode_start_x = x_pos
            info["x_pos"] = max(0, x_pos - self._episode_start_x)
        return info

    def reset(self, **kwargs):
        self._episode_start_x = None
        obs, info = self.env.reset(**kwargs)
        info = self._inject_extra(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._inject_extra(info)
        return obs, reward, terminated, truncated, info


class RewardOverrideWrapper(gym.Wrapper):
    """
    Replace environment reward with custom shaping
    """

    def __init__(
        self,
        env,
        progress_scale: float = 5e-4,
        time_penalty: float = -0.5,
        score_scale: float = 1.0,
        death_penalty: float = -50.0,
        timeout_penalty: float = -10.0,
        win_reward: float = 400.0,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.time_penalty = time_penalty
        self.score_scale = score_scale
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        self.win_reward = win_reward
        self._best_x = None
        self._prev_time = None
        self._prev_score = None

    def _reset_trackers(self, info):
        self._best_x = info.get("x_pos", 0)
        self._prev_time = info.get("time_left")
        self._prev_score = info.get("score", 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        self._reset_trackers(info)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        reward = 0.0

        # Positive reward for surpassing previous best x_pos
        x_pos = info.get("x_pos")
        if x_pos is not None:
            if self._best_x is None:
                self._best_x = x_pos
            delta = x_pos - self._best_x
            if delta > 0:
                reward += delta * self.progress_scale
                self._best_x = x_pos

        # Negative reward as time counts down
        time_left = info.get("time_left")
        if time_left is not None:
            if self._prev_time is None:
                self._prev_time = time_left
            else:
                delta_time = self._prev_time - time_left
                if delta_time > 0:
                    reward += self.time_penalty * delta_time
                self._prev_time = time_left

        # Reward for score increments
        score = info.get("score")
        if score is not None:
            if self._prev_score is None:
                self._prev_score = score
            else:
                score_delta = score - self._prev_score
                if score_delta > 0:
                    reward += score_delta * self.score_scale
                self._prev_score = score

        if terminated or truncated:
            if info.get("death"):
                reward += self.death_penalty
            elif time_left is not None and time_left <= 0:
                reward += self.timeout_penalty
            else:
                reward += self.win_reward
            self._reset_trackers(info)

        return obs, reward, terminated, truncated, info


class InfoLogger(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if reward > 0 or terminated:
            print(info)
        return obs, reward, terminated, truncated, info

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
    ["LEFT", "A"],       # 10: 左 + 跳
    ["LEFT", "B"],       # 11: 左 + 跑
    ["LEFT", "A", "B"],  # 12: 左 + 跳 + 跑
]
