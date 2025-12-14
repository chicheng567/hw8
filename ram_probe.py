import argparse
import numpy as np
import retro


def make_action(buttons, pressed):
    action = np.zeros(len(buttons), dtype=np.uint8)
    for name in pressed:
        if name not in buttons:
            raise ValueError(f"Button '{name}' not available. Options: {buttons}")
        action[buttons.index(name)] = 1
    return action


def capture_ram(env, action, settle_steps, hold_steps):
    obs, info = env.reset()
    for _ in range(settle_steps):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape[0], dtype=np.uint8))
        if terminated or truncated:
            obs, info = env.reset()
    for _ in range(hold_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    return env.get_ram().copy()


def main():
    parser = argparse.ArgumentParser(description="Probe RAM addresses affected by horizontal movement.")
    parser.add_argument("--game", default="SuperMarioWorld-Snes")
    parser.add_argument("--state", default="Start")
    parser.add_argument("--settle-steps", type=int, default=120, help="Frames to play NOOP after reset so the level is ready.")
    parser.add_argument("--hold-steps", type=int, default=60, help="Frames to hold each action.")
    parser.add_argument("--top-k", type=int, default=20, help="How many candidate addresses to print.")
    parser.add_argument("--right-action", nargs="*", default=["RIGHT", "B"], help="Buttons to hold for moving right.")
    parser.add_argument("--left-action", nargs="*", default=["LEFT", "B"], help="Buttons to hold for moving left.")
    args = parser.parse_args()

    env = retro.make(game=args.game, state=args.state, render_mode="rgb_array")
    buttons = list(env.unwrapped.buttons)
    print("Buttons:", buttons)

    noop = np.zeros(len(buttons), dtype=np.uint8)
    right = make_action(buttons, args.right_action)
    left = make_action(buttons, args.left_action)

    base_ram = capture_ram(env, noop, args.settle_steps, args.hold_steps)
    right_ram = capture_ram(env, right, args.settle_steps, args.hold_steps)
    left_ram = capture_ram(env, left, args.settle_steps, args.hold_steps)

    env.close()

    delta_right = right_ram - base_ram
    delta_left = left_ram - base_ram

    candidates = []
    for idx in range(len(delta_right)):
        dr = int(delta_right[idx])
        dl = int(delta_left[idx])
        if dr == 0 and dl == 0:
            continue
        # look for addresses that move in opposite directions when walking
        if dr == -dl and dr != 0:
            score = abs(dr)
        else:
            score = abs(dr) + abs(dl)
        candidates.append((score, idx, dr, dl, int(base_ram[idx]), int(right_ram[idx]), int(left_ram[idx])))

    candidates.sort(reverse=True)

    print(f"Top {args.top_k} candidate RAM addresses (index, delta_right, delta_left, base, right, left):")
    for score, idx, dr, dl, base_val, right_val, left_val in candidates[: args.top_k]:
        print(f"0x{idx:03X} ({idx:4d}): ΔR={dr:4d} ΔL={dl:4d} base={base_val:4d} right={right_val:4d} left={left_val:4d}")


if __name__ == "__main__":
    main()
