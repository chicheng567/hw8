import os
from pathlib import Path

import numpy as np
import retro
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

from control import ExtraInfoWrapper


def annotate_frame(frame: np.ndarray, step: int, x_pos: int | None, raw_x: int | None) -> np.ndarray:
    """Overlay step/x-position text onto an RGB frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = f"step={step:04d}  rel_x={x_pos if x_pos is not None else 'N/A'}  raw_x={raw_x if raw_x is not None else 'N/A'}"
    font = ImageFont.load_default()
    text_size = draw.textbbox((0, 0), text, font=font)
    padding = 4
    bg_rect = (
        0,
        0,
        text_size[2] + padding * 2,
        text_size[3] + padding * 2,
    )
    draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def read_raw_x_pos(ram: np.ndarray) -> int:
    low = int(ram[0x0094])
    high = int(ram[0x0095])
    return (high << 8) | low


def main():
    out_dir = Path("videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xpos_debug.mp4"

    env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland1", render_mode="rgb_array")
    env = ExtraInfoWrapper(env)

    buttons = list(env.unwrapped.buttons)
    noop = np.zeros((len(buttons),), dtype=np.uint8)
    run_right = noop.copy()
    run_right[buttons.index("RIGHT")] = 1
    run_right[buttons.index("B")] = 1
    run_left = noop.copy()
    run_left[buttons.index("LEFT")] = 1
    run_left[buttons.index("B")] = 1

    obs, info = env.reset()

    writer = imageio.get_writer(out_path, fps=60)
    max_steps = 600
    for step in range(max_steps):
        if step < 30:
            action = noop
        elif step < 210:
            action = run_right
        else:
            action = run_left
        obs, reward, terminated, truncated, info = env.step(action)
        ram = env.unwrapped.get_ram()
        raw_x = read_raw_x_pos(ram)
        frame = annotate_frame(obs, step, info.get("x_pos"), raw_x)
        writer.append_data(frame)
        if terminated or truncated:
            break

    writer.close()
    env.close()

    print(f"Wrote video to {out_path.resolve()}")


if __name__ == "__main__":
    main()
