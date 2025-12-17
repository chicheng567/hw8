#!/usr/bin/env python3
"""Evaluate a submitted policy bundle without touching the working tree."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_MEMBERS = {"custom_policy.py", "wrappers.py"}


def _compute_custom_score(info: dict | None) -> float:
    if not isinstance(info, dict):
        return 0.0
    x_pos = float(info.get("x_pos", 0.0))
    score = float(info.get("score", 0.0))
    coins = float(info.get("coins", info.get("coin", 0.0)))
    return x_pos * 0.01 + score * 0.1 + coins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Path to the student's zip file")
    parser.add_argument("--weights-name", default="policy.zip", help="File name of the checkpoint inside the zip")
    parser.add_argument("--handle", default=None, help="Label to display in results (defaults to submission stem)")
    parser.add_argument("--game", default="SuperMarioWorld-Snes")
    parser.add_argument("--state", default="YoshiIsland1")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--record-video-len", type=int, default=0, help="Number of frames to record (0 disables video)")
    parser.add_argument("--videos-dir", default="videos/submissions", help="Folder for recorded videos")
    parser.add_argument("--device", default="cpu", help="Torch device used when loading the checkpoint")
    parser.add_argument("--output-json", default=None, help="Optional path to write the result summary (JSON)")
    return parser.parse_args()


def _slugify(value: str) -> str:
    import re

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "submission"


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target_path = (dest / member.filename).resolve()
            if not str(target_path).startswith(str(dest)):
                raise ValueError(f"Blocked suspicious path {member.filename}")
        zf.extractall(dest)


def _find_unique(root: Path, target: str) -> Path:
    matches = [p for p in root.rglob(target) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find {target} in {root}")
    if len(matches) > 1:
        raise ValueError(f"Found multiple copies of {target}; ensure only one exists")
    return matches[0]


def _prep_module_dirs(*paths: Path) -> List[str]:
    seen = []
    for path in paths:
        as_str = str(path)
        if as_str not in seen:
            seen.append(as_str)
    return seen


@contextlib.contextmanager
def submission_module_context(extra_paths: Iterable[str]):
    original_path = list(sys.path)
    sys.path = list(extra_paths) + original_path
    preserved = {}
    for name in ("custom_policy", "wrappers", "eval"):
        if name in sys.modules:
            preserved[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        for name in ("custom_policy", "wrappers", "eval"):
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path = original_path


def evaluate_submission(args: argparse.Namespace) -> dict:
    submission_path = Path(args.submission).expanduser().resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission zip not found: {submission_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_root = Path(tmpdir)
        _safe_extract(submission_path, extract_root)
        custom_policy_path = _find_unique(extract_root, "custom_policy.py")
        wrappers_path = _find_unique(extract_root, "wrappers.py")
        weights_path = _find_unique(extract_root, args.weights_name)

        module_dirs = _prep_module_dirs(custom_policy_path.parent, wrappers_path.parent)
        raw_team = (args.handle or submission_path.stem).strip()
        slug = _slugify(raw_team or submission_path.stem)
        timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        with submission_module_context(module_dirs):
            custom_policy = importlib.import_module("custom_policy")
            eval_module = importlib.import_module("eval")
            CustomPPO = getattr(custom_policy, "CustomPPO")

            model = CustomPPO.load(str(weights_path), device=args.device)
            env = eval_module.make_base_env(args.game, args.state)
            scores: list[float] = []
            try:
                for episode_idx in range(args.n_episodes):
                    obs, info = env.reset()
                    cumulative = 0.0
                    steps = 0
                    while steps < args.max_steps:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        cumulative = _compute_custom_score(info)
                        if terminated or truncated:
                            break
                        steps += 1
                    scores.append(cumulative)
            finally:
                env.close()

            mean_ret = float(sum(scores) / len(scores)) if scores else 0.0
            if args.record_video_len > 0:
                video_prefix = f"{slug}_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                eval_module.record_video(
                    model=model,
                    game=args.game,
                    state=args.state,
                    out_dir=args.videos_dir,
                    video_len=args.record_video_len,
                    prefix=video_prefix,
                )

        return {
            "timestamp": timestamp,
            "submission": submission_path.name,
            "team": raw_team or slug,
            "mean_return": mean_ret,
        }


def main():
    args = parse_args()
    result = evaluate_submission(args)
    json_text = json.dumps(result, ensure_ascii=False)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json_text + "\n")
    print(json_text)


if __name__ == "__main__":
    main()
