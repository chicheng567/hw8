#!/usr/bin/env python3
"""Automatically pull new submissions from Hugging Face Hub and run evaluations."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

DEFAULT_REPO_ID = "SamChen567/smw-submissions"
STATE_FILE = Path("hf_space/auto_judge_state.json")
LEADERBOARD_PATH = Path("hf_space/leaderboard.jsonl")
DEFAULT_VIDEOS_DIR = Path("videos/submissions")
SUBMISSION_PREFIX = "submissions/"
METADATA_FILENAME = "metadata.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("SUBMISSION_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--run-script", default="hf_space/run_local_eval.py", help="Path to run_local_eval.py")
    parser.add_argument("--weights-name", default="policy.zip")
    parser.add_argument("--game", default="SuperMarioWorld-Snes")
    parser.add_argument("--state", default="YoshiIsland1")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=18000)
    parser.add_argument("--record-video-len", type=int, default=4000)
    parser.add_argument("--videos-dir", default=str(DEFAULT_VIDEOS_DIR))
    parser.add_argument("--leaderboard", default=str(LEADERBOARD_PATH))
    parser.add_argument("--state-file", default=str(STATE_FILE))
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds")
    parser.add_argument("--loop", action="store_true", help="Continuously watch for new submissions")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--device", default="cpu", help="Torch device for evaluation")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[auto_judge] {message}", flush=True)


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    return set(data)


def save_state(path: Path, processed: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(processed), indent=2))


def discover_submissions(repo_id: str) -> list[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    submissions = [f for f in files if f.startswith(SUBMISSION_PREFIX) and f.endswith(".zip")]
    submissions.sort()
    return submissions


def download_submission(repo_id: str, remote_path: str, target_dir: Path) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=remote_path,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def load_leaderboard_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _read_last_entry(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _load_metadata_entries_remote(repo_id: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=METADATA_FILENAME,
                local_dir=tmpdir,
                local_dir_use_symlinks=False,
            )
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return []
            raise
        entries: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return entries


def _build_repo_map(entries: list[dict]) -> dict[str, dict]:
    repo_map: dict[str, dict] = {}
    for item in entries:
        repo_path = item.get("repo_path")
        if repo_path:
            repo_map[repo_path] = item
    return repo_map


def _save_metadata_entries(api: HfApi, repo_id: str, entries: list[dict]) -> None:
    entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / METADATA_FILENAME
        with open(out_path, "w", encoding="utf-8") as f:
            for record in entries:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=METADATA_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update metadata with scores",
        )


def _update_metadata_entry(entries: list[dict], repo_path: str, result: dict) -> dict:
    team = result.get("team")
    score = result.get("mean_return")
    evaluated_at = result.get("timestamp")
    for item in entries:
        if item.get("repo_path") == repo_path:
            if team:
                item["team"] = team
                item["handle"] = item.get("handle") or team
            item["score"] = score
            item["evaluated_at"] = evaluated_at
            return item
    entry = {
        "timestamp": evaluated_at,
        "repo_path": repo_path,
        "handle": team,
        "team": team,
        "notes": "",
        "score": score,
        "evaluated_at": evaluated_at,
    }
    entries.append(entry)
    return entry


def _guess_handle_from_path(remote_path: str) -> str:
    stem = Path(remote_path).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return stem


def _normalize_team(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def _sanitize_entry(entry: dict) -> dict:
    team = entry.get("team") or entry.get("handle") or ""
    sanitized = {
        "timestamp": entry.get("timestamp", ""),
        "submission": entry.get("submission", ""),
        "team": team,
        "mean_return": float(entry.get("mean_return", 0.0)),
    }
    return sanitized


def _simplify_entries(entries: list[dict]) -> list[dict]:
    return [_sanitize_entry(item) for item in entries]


def run_evaluation(args: argparse.Namespace, submission_path: Path, team_name: str | None = None) -> dict | None:
    leaderboard_path = Path(args.leaderboard)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    existing_entries = _simplify_entries(load_leaderboard_entries(leaderboard_path))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmpfile:
        result_path = Path(tmpfile.name)
    try:
        cmd = [
            sys.executable,
            args.run_script,
            "--submission",
            str(submission_path),
            "--weights-name",
            args.weights_name,
            "--game",
            args.game,
            "--state",
            args.state,
            "--n-episodes",
            str(args.n_episodes),
            "--max-steps",
            str(args.max_steps),
            "--record-video-len",
            str(args.record_video_len),
            "--videos-dir",
            str(videos_dir),
            "--device",
            args.device,
            "--output-json",
            str(result_path),
        ]
        if team_name:
            cmd.extend(["--handle", team_name])
        log(f"Running evaluation: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"Evaluation failed (exit {result.returncode}): {result.stderr}\n{result.stdout}")
            return None
        try:
            raw_text = result_path.read_text(encoding="utf-8").strip()
            new_entry = json.loads(raw_text)
        except (json.JSONDecodeError, OSError) as exc:
            log(f"Could not parse evaluation JSON ({exc}); check logs.")
            return None
    finally:
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass

    sanitized_new = _sanitize_entry(new_entry)
    new_score = float(sanitized_new.get("mean_return", 0.0))
    team = sanitized_new.get("team", "")
    team_key = _normalize_team(team)
    current_entry = next((item for item in existing_entries if _normalize_team(item.get("team", "")) == team_key), None)
    current_score = float(current_entry.get("mean_return", 0.0)) if current_entry else None

    if current_entry is not None and current_score >= new_score:
        log(f"No improvement for {team or 'unknown team'}: {new_score:.3f} <= {current_score:.3f}")
        return current_entry

    updated_entries = [item for item in existing_entries if _normalize_team(item.get("team", "")) != team_key]
    updated_entries.append(sanitized_new)
    updated_entries.sort(key=lambda item: item.get("mean_return", 0.0), reverse=True)
    leaderboard_path.write_text("\n".join(json.dumps(item) for item in updated_entries) + "\n", encoding="utf-8")
    return sanitized_new


def upload_leaderboard(api: HfApi, repo_id: str, leaderboard_path: Path) -> None:
    if leaderboard_path.exists():
        api.upload_file(
            path_or_fileobj=str(leaderboard_path),
            path_in_repo="leaderboard.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update leaderboard",
        )
        log("Uploaded leaderboard.jsonl")


def process_once(args: argparse.Namespace, api: HfApi) -> None:
    processed = load_state(Path(args.state_file))
    submissions = discover_submissions(args.repo_id)
    new_items = [item for item in submissions if item not in processed]
    if not new_items:
        log("No new submissions found")
        return

    metadata_entries = _load_metadata_entries_remote(args.repo_id)
    repo_entry_map = _build_repo_map(metadata_entries)
    for remote_path in tqdm(new_items, desc="Evaluating", unit="zip"):
        log(f"Processing {remote_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_zip = download_submission(args.repo_id, remote_path, Path(tmpdir))
            mapped_entry = repo_entry_map.get(remote_path)
            team_name = (mapped_entry.get("team") if mapped_entry else None) or (mapped_entry.get("handle") if mapped_entry else None) or _guess_handle_from_path(remote_path)
            result = run_evaluation(args, local_zip, team_name=team_name)
            if result is None:
                log(f"Skipping upload for {remote_path} due to evaluation failure")
                continue
            processed.add(remote_path)
            save_state(Path(args.state_file), processed)
            upload_leaderboard(api, args.repo_id, Path(args.leaderboard))
            updated_entry = _update_metadata_entry(metadata_entries, remote_path, result)
            repo_entry_map[remote_path] = updated_entry
            _save_metadata_entries(api, args.repo_id, metadata_entries)
            log(f"Finished {remote_path}: mean_return={result.get('mean_return')}")


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit("HF_TOKEN is required (export it or pass --token)")
    api = HfApi(token=args.token)

    if args.loop:
        log("Entering watch mode")
        while True:
            try:
                process_once(args, api)
            except Exception as exc:  # noqa: BLE001
                log(f"Error during processing: {exc}")
            time.sleep(max(1, args.interval))
    else:
        process_once(args, api)


if __name__ == "__main__":
    main()
