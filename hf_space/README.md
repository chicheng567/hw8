# Evaluation utilities

This folder contains helper scripts for grading submissions stored in the `SamChen567/smw-submissions` dataset repository.

## run_local_eval.py
Run this script when you want to evaluate one submission manually:
```bash
python hf_space/run_local_eval.py \
  --submission submissions_cache/submissions/2025-05-30_team.zip \
  --weights-name policy.zip \
  --game SuperMarioWorld-Snes \
  --state YoshiIsland1 \
  --n-episodes 3 \
  --max-steps 2500 \
  --record-video-len 1200 \
  --videos-dir videos/submissions
```
It loads the student's `custom_policy.py`/`wrappers.py`, evaluates `policy.zip`, appends the JSON result to `leaderboard.jsonl`, and (optionally) records a video.

## auto_judge.py
`auto_judge.py` fully automates the process: it lists new uploads inside `SamChen567/smw-submissions`, downloads them, runs `run_local_eval.py`, and pushes the updated leaderboard back to the dataset.

Basic usage:
```bash
export HF_TOKEN=YOUR_HF_TOKEN_HERE
python hf_space/auto_judge.py --loop --interval 300 \
  --repo-id SamChen567/smw-submissions \
  --videos-dir videos/submissions
```
Key options:
- `--loop` keeps the watcher running; omit it to process outstanding submissions once.
- `--interval` controls how often the dataset is scanned.
- `--leaderboard` and `--state-file` let you relocate `leaderboard.jsonl` and the processed-cache.
- `--game`, `--state`, `--n-episodes`, etc. are passed through to `run_local_eval.py`.

The script records which zips have been graded in `hf_space/auto_judge_state.json`. If you ever need to regrade everything, delete that file and rerun the watcher.
