"""Validate the Puzzle A Python port against the human logs in Combined Data/,
and compute per-level human action baselines for RHAE scoring.

Replays each participant's logged keystream level-by-level through PuzzleAEnv and
checks, before every move, that the simulated positionBefore / colorBefore /
letter matrix match the log. Restarts ('r') were not logged by game.html, so on a
state mismatch we check whether the log matches the level's initial state and
resync (counted as a detected restart), otherwise it's a real divergence.

Outputs llm_eval/human_baseline.json:
  {"per_level": {"1": {"n": ..., "median_actions": ..., "counts": [...]}, ...}}

A level counts as completed by a participant if a later-level movement exists
(levels 1-4) or if the replay reaches the win condition (level 5).
"""

import glob
import json
import os
import statistics
from collections import defaultdict

from puzzle_a_env import PuzzleAEnv, KEY_TO_ACTION, LEVELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Combined Data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "human_baseline.json")


def replay_level(env, level_idx, moves):
    """Replay one level's movements. Returns (mismatches, restarts, won)."""
    env.level_index = level_idx
    env._load_level(level_idx)
    env.game_over = False
    start_pos = list(LEVELS[level_idx]["avatar"])
    start_color = LEVELS[level_idx]["initialAvatarColor"]
    mismatches = 0
    restarts = 0
    won = False
    for mv in moves:
        logged_pos = [mv["positionBefore"]["x"], mv["positionBefore"]["y"]]
        logged_color = mv["colorBefore"]
        if logged_pos != env.avatar or logged_color != env.avatar_color:
            if logged_pos == start_pos and logged_color == start_color:
                env._load_level(level_idx)  # undetected 'r' press
                restarts += 1
            else:
                mismatches += 1
                # resync position so one error doesn't cascade
                env.avatar = logged_pos
                env.avatar_color = logged_color
        if mv.get("gameStateBefore") and env.letter_matrix() != mv["gameStateBefore"]:
            mismatches += 1
        prev_score = env.score
        env.step(KEY_TO_ACTION[mv["key"]])
        if env.score > prev_score or env.game_over:
            won = True
            break
    return mismatches, restarts, won


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_gA_gamestate_*.json")))
    print(f"Found {len(files)} Puzzle A logs\n")
    level_counts = defaultdict(list)
    total_mm = total_rs = 0

    for path in files:
        pid = os.path.basename(path).split("_")[0]
        with open(path) as f:
            data = json.load(f)
        by_level = defaultdict(list)
        for mv in data.get("movements", []):
            by_level[mv["level"]].append(mv)

        env = PuzzleAEnv()
        parts = []
        for lvl in range(1, 6):
            moves = by_level.get(lvl)
            if not moves:
                continue
            mm, rs, won = replay_level(env, lvl - 1, moves)
            total_mm += mm
            total_rs += rs
            completed = won or (lvl < 5 and by_level.get(lvl + 1))
            if completed:
                level_counts[lvl].append(len(moves))
            parts.append(f"L{lvl}:{len(moves)}{'✓' if completed else '✗'}"
                         + (f" mm={mm}" if mm else "") + (f" r={rs}" if rs else ""))
        print(f"{pid}: " + "  ".join(parts))

    print(f"\nTotal state mismatches across all replays: {total_mm}")
    print(f"Total detected (unlogged) restarts: {total_rs}")

    baseline = {"per_level": {}}
    print("\nHuman baseline (actions per completed level):")
    for lvl in range(1, 6):
        counts = sorted(level_counts[lvl])
        if counts:
            med = statistics.median(counts)
            baseline["per_level"][str(lvl)] = {
                "n": len(counts), "median_actions": med,
                "min": counts[0], "max": counts[-1], "counts": counts,
            }
            print(f"  Level {lvl}: n={len(counts)} median={med} min={counts[0]} max={counts[-1]}")

    with open(OUT_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
