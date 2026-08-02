"""Run an LLM agent (via OpenRouter) on Puzzle A, ARC-AGI-3 style.

The agent is told nothing about the game: it sees a 6x6 frame of colors, a score,
and an abstract action vocabulary (ACTION1-4, RESET). Each turn it returns JSON
with its reasoning, updated persistent notes (its only long-term memory), and one
action. Every emitted action counts toward the budget, matching how human
keypresses were logged.

Scoring follows ARC-AGI-3: per-level RHAE = min(1, (human_median / agent_actions)^2),
game score = mean over all levels (uncompleted levels score 0).

Usage:
    python3 run_agent.py [--model openai/gpt-4o-mini] [--max-actions 400] [--runs 1]
"""

import argparse
import json
import os
import re
import time
import urllib.request

from puzzle_a_env import PuzzleAEnv, ACTIONS, LEVELS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """\
You are playing an unknown turn-based game. You are given no instructions: you must \
discover the controls, the rules, and the objective purely by experimenting.

Each turn you see:
- FRAME: a 6x6 grid of colors (row 0 is the top row; each row is listed left to right)
- SCORE: starts at 0; it increases when you accomplish something the game considers a win
- your persistent NOTES from previous turns (your only long-term memory)
- a short history of your recent actions and whether the frame changed

Available actions: ACTION1, ACTION2, ACTION3, ACTION4, RESET. Their effects are \
unknown and it is up to you to figure out what each one does. When the score \
increases, the game advances to a new configuration (a new level of the same game).

Strategy advice: experiment systematically, form hypotheses about what the actions \
do and what causes the score to increase, record what you learn in your notes, and \
exploit what you learn to increase the score in as few actions as possible.

Respond with ONLY a JSON object, no other text:
{"reasoning": "<brief thinking about this turn>",
 "notes": "<your updated persistent notes: rules discovered, hypotheses, current plan. Keep under 250 words.>",
 "action": "<one of ACTION1|ACTION2|ACTION3|ACTION4|RESET>"}
"""

SYSTEM_PROMPT_SCRATCHPAD = """\
You are playing an unknown turn-based game. You are given no instructions: you must \
discover the controls, the rules, and the objective purely by experimenting.

Each turn you see:
- FRAME: a 6x6 grid of colors (row 0 is the top row; each row is listed left to right)
- SCORE: starts at 0; it increases when you accomplish something the game considers a win
- DISCOVERIES: an append-only log of facts you have confirmed. Entries are permanent: \
they are never edited or deleted, and the full log is shown to you every turn. This is \
your reliable long-term memory.
- SCRATCHPAD: your rewriteable working memory for current hypotheses and plans
- a short history of your recent actions and whether the frame changed

Available actions: ACTION1, ACTION2, ACTION3, ACTION4, RESET. Their effects are \
unknown and it is up to you to figure out what each one does. When the score \
increases, the game advances to a new configuration (a new level of the same game).

Strategy advice: experiment systematically. When you CONFIRM a fact through repeated \
observation (e.g. what an action does, what a cell type does, what made the score \
increase), add it to DISCOVERIES so it can never be forgotten. Only add facts you are \
confident in, and never re-add a fact already in the log. Use the SCRATCHPAD for \
tentative hypotheses and your current plan. Exploit what you learn to increase the \
score in as few actions as possible.

Respond with ONLY a JSON object, no other text:
{"reasoning": "<brief thinking about this turn>",
 "discoveries": ["<zero or more NEW confirmed facts, each under 25 words>"],
 "scratchpad": "<rewriteable working memory: hypotheses and current plan. Keep under 200 words.>",
 "action": "<one of ACTION1|ACTION2|ACTION3|ACTION4|RESET>"}
"""


def call_openrouter(api_key, model, messages, max_retries=4):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 24000,
        "response_format": {"type": "json_object"},
    }).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(API_URL, data=body, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            if "choices" not in data:
                raise RuntimeError(f"Bad response: {str(data)[:300]}")
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # Some reasoning models (e.g. GLM) sometimes return the answer only in
            # the reasoning channel with empty content — fall back to it.
            if not content.strip():
                content = msg.get("reasoning") or ""
            cost = (data.get("usage") or {}).get("cost", 0) or 0
            return content, cost
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_response(content):
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    action = str(obj.get("action", "")).strip().upper()
    if action not in ACTIONS:
        return None
    discoveries = obj.get("discoveries", [])
    if not isinstance(discoveries, list):
        discoveries = [str(discoveries)]
    return {
        "reasoning": obj.get("reasoning", ""),
        "notes": obj.get("notes", "") or obj.get("scratchpad", ""),
        "discoveries": [str(x)[:300] for x in discoveries if x],
        "action": action,
    }


def salvage_response(content):
    """Last resort for malformed JSON: regex out the action (and notes if possible)."""
    if not content:
        return None
    actions = re.findall(r'"action"\s*:\s*"(ACTION[1-4]|RESET)"', content)
    if not actions:
        return None
    m = re.search(r'"notes"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
    notes = m.group(1).replace('\\"', '"').replace("\\n", "\n") if m else ""
    return {"reasoning": "(salvaged from malformed response)", "notes": notes,
            "action": actions[-1]}


def frame_text(frame):
    return "\n".join(f"row {y}: " + " ".join(row) for y, row in enumerate(frame))


def build_user_message(obs, notes, history, actions_used, max_actions, discoveries=None):
    recent = history[-12:]
    hist_lines = []
    for h in recent:
        hist_lines.append(f"  turn {h['turn']}: {h['action']} -> "
                          + ("frame changed" if h["changed"] else "frame UNCHANGED")
                          + (f", SCORE increased to {h['score']}" if h.get("scored") else ""))
    parts = []
    if discoveries is not None:
        disc = "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(discoveries)) or "  (none yet)"
        parts.append(f"DISCOVERIES (append-only, permanent):\n{disc}\n")
        parts.append(f"SCRATCHPAD (from your previous turn):\n{notes or '(empty)'}\n")
    else:
        parts.append(f"NOTES (from your previous turns):\n{notes or '(none yet)'}\n")
    parts.append("RECENT ACTIONS:\n" + ("\n".join(hist_lines) or "  (none yet)") + "\n")
    parts.append(f"SCORE: {obs['score']}\n"
                 f"ACTIONS USED: {actions_used} / {max_actions}\n")
    parts.append(f"FRAME:\n{frame_text(obs['frame'])}\n")
    parts.append("Choose your next action. Respond with JSON only.")
    return "\n".join(parts)


def rhae_scorecard(env, baseline):
    per_level = {}
    total = 0.0
    for i in range(len(LEVELS)):
        lvl = str(i + 1)
        completed = env.score > i
        actions = env.actions_per_level[i]
        entry = {"completed": completed, "agent_actions": actions}
        if completed and lvl in baseline["per_level"]:
            human = baseline["per_level"][lvl]["median_actions"]
            entry["human_median_actions"] = human
            entry["rhae"] = min(1.0, (human / actions) ** 2) if actions else 1.0
        else:
            entry["rhae"] = 0.0
        total += entry["rhae"]
        per_level[lvl] = entry
    return {"per_level": per_level, "game_score": total / len(LEVELS)}


def checkpoint_path(model, run_id, memory="notes"):
    tag = "" if memory == "notes" else f"_{memory}"
    return os.path.join(RESULTS_DIR, f"checkpoint_{model.replace('/', '_')}{tag}_run{run_id}.json")


def save_checkpoint(path, state):
    """Atomic write so a crash mid-write can't corrupt the checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def run_episode(api_key, model, max_actions, run_id, verbose=True, resume=None,
                memory="notes"):
    env = PuzzleAEnv()
    obs = env.reset()
    with open(os.path.join(HERE, "human_baseline.json")) as f:
        baseline = json.load(f)

    notes = ""
    discoveries = []
    history = []
    transcript = []
    total_cost = 0.0
    turn = 0
    parse_failures = 0
    consecutive_failures = 0

    if resume:
        with open(resume) as f:
            ck = json.load(f)
        notes, history, transcript = ck["notes"], ck["history"], ck["transcript"]
        discoveries = ck.get("discoveries", [])
        total_cost, turn, parse_failures = ck["total_cost"], ck["turn"], ck["parse_failures"]
        # The env is deterministic: rebuild state by replaying the transcript's actions.
        for t in transcript:
            obs = env.step(t["action"])
        print(f"Resumed at turn {turn}, score={obs['score']}, cost so far ${total_cost:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ck_path = checkpoint_path(model, run_id, memory)
    system_prompt = SYSTEM_PROMPT_SCRATCHPAD if memory == "scratchpad" else SYSTEM_PROMPT

    while turn < max_actions and not env.game_over:
        turn += 1
        user_msg = build_user_message(obs, notes, history, turn - 1, max_actions,
                                      discoveries if memory == "scratchpad" else None)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}]
        if consecutive_failures:
            messages.append({"role": "user", "content":
                "Your previous response was not valid: it echoed the template "
                "placeholders instead of real values. Reply with concrete JSON, "
                'for example: {"reasoning": "testing movement", "notes": "so far...", '
                '"action": "ACTION2"}. The action field must be exactly one of '
                "ACTION1, ACTION2, ACTION3, ACTION4, RESET."})
        content, cost = call_openrouter(api_key, model, messages)
        total_cost += cost
        parsed = parse_response(content) or salvage_response(content)
        if parsed is None:
            parse_failures += 1
            consecutive_failures += 1
            with open(os.path.join(RESULTS_DIR, "parse_failures.log"), "a") as f:
                f.write(f"{model} run{run_id} turn{turn}: {repr(content)[:1000]}\n---\n")
            if consecutive_failures >= 8:
                raise RuntimeError("8 consecutive unparseable responses")
            turn -= 1
            continue
        consecutive_failures = 0

        prev_frame = obs["frame"]
        prev_score = obs["score"]
        obs = env.step(parsed["action"])
        changed = obs["frame"] != prev_frame
        scored = obs["score"] > prev_score
        notes = parsed["notes"][:2000]
        for d in parsed.get("discoveries", []):
            if d not in discoveries:
                discoveries.append(d)
        history.append({"turn": turn, "action": parsed["action"],
                        "changed": changed, "scored": scored, "score": obs["score"]})
        transcript.append({
            "turn": turn, "action": parsed["action"], "reasoning": parsed["reasoning"],
            "notes": notes, "new_discoveries": parsed.get("discoveries", []),
            "frame_after": obs["frame"], "score": obs["score"],
            "level": env.level_index + 1, "frame_changed": changed,
        })
        save_checkpoint(ck_path, {
            "model": model, "max_actions": max_actions, "run_id": run_id,
            "notes": notes, "discoveries": discoveries, "history": history,
            "transcript": transcript,
            "total_cost": total_cost, "turn": turn, "parse_failures": parse_failures,
        })
        if verbose and (scored or turn % 10 == 0):
            print(f"  turn {turn}: score={obs['score']} level={env.level_index + 1} "
                  f"cost=${total_cost:.4f}" + ("  ** LEVEL SOLVED **" if scored else ""),
                  flush=True)

    scorecard = rhae_scorecard(env, baseline)
    result = {
        "model": model, "memory": memory, "discoveries": discoveries,
        "max_actions": max_actions, "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "levels_completed": env.score, "total_actions": turn,
        "actions_per_level": env.actions_per_level,
        "scorecard": scorecard, "cost_usd": round(total_cost, 4),
        "parse_failures": parse_failures,
        "transcript": transcript,
    }
    safe_model = model.replace("/", "_")
    out = os.path.join(RESULTS_DIR, f"{result['timestamp'].replace(':', '')}_{safe_model}_run{run_id}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if os.path.exists(ck_path):
        os.remove(ck_path)
    return result, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--max-actions", type=int, default=400)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--resume", help="Path to a checkpoint file to resume from")
    ap.add_argument("--memory", choices=["notes", "scratchpad"], default="notes",
                    help="notes: rewriteable notes blob; scratchpad: append-only "
                         "discoveries log + rewriteable scratchpad")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_file = os.path.join(HERE, "..", ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or put it in ../.env")

    for run_id in range(1, args.runs + 1):
        print(f"\n=== Run {run_id}/{args.runs}: {args.model}, budget {args.max_actions} actions ===", flush=True)
        result, out = run_episode(api_key, args.model, args.max_actions, run_id,
                                  resume=args.resume if run_id == 1 else None,
                                  memory=args.memory)
        sc = result["scorecard"]
        print(f"\nLevels completed: {result['levels_completed']}/5 in {result['total_actions']} actions")
        for lvl, e in sc["per_level"].items():
            status = "solved" if e["completed"] else "not solved"
            print(f"  Level {lvl}: {status}, {e['agent_actions']} actions, RHAE={e['rhae']:.3f}")
        print(f"Game score (ARC-style): {sc['game_score'] * 100:.1f}%")
        print(f"API cost: ${result['cost_usd']:.4f}")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
