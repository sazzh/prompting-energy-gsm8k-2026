"""
Runs the full experiment (3 trials, 600 requests) against an already-running llama-server.
Process: wait for server -> save /props -> warm-up -> for each trial: idle window, then requests.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "Final Prompts"
OUTPUT_DIR = PROJECT_ROOT / "Experiment"
LLAMA_BASE = "http://127.0.0.1:9931"

SETTINGS = {
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "min_p": 0,
    "max_tokens": 1024,
    "repeat_penalty": 1.0,
    "seed": 325,
    "cache_prompt": False,
}

IDLE_SECONDS = 300  # idle baseline before each trials
REQUEST_GAP_SECONDS = 2  # pause between requests
WARMUP_NUMBER = 8  # first 8 spare prompts = 2 questions x 4 techniques

FIELDS = [
    "phase",
    "trial",
    "position",
    "request_id",
    "qid",
    "technique",
    "start",
    "end",
    "prompt_tokens",
    "completion_tokens",
    "cache_n",
    "finish_reason",
    "prompt_ms",
    "predicted_ms",
    "flag",
    "output",
]

session = requests.Session()


def send(messages: list) -> dict:
    body = {
        "messages": messages,
        **SETTINGS,
    }
    start = now()
    res = session.post(f"{LLAMA_BASE}/v1/chat/completions", json=body, timeout=600)
    end = now()
    res.raise_for_status()
    data = res.json()
    return {
        "start": start,
        "end": end,
        "output": data["choices"][0]["message"]["content"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "cache_n": data["timings"]["cache_n"],
        "prompt_ms": data["timings"]["prompt_ms"],
        "predicted_ms": data["timings"]["predicted_ms"],
    }


def wait_for_server() -> None:
    while True:
        try:
            res = session.get(f"{LLAMA_BASE}/health", timeout=5)
            if res.status_code == 200:
                print("Server is ready.")
                return
        except requests.exceptions.RequestException:
            pass
        print("Waiting for the server...")
        time.sleep(5)


def save_model_properties() -> None:
    res = session.get(f"{LLAMA_BASE}/props", timeout=10)
    res.raise_for_status()
    file_name = (
        f"llama_model_properties_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(OUTPUT_DIR / file_name, "w", encoding="utf-8") as f:
        json.dump(res.json(), f, indent=4)
    print("Model properties saved")


def now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


if __name__ == "__main__":
    with open(DATA_DIR / "prompts_spare.json", encoding="utf-8") as f:
        spare = json.load(f)
    with open(DATA_DIR / "prompts_main.json", encoding="utf-8") as f:
        prompts = json.load(f)
    with open(DATA_DIR / "schedule.csv", encoding="utf-8") as f:
        schedule = [
            (int(r["trial"]), int(r["position"]), r["request_id"])
            for r in csv.DictReader(f)
        ]
    out = OUTPUT_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    first_tokens = {}

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        def log(row):
            w.writerow(row)
            f.flush()

        wait_for_server()
        save_model_properties()

        print(f"Warm-up: {WARMUP_NUMBER} spare requests")
        for rid in list(spare)[:WARMUP_NUMBER]:
            p = spare[rid]
            row = {
                "phase": "warmup",
                "request_id": rid,
                "qid": p["qid"],
                "technique": p["technique"],
            }
            row.update(send(p["messages"]))
            log(row)
            time.sleep(REQUEST_GAP_SECONDS)

        for t in sorted({s[0] for s in schedule}):
            todo = [s for s in schedule if s[0] == t]

            print(f"Trial {t}: idle for {IDLE_SECONDS} s")
            start = now()
            time.sleep(IDLE_SECONDS)
            log({"phase": "idle", "trial": t, "start": start, "end": now()})

            for i, (_, pos, rid) in enumerate(todo, start=1):
                p = prompts[rid]
                row = {
                    "phase": "run",
                    "trial": t,
                    "position": pos,
                    "request_id": rid,
                    "qid": p["qid"],
                    "technique": p["technique"],
                }
                row.update(send(p["messages"]))
                flags = []
                if row["finish_reason"] == "length":
                    flags.append("length")
                if row["cache_n"] != 0:
                    flags.append("cache")
                if (
                    first_tokens.setdefault(rid, row["prompt_tokens"])
                    != row["prompt_tokens"]
                ):
                    flags.append("prompt_tokens_changed")
                row["flag"] = ";".join(flags)
                log(row)
                print(
                    f"trial {t}  {i}/{len(todo)}  {rid}  {row['completion_tokens']} tok  {row['flag']}"
                )
                time.sleep(REQUEST_GAP_SECONDS)

        print(f"Final idle for {IDLE_SECONDS} s")
        start = now()
        time.sleep(IDLE_SECONDS)
        log({"phase": "idle", "trial": "final", "start": start, "end": now()})

        print(f"Experiment finished! Results in {out}")
