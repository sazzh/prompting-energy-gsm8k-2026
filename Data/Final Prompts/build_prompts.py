"""
Builds the prompts (as chat message lists) and a randomised schedule.
"""

import csv
import json
import random
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "Dataset Questions"

SEED = 325
NUM_TRIALS = 3
TECHNIQUES = ["zero_shot", "one_shot", "few_shot", "cot"]


def load(path) -> Any:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def build_messages(question, technique, examples) -> list[dict[str, str]]:
    raw_format = {"role": "user", "content": question}

    if technique == "zero_shot":
        return [raw_format]

    if technique == "cot":
        return [{"role": "user", "content": question + "\nLet's think step by step."}]

    # only use first example for one-shot, all examples for few-shot
    prompt = examples[:1] if technique == "one_shot" else examples
    msgs = []
    for example in prompt:
        msgs.append({"role": "user", "content": example["question"]})
        msgs.append({"role": "assistant", "content": example["answer"]})
    return msgs + [raw_format]


def build_prompts(questions, examples) -> dict[str, dict[str, object]]:
    prompts = {}
    for q in questions:
        for technique in TECHNIQUES:
            prompts[f"{q['id']}_{technique}"] = {
                "qid": q["id"],
                "technique": technique,
                "answer_value": q["answer_value"],
                "messages": build_messages(q["question"], technique, examples),
            }
    return prompts


def make_prompt_order(prompts, rng) -> list[str]:
    """
    Random order where adjacent prompts do not share a technique (or question).
    Reduce ordering bias and make energy comparisons across techniques more fair.
    """
    while True:
        prompt_order = []
        remaining = list(prompts)

        while remaining:
            if prompt_order:
                last_prompt_id = prompt_order[-1]
                last_prompt = prompts[last_prompt_id]

                eligible_prompts = [
                    pid
                    for pid in remaining
                    if prompts[pid]["qid"] != last_prompt["qid"]
                    and prompts[pid]["technique"] != last_prompt["technique"]
                ]
            else:
                eligible_prompts = remaining

            if not eligible_prompts:
                break

            random_selection = rng.choice(eligible_prompts)
            prompt_order.append(random_selection)
            remaining.remove(random_selection)

        if not remaining:
            return prompt_order


if __name__ == "__main__":
    main_questions, spare_questions, examples = (
        load(DATA_DIR / "dataset_test_50.json"),
        load(DATA_DIR / "dataset_test_spare_10.json"),
        load(DATA_DIR / "dataset_train_fewshot_3.json"),
    )

    prompts_main = build_prompts(main_questions, examples)
    prompts_spare = build_prompts(spare_questions, examples)

    rng = random.Random(SEED)
    with open(SCRIPT_DIR / "schedule.csv", "w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["trial", "position", "request_id"])
        for trial in range(1, NUM_TRIALS + 1):
            for pos, rid in enumerate(make_prompt_order(prompts_main, rng), start=1):
                w.writerow([trial, pos, rid])

    for name, data in [
        ("prompts_main.json", prompts_main),
        ("prompts_spare.json", prompts_spare),
    ]:
        with open(SCRIPT_DIR / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"{len(prompts_main)} main prompts, {len(prompts_spare)} spare prompts, "
        f"{NUM_TRIALS * len(prompts_main)} scheduled requests"
    )
