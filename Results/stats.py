"""Summarise per-request results for the GSM8K experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent
TECHNIQUES = ["zero_shot", "one_shot", "few_shot", "cot"]
COMPARISON_FAMILIES = {
    # comparison against zero shot as baseline
    "Family 1 (primary)": [
        ("one_shot", "zero_shot"),
        ("few_shot", "zero_shot"),
        ("cot", "zero_shot"),
    ],
    "Family 2 (exploratory)": [
        ("few_shot", "one_shot"),
        ("few_shot", "cot"),
        ("one_shot", "cot"),
    ],
}


def load_data():
    df = pd.read_csv(DATA_DIR / "per_request.csv")
    required = {
        "qid",
        "technique",
        "net_J",
        "prompt_tokens",
        "completion_tokens",
        "net_prompt_J",
        "net_gen_J",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def build_question_table(df):
    # net energy over three trials for each question and technique
    question_table = df.pivot_table(
        index="qid", columns="technique", values="net_J", aggfunc="mean"
    )[TECHNIQUES]
    question_table.to_csv(DATA_DIR / "per_question.csv")

    # descriptive statistics
    descriptives = pd.DataFrame(
        {
            "num_questions": question_table.count(),
            "mean_J": question_table.mean(),
            "std_J": question_table.std(ddof=1),
            "median_J": question_table.median(),
        }
    )
    print(descriptives.round(1))
    return question_table


def compare_techniques(question_table, technique_a, technique_b):
    diff = (question_table[technique_a] - question_table[technique_b]).to_numpy()

    return {
        "contrast": f"{technique_a} - {technique_b}",
        "mean_J": diff.mean(),
        "median_J": np.median(diff),
        "pct": 100 * diff.mean() / question_table[technique_b].mean(),
        "d_z": diff.mean() / diff.std(ddof=1),
        "p_raw": stats.wilcoxon(diff)[1],
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0

    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)

    return adjusted


def print_pairwise_comparisons(question_table):
    for family_name, pairs in COMPARISON_FAMILIES.items():
        comparison_table = pd.DataFrame(
            [compare_techniques(question_table, a, b) for a, b in pairs]
        )
        comparison_table["p_holm"] = holm_adjust(comparison_table["p_raw"])
        display = comparison_table.round(
            {"mean_J": 1, "median_J": 1, "pct": 1, "d_z": 2}
        )
        for column in ["p_raw", "p_holm"]:
            display[column] = comparison_table[column].map(
                lambda p: f"{p:.4f}" if p > 0.001 else f"{p:.1e}"
            )
        print(f"\n{family_name}")
        print(display.to_string(index=False))


def describe_mechanism(df):
    means_by_technique = (
        df.groupby("technique")[
            ["net_prompt_J", "net_gen_J", "prompt_tokens", "completion_tokens"]
        ]
        .mean()
        .loc[TECHNIQUES]
    )

    print("\nMeans by technique")
    print(means_by_technique.round(1))
    print("\nChange relative to zero-shot")
    print((means_by_technique - means_by_technique.loc["zero_shot"]).round(1))

    # energy per token
    prompt_j_per_token = df["net_prompt_J"].sum() / df["prompt_tokens"].sum()
    output_j_per_token = df["net_gen_J"].sum() / df["completion_tokens"].sum()
    print(f"\nJ per prompt token: {prompt_j_per_token:.3f}")
    print(f"J per output token: {output_j_per_token:.3f}")


def main():
    df = load_data()
    question_table = build_question_table(df)
    print_pairwise_comparisons(question_table)
    describe_mechanism(df)


if __name__ == "__main__":
    main()
