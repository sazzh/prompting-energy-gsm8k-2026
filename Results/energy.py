"""Compute per-request energy usage from power and experiment logs."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results_20261104_175031.csv"
HWINFO = BASE_DIR / "experiment-system-logs-HWiNFO.CSV"
POWER = "CPU Package Power [W]"


def load_power_log(path: Path) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Load the hardware power log and return time offsets and power samples."""
    power_log = pd.read_csv(
        path,
        encoding="latin-1",
        usecols=["Date", "Time", POWER],
        index_col=False,
        low_memory=False,
    )

    # Drop footer rows that do not follow the timestamp pattern used in log
    power_log = power_log[power_log["Date"].astype(str).str.match(r"^\d+\.\d+\.\d{4}$")]
    power_log["ts"] = pd.to_datetime(
        power_log["Date"] + " " + power_log["Time"],
        format="%d.%m.%Y %H:%M:%S.%f",
    )
    power_log[POWER] = pd.to_numeric(power_log[POWER])

    start_time = power_log["ts"].iloc[0]
    elapsed_s = (power_log["ts"] - start_time).dt.total_seconds().to_numpy()
    power_w = power_log[POWER].to_numpy()
    return elapsed_s, power_w, start_time


def load_request_log(path: Path, start_time: pd.Timestamp) -> pd.DataFrame:
    """Load the request metadata and append request start/end offsets in seconds."""
    request_log = pd.read_csv(path)
    request_log["start_s"] = (
        pd.to_datetime(request_log["start"]) - start_time
    ).dt.total_seconds()
    request_log["end_s"] = (
        pd.to_datetime(request_log["end"]) - start_time
    ).dt.total_seconds()
    return request_log


def compute_idle_power(
    power_w: np.ndarray, elapsed_s: np.ndarray, request_log: pd.DataFrame
) -> float:
    """Median idle power across all samples inside the idle request windows."""
    idle_rows = request_log[request_log["phase"] == "idle"]
    if idle_rows.empty:
        raise ValueError("Request log contains no idle windows.")

    idle_samples = np.concatenate(
        [
            power_w[(elapsed_s >= start_s) & (elapsed_s < end_s)]
            for start_s, end_s in zip(idle_rows["start_s"], idle_rows["end_s"])
        ]
    )
    if idle_samples.size == 0:
        raise ValueError("No power samples fall within the idle windows.")

    return round(float(np.median(idle_samples)), 2)


def energy_between(
    start_s: float, end_s: float, elapsed_s: np.ndarray, power_w: np.ndarray
) -> float:
    """Approximate energy in joules between two timestamps using custom trapezoid rule."""
    inside = (elapsed_s > start_s) & (elapsed_s < end_s)
    timestamps = np.concatenate([[start_s], elapsed_s[inside], [end_s]])
    powers = np.concatenate(
        [
            [np.interp(start_s, elapsed_s, power_w)],
            power_w[inside],
            [np.interp(end_s, elapsed_s, power_w)],
        ]
    )
    return float(np.sum((powers[1:] + powers[:-1]) / 2 * np.diff(timestamps)))


def main() -> None:
    elapsed_s, power_w, start_time = load_power_log(HWINFO)
    request_log = load_request_log(RESULTS, start_time)

    idle_power_w = compute_idle_power(power_w, elapsed_s, request_log)
    print("idle power (W):", idle_power_w)

    run = request_log[request_log["phase"] == "run"].copy()
    run["duration_s"] = run["end_s"] - run["start_s"]
    run["gross_J"] = [
        energy_between(start_s, end_s, elapsed_s, power_w)
        for start_s, end_s in zip(run["start_s"], run["end_s"])
    ]
    run["net_J"] = run["gross_J"] - idle_power_w * run["duration_s"]

    # Prompt processing ends when the server reports prompt_ms
    run["prompt_s"] = run["prompt_ms"] / 1000
    run["net_prompt_J"] = [
        energy_between(start_s, start_s + prompt_s, elapsed_s, power_w)
        - idle_power_w * prompt_s
        for start_s, prompt_s in zip(run["start_s"], run["prompt_s"])
    ]
    run["net_gen_J"] = run["net_J"] - run["net_prompt_J"]

    output_columns = [
        "trial",
        "position",
        "request_id",
        "qid",
        "technique",
        "start",
        "end",
        "duration_s",
        "prompt_tokens",
        "completion_tokens",
        "gross_J",
        "net_J",
        "net_prompt_J",
        "net_gen_J",
    ]
    run[output_columns].to_csv(BASE_DIR / "per_request.csv", index=False)

    summary = (
        run.groupby("technique")[["net_J", "net_prompt_J", "net_gen_J"]].mean().round(1)
    )
    print(summary)


if __name__ == "__main__":
    main()
