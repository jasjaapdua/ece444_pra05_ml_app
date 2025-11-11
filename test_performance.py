"""
ECE444 PRA5 - Performance / Latency Testing
Author: Jasjaap Dua
"""

import pytest
import requests
import time
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pytz")

BASE_URL = "http://ece444-pra05-env.eba-vraaipee.us-east-2.elasticbeanstalk.com/predict"

TEST_CASES = [
    ("fake_1", "Eiffel Tower washes up on Delaware Beach."),
    ("fake_2", "University of Toronto moved to Montreal"),
    ("real_1", "The Prime Minister announced new economic policies"),
    ("real_2", "Syria's President Meets Trump at White House for First Time"),
]

CSV_FILE = "latency_results.csv"
SUMMARY_FILE = "latency_summary.csv"
PLOT_FILE = "latency_boxplot.png"

# remove old files so each run starts clean
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)
if os.path.exists(SUMMARY_FILE):
    os.remove(SUMMARY_FILE)
if os.path.exists(PLOT_FILE):
    os.remove(PLOT_FILE)


@pytest.mark.parametrize("case_name,text", TEST_CASES)
def test_api_latency(case_name, text):
    """Run exactly 100 requests per test case and record latencies."""
    results = []
    print(f"\nRunning latency test for {case_name}")

    for i in range(100):
        start = time.time()
        resp = requests.post(BASE_URL, json={"message": text})
        end = time.time()

        latency_ms = (end - start) * 1000
        assert resp.status_code == 200, f"{case_name} failed at request {i}"
        results.append({"case": case_name, "latency_ms": latency_ms})

        time.sleep(0.05)

    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "latency_ms"])
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"Completed 100 calls for {case_name}")


def test_generate_mean_and_plot():
    """Compute mean latency per test case and generate boxplot."""
    assert os.path.exists(CSV_FILE), "latency_results.csv not found"

    df = pd.read_csv(CSV_FILE)

    # check counts
    counts = df["case"].value_counts()
    for case_name, count in counts.items():
        assert count == 100, f"{case_name} ran {count} times instead of 100"

    mean_latency = df.groupby("case")["latency_ms"].mean().reset_index()

    # print mean values
    print("Mean latency (ms) per test case:")
    for _, row in mean_latency.iterrows():
        print(f"{row['case']},{row['latency_ms']:.4f}")

    mean_latency[["case", "latency_ms"]].to_csv(SUMMARY_FILE, index=False)

    plt.figure(figsize=(8, 5))
    df.boxplot(by="case", column=["latency_ms"], grid=False)
    plt.title("Fake News API Latency (100 requests per case)")
    plt.suptitle("")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, bbox_inches="tight")
    plt.close()

    print(f"\nSaved {SUMMARY_FILE} and {PLOT_FILE}")
