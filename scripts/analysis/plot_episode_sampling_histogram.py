#!/usr/bin/env python3
"""Parse training logs and plot which episodes were sampled.

This script looks for structured lines emitted by the value-finetuning
training loop and turns them into a per-episode count histogram.

Example log line:
    EPISODE_SAMPLE epoch=3 dataset=droid episode=42 step=128
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


EPISODE_PATTERN = re.compile(r"\bepisode=(\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot episode-sampling frequency from DreamZero training logs."
    )
    parser.add_argument("log_file", type=Path, help="Path to the captured training log file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the CSV and PNG outputs. Defaults to the log file directory.",
    )
    parser.add_argument(
        "--csv-name",
        default="episode_sampling_counts.csv",
        help="Output CSV filename inside the output directory.",
    )
    parser.add_argument(
        "--png-name",
        default="episode_sampling_histogram.png",
        help="Output PNG filename inside the output directory.",
    )
    return parser.parse_args()


def load_episode_counts(log_file: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "EPISODE_SAMPLE" not in line:
                continue
            match = EPISODE_PATTERN.search(line)
            if match is None:
                continue
            counts[int(match.group(1))] += 1
    return counts


def write_counts_csv(counts: Counter[int], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "sample_count"])
        for episode, sample_count in sorted(counts.items()):
            writer.writerow([episode, sample_count])


def save_histogram_plot(counts: Counter[int], png_path: Path) -> None:
    if not counts:
        raise ValueError("No EPISODE_SAMPLE lines were found in the log file.")

    import matplotlib.pyplot as plt

    episodes = sorted(counts)
    sample_counts = [counts[episode] for episode in episodes]

    fig_width = max(12.0, min(40.0, len(episodes) * 0.25))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.bar(episodes, sample_counts, width=0.9, color="#d97706", edgecolor="#1f2937", linewidth=0.5)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Sample count")
    ax.set_title("Training episode sampling histogram")
    if len(episodes) > 40:
        tick_step = max(1, len(episodes) // 20)
        ax.set_xticks(episodes[::tick_step])
        ax.tick_params(axis="x", labelrotation=45)
    else:
        ax.set_xticks(episodes)
        ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.log_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = load_episode_counts(args.log_file)
    if not counts:
        raise SystemExit(f"No EPISODE_SAMPLE lines found in {args.log_file}")

    csv_path = output_dir / args.csv_name
    png_path = output_dir / args.png_name

    write_counts_csv(counts, csv_path)
    save_histogram_plot(counts, png_path)

    total_samples = sum(counts.values())
    unique_episodes = len(counts)
    top_five = counts.most_common(5)

    print(f"Parsed {total_samples} samples across {unique_episodes} episodes from {args.log_file}")
    print(f"Wrote counts to {csv_path}")
    print(f"Wrote histogram to {png_path}")
    print("Top episodes:")
    for episode, sample_count in top_five:
        print(f"  episode {episode}: {sample_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
