"""
Lotto 6/49 Analysis Tools
Basic analysis functions for the historical dataset
example
second change to test
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover - optional plotting dependency
    HAS_MPL = False

# disable showing charts by default
SHOW_PLOTS = False


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
DATASET_NAME = "lotto_649_complete.csv"
NUMBER_RANGE = range(1, 50)
 
NUMBERS_PER_DRAW = 6  # main numbers per draw
BONUS_COLUMN = "Bonus"


# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------
def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Lotto 6/49 dataset.

    Search order when `path` is not provided:
    1. Current working directory
    2. Project parent directory
    3. User home directory
    4. Known local project path

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    """
    candidates: list[Path] = []

    if path:
        candidates.append(Path(path))
    else:
        candidates.extend(
            [
                Path.cwd() / DATASET_NAME,
                Path(__file__).resolve().parent.parent / DATASET_NAME,
                Path.home() / DATASET_NAME,
                Path(r"C:/Users/ZHIZHANG/lotto-649-historical-dataset") / DATASET_NAME,
            ]
        )

    for p in candidates:
        if p.exists():
            return pd.read_csv(p)

    searched = "\n".join(f" - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find '{DATASET_NAME}'.\n"
        f"Searched:\n{searched}\n\n"
        "Place the CSV in your project folder or pass its path to load_dataset(path)."
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def extract_all_numbers(df: pd.DataFrame) -> list[int]:
    """Extract all drawn numbers from the dataframe."""
    nums: list[int] = []
    for i in range(1, NUMBERS_PER_DRAW + 1):
        nums.extend(df[f"Num{i}"].tolist())
    # include bonus column when present
    if BONUS_COLUMN in df.columns:
        nums.extend(df[BONUS_COLUMN].tolist())
    return nums


def build_frequency(df: pd.DataFrame) -> Counter:
    """Build frequency counter of all drawn numbers."""
    return Counter(extract_all_numbers(df))


# -------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------
def basic_statistics(df: pd.DataFrame) -> Counter:
    """Print basic dataset statistics and return number frequency."""
    freq = build_frequency(df)

    print("LOTTO 6/49 BASIC STATISTICS")
    print("=" * 50)
    print(f"Total Draws : {len(df):,}")
    print(f"Date Range : {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")

    # total drawn numbers includes bonus numbers
    total_numbers = len(df) * (NUMBERS_PER_DRAW + (1 if BONUS_COLUMN in df.columns else 0))

    print("\nMOST FREQUENT NUMBERS:")
    for num, count in freq.most_common(10):
        print(f"  #{num:2d}: {count:4d} times ({count / total_numbers:.4%})")

    print("\nLEAST FREQUENT NUMBERS:")
    for num, count in freq.most_common()[-10:]:
        print(f"  #{num:2d}: {count:4d} times ({count / total_numbers:.4%})")

    print("\nPERCENTAGE OF EACH NUMBER (overall):")
    for n in sorted(NUMBER_RANGE):
        c = freq.get(n, 0)
        print(f"  #{n:2d}: {c:4d} times ({c / total_numbers:.4%})")

    # Histogram chart of percentages
    percentages = [freq.get(n, 0) / total_numbers for n in sorted(NUMBER_RANGE)]
    if HAS_MPL and SHOW_PLOTS:
        x = list(sorted(NUMBER_RANGE))
        y = [p * 100 for p in percentages]
        plt.figure(figsize=(12, 5))
        plt.bar(x, y, color="tab:blue")
        plt.xlabel("Number")
        plt.ylabel("Percentage (%)")
        plt.title("Percentage of Each Number (including bonus)")
        plt.xticks(x)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
    else:
        print("\nMatplotlib not installed — install with `pip install matplotlib` or `conda install matplotlib` to see the histogram.")

    return freq


# -------------------------------------------------------------------
# Candidate Generation
# -------------------------------------------------------------------
def generate_candidate_sets(
    df: pd.DataFrame,
    method: str = "most_frequent",
    count: int = 10,
    top_k: int = 15,
) -> list[tuple[int, ...]]:
    """Generate candidate Lotto 6/49 number sets.

    Methods
    -------
    most_frequent
        Sample from the top-k most frequent numbers.
    weighted_random
        Random sampling weighted by historical frequency.
    hot_and_cold
        Mix of frequent and infrequent numbers.
    """
    freq = build_frequency(df)
    numbers = list(NUMBER_RANGE)
    candidates: list[tuple[int, ...]] = []

    if method == "most_frequent":
        pool = [n for n, _ in freq.most_common(top_k)] or numbers
        for _ in range(count):
            main = sorted(random.sample(pool, NUMBERS_PER_DRAW))
            bonus_pool = [n for n in numbers if n not in main]
            bonus = random.choice(bonus_pool)
            candidates.append(tuple(main + [bonus]))

    elif method == "weighted_random":
        weights = [freq.get(n, 1) for n in numbers]
        for _ in range(count):
            pick = set()
            while len(pick) < NUMBERS_PER_DRAW:
                pick.add(random.choices(numbers, weights=weights, k=1)[0])
            main = sorted(pick)
            bonus_pool = [n for n in numbers if n not in main]
            # weight the bonus by same frequency weights
            bonus_weights = [freq.get(n, 1) for n in bonus_pool]
            bonus = random.choices(bonus_pool, weights=bonus_weights, k=1)[0]
            candidates.append(tuple(main + [bonus]))

    elif method == "hot_and_cold":
        hot = [n for n, _ in freq.most_common(10)] or numbers
        cold = [n for n, _ in freq.most_common()[:-11:-1]] or numbers
        for _ in range(count):
            pick = set(random.sample(hot, 3))
            pick.update(random.sample(cold, 3))
            main = sorted(pick)
            bonus_pool = [n for n in numbers if n not in main]
            bonus = random.choice(bonus_pool)
            candidates.append(tuple(main + [bonus]))

    else:
        raise ValueError(f"Unknown method: {method}")

    return candidates


# -------------------------------------------------------------------
# Monte Carlo Simulation
# -------------------------------------------------------------------
def monte_carlo(
    df: pd.DataFrame, trials: int = 20_000, top_n: int = 10
) -> list[tuple[int, ...]]:
    """Improved Monte Carlo: accumulate total score per ticket and rank by accumulated score
    so tickets that appear often and have high scores are preferred."""
    freq = build_frequency(df)
    numbers = list(NUMBER_RANGE)
    weights = [freq.get(n, 1) for n in numbers]

    seen_counts: dict[tuple[int, ...], int] = {}
    sum_scores: dict[tuple[int, ...], int] = {}

    for _ in range(trials):
        pick = set()
        while len(pick) < NUMBERS_PER_DRAW:
            pick.add(random.choices(numbers, weights=weights, k=1)[0])

        main = sorted(pick)
        bonus_pool = [n for n in numbers if n not in main]
        bonus_weights = [freq.get(n, 1) for n in bonus_pool]
        bonus = random.choices(bonus_pool, weights=bonus_weights, k=1)[0]

        ticket = tuple(main + [bonus])
        score = sum(freq.get(n, 0) for n in ticket)

        seen_counts[ticket] = seen_counts.get(ticket, 0) + 1
        sum_scores[ticket] = sum_scores.get(ticket, 0) + score

    # rank by total accumulated score (occurrence-weighted), tiebreak by occurrence count
    ranked = sorted(
        sum_scores.items(),
        key=lambda kv: (kv[1], seen_counts.get(kv[0], 0)),
        reverse=True,
    )[:top_n]

    return [ticket for ticket, _ in ranked]


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------
def print_candidate_sets(sets: Iterable[tuple[int, ...]], df: pd.DataFrame | None = None) -> None:
    """Print candidate sets and show per-number percentage-weights and total percentage.

    If `df` is provided the function uses its historical frequencies (including bonus)
    to compute weights as the percentage of occurrences over the whole period.
    """
    freq: Counter = build_frequency(df) if df is not None else Counter()

    # compute total numbers for percentage base when df provided
    if df is not None:
        total_numbers = len(df) * (NUMBERS_PER_DRAW + (1 if BONUS_COLUMN in df.columns else 0))
    else:
        total_numbers = 1  # avoid division by zero; weights will be zero

    for i, s in enumerate(sets, 1):
        # per-number weight as fraction of total numbers
        per_weights = [freq.get(n, 0) / total_numbers for n in s]
        total_weight = sum(per_weights)

        if len(s) == NUMBERS_PER_DRAW + 1:
            main_nums = s[:NUMBERS_PER_DRAW]
            bonus = s[-1]
            main_parts = [f"{n}({(freq.get(n,0)/total_numbers):.3%})" for n in main_nums]
            bonus_part = f"{bonus}({(freq.get(bonus,0)/total_numbers):.3%})"
            print(f"#{i:02d}: {', '.join(main_parts)}  (Bonus: {bonus_part})  Total weight: {total_weight:.3%}")
        else:
            parts = [f"{n}({(freq.get(n,0)/total_numbers):.3%})" for n in s]
            print(f"#{i:02d}: {', '.join(parts)}  Total weight: {total_weight:.3%}")


# -------------------------------------------------------------------
# Consecutive-number analysis
# -------------------------------------------------------------------
def analyze_consecutives(df: pd.DataFrame, include_bonus: bool = False) -> None:
    """Analyze consecutive numbers in the historical draws.

    Prints summary statistics, run-length distribution, most common consecutive
    pairs and sequences. When `include_bonus` is True the Bonus column is
    included in each draw's number set for the analysis.
    """
    total_draws = len(df)
    runs_counter = Counter()  # counts of run lengths (>=2)
    pair_counter = Counter()  # counts of consecutive pairs (n, n+1)
    seq3_counter = Counter()  # counts of length-3 consecutive sequences
    seq4_counter = Counter()  # counts of length-4 consecutive sequences
    seq5_counter = Counter()  # counts of length-5 consecutive sequences
    draws_with_any = 0

    for row in df.itertuples(index=False):
        nums = [getattr(row, f"Num{i}") for i in range(1, NUMBERS_PER_DRAW + 1)]
        if include_bonus and BONUS_COLUMN in df.columns:
            nums.append(getattr(row, BONUS_COLUMN))
        nums = sorted(int(n) for n in nums)

        # find runs
        cur_run = [nums[0]]
        any_run = False
        for a, b in zip(nums, nums[1:]):
            if b == a + 1:
                cur_run.append(b)
            else:
                if len(cur_run) >= 2:
                    runs_counter[len(cur_run)] += 1
                    any_run = True
                    # register pairs and seq3 from this run
                    for i in range(len(cur_run) - 1):
                        pair_counter[(cur_run[i], cur_run[i + 1])] += 1
                    if len(cur_run) >= 3:
                        for i in range(len(cur_run) - 2):
                            seq3_counter[tuple(cur_run[i : i + 3])] += 1
                    if len(cur_run) >= 4:
                        for i in range(len(cur_run) - 3):
                            seq4_counter[tuple(cur_run[i : i + 4])] += 1
                    if len(cur_run) >= 5:
                        for i in range(len(cur_run) - 4):
                            seq5_counter[tuple(cur_run[i : i + 5])] += 1
                cur_run = [b]

        # final run check
        if len(cur_run) >= 2:
            runs_counter[len(cur_run)] += 1
            any_run = True
            for i in range(len(cur_run) - 1):
                pair_counter[(cur_run[i], cur_run[i + 1])] += 1
            if len(cur_run) >= 3:
                for i in range(len(cur_run) - 2):
                    seq3_counter[tuple(cur_run[i : i + 3])] += 1
            if len(cur_run) >= 4:
                for i in range(len(cur_run) - 3):
                    seq4_counter[tuple(cur_run[i : i + 4])] += 1
            if len(cur_run) >= 5:
                for i in range(len(cur_run) - 4):
                    seq5_counter[tuple(cur_run[i : i + 5])] += 1

        if any_run:
            draws_with_any += 1

    total_runs = sum(runs_counter.values())
    total_pairs = sum(pair_counter.values())

    # total numbers that are part of consecutive runs (sum of run lengths)
    total_consecutive_numbers = sum(length * cnt for length, cnt in runs_counter.items())
    total_numbers_drawn = total_draws * (NUMBERS_PER_DRAW + (1 if include_bonus and BONUS_COLUMN in df.columns else 0))

    print("\nCONSECUTIVE-NUMBERS ANALYSIS")
    print("=" * 50)
    print(f"Total draws analyzed: {total_draws:,}")
    print(f"Draws with at least one consecutive run: {draws_with_any:,} ({draws_with_any/total_draws:.2%})")
    print(f"Total consecutive runs found: {total_runs:,}")
    print(f"Total consecutive pairs found: {total_pairs:,}")
    print(f"Total consecutive numbers in runs: {total_consecutive_numbers:,} ({total_consecutive_numbers/total_numbers_drawn:.2%} of drawn numbers)")

    if total_runs:
        print("\nRun-length distribution (length: count):")
        for length, cnt in sorted(runs_counter.items(), reverse=True):
            print(f"  {length:2d}: {cnt:4d} ({cnt/total_runs:.2%} of runs)")

    if total_pairs:
        print("\nMost common consecutive pairs:")
        for (a, b), cnt in pair_counter.most_common(10):
            print(f"  ({a},{b}): {cnt:4d} times ({cnt/total_pairs:.2%} of pairs)")

    if seq3_counter:
        print("\nMost common 3-number consecutive sequences:")
        for seq, cnt in seq3_counter.most_common(10):
            print(f"  {seq}: {cnt:4d} times")

    if seq4_counter:
        print("\nMost common 4-number consecutive sequences:")
        for seq, cnt in seq4_counter.most_common(10):
            print(f"  {seq}: {cnt:4d} times")

    if seq5_counter:
        print("\nMost common 5-number consecutive sequences:")
        for seq, cnt in seq5_counter.most_common(5):
            print(f"  {seq}: {cnt:4d} times")

    # optional plot: run-length histogram
    if HAS_MPL and SHOW_PLOTS and total_runs:
         lengths = []
         for length, cnt in runs_counter.items():
             lengths.extend([length] * cnt)
         bins = list(range(2, max(lengths) + 2))
         plt.figure(figsize=(8, 4))
         counts, bins_out, patches = plt.hist(lengths, bins=bins, align='left', rwidth=0.8)
         plt.xlabel('Run length')
         plt.ylabel('Count')
         plt.title('Distribution of consecutive run lengths')
         plt.grid(axis='y', linestyle='--', alpha=0.4)
 
         # annotate each bar with its count
         for rect, count in zip(patches, counts):
             height = rect.get_height()
             if height > 0:
                 plt.text(rect.get_x() + rect.get_width() / 2, height, f"{int(count)}",
                          ha='center', va='bottom', fontsize=9)
 
         plt.tight_layout()
         plt.show()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    df = load_dataset()
    basic_statistics(df)
    try:
        analyze_consecutives(df, include_bonus=False)
    except NameError:
        print("\nConsecutive analysis function not available. Skipping.")

    print("\nSuggested candidate sets (most_frequent):")
    print_candidate_sets(generate_candidate_sets(df), df)

    print("\nMonte Carlo suggestions:")
    print_candidate_sets(monte_carlo(df, trials=5_000), df)


def match_numbers(candidate: Iterable[int] | tuple[int, ...], actual: Iterable[int] | tuple[int, ...]) -> dict:
    """Return match counts between a candidate (main + optional bonus) and an actual draw.

    Returns:
      {"main_matches": int, "bonus_match": bool, "total_matches": int}

    Both `candidate` and `actual` may be length-6 (no bonus) or length-7 (bonus as last element).
    """
    cand = list(candidate)
    act = list(actual)

    cand_main = set(cand[:NUMBERS_PER_DRAW])
    cand_bonus = cand[NUMBERS_PER_DRAW] if len(cand) > NUMBERS_PER_DRAW else None

    act_main = set(act[:NUMBERS_PER_DRAW])
    act_bonus = act[NUMBERS_PER_DRAW] if len(act) > NUMBERS_PER_DRAW else None

    main_matches = len(cand_main & act_main)
    bonus_match = (cand_bonus is not None and act_bonus is not None and cand_bonus == act_bonus)
    total_matches = main_matches + (1 if bonus_match else 0)

    return {"main_matches": main_matches, "bonus_match": bonus_match, "total_matches": total_matches}


def evaluate_candidates(candidates: Iterable[tuple[int, ...]], actual: Iterable[int] | tuple[int, ...]) -> list[dict]:
    """Evaluate multiple candidate tickets against an actual draw.

    Returns a list of dicts with keys: "candidate", "main_matches", "bonus_match", "total_matches".
    """
    results: list[dict] = []
    for c in candidates:
        m = match_numbers(c, actual)
        results.append({"candidate": tuple(c), **m})
    return results


def update_model_and_regenerate(actual: Iterable[int] | tuple[int, ...], df: pd.DataFrame, persist: bool = False, csv_path: str | Path | None = None, regen_count: int = 10, mc_trials: int = 5_000) -> dict:
    """Append actual result, rebuild frequencies and return regenerated candidate lists.

    Returns dict with keys: "df", "mc_candidates", "rand_candidates".
    """
    df_updated = add_actual_result(actual, df, persist=persist, csv_path=csv_path)
    mc_candidates = monte_carlo(df_updated, trials=mc_trials, top_n=regen_count)
    rand_candidates = generate_candidate_sets(df_updated, method="weighted_random", count=regen_count)
    return {"df": df_updated, "mc_candidates": mc_candidates, "rand_candidates": rand_candidates}

# example usage (moved below function definitions so update_model_and_regenerate exists)
from pprint import pprint

actual = (13, 22, 24, 28, 29, 46, 16)  # main 6 + bonus

df = load_dataset()  # or use existing df variable

# generate candidates
mc_candidates = monte_carlo(df, trials=5_000, top_n=10)
rand_candidates = generate_candidate_sets(df, method="weighted_random", count=10)

# evaluate
mc_results = evaluate_candidates(mc_candidates, actual)
rand_results = evaluate_candidates(rand_candidates, actual)

# print all results (sorted by total_matches desc)
print("Monte Carlo results (sorted):")
for r in sorted(mc_results, key=lambda x: x["total_matches"], reverse=True):
    print(r)

print("\nRandom-weighted results (sorted):")
for r in sorted(rand_results, key=lambda x: x["total_matches"], reverse=True):
    print(r)

updated = update_model_and_regenerate((13,22,24,28,29,46,16), df, persist=False, regen_count=10)
print_candidate_sets(updated["mc_candidates"], updated["df"])
print_candidate_sets(updated["rand_candidates"], updated["df"])


def add_actual_result(actual: Iterable[int] | tuple[int, ...], df: pd.DataFrame, date: str | None = None, persist: bool = False, csv_path: str | Path | None = None) -> pd.DataFrame:
    """Append an actual draw to `df`. `actual` is length-6 or length-7 (bonus last).
    If persist=True the updated dataframe is written to `csv_path` or the default dataset path.
    """
    actual = list(actual)
    if len(actual) < NUMBERS_PER_DRAW:
        raise ValueError("actual must contain at least the 6 main numbers")
    row = {}
    row["Date"] = date or pd.Timestamp.now().strftime("%Y-%m-%d")
    for i in range(NUMBERS_PER_DRAW):
        row[f"Num{i+1}"] = int(actual[i])
    if len(actual) > NUMBERS_PER_DRAW:
        row[BONUS_COLUMN] = int(actual[NUMBERS_PER_DRAW])
    elif BONUS_COLUMN in df.columns:
        row[BONUS_COLUMN] = pd.NA

    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    if persist:
        csv_target = Path(csv_path) if csv_path else Path(__file__).resolve().parent.parent / DATASET_NAME
        new_df.to_csv(csv_target, index=False)
    return new_df


def update_model_and_regenerate(actual: Iterable[int] | tuple[int, ...], df: pd.DataFrame, persist: bool = False, csv_path: str | Path | None = None, regen_count: int = 10, mc_trials: int = 5_000) -> dict:
    """Append actual result, rebuild frequencies and return regenerated candidate lists.

    Returns dict with keys: "df", "mc_candidates", "rand_candidates".
    """
    df_updated = add_actual_result(actual, df, persist=persist, csv_path=csv_path)
    mc_candidates = monte_carlo(df_updated, trials=mc_trials, top_n=regen_count)
    rand_candidates = generate_candidate_sets(df_updated, method="weighted_random", count=regen_count)
    return {"df": df_updated, "mc_candidates": mc_candidates, "rand_candidates": rand_candidates}
