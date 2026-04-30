"""
Entropy Analysis for Shannon's Guessing Game Experiment.

Based on methodology from:
Ren, G., Takahashi, S., & Tanaka-Ishii, K. (2019).
"Entropy Rate Estimation for English via a Large Cognitive Experiment Using Mechanical Turk"
https://pmc.ncbi.nlm.nih.gov/articles/PMC7514546/
"""

import sys
import math
import csv
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ALPHABET_SIZE = 34   # 33 Ukrainian letters + space
MIN_POSITION = 70    # first guessed position (initial 70 chars revealed)

DATA_DIR = Path(__file__).parent.parent / "dataset_export"


@dataclass
class User:
    id: int


@dataclass
class Sentence:
    id: int
    text: str
    dataset_type: str
    length: int


@dataclass
class ExperimentResult:
    id: int
    anon_user_id: int
    sentence_id: int
    finished: bool


@dataclass
class CharacterGuess:
    id: int
    experiment_result_id: int
    position: int
    guess_number: int
    guessed_character: str
    is_correct: bool


def load_data():
    print("Loading data from CSVs...")

    with open(DATA_DIR / "sentences.csv", encoding="utf-8") as f:
        sentences = [
            Sentence(int(r["id"]), r["text"], r["dataset_type"], int(r["length"]))
            for r in csv.DictReader(f)
        ]

    with open(DATA_DIR / "sessions.csv", encoding="utf-8") as f:
        experiments = [
            ExperimentResult(int(r["id"]), int(r["anon_user_id"]), int(r["sentence_id"]), r["finished"] == "True")
            for r in csv.DictReader(f)
        ]

    with open(DATA_DIR / "guesses.csv", encoding="utf-8") as f:
        guesses = sorted(
            [
                CharacterGuess(int(r["id"]), int(r["session_id"]), int(r["position"]),
                               int(r["guess_number"]), r["guessed_character"], r["is_correct"] == "True")
                for r in csv.DictReader(f)
            ],
            key=lambda g: (g.experiment_result_id, g.position, g.guess_number)
        )

    user_ids = {e.anon_user_id for e in experiments}
    users = [User(uid) for uid in user_ids]

    print("Data loaded.\n")
    return users, sentences, experiments, guesses


def print_basic_info(users, sentences, experiments, guesses):
    """Print basic statistics about the dataset."""
    print("=" * 60)
    print("BASIC DATASET INFORMATION")
    print("=" * 60)

    print(f"\n{'Metric':<40} {'Value':>12}")
    print("-" * 54)

    # Counts
    print(f"{'Total users':<40} {len(users):>12,}")
    print(f"{'Total sentences':<40} {len(sentences):>12,}")
    print(f"{'Total experiments':<40} {len(experiments):>12,}")
    print(f"{'Total guesses (position >= 70)':<40} {len(guesses):>12,}")

    # Experiment completion
    finished = sum(1 for e in experiments if e.finished)
    print(f"\n{'Finished experiments':<40} {finished:>12,}")
    print(f"{'Unfinished experiments':<40} {len(experiments) - finished:>12,}")
    if experiments:
        print(f"{'Completion rate':<40} {finished/len(experiments)*100:>11.1f}%")

    # Position coverage
    positions = set(g.position for g in guesses)
    if positions:
        print(f"\n{'Position range':<40} {min(positions):>5} - {max(positions):<5}")
        print(f"{'Unique positions covered':<40} {len(positions):>12}")

    # Guess correctness
    if guesses:
        correct = sum(1 for g in guesses if g.is_correct)
        print(f"\n{'Correct guesses':<40} {correct:>12,}")
        print(f"{'Incorrect guesses':<40} {len(guesses) - correct:>12,}")
        print(f"{'Overall accuracy':<40} {correct/len(guesses)*100:>11.1f}%")

    # Configuration
    print(f"\n{'Alphabet size (K)':<40} {ALPHABET_SIZE:>12}")
    print(f"{'Initial reveal (first guessed pos)':<40} {MIN_POSITION:>12}")

    print("\n" + "=" * 60)


def compute_guesses_to_correct(guesses):
    """
    Compute the number of guesses needed to get each position correct.
    Returns dict: position -> list of guess counts (one per experiment that reached that position)
    """
    guesses_per_position = defaultdict(list)

    current_exp = None
    current_pos = None
    count = 0

    for g in guesses:
        # New experiment or new position
        if g.experiment_result_id != current_exp or g.position != current_pos:
            current_exp = g.experiment_result_id
            current_pos = g.position
            count = 0

        count += 1

        if g.is_correct:
            guesses_per_position[g.position].append(count)

    return guesses_per_position


def compute_upper_bound(guess_counts):
    """
    Compute Shannon upper bound: -Σ qᵢ log₂(qᵢ)
    where qᵢ = probability of needing exactly i guesses
    """
    if not guess_counts:
        return None

    n = len(guess_counts)
    freq = defaultdict(int)
    for g in guess_counts:
        freq[g] += 1

    entropy = 0.0
    for i in range(1, ALPHABET_SIZE + 1):
        q_i = freq[i] / n
        if q_i > 0:
            entropy -= q_i * math.log2(q_i)

    return entropy


def compute_lower_bound(guess_counts):
    """
    Compute Shannon lower bound: Σ i(qᵢ - qᵢ₊₁) log₂(i)
    """
    if not guess_counts:
        return None

    n = len(guess_counts)
    freq = defaultdict(int)
    for g in guess_counts:
        freq[g] += 1

    q = {i: freq[i] / n for i in range(1, ALPHABET_SIZE + 2)}

    lower = 0.0
    for i in range(1, ALPHABET_SIZE + 1):
        q_i = q[i]
        q_i_next = q[i + 1]
        if i > 0:
            lower += i * (q_i - q_i_next) * math.log2(i) if i > 1 else 0

    return lower


def plot_entropy_by_position(guesses_per_position, max_position=110):
    """Plot entropy bounds by position."""
    print("\n" + "=" * 60)
    print("ENTROPY BOUNDS BY POSITION")
    print("=" * 60)

    positions = sorted(p for p in guesses_per_position.keys() if p <= max_position)
    upper_bounds = []
    lower_bounds = []
    sample_counts = []

    for pos in positions:
        counts = guesses_per_position[pos]
        upper_bounds.append(compute_upper_bound(counts))
        lower_bounds.append(compute_lower_bound(counts))
        sample_counts.append(len(counts))

    plot_positions = positions
    plot_upper = upper_bounds
    plot_lower = lower_bounds

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(plot_positions, plot_upper, 'b-', linewidth=1.5, label='Upper Bound', alpha=0.8)
    ax.plot(plot_positions, plot_lower, 'r-', linewidth=1.5, label='Lower Bound', alpha=0.8)

    # Reference lines
    max_entropy = math.log2(ALPHABET_SIZE)
    ax.axhline(y=max_entropy, color='gray', linestyle='--', linewidth=1,
               label=f'Max entropy (log₂{ALPHABET_SIZE} = {max_entropy:.2f})', alpha=0.5)

    ax.set_xlabel('Position in sentence', fontsize=12)
    ax.set_ylabel('Entropy (bits per character)', fontsize=12)
    ax.set_title(f'Ukrainian Language Entropy Bounds (positions 70–{max_position})', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'entropy_by_position.pdf'
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved: {output_path}")

    return positions, upper_bounds, lower_bounds


# === Fitting function for smoothed extrapolation ===

def tail_power(n, a, b, h):
    """f(n) = h + a/n^b — power decay"""
    return h + a / np.power(n, b)




def weighted_regression_extrapolation(positions, upper_bounds, guesses_per_position):
    """Fit using sample counts as weights — high-sample positions matter more."""
    print("\n" + "=" * 60)
    print("WEIGHTED REGRESSION EXTRAPOLATION")
    print("=" * 60)

    x = np.array(positions, dtype=float)
    y = np.array(upper_bounds, dtype=float)
    weights = np.array([len(guesses_per_position[p]) for p in positions], dtype=float)

    # Normalize weights
    weights = weights / weights.max()

    print(f"\nSample counts: {weights.min()*weights.max():.0f} - {weights.max()*weights.max():.0f}")
    print(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")

    results = []

    # 1. Weighted hyperbolic: h + a/n
    try:
        def weighted_loss_hyp(params):
            a, h = params
            pred = h + a / x
            return np.sum(weights * (y - pred) ** 2)

        from scipy.optimize import minimize
        res = minimize(weighted_loss_hyp, [50, 0.5], bounds=[(0, None), (0, None)])
        a, h = res.x
        pred = h + a / x
        ss_res = np.sum(weights * (y - pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results.append({'name': 'Weighted Hyperbolic', 'h': h, 'a': a, 'r2': r2, 'func': lambda n, a=a, h=h: h + a/n})
    except Exception as e:
        print(f"Weighted Hyperbolic failed: {e}")

    # 2. Weighted power: h + a/n^b
    try:
        def weighted_loss_pow(params):
            a, b, h = params
            pred = h + a / np.power(x, b)
            return np.sum(weights * (y - pred) ** 2)

        res = minimize(weighted_loss_pow, [50, 1.0, 0.5], bounds=[(0, None), (0.1, 3), (0, None)])
        a, b, h = res.x
        pred = h + a / np.power(x, b)
        ss_res = np.sum(weights * (y - pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results.append({'name': 'Weighted Power', 'h': h, 'a': a, 'b': b, 'r2': r2, 'func': lambda n, a=a, b=b, h=h: h + a/np.power(n, b)})
    except Exception as e:
        print(f"Weighted Power failed: {e}")

    # Print results
    print(f"\n{'Function':<25} {'h (bpc)':>10} {'R²':>10} {'Params':>25}")
    print("-" * 75)
    for r in results:
        params = f"a={r['a']:.2f}" + (f", b={r.get('b', 1):.2f}" if 'b' in r else "")
        print(f"{r['name']:<25} {r['h']:>10.3f} {r['r2']:>10.4f} {params:>25}")

    return results


def smoothed_fit_extrapolation(positions, upper_bounds):
    """Smooth the data first, then fit."""
    print("\n" + "=" * 60)
    print("SMOOTHED FIT EXTRAPOLATION")
    print("=" * 60)

    x = np.array(positions, dtype=float)
    y = np.array(upper_bounds, dtype=float)

    # Apply different smoothing windows
    results = []

    for window in [3, 5, 7]:
        # Simple moving average
        y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
        x_smooth = x[(window-1)//2 : -(window-1)//2] if window > 1 else x

        if len(x_smooth) != len(y_smooth):
            x_smooth = x[(window-1)//2 : (window-1)//2 + len(y_smooth)]

        # Fit power decay to smoothed data
        try:
            popt, _ = curve_fit(
                tail_power, x_smooth, y_smooth,
                p0=[50, 1.0, 0.5],
                bounds=([0, 0.1, 0], [np.inf, 3, np.inf]),
                maxfev=10000
            )
            a, b, h = popt
            y_pred = tail_power(x_smooth, *popt)
            r2 = 1 - np.sum((y_smooth - y_pred)**2) / np.sum((y_smooth - np.mean(y_smooth))**2)
            results.append({
                'name': f'Smoothed (w={window})',
                'h': h, 'a': a, 'b': b, 'r2': r2,
                'x_smooth': x_smooth, 'y_smooth': y_smooth,
                'func': lambda n, a=a, b=b, h=h: h + a/np.power(n, b)
            })
        except Exception as e:
            print(f"Smoothed (w={window}) failed: {e}")

    # Print results
    print(f"\n{'Function':<25} {'h (bpc)':>10} {'R²':>10} {'Params':>25}")
    print("-" * 75)
    for r in results:
        params = f"a={r['a']:.2f}, b={r['b']:.2f}"
        print(f"{r['name']:<25} {r['h']:>10.3f} {r['r2']:>10.4f} {params:>25}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: smoothed data
    ax1.plot(x, y, 'ko', markersize=4, alpha=0.4, label='Raw data')
    colors = ['blue', 'green', 'red']
    for r, c in zip(results, colors):
        ax1.plot(r['x_smooth'], r['y_smooth'], 'o-', color=c, markersize=3, label=r['name'], alpha=0.8)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Entropy (bpc)')
    ax1.set_title('Smoothed Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: extrapolation
    x_extrap = np.linspace(min(x), 200, 200)
    ax2.plot(x, y, 'ko', markersize=4, alpha=0.4, label='Raw data')
    for r, c in zip(results, colors):
        y_extrap = r['func'](x_extrap)
        ax2.plot(x_extrap, y_extrap, '-', color=c, linewidth=2, label=f"{r['name']}: h={r['h']:.2f}", alpha=0.8)
        ax2.axhline(y=r['h'], color=c, linestyle=':', alpha=0.4)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Entropy (bpc)')
    ax2.set_title('Extrapolation from Smoothed Fits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0, top=2.5)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'smoothed_extrapolation.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved: {output_path}")

    return results


def compute_entropy_for_experiments(guesses, exp_ids, min_samples=50):
    """Compute entropy bounds for a subset of experiments."""
    filtered_guesses = [g for g in guesses if g.experiment_result_id in exp_ids]

    guesses_per_pos = defaultdict(list)
    current_exp = None
    current_pos = None
    count = 0

    for g in filtered_guesses:
        if g.experiment_result_id != current_exp or g.position != current_pos:
            current_exp = g.experiment_result_id
            current_pos = g.position
            count = 0
        count += 1
        if g.is_correct:
            guesses_per_pos[g.position].append(count)

    bounds_at_70 = compute_upper_bound(guesses_per_pos.get(70, []))
    bounds_at_80 = compute_upper_bound(guesses_per_pos.get(80, []))
    bounds_at_90 = compute_upper_bound(guesses_per_pos.get(90, []))

    samples_at_70 = len(guesses_per_pos.get(70, []))
    samples_at_80 = len(guesses_per_pos.get(80, []))
    samples_at_90 = len(guesses_per_pos.get(90, []))

    return {
        'upper_70': bounds_at_70,
        'upper_80': bounds_at_80,
        'upper_90': bounds_at_90,
        'samples_70': samples_at_70,
        'samples_80': samples_at_80,
        'samples_90': samples_at_90,
    }


def compute_full_curve_for_experiments(guesses, exp_ids, min_samples=30):
    """Compute full entropy curve for a subset of experiments."""
    filtered_guesses = [g for g in guesses if g.experiment_result_id in exp_ids]

    guesses_per_pos = defaultdict(list)
    current_exp = None
    current_pos = None
    count = 0

    for g in filtered_guesses:
        if g.experiment_result_id != current_exp or g.position != current_pos:
            current_exp = g.experiment_result_id
            current_pos = g.position
            count = 0
        count += 1
        if g.is_correct:
            guesses_per_pos[g.position].append(count)

    positions = []
    upper_bounds = []
    sample_counts = []

    for pos in sorted(guesses_per_pos.keys()):
        counts = guesses_per_pos[pos]
        if len(counts) >= min_samples:
            positions.append(pos)
            upper_bounds.append(compute_upper_bound(counts))
            sample_counts.append(len(counts))

    return positions, upper_bounds, sample_counts


def plot_trimming_curves(guesses, sorted_exps, n_total):
    """Plot entropy curves for different trimming approaches."""
    print("\n" + "=" * 60)
    print("PLOTTING TRIMMING CURVES")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Plot 1: Remove worst performers ===
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    for i, keep_pct in enumerate([100, 90, 80, 70, 60, 50]):
        n_keep = int(n_total * keep_pct / 100)
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[:n_keep]}
        positions, upper_bounds, _ = compute_full_curve_for_experiments(guesses, exp_ids)
        if positions:
            ax.plot(positions, upper_bounds, '-', color=colors[i], linewidth=1.5,
                    label=f'Top {keep_pct}%', alpha=0.8)

    ax.set_xlabel('Position')
    ax.set_ylabel('Upper Bound (bpc)')
    ax.set_title('Remove Worst Performers')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.5)

    # === Plot 2: Remove best performers ===
    ax = axes[0, 1]
    colors = plt.cm.plasma(np.linspace(0, 1, 6))

    for i, remove_pct in enumerate([0, 5, 10, 15, 20, 25]):
        n_remove = int(n_total * remove_pct / 100)
        n_keep = n_total - n_remove
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[n_remove:]}
        positions, upper_bounds, _ = compute_full_curve_for_experiments(guesses, exp_ids)
        if positions:
            label = f'Remove {remove_pct}%' if remove_pct > 0 else 'All data'
            ax.plot(positions, upper_bounds, '-', color=colors[i], linewidth=1.5,
                    label=label, alpha=0.8)

    ax.set_xlabel('Position')
    ax.set_ylabel('Upper Bound (bpc)')
    ax.set_title('Remove Best Performers (Lucky Guessers)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.5)

    # === Plot 3: Symmetric trim ===
    ax = axes[1, 0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, 6))

    for i, trim_pct in enumerate([0, 5, 10, 15, 20, 25]):
        n_trim = int(n_total * trim_pct / 100)
        n_keep = n_total - 2 * n_trim
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[n_trim:n_total - n_trim]}
        positions, upper_bounds, _ = compute_full_curve_for_experiments(guesses, exp_ids)
        if positions:
            label = f'{trim_pct}% each side' if trim_pct > 0 else 'All data'
            ax.plot(positions, upper_bounds, '-', color=colors[i], linewidth=1.5,
                    label=label, alpha=0.8)

    ax.set_xlabel('Position')
    ax.set_ylabel('Upper Bound (bpc)')
    ax.set_title('Symmetric Trim (Both Edges)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.5)

    # === Plot 4: Asymmetric trim ===
    ax = axes[1, 1]
    asymmetric_trims = [
        (0, 0, 'All data'),
        (5, 20, '-5% best, -20% worst'),
        (5, 30, '-5% best, -30% worst'),
        (10, 20, '-10% best, -20% worst'),
        (10, 30, '-10% best, -30% worst'),
        (10, 40, '-10% best, -40% worst'),
    ]
    colors = plt.cm.cividis(np.linspace(0, 1, len(asymmetric_trims)))

    for i, (trim_top, trim_bottom, label) in enumerate(asymmetric_trims):
        n_trim_top = int(n_total * trim_top / 100)
        n_trim_bottom = int(n_total * trim_bottom / 100)
        n_keep = n_total - n_trim_top - n_trim_bottom
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[n_trim_top:n_total - n_trim_bottom]}
        positions, upper_bounds, _ = compute_full_curve_for_experiments(guesses, exp_ids)
        if positions:
            ax.plot(positions, upper_bounds, '-', color=colors[i], linewidth=1.5,
                    label=label, alpha=0.8)

    ax.set_xlabel('Position')
    ax.set_ylabel('Upper Bound (bpc)')
    ax.set_title('Asymmetric Trim')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.5)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'trimming_curves.pdf'
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


def analyze_by_performance_tier(guesses, experiments, min_samples=200):
    """
    Analyze entropy by user performance tier with various trimming approaches.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS BY PERFORMANCE TIER")
    print("=" * 60)

    # Calculate performance per experiment (average guesses per character)
    exp_performance = defaultdict(lambda: {'total_guesses': 0, 'positions': 0})

    current_exp = None
    current_pos = None
    count = 0

    for g in guesses:
        if g.experiment_result_id != current_exp or g.position != current_pos:
            if current_exp is not None and count > 0:
                exp_performance[current_exp]['total_guesses'] += count
                exp_performance[current_exp]['positions'] += 1
            current_exp = g.experiment_result_id
            current_pos = g.position
            count = 0
        count += 1
        if g.is_correct:
            exp_performance[current_exp]['total_guesses'] += count
            exp_performance[current_exp]['positions'] += 1
            count = 0

    # Calculate average guesses per position for each experiment
    exp_avg_guesses = {}
    for exp_id, data in exp_performance.items():
        if data['positions'] > 0:
            exp_avg_guesses[exp_id] = data['total_guesses'] / data['positions']

    if not exp_avg_guesses:
        print("No experiment data to analyze.")
        return

    # Sort by performance (lower avg guesses = better)
    sorted_exps = sorted(exp_avg_guesses.items(), key=lambda x: x[1])
    n_total = len(sorted_exps)

    print(f"\nTotal experiments with data: {n_total}")
    print(f"Best performer: {sorted_exps[0][1]:.2f} avg guesses/position")
    print(f"Worst performer: {sorted_exps[-1][1]:.2f} avg guesses/position")
    print(f"Median performer: {sorted_exps[n_total//2][1]:.2f} avg guesses/position")

    # Distribution of performance
    perfs = [p for _, p in sorted_exps]
    print(f"25th percentile: {perfs[n_total//4]:.2f}")
    print(f"75th percentile: {perfs[3*n_total//4]:.2f}")

    english_h90 = 1.405  # English upper bound at position 90
    english_h_final = 1.22  # English final entropy estimate

    # === APPROACH 1: Remove worst performers (keep top X%) ===
    print("\n" + "-" * 60)
    print("APPROACH 1: Remove worst performers (keep top X%)")
    print("-" * 60)
    print(f"{'Keep':<12} {'Exps':>6} {'n@90':>6} {'H@70':>8} {'H@80':>8} {'H@90':>8} {'Est.h':>8}")
    print("-" * 60)

    for keep_pct in [100, 90, 80, 70, 60, 50]:
        n_keep = int(n_total * keep_pct / 100)
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[:n_keep]}
        r = compute_entropy_for_experiments(guesses, exp_ids)
        h90_str = f"{r['upper_90']:.3f}" if r['upper_90'] else "N/A"
        h80_str = f"{r['upper_80']:.3f}" if r['upper_80'] else "N/A"
        h70_str = f"{r['upper_70']:.3f}" if r['upper_70'] else "N/A"
        est_h = f"{english_h_final * r['upper_90'] / english_h90:.2f}" if r['upper_90'] else "N/A"
        print(f"Top {keep_pct}%{'':<6} {n_keep:>6} {r['samples_90']:>6} {h70_str:>8} {h80_str:>8} {h90_str:>8} {est_h:>8}")

    # === APPROACH 2: Remove best performers (remove lucky guessers) ===
    print("\n" + "-" * 60)
    print("APPROACH 2: Remove best performers (remove top X%)")
    print("-" * 60)
    print(f"{'Remove':<12} {'Exps':>6} {'n@90':>6} {'H@70':>8} {'H@80':>8} {'H@90':>8} {'Est.h':>8}")
    print("-" * 60)

    for remove_pct in [0, 5, 10, 15, 20, 25]:
        n_remove = int(n_total * remove_pct / 100)
        n_keep = n_total - n_remove
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[n_remove:]}
        r = compute_entropy_for_experiments(guesses, exp_ids)
        h90_str = f"{r['upper_90']:.3f}" if r['upper_90'] else "N/A"
        h80_str = f"{r['upper_80']:.3f}" if r['upper_80'] else "N/A"
        h70_str = f"{r['upper_70']:.3f}" if r['upper_70'] else "N/A"
        est_h = f"{english_h_final * r['upper_90'] / english_h90:.2f}" if r['upper_90'] else "N/A"
        label = f"Remove {remove_pct}%" if remove_pct > 0 else "None"
        print(f"{label:<12} {n_keep:>6} {r['samples_90']:>6} {h70_str:>8} {h80_str:>8} {h90_str:>8} {est_h:>8}")

    # === APPROACH 3: Trim from both edges ===
    print("\n" + "-" * 60)
    print("APPROACH 3: Trim from both edges (symmetric)")
    print("-" * 60)
    print(f"{'Trim':<12} {'Exps':>6} {'n@90':>6} {'H@70':>8} {'H@80':>8} {'H@90':>8} {'Est.h':>8}")
    print("-" * 60)

    for trim_pct in [0, 5, 10, 15, 20, 25]:
        n_trim = int(n_total * trim_pct / 100)
        n_keep = n_total - 2 * n_trim
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[n_trim:n_total - n_trim]}
        r = compute_entropy_for_experiments(guesses, exp_ids)
        h90_str = f"{r['upper_90']:.3f}" if r['upper_90'] else "N/A"
        h80_str = f"{r['upper_80']:.3f}" if r['upper_80'] else "N/A"
        h70_str = f"{r['upper_70']:.3f}" if r['upper_70'] else "N/A"
        est_h = f"{english_h_final * r['upper_90'] / english_h90:.2f}" if r['upper_90'] else "N/A"
        label = f"{trim_pct}% each" if trim_pct > 0 else "None"
        print(f"{label:<12} {n_keep:>6} {r['samples_90']:>6} {h70_str:>8} {h80_str:>8} {h90_str:>8} {est_h:>8}")

    # === APPROACH 4: Asymmetric trim (more from bottom, less from top) ===
    print("\n" + "-" * 60)
    print("APPROACH 4: Asymmetric trim (more worst, less best)")
    print("-" * 60)
    print(f"{'Trim':<16} {'Exps':>6} {'n@90':>6} {'H@70':>8} {'H@80':>8} {'H@90':>8} {'Est.h':>8}")
    print("-" * 64)

    asymmetric_trims = [
        (5, 20),   # remove 5% best, 20% worst
        (5, 30),   # remove 5% best, 30% worst
        (10, 20),  # remove 10% best, 20% worst
        (10, 30),  # remove 10% best, 30% worst
        (10, 40),  # remove 10% best, 40% worst
    ]

    for trim_top, trim_bottom in asymmetric_trims:
        n_trim_top = int(n_total * trim_top / 100)
        n_trim_bottom = int(n_total * trim_bottom / 100)
        n_keep = n_total - n_trim_top - n_trim_bottom
        if n_keep < 50:
            continue
        exp_ids = {exp_id for exp_id, _ in sorted_exps[n_trim_top:n_total - n_trim_bottom]}
        r = compute_entropy_for_experiments(guesses, exp_ids)
        h90_str = f"{r['upper_90']:.3f}" if r['upper_90'] else "N/A"
        h80_str = f"{r['upper_80']:.3f}" if r['upper_80'] else "N/A"
        h70_str = f"{r['upper_70']:.3f}" if r['upper_70'] else "N/A"
        est_h = f"{english_h_final * r['upper_90'] / english_h90:.2f}" if r['upper_90'] else "N/A"
        label = f"-{trim_top}%top -{trim_bottom}%bot"
        print(f"{label:<16} {n_keep:>6} {r['samples_90']:>6} {h70_str:>8} {h80_str:>8} {h90_str:>8} {est_h:>8}")

    print("\n" + "-" * 60)
    print("Note: Est.h = estimated Ukrainian entropy using ratio to English")
    print(f"      English H@90 ≈ {english_h90:.3f} bpc, English h ≈ {english_h_final:.2f} bpc")
    print("-" * 60)

    # Plot curves for different trimming approaches
    plot_trimming_curves(guesses, sorted_exps, n_total)


def analyze_sentence_usage(sentences, experiments):
    """Analyze how many times each sentence was used in finished/unfinished experiments."""
    print("\n" + "=" * 60)
    print("SENTENCE USAGE ANALYSIS")
    print("=" * 60)

    # Count usage per sentence
    sentence_stats = {s.id: {'text': s.text[:50] + '...' if len(s.text) > 50 else s.text,
                             'length': len(s.text),
                             'finished': 0,
                             'unfinished': 0,
                             'total': 0}
                      for s in sentences}

    for exp in experiments:
        if exp.sentence_id in sentence_stats:
            sentence_stats[exp.sentence_id]['total'] += 1
            if exp.finished:
                sentence_stats[exp.sentence_id]['finished'] += 1
            else:
                sentence_stats[exp.sentence_id]['unfinished'] += 1

    # Print summary only (full table omitted for brevity)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sentence_ids = list(sentence_stats.keys())
    finished_counts = [sentence_stats[sid]['finished'] for sid in sentence_ids]
    unfinished_counts = [sentence_stats[sid]['unfinished'] for sid in sentence_ids]
    total_counts = [sentence_stats[sid]['total'] for sid in sentence_ids]
    lengths = [sentence_stats[sid]['length'] for sid in sentence_ids]

    # Plot 1: Bar chart of usage by sentence
    ax = axes[0, 0]
    x = np.arange(len(sentence_ids))
    width = 0.35
    ax.bar(x - width/2, finished_counts, width, label='Finished', color='green', alpha=0.7)
    ax.bar(x + width/2, unfinished_counts, width, label='Unfinished', color='red', alpha=0.7)
    ax.set_xlabel('Sentence ID')
    ax.set_ylabel('Number of experiments')
    ax.set_title('Experiment Count by Sentence')
    ax.set_xticks(x)
    ax.set_xticklabels(sentence_ids)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Completion rate by sentence
    ax = axes[0, 1]
    completion_rates = [f / t * 100 if t > 0 else 0 for f, t in zip(finished_counts, total_counts)]
    bars = ax.bar(sentence_ids, completion_rates, color='blue', alpha=0.7)
    ax.axhline(y=np.mean(completion_rates), color='red', linestyle='--', label=f'Mean: {np.mean(completion_rates):.1f}%')
    ax.set_xlabel('Sentence ID')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Completion Rate by Sentence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Sentence length vs total usage
    ax = axes[1, 0]
    ax.scatter(lengths, total_counts, s=100, alpha=0.7, c=completion_rates, cmap='RdYlGn')
    for i, sid in enumerate(sentence_ids):
        ax.annotate(str(sid), (lengths[i], total_counts[i]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel('Sentence Length (characters)')
    ax.set_ylabel('Total Experiments')
    ax.set_title('Sentence Length vs Usage (color = completion rate)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Sentence length vs completion rate
    ax = axes[1, 1]
    ax.scatter(lengths, completion_rates, s=100, alpha=0.7, c='blue')
    for i, sid in enumerate(sentence_ids):
        ax.annotate(str(sid), (lengths[i], completion_rates[i]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel('Sentence Length (characters)')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Sentence Length vs Completion Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'sentence_usage.pdf'
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved: {output_path}")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Total sentences: {len(sentences)}")
    print(f"Total experiments: {sum(total_counts)}")
    print(f"Mean experiments per sentence: {np.mean(total_counts):.1f}")
    print(f"Std dev: {np.std(total_counts):.1f}")
    print(f"Min/Max usage: {min(total_counts)} / {max(total_counts)}")

    return sentence_stats


def plot_observations_by_position(guesses, experiments):
    """Plot observations (unique experiment×position pairs) by position, split by finished/unfinished."""
    print("\n" + "=" * 60)
    print("OBSERVATIONS BY POSITION")
    print("=" * 60)

    finished_exp_ids = {e.id for e in experiments if e.finished}

    # Count distinct experiments that correctly guessed each position
    # (only correct guesses constitute observations)
    from collections import defaultdict
    finished_obs = defaultdict(set)
    unfinished_obs = defaultdict(set)

    for g in guesses:
        if not g.is_correct:
            continue
        if g.experiment_result_id in finished_exp_ids:
            finished_obs[g.position].add(g.experiment_result_id)
        else:
            unfinished_obs[g.position].add(g.experiment_result_id)

    all_positions = sorted(set(finished_obs.keys()) | set(unfinished_obs.keys()))
    finished_counts = [len(finished_obs[p]) for p in all_positions]
    unfinished_counts = [len(unfinished_obs[p]) for p in all_positions]
    total_counts = [f + u for f, u in zip(finished_counts, unfinished_counts)]

    total_finished = sum(finished_counts)
    total_unfinished = sum(unfinished_counts)
    print(f"Total observations from finished experiments:   {total_finished:,}")
    print(f"Total observations from unfinished experiments: {total_unfinished:,}")
    print(f"Total observations:                             {total_finished + total_unfinished:,}")

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(all_positions, finished_counts, label=f'Finished ({total_finished:,})', color='green', alpha=0.8)
    ax.bar(all_positions, unfinished_counts, bottom=finished_counts,
           label=f'Unfinished ({total_unfinished:,})', color='red', alpha=0.8)

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Observations (experiments reaching position)', fontsize=12)
    ax.set_title('Observations by Position and Completion Status', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    output_path = Path(__file__).parent / 'graphs' / 'observations_by_position.pdf'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


def plot_first_guess_accuracy(guesses):
    """Plot first-guess accuracy (P(correct on guess 1)) by position."""
    print("\n" + "=" * 60)
    print("FIRST-GUESS ACCURACY BY POSITION")
    print("=" * 60)

    # For each position, count experiments where guess_number=1 was correct vs total experiments
    first_correct = defaultdict(int)
    first_total = defaultdict(int)

    seen = set()
    for g in guesses:
        key = (g.experiment_result_id, g.position)
        if key in seen:
            continue
        seen.add(key)
        # First guess at this position for this experiment
        first_total[g.position] += 1
        if g.is_correct:
            first_correct[g.position] += 1

    positions = sorted(first_total.keys())
    accuracy = [first_correct[p] / first_total[p] * 100 for p in positions]
    samples = [first_total[p] for p in positions]

    # Print summary for positions with 200+ samples
    cutoff = next((p for p in positions if first_total[p] < 200), positions[-1] + 1)
    print(f"\n{'Pos':>4} {'Samples':>8} {'Accuracy':>9}")
    print("-" * 24)
    for p, acc, n in zip(positions, accuracy, samples):
        if p <= min(positions[0] + 4, cutoff) or p == cutoff - 1:
            print(f"{p:>4} {n:>8} {acc:>8.1f}%")
    print(f"  ...")
    print(f"Mean accuracy (pos {positions[0]}-{cutoff-1}): {sum(accuracy[:positions.index(cutoff)])/len(accuracy[:positions.index(cutoff)]):.1f}%")

    # Match Ren et al. Figure 3 style: scatter, probability (0-1), trimmed to good data
    max_pos = 110
    plot_positions = [p for p in positions if p <= max_pos]
    plot_accuracy = [first_correct[p] / first_total[p] for p in plot_positions]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_positions, plot_accuracy, color='blue', s=15, zorder=5)
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel('probability', fontsize=12)
    ax.set_ylim(0.15, 0.85)
    ax.set_xlim(68, max_pos + 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'first_guess_accuracy.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def precompute_guess_counts_by_experiment(guesses):
    """
    Pre-compute position -> guess_count for each experiment.
    Returns dict: exp_id -> {position: guess_count}
    This is the expensive step - done ONCE, then reused across all bootstrap iterations.
    """
    exp_pos_counts = defaultdict(dict)
    current_exp = None
    current_pos = None
    count = 0

    for g in guesses:
        if g.experiment_result_id != current_exp or g.position != current_pos:
            current_exp = g.experiment_result_id
            current_pos = g.position
            count = 0
        count += 1
        if g.is_correct:
            exp_pos_counts[g.experiment_result_id][g.position] = count

    return dict(exp_pos_counts)


def compute_h_from_precomputed(exp_ids_resampled, exp_pos_counts, min_samples=50):
    """
    Compute entropy estimate from pre-computed per-experiment data.
    exp_ids_resampled: list of experiment IDs (may contain duplicates from resampling)
    exp_pos_counts: dict from precompute_guess_counts_by_experiment()
    """
    # Merge: position -> [guess_counts] across all resampled experiments
    pos_counts = defaultdict(list)
    for eid in exp_ids_resampled:
        for pos, gc in exp_pos_counts.get(eid, {}).items():
            pos_counts[pos].append(gc)

    # Compute upper bounds at positions with enough samples
    position_bounds = {}
    for pos, counts in pos_counts.items():
        if len(counts) >= min_samples:
            position_bounds[pos] = compute_upper_bound(counts)

    if not position_bounds:
        return None, None, None

    mean_upper = np.mean(list(position_bounds.values()))
    upper_at_90 = position_bounds.get(90, None)
    upper_at_70 = position_bounds.get(70, None)

    return mean_upper, upper_at_90, upper_at_70


def detect_cheaters_binomial(guesses, exp_pos_counts, p_threshold=0.01, min_positions=5):
    """
    Detect suspiciously good sessions via binomial test.
    For each session, tests whether the number of first-try correct guesses
    is significantly higher than expected given the per-position population accuracy.
    Returns set of flagged experiment IDs.
    """
    print("\n" + "=" * 60)
    print(f"BINOMIAL CHEATER DETECTION (p < {p_threshold})")
    print("=" * 60)

    # Compute empirical first-guess accuracy per position
    first_correct = defaultdict(int)
    first_total = defaultdict(int)
    current_exp = None
    current_pos = None
    is_first = True
    for g in guesses:
        if g.experiment_result_id != current_exp or g.position != current_pos:
            current_exp = g.experiment_result_id
            current_pos = g.position
            is_first = True
        if is_first:
            first_total[g.position] += 1
            if g.is_correct:
                first_correct[g.position] += 1
            is_first = False

    cheater_ids = set()
    flagged_details = []
    for exp_id, pos_counts in exp_pos_counts.items():
        if not pos_counts or len(pos_counts) < min_positions:
            continue
        total = len(pos_counts)
        first_try = sum(1 for gc in pos_counts.values() if gc == 1)
        positions = list(pos_counts.keys())
        avg_p = sum(first_correct[p] / first_total[p] for p in positions) / len(positions)
        p_value = binom.sf(first_try - 1, total, avg_p)
        if p_value < p_threshold:
            cheater_ids.add(exp_id)
            flagged_details.append((exp_id, total, first_try, first_try / total, avg_p, p_value))

    flagged_details.sort(key=lambda x: x[5])

    print(f"\nFlagged: {len(cheater_ids)} sessions ({len(cheater_ids) / len(exp_pos_counts) * 100:.1f}%)")
    print(f"\n{'ExpID':>6} {'Pos':>4} {'1st try':>7} {'Rate':>6} {'Avg p':>6} {'P-value':>12}")
    print("-" * 48)
    for eid, total, ft, rate, avg_p, pv in flagged_details:
        print(f"{eid:>6} {total:>4} {ft:>7} {rate:>6.1%} {avg_p:>6.1%} {pv:>12.2e}")

    return cheater_ids


def bootstrap_obs_weighted(exp_pos_counts, sorted_exps, bottom_trim_pcts,
                           n_bootstrap=2000, min_samples=50, seed=42):
    """
    Observation-weighted bootstrap with cheater-removed pool.
    sorted_exps: list of (exp_id, avg_guesses) already sorted and cheater-free.
    bottom_trim_pcts: list of bottom trim percentages to test.
    """
    print("\n" + "=" * 60)
    print("OBSERVATION-WEIGHTED BOOTSTRAP")
    print(f"({n_bootstrap} iterations)")
    print("=" * 60)

    rng = np.random.default_rng(seed=seed)
    n_remaining = len(sorted_exps)
    results = {}

    for bot_pct in bottom_trim_pcts:
        nb = int(n_remaining * bot_pct / 100)
        pool = sorted_exps[:n_remaining - nb if nb > 0 else n_remaining]
        pool_ids = [eid for eid, _ in pool]
        pool_arr = np.array(pool_ids)
        n_pool = len(pool_arr)

        # Point estimate: full-pool weighted mean (no resampling)
        full_pos_counts = defaultdict(list)
        for eid in pool_ids:
            for pos, gc in exp_pos_counts.get(eid, {}).items():
                full_pos_counts[pos].append(gc)
        full_bounds = {p: compute_upper_bound(c) for p, c in full_pos_counts.items() if len(c) >= min_samples}
        full_tot = sum(len(full_pos_counts[p]) for p in full_bounds)
        point_est = sum(full_bounds[p] * len(full_pos_counts[p]) for p in full_bounds) / full_tot if full_tot else float('nan')

        samples = []
        for i in range(n_bootstrap):
            if (i + 1) % 500 == 0:
                print(f"  {bot_pct}% trim: iteration {i + 1}/{n_bootstrap}...")
            indices = rng.integers(0, n_pool, size=n_pool)
            resampled = pool_arr[indices]

            pos_counts = defaultdict(list)
            for eid in resampled:
                for pos, gc in exp_pos_counts.get(eid, {}).items():
                    pos_counts[pos].append(gc)

            position_bounds = {}
            for pos, counts in pos_counts.items():
                if len(counts) >= min_samples:
                    position_bounds[pos] = compute_upper_bound(counts)

            if not position_bounds:
                continue

            total_obs = sum(len(pos_counts[p]) for p in position_bounds)
            wtd_mean = sum(position_bounds[p] * len(pos_counts[p]) for p in position_bounds) / total_obs
            samples.append(wtd_mean)

        arr = np.array(samples)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        med = np.median(arr)
        results[bot_pct] = {'point': point_est, 'median': med, 'ci_low': lo, 'ci_high': hi,
                            'width': hi - lo, 'pool': n_pool}
        print(f"  Bottom {bot_pct}%: pool={n_pool}, point={point_est:.3f}, median={med:.3f} [{lo:.3f}, {hi:.3f}] width={hi - lo:.3f}")

    # Summary table
    print(f"\n{'Bottom trim':<12} {'Pool':>5} {'Point':>8} {'Median':>8} {'95% CI':>20} {'Width':>7}")
    print("-" * 65)
    for bot_pct in bottom_trim_pcts:
        r = results[bot_pct]
        print(f"{bot_pct:>10}%  {r['pool']:>5} {r['point']:>8.3f} {r['median']:>8.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['width']:>7.3f}")

    return results


def plot_trim_sensitivity(obs_weighted_results):
    """Plot point estimate with 95% CI error bars across trim levels."""
    trims = sorted(obs_weighted_results.keys())
    points = [obs_weighted_results[t]['point'] for t in trims]
    medians = [obs_weighted_results[t]['median'] for t in trims]
    ci_low = [obs_weighted_results[t]['ci_low'] for t in trims]
    ci_high = [obs_weighted_results[t]['ci_high'] for t in trims]
    pools = [obs_weighted_results[t]['pool'] for t in trims]

    err_low = [m - l for m, l in zip(medians, ci_low)]
    err_high = [h - m for m, h in zip(medians, ci_high)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(trims, medians, yerr=[err_low, err_high], fmt='s-', color='coral',
                ecolor='gray', elinewidth=1.2, capsize=4, markersize=5,
                label='Bootstrap median (95% CI)')
    ax.plot(trims, points, 'o', color='steelblue', markersize=6, label='Point estimate')

    for t, m, n in zip(trims, medians, pools):
        ax.annotate(f'n={n}', xy=(t, m), xytext=(0, -14), textcoords='offset points',
                    ha='center', fontsize=8, color='dimgray')

    ax.set_xlabel('Bottom trim (%)')
    ax.set_ylabel('Upper bound $H_{upper}$ (bpc)')
    ax.set_title('Trim sensitivity: entropy upper bound vs bottom-trim level')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xticks(trims)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'trim_sensitivity.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def _run_bootstrap_loop(rng, pool_ids, exp_pos_counts, n_bootstrap, min_samples, label=""):
    """Core bootstrap loop shared by all bootstrap variants."""
    n_pool = len(pool_ids)
    pool_arr = np.array(pool_ids)

    mean_upper_samples = []
    upper_90_samples = []
    upper_70_samples = []

    for i in range(n_bootstrap):
        if label and (i + 1) % 500 == 0:
            print(f"  {label}: iteration {i+1}/{n_bootstrap}...")

        # Resample with replacement — numpy generates indices fast
        indices = rng.integers(0, n_pool, size=n_pool)
        resampled_ids = pool_arr[indices]

        mean_u, u90, u70 = compute_h_from_precomputed(resampled_ids, exp_pos_counts, min_samples)
        if mean_u is not None:
            mean_upper_samples.append(mean_u)
        if u90 is not None:
            upper_90_samples.append(u90)
        if u70 is not None:
            upper_70_samples.append(u70)

    return mean_upper_samples, upper_90_samples, upper_70_samples


def bootstrap_by_session(guesses, experiments, n_bootstrap=2000, min_samples=50):
    """
    Bootstrap resampling at the session (experiment) level.
    Answers: how stable is our entropy estimate given participant variability?
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAP: SESSION-LEVEL RESAMPLING")
    print(f"({n_bootstrap} iterations)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)
    exp_ids = list(exp_pos_counts.keys())

    print(f"Total experiments to resample from: {len(exp_ids)}")

    mean_upper_samples, upper_90_samples, upper_70_samples = _run_bootstrap_loop(
        rng, exp_ids, exp_pos_counts, n_bootstrap, min_samples, label="Session"
    )

    results = _print_bootstrap_results("Session-level", mean_upper_samples, upper_90_samples, upper_70_samples)
    return results


def bootstrap_by_trimming(guesses, experiments, n_bootstrap=2000, min_samples=50):
    """
    Bootstrap within different trimming levels.
    Answers: does trimming reduce or increase estimate variance?
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAP: TRIMMING COMPARISON")
    print(f"({n_bootstrap} iterations)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)

    # Compute performance per experiment (average guesses per position)
    exp_avg_guesses = {}
    for exp_id, pos_counts in exp_pos_counts.items():
        if pos_counts:
            exp_avg_guesses[exp_id] = sum(pos_counts.values()) / len(pos_counts)

    sorted_exps = sorted(exp_avg_guesses.items(), key=lambda x: x[1])
    n_total = len(sorted_exps)

    trim_configs = [
        # No trim
        ('All data', 0, 0),
        # Symmetric
        ('Sym 5%', 5, 5),
        ('Sym 10%', 10, 10),
        ('Sym 15%', 15, 15),
        ('Sym 20%', 20, 20),
        ('Sym 25%', 25, 25),
        ('Sym 30%', 30, 30),
        # Asymmetric (top% best / bottom% worst)
        ('-5%top -10%bot', 5, 10),
        ('-5%top -15%bot', 5, 15),
        ('-5%top -20%bot', 5, 20),
        ('-5%top -25%bot', 5, 25),
        ('-5%top -30%bot', 5, 30),
        ('-10%top -15%bot', 10, 15),
        ('-10%top -20%bot', 10, 20),
        ('-10%top -25%bot', 10, 25),
        ('-10%top -30%bot', 10, 30),
        ('-10%top -40%bot', 10, 40),
        # Bottom-only (remove worst X%)
        ('Top 90%', 0, 10),
        ('Top 80%', 0, 20),
        ('Top 70%', 0, 30),
        ('Top 60%', 0, 40),
        ('Top 50%', 0, 50),
    ]

    all_results = {}

    for label, trim_top_pct, trim_bottom_pct in trim_configs:
        n_trim_top = int(n_total * trim_top_pct / 100)
        n_trim_bottom = int(n_total * trim_bottom_pct / 100)

        pool_exps = sorted_exps[n_trim_top:n_total - n_trim_bottom if n_trim_bottom > 0 else n_total]
        pool_ids = [eid for eid, _ in pool_exps]
        n_pool = len(pool_ids)

        if n_pool < 50:
            print(f"\n{label}: Too few experiments ({n_pool}), skipping")
            continue

        print(f"\n--- {label} (pool: {n_pool} experiments) ---")

        mean_upper_samples, upper_90_samples, _ = _run_bootstrap_loop(
            rng, pool_ids, exp_pos_counts, n_bootstrap, min_samples
        )

        if mean_upper_samples:
            ci_low = np.percentile(mean_upper_samples, 2.5)
            ci_high = np.percentile(mean_upper_samples, 97.5)
            median = np.median(mean_upper_samples)
            ci_width = ci_high - ci_low
            print(f"  Mean upper bound: {median:.3f} [{ci_low:.3f}, {ci_high:.3f}] (width: {ci_width:.3f})")
            all_results[label] = {
                'median': median, 'ci_low': ci_low, 'ci_high': ci_high,
                'ci_width': ci_width, 'samples': mean_upper_samples,
                'n_valid': len(mean_upper_samples)
            }
        if upper_90_samples:
            ci_low = np.percentile(upper_90_samples, 2.5)
            ci_high = np.percentile(upper_90_samples, 97.5)
            print(f"  Upper bound @90:  {np.median(upper_90_samples):.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Trimming':<28} {'Median':>8} {'95% CI':>18} {'Width':>8}")
    print("-" * 70)
    for label, r in all_results.items():
        print(f"{label:<28} {r['median']:>8.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['ci_width']:>8.3f}")

    return all_results


def bootstrap_by_sentence(guesses, experiments, sentences, n_bootstrap=2000, min_samples=50):
    """
    Bootstrap resampling at the sentence level.
    Answers: how much does sentence/text choice drive the entropy estimate?
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAP: SENTENCE-LEVEL RESAMPLING")
    print(f"({n_bootstrap} iterations)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # Pre-compute per-experiment position counts
    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)

    # Map experiment -> sentence
    exp_to_sentence = {exp.id: exp.sentence_id for exp in experiments}

    # Group experiment IDs by sentence
    exps_by_sentence = defaultdict(list)
    for exp_id in exp_pos_counts:
        sid = exp_to_sentence.get(exp_id)
        if sid is not None:
            exps_by_sentence[sid].append(exp_id)

    sentence_ids = [sid for sid in exps_by_sentence if len(exps_by_sentence[sid]) > 0]
    n_sentences = len(sentence_ids)
    sentence_arr = np.array(sentence_ids)

    print(f"Total sentences with data: {n_sentences}")
    exp_counts = [len(exps_by_sentence[s]) for s in sentence_ids]
    print(f"Experiments per sentence: {min(exp_counts)} - {max(exp_counts)}")

    mean_upper_samples = []
    upper_90_samples = []
    upper_70_samples = []

    for i in range(n_bootstrap):
        if (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}/{n_bootstrap}...")

        # Resample sentence IDs with replacement
        indices = rng.integers(0, n_sentences, size=n_sentences)
        resampled_sids = sentence_arr[indices]

        # Collect all experiment IDs from resampled sentences
        resampled_exp_ids = []
        for sid in resampled_sids:
            resampled_exp_ids.extend(exps_by_sentence[sid])

        mean_u, u90, u70 = compute_h_from_precomputed(resampled_exp_ids, exp_pos_counts, min_samples)
        if mean_u is not None:
            mean_upper_samples.append(mean_u)
        if u90 is not None:
            upper_90_samples.append(u90)
        if u70 is not None:
            upper_70_samples.append(u70)

    results = _print_bootstrap_results("Sentence-level", mean_upper_samples, upper_90_samples, upper_70_samples)
    return results


def _print_bootstrap_results(label, mean_upper_samples, upper_90_samples, upper_70_samples):
    """Print and return bootstrap results."""
    english_h90 = 1.405
    english_h_final = 1.22

    results = {}

    if mean_upper_samples:
        arr = np.array(mean_upper_samples)
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        median = np.median(arr)
        print(f"\n  Mean upper bound across positions:")
        print(f"    Median:  {median:.4f} bpc")
        print(f"    95% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"    Width:   {ci_high - ci_low:.4f}")
        print(f"    Std:     {np.std(arr):.4f}")
        print(f"    Valid iterations: {len(arr)}/{len(mean_upper_samples)}")
        results['mean_upper'] = {'median': median, 'ci_low': ci_low, 'ci_high': ci_high, 'samples': arr}

    if upper_90_samples:
        arr = np.array(upper_90_samples)
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        median = np.median(arr)
        # Ratio-based estimate
        est_h_samples = english_h_final * arr / english_h90
        est_ci_low, est_ci_high = np.percentile(est_h_samples, [2.5, 97.5])
        est_median = np.median(est_h_samples)
        print(f"\n  Upper bound at position 90:")
        print(f"    Median:  {median:.4f} bpc")
        print(f"    95% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"    Width:   {ci_high - ci_low:.4f}")
        print(f"    Ratio-based h estimate: {est_median:.3f} [{est_ci_low:.3f}, {est_ci_high:.3f}]")
        results['upper_90'] = {'median': median, 'ci_low': ci_low, 'ci_high': ci_high, 'samples': arr}
        results['ratio_h'] = {'median': est_median, 'ci_low': est_ci_low, 'ci_high': est_ci_high, 'samples': est_h_samples}

    if upper_70_samples:
        arr = np.array(upper_70_samples)
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        median = np.median(arr)
        print(f"\n  Upper bound at position 70:")
        print(f"    Median:  {median:.4f} bpc")
        print(f"    95% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"    Width:   {ci_high - ci_low:.4f}")
        results['upper_70'] = {'median': median, 'ci_low': ci_low, 'ci_high': ci_high, 'samples': arr}

    return results


def plot_bootstrap_results(session_results, trimming_results, sentence_results):
    """Create comprehensive bootstrap visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Plot 1: Session-level bootstrap distributions ===
    ax = axes[0, 0]
    if 'mean_upper' in session_results:
        ax.hist(session_results['mean_upper']['samples'], bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        r = session_results['mean_upper']
        ax.axvline(r['median'], color='red', linewidth=2, label=f"Median: {r['median']:.3f}")
        ax.axvline(r['ci_low'], color='red', linestyle='--', linewidth=1, label=f"95% CI: [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
        ax.axvline(r['ci_high'], color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Upper Bound (bpc)')
    ax.set_ylabel('Count')
    ax.set_title('Session Bootstrap: Mean Upper Bound')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Plot 2: Session vs Sentence comparison at position 90 ===
    ax = axes[0, 1]
    plotted = False
    if 'upper_90' in session_results:
        ax.hist(session_results['upper_90']['samples'], bins=50, alpha=0.5, color='steelblue',
                label=f"Session (w={session_results['upper_90']['ci_high'] - session_results['upper_90']['ci_low']:.3f})", edgecolor='black', linewidth=0.3)
        plotted = True
    if 'upper_90' in sentence_results:
        ax.hist(sentence_results['upper_90']['samples'], bins=50, alpha=0.5, color='coral',
                label=f"Sentence (w={sentence_results['upper_90']['ci_high'] - sentence_results['upper_90']['ci_low']:.3f})", edgecolor='black', linewidth=0.3)
        plotted = True
    if plotted:
        ax.axvline(1.405, color='green', linestyle='--', linewidth=1.5, label='English H@90 (1.405)')
    ax.set_xlabel('Upper Bound at Position 90 (bpc)')
    ax.set_ylabel('Count')
    ax.set_title('Session vs Sentence Bootstrap @ Pos 90')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Plot 3: Trimming bootstrap comparison ===
    ax = axes[1, 0]
    if trimming_results:
        labels = []
        medians = []
        ci_lows = []
        ci_highs = []
        for label, r in trimming_results.items():
            labels.append(label)
            medians.append(r['median'])
            ci_lows.append(r['ci_low'])
            ci_highs.append(r['ci_high'])

        y_pos = np.arange(len(labels))
        ci_err_low = [m - l for m, l in zip(medians, ci_lows)]
        ci_err_high = [h - m for m, h in zip(medians, ci_highs)]

        ax.barh(y_pos, medians, xerr=[ci_err_low, ci_err_high], align='center',
                color='steelblue', alpha=0.7, capsize=5, ecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Mean Upper Bound (bpc)')
        ax.set_title('Trimming: Bootstrap 95% CI Comparison')
        ax.grid(True, alpha=0.3, axis='x')

        # Annotate widths
        for i, (m, l, h) in enumerate(zip(medians, ci_lows, ci_highs)):
            ax.annotate(f'w={h-l:.3f}', xy=(h + 0.005, i), va='center', fontsize=8, color='gray')

    # === Plot 4: Ratio-based h estimate distributions ===
    ax = axes[1, 1]
    plotted = False
    if 'ratio_h' in session_results:
        ax.hist(session_results['ratio_h']['samples'], bins=50, alpha=0.5, color='steelblue',
                label=f"Session: {session_results['ratio_h']['median']:.3f}", edgecolor='black', linewidth=0.3)
        r = session_results['ratio_h']
        ax.axvline(r['ci_low'], color='steelblue', linestyle='--', linewidth=1)
        ax.axvline(r['ci_high'], color='steelblue', linestyle='--', linewidth=1)
        plotted = True
    if 'ratio_h' in sentence_results:
        ax.hist(sentence_results['ratio_h']['samples'], bins=50, alpha=0.5, color='coral',
                label=f"Sentence: {sentence_results['ratio_h']['median']:.3f}", edgecolor='black', linewidth=0.3)
        r = sentence_results['ratio_h']
        ax.axvline(r['ci_low'], color='coral', linestyle='--', linewidth=1)
        ax.axvline(r['ci_high'], color='coral', linestyle='--', linewidth=1)
        plotted = True
    if plotted:

        ax.axvline(1.22, color='green', linestyle='--', linewidth=1.5, label='English h (1.22)')
    ax.set_xlabel('Estimated h (bpc)')
    ax.set_ylabel('Count')
    ax.set_title('Ratio-Based Ukrainian h Estimate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Bootstrap Confidence Intervals for Ukrainian Entropy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'bootstrap_ci.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_path}")


def bootstrap_per_position(guesses, experiments, n_bootstrap=2000, min_samples=50):
    """
    Bootstrap to get cleaned per-position entropy curves under different trimming approaches.
    Returns bootstrap median and CI at each position for each trimming config.
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAPPED ENTROPY BY POSITION")
    print(f"({n_bootstrap} iterations)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)

    # Compute performance per experiment
    exp_avg_guesses = {}
    for exp_id, pos_counts in exp_pos_counts.items():
        if pos_counts:
            exp_avg_guesses[exp_id] = sum(pos_counts.values()) / len(pos_counts)

    sorted_exps = sorted(exp_avg_guesses.items(), key=lambda x: x[1])
    n_total = len(sorted_exps)

    # All positions present in the data
    all_positions = sorted(set(
        pos for pc in exp_pos_counts.values() for pos in pc.keys()
    ))

    trim_configs = [
        ('All data', 0, 0, 'black'),
        ('Sym 10%', 10, 10, 'blue'),
        ('Sym 20%', 20, 20, 'green'),
        ('-10% best, -30% worst', 10, 30, 'red'),
        ('-5% best, -20% worst', 5, 20, 'purple'),
        ('Top 50%', 0, 50, 'orange'),
    ]

    all_results = {}

    for label, trim_top_pct, trim_bottom_pct, color in trim_configs:
        n_trim_top = int(n_total * trim_top_pct / 100)
        n_trim_bottom = int(n_total * trim_bottom_pct / 100)
        end_idx = n_total - n_trim_bottom if n_trim_bottom > 0 else n_total
        pool_ids = [eid for eid, _ in sorted_exps[n_trim_top:end_idx]]
        n_pool = len(pool_ids)

        if n_pool < 50:
            print(f"\n{label}: Too few experiments ({n_pool}), skipping")
            continue

        print(f"\n--- {label} (pool: {n_pool} experiments) ---")
        pool_arr = np.array(pool_ids)

        # Collect per-position upper bounds across all bootstrap iterations
        # position -> list of upper bound values
        pos_bootstrap = defaultdict(list)

        for i in range(n_bootstrap):
            if (i + 1) % 500 == 0:
                print(f"  Iteration {i+1}/{n_bootstrap}...")

            indices = rng.integers(0, n_pool, size=n_pool)
            resampled_ids = pool_arr[indices]

            # Merge position -> [guess_counts]
            pos_counts = defaultdict(list)
            for eid in resampled_ids:
                for pos, gc in exp_pos_counts.get(eid, {}).items():
                    pos_counts[pos].append(gc)

            for pos, counts in pos_counts.items():
                if len(counts) >= min_samples:
                    pos_bootstrap[pos].append(compute_upper_bound(counts))

        # Compute median and CI at each position
        positions = []
        medians = []
        ci_lows = []
        ci_highs = []

        for pos in sorted(pos_bootstrap.keys()):
            vals = np.array(pos_bootstrap[pos])
            if len(vals) >= n_bootstrap * 0.5:  # position appeared in at least half the iterations
                positions.append(pos)
                medians.append(np.median(vals))
                ci_lows.append(np.percentile(vals, 2.5))
                ci_highs.append(np.percentile(vals, 97.5))

        all_results[label] = {
            'positions': positions, 'medians': medians,
            'ci_lows': ci_lows, 'ci_highs': ci_highs,
            'color': color, 'n_pool': n_pool,
        }

        # Print summary for a few key positions
        for target_pos in [70, 80, 90, 100]:
            if target_pos in positions:
                idx = positions.index(target_pos)
                print(f"  Pos {target_pos}: {medians[idx]:.3f} [{ci_lows[idx]:.3f}, {ci_highs[idx]:.3f}]")

    # === Plot ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: all curves with CI bands
    for label, r in all_results.items():
        ax1.plot(r['positions'], r['medians'], '-', color=r['color'], linewidth=2,
                 label=f"{label} (n={r['n_pool']})", alpha=0.9)
        ax1.fill_between(r['positions'], r['ci_lows'], r['ci_highs'],
                         color=r['color'], alpha=0.1)

    ax1.set_xlabel('Position in sentence', fontsize=12)
    ax1.set_ylabel('Upper Bound (bpc)', fontsize=12)
    ax1.set_title('Bootstrapped Entropy by Position (median + 95% CI)', fontsize=13)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 2.8)

    # Right: CI width by position for each method
    for label, r in all_results.items():
        widths = [h - l for h, l in zip(r['ci_highs'], r['ci_lows'])]
        ax2.plot(r['positions'], widths, '-', color=r['color'], linewidth=1.5,
                 label=label, alpha=0.8)

    ax2.set_xlabel('Position in sentence', fontsize=12)
    ax2.set_ylabel('95% CI Width (bpc)', fontsize=12)
    ax2.set_title('Confidence Interval Width by Position', fontsize=13)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'bootstrapped_by_position.pdf'
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved: {output_path}")
    return all_results


def plot_character_frequencies():
    """Compare Ukrainian and English character frequency distributions and compute H1."""
    print("\n" + "=" * 60)
    print("CHARACTER FREQUENCY COMPARISON: UKRAINIAN vs ENGLISH")
    print("=" * 60)

    # Ukrainian letter frequencies (source: sttmedia.com, 2.66M char corpus)
    ukr_freqs = {
        'О': 9.28, 'А': 8.34, 'Н': 7.10, 'І': 6.23, 'И': 6.00,
        'В': 5.50, 'Р': 5.48, 'Т': 4.77, 'Е': 4.59, 'С': 4.57,
        'К': 4.00, 'Л': 3.93, 'У': 3.38, 'Д': 3.06, 'М': 3.02,
        'П': 2.84, 'Я': 2.16, 'З': 2.10, 'Ь': 1.83, 'Г': 1.59,
        'Б': 1.53, 'Й': 1.24, 'Х': 1.17, 'Ч': 1.15, 'Ц': 1.02,
        'Ї': 0.84, 'Ж': 0.71, 'Ш': 0.71, 'Ю': 0.70, 'Є': 0.39,
        'Ф': 0.35, 'Щ': 0.32, 'Ґ': 0.01,
    }

    # English letter frequencies (source: sttmedia.com, 2.10M char corpus)
    eng_freqs = {
        'E': 12.60, 'T': 9.37, 'A': 8.34, 'O': 7.70, 'N': 6.80,
        'I': 6.71, 'H': 6.11, 'S': 6.11, 'R': 5.68, 'L': 4.24,
        'D': 4.14, 'U': 2.85, 'C': 2.73, 'M': 2.53, 'W': 2.34,
        'Y': 2.04, 'F': 2.03, 'G': 1.92, 'P': 1.66, 'B': 1.54,
        'V': 1.06, 'K': 0.87, 'J': 0.23, 'X': 0.20, 'Q': 0.09,
        'Z': 0.06,
    }

    # Compute H1 (without space — matching the raw frequency data)
    def compute_h1(freqs):
        h1 = 0.0
        for letter, pct in freqs.items():
            p = pct / 100.0
            if p > 0:
                h1 -= p * math.log2(p)
        return h1

    ukr_h1_no_space = compute_h1(ukr_freqs)
    eng_h1_no_space = compute_h1(eng_freqs)

    # H0 values
    ukr_h0 = math.log2(33)  # without space
    eng_h0 = math.log2(26)  # without space
    ukr_h0_space = math.log2(34)
    eng_h0_space = math.log2(27)

    print(f"\n{'Metric':<30} {'Ukrainian':>12} {'English':>12} {'Difference':>12}")
    print("-" * 70)
    print(f"{'Alphabet size (no space)':<30} {33:>12} {26:>12} {7:>12}")
    print(f"{'Alphabet size (with space)':<30} {34:>12} {27:>12} {7:>12}")
    print(f"{'H0 no space (bpc)':<30} {ukr_h0:>12.3f} {eng_h0:>12.3f} {ukr_h0 - eng_h0:>12.3f}")
    print(f"{'H0 with space (bpc)':<30} {ukr_h0_space:>12.3f} {eng_h0_space:>12.3f} {ukr_h0_space - eng_h0_space:>12.3f}")
    print(f"{'H1 no space (bpc)':<30} {ukr_h1_no_space:>12.3f} {eng_h1_no_space:>12.3f} {ukr_h1_no_space - eng_h1_no_space:>12.3f}")
    print(f"{'Top letter frequency':<30} {'О 9.28%':>12} {'E 12.60%':>12}")
    print(f"{'Bottom letter frequency':<30} {'Ґ 0.01%':>12} {'Z 0.06%':>12}")

    # Sort both by frequency descending
    ukr_sorted = sorted(ukr_freqs.items(), key=lambda x: -x[1])
    eng_sorted = sorted(eng_freqs.items(), key=lambda x: -x[1])

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Side-by-side frequency bars (sorted by rank)
    ax = axes[0]
    max_len = max(len(ukr_sorted), len(eng_sorted))
    x = np.arange(max_len)
    width = 0.35

    ukr_vals = [f for _, f in ukr_sorted] + [0] * (max_len - len(ukr_sorted))
    eng_vals = [f for _, f in eng_sorted] + [0] * (max_len - len(eng_sorted))
    ukr_labels = [l for l, _ in ukr_sorted] + [''] * (max_len - len(ukr_sorted))
    eng_labels = [l for l, _ in eng_sorted] + [''] * (max_len - len(eng_sorted))

    ax.bar(x - width/2, ukr_vals, width, label='Ukrainian (33)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, eng_vals, width, label='English (26)', color='coral', alpha=0.8)
    ax.set_xlabel('Rank (by frequency)')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Letter Frequencies by Rank')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add letter labels on top
    for i in range(min(10, len(ukr_sorted))):
        ax.text(i - width/2, ukr_vals[i] + 0.15, ukr_labels[i], ha='center', va='bottom', fontsize=6, color='steelblue')
    for i in range(min(10, len(eng_sorted))):
        ax.text(i + width/2, eng_vals[i] + 0.15, eng_labels[i], ha='center', va='bottom', fontsize=6, color='coral')

    # Plot 2: Cumulative frequency
    ax = axes[1]
    ukr_cumul = np.cumsum([f for _, f in ukr_sorted])
    eng_cumul = np.cumsum([f for _, f in eng_sorted])
    ax.plot(range(1, len(ukr_cumul) + 1), ukr_cumul, 'o-', color='steelblue', markersize=4, label=f'Ukrainian ({len(ukr_sorted)} letters)')
    ax.plot(range(1, len(eng_cumul) + 1), eng_cumul, 'o-', color='coral', markersize=4, label=f'English ({len(eng_sorted)} letters)')
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% coverage')
    ax.set_xlabel('Number of letters (ranked by frequency)')
    ax.set_ylabel('Cumulative frequency (%)')
    ax.set_title('Cumulative Letter Coverage')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-letter entropy contribution (-p*log2(p))
    ax = axes[2]
    ukr_entropy_contrib = [-f/100 * math.log2(f/100) if f > 0 else 0 for _, f in ukr_sorted]
    eng_entropy_contrib = [-f/100 * math.log2(f/100) if f > 0 else 0 for _, f in eng_sorted]

    ax.bar(x[:len(ukr_sorted)] - width/2, ukr_entropy_contrib, width, label='Ukrainian', color='steelblue', alpha=0.8)
    ax.bar(x[:len(eng_sorted)] + width/2, eng_entropy_contrib, width, label='English', color='coral', alpha=0.8)
    ax.set_xlabel('Rank (by frequency)')
    ax.set_ylabel('Entropy contribution (bits)')
    ax.set_title('Per-Letter Entropy Contribution (-p log₂ p)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight Ukrainian-only letters
    ukr_only = {'Ї', 'Є', 'Ґ', 'Щ', 'И', 'І', 'Ь'}
    for i, (letter, _) in enumerate(ukr_sorted):
        if letter in ukr_only:
            ax.bar(i - width/2, ukr_entropy_contrib[i], width, color='gold', alpha=0.9, edgecolor='steelblue')
            ax.text(i - width/2, ukr_entropy_contrib[i] + 0.002, letter, ha='center', va='bottom', fontsize=6, fontweight='bold', color='darkgoldenrod')

    plt.suptitle('Ukrainian vs English: Character Frequencies and Entropy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'character_frequency_comparison.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_path}")


def weighted_average_upper_bound(guesses, experiments, guesses_per_position=None):
    """
    Compute weighted average of upper bounds across positions.
    Two weighting schemes: observation-count and inverse-variance (from bootstrap).
    Applied across trimming approaches, with and without first 5 positions.
    """
    print("\n" + "=" * 60)
    print("WEIGHTED AVERAGE UPPER BOUND")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)

    # Compute performance per experiment
    exp_avg_guesses = {}
    for exp_id, pos_counts in exp_pos_counts.items():
        if pos_counts:
            exp_avg_guesses[exp_id] = sum(pos_counts.values()) / len(pos_counts)

    sorted_exps = sorted(exp_avg_guesses.items(), key=lambda x: x[1])
    n_total = len(sorted_exps)

    trim_configs = [
        ('All data', 0, 0),
        ('Sym 10%', 10, 10),
        ('Sym 20%', 20, 20),
        ('-5%top -20%bot', 5, 20),
        ('-10%top -30%bot', 10, 30),
        ('Top 50%', 0, 50),
    ]

    newcomer_configs = [
        ('All positions', 0, None),
        ('Skip first 5', 5, None),
        ('75-100', 5, 100),
        ('75-105', 5, 105),
        ('75-110', 5, 110),
    ]

    # Bootstrap per-position variance for inverse-variance weights
    n_bootstrap = 500  # enough for variance estimation
    min_samples_boot = 30

    print(f"\nComputing per-position bootstrap variance ({n_bootstrap} iterations)...")

    results = []

    for trim_label, trim_top_pct, trim_bottom_pct in trim_configs:
        n_trim_top = int(n_total * trim_top_pct / 100)
        n_trim_bottom = int(n_total * trim_bottom_pct / 100)
        pool_exps = sorted_exps[n_trim_top:n_total - n_trim_bottom if n_trim_bottom > 0 else n_total]
        pool_ids = [eid for eid, _ in pool_exps]
        n_pool = len(pool_ids)

        if n_pool < 50:
            continue

        # Compute upper bounds at each position for this pool
        filtered_guesses = [g for g in guesses if g.experiment_result_id in set(pool_ids)]
        gpc = defaultdict(list)
        current_exp = None
        current_pos = None
        count = 0
        for g in filtered_guesses:
            if g.experiment_result_id != current_exp or g.position != current_pos:
                current_exp = g.experiment_result_id
                current_pos = g.position
                count = 0
            count += 1
            if g.is_correct:
                gpc[g.position].append(count)

        # Bootstrap per-position variance
        pool_ids_arr = np.array(pool_ids)
        pool_exp_pos = {eid: exp_pos_counts[eid] for eid in pool_ids}

        pos_bootstrap_samples = defaultdict(list)
        for b in range(n_bootstrap):
            resampled = rng.choice(pool_ids_arr, size=len(pool_ids_arr), replace=True)
            merged = defaultdict(list)
            for eid in resampled:
                for pos, gc in pool_exp_pos[eid].items():
                    merged[pos].append(gc)
            for pos, counts in merged.items():
                if len(counts) >= min_samples_boot:
                    ub = compute_upper_bound(counts)
                    if ub is not None and ub > 0:
                        pos_bootstrap_samples[pos].append(ub)

        # Compute variance per position
        pos_variance = {}
        for pos, samples in pos_bootstrap_samples.items():
            if len(samples) > 10:
                pos_variance[pos] = np.var(samples)

        for nc_label, skip_n, max_pos in newcomer_configs:
            # Get positions with enough data
            valid_positions = sorted(p for p in gpc.keys()
                                     if p >= MIN_POSITION + skip_n
                                     and (max_pos is None or p <= max_pos)
                                     and len(gpc[p]) >= 1)

            if not valid_positions:
                continue

            # Observation-weighted average
            obs_weights = []
            uppers = []
            for p in valid_positions:
                ub = compute_upper_bound(gpc[p])
                if ub is not None and ub > 0:
                    uppers.append(ub)
                    obs_weights.append(len(gpc[p]))

            if not uppers:
                continue

            obs_weights = np.array(obs_weights, dtype=float)
            uppers = np.array(uppers)
            obs_avg = np.average(uppers, weights=obs_weights)

            # Inverse-variance weighted average
            iv_positions = [p for p in valid_positions if p in pos_variance and pos_variance[p] > 0]
            iv_uppers = []
            iv_weights = []
            for p in iv_positions:
                ub = compute_upper_bound(gpc[p])
                if ub is not None and ub > 0:
                    iv_uppers.append(ub)
                    iv_weights.append(1.0 / pos_variance[p])

            n_iv = len(iv_uppers)
            if n_iv > 0:
                iv_weights = np.array(iv_weights)
                iv_uppers = np.array(iv_uppers)
                iv_avg = np.average(iv_uppers, weights=iv_weights)
            else:
                iv_avg = None

            # Simple (unweighted) average
            simple_avg = np.mean(uppers)

            results.append({
                'trim': trim_label,
                'newcomer': nc_label,
                'n_pool': n_pool,
                'n_positions': len(uppers),
                'simple_avg': simple_avg,
                'obs_avg': obs_avg,
                'iv_avg': iv_avg,
                'iv_positions': n_iv,
            })

    # Print results table
    print(f"\n{'Trim':<20} {'Positions':<16} {'N pos':>6} {'Simple':>8} {'Obs-wtd':>8} {'IV-wtd':>8}")
    print("-" * 72)
    for r in results:
        iv_str = f"{r['iv_avg']:.3f}" if r['iv_avg'] is not None else "N/A"
        print(f"{r['trim']:<20} {r['newcomer']:<16} {r['n_positions']:>6} "
              f"{r['simple_avg']:>8.3f} {r['obs_avg']:>8.3f} {iv_str:>8}")

    # Ratio-based h estimates
    english_h90 = 1.405
    english_h_final = 1.22

    print(f"\n--- Ratio-based h estimates (× {english_h_final}/{english_h90:.3f}) ---")
    print(f"{'Trim':<20} {'Positions':<16} {'Simple':>8} {'Obs-wtd':>8} {'IV-wtd':>8}")
    print("-" * 66)
    ratio = english_h_final / english_h90
    for r in results:
        simple_h = r['simple_avg'] * ratio
        obs_h = r['obs_avg'] * ratio
        iv_h = f"{r['iv_avg'] * ratio:.3f}" if r['iv_avg'] is not None else "N/A"
        print(f"{r['trim']:<20} {r['newcomer']:<16} {simple_h:>8.3f} {obs_h:>8.3f} {iv_h:>8}")

    return results



def bootstrap_sample_size_convergence(guesses, experiments, n_bootstrap=1000):
    """
    Ren et al. Table 3 equivalent: bootstrap at different sample sizes S,
    compute mean upper bound and pooled upper bound.
    Samples from the full pool (all experiments with data), no pre-trimming.
    """
    print("\n" + "=" * 60)
    print("SAMPLE SIZE CONVERGENCE (cf. Ren et al. Table 3)")
    print(f"({n_bootstrap} iterations per sample size)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)

    pool_ids = np.array(list(exp_pos_counts.keys()))
    n_pool = len(pool_ids)

    # Position range: 75-110
    min_pos = MIN_POSITION + 5  # skip first 5 (newcomer effect)
    max_pos = 110

    # --- Pooled calculation with trimming ---
    print(f"\n--- Pooled upper bound (positions {min_pos}-{max_pos}, all observations into one distribution) ---")

    # Compute performance per experiment for trimming
    exp_avg_guesses = {}
    for exp_id, pos_counts in exp_pos_counts.items():
        if pos_counts:
            exp_avg_guesses[exp_id] = sum(pos_counts.values()) / len(pos_counts)
    sorted_exps = sorted(exp_avg_guesses.items(), key=lambda x: x[1])
    n_total = len(sorted_exps)

    trim_configs = [
        ('All data', 0, 0),
        ('Sym 10%', 10, 10),
        ('Sym 20%', 20, 20),
        ('-5%top -20%bot', 5, 20),
        ('-10%top -30%bot', 10, 30),
        ('Top 50%', 0, 50),
    ]

    english_h90 = 1.405
    english_h_final = 1.22
    ratio = english_h_final / english_h90

    print(f"\n{'Trim':<20} {'Exps':>5} {'Obs':>7} {'Pool UB':>8} {'Pool LB':>8} {'Gap':>6} {'Mean UB':>8} {'Pool h':>7}")
    print("-" * 78)

    for trim_label, trim_top_pct, trim_bottom_pct in trim_configs:
        n_trim_top = int(n_total * trim_top_pct / 100)
        n_trim_bottom = int(n_total * trim_bottom_pct / 100)
        trimmed = sorted_exps[n_trim_top:n_total - n_trim_bottom if n_trim_bottom > 0 else n_total]
        trimmed_ids = set(eid for eid, _ in trimmed)

        pooled = []
        per_pos = defaultdict(list)
        for eid in trimmed_ids:
            for pos, gc in exp_pos_counts.get(eid, {}).items():
                if min_pos <= pos <= max_pos:
                    pooled.append(gc)
                    per_pos[pos].append(gc)

        pool_ub = compute_upper_bound(pooled)
        pool_lb = compute_lower_bound(pooled)
        mean_ub = np.mean([compute_upper_bound(per_pos[p]) for p in sorted(per_pos) if compute_upper_bound(per_pos[p]) is not None])

        print(f"{trim_label:<20} {len(trimmed_ids):>5} {len(pooled):>7} {pool_ub:>8.4f} {pool_lb:>8.4f} "
              f"{pool_ub - pool_lb:>6.3f} {mean_ub:>8.4f} {pool_ub * ratio:>7.3f}")

    # --- Bootstrap convergence table ---
    print(f"\n--- Bootstrap convergence (per-position mean vs pooled) ---")
    print(f"Pool: {n_pool} experiments (all with data, no trim)")

    sample_sizes = [50, 100, 150, 200, 300, 400, 500, 600, n_pool]

    print(f"\n{'S':>6} {'Mean UB':>8} {'5% Lo':>7} {'5% Hi':>7} {'Width':>6}"
          f" {'Pool UB':>8} {'5% Lo':>7} {'5% Hi':>7} {'Width':>6}")
    print("-" * 76)

    for S in sample_sizes:
        if S > n_pool:
            continue

        mean_ub_samples = []
        pooled_ub_samples = []

        for b in range(n_bootstrap):
            resampled = rng.choice(pool_ids, size=S, replace=True)

            merged = defaultdict(list)
            pooled = []
            for eid in resampled:
                for pos, gc in exp_pos_counts.get(eid, {}).items():
                    if min_pos <= pos <= max_pos:
                        merged[pos].append(gc)
                        pooled.append(gc)

            # Per-position mean
            uppers = []
            for pos in sorted(merged.keys()):
                if len(merged[pos]) >= 20:
                    ub = compute_upper_bound(merged[pos])
                    if ub is not None and ub > 0:
                        uppers.append(ub)
            if uppers:
                mean_ub_samples.append(np.mean(uppers))

            # Pooled
            if len(pooled) >= 50:
                pub = compute_upper_bound(pooled)
                if pub is not None and pub > 0:
                    pooled_ub_samples.append(pub)

        if not mean_ub_samples:
            print(f"{S:>6}   (insufficient data)")
            continue

        m_arr = np.array(mean_ub_samples)
        p_arr = np.array(pooled_ub_samples)

        m_mean, m_lo, m_hi = np.mean(m_arr), np.percentile(m_arr, 5), np.percentile(m_arr, 95)
        p_mean, p_lo, p_hi = np.mean(p_arr), np.percentile(p_arr, 5), np.percentile(p_arr, 95)

        print(f"{S:>6} {m_mean:>8.3f} {m_lo:>7.3f} {m_hi:>7.3f} {m_hi-m_lo:>6.3f}"
              f" {p_mean:>8.3f} {p_lo:>7.3f} {p_hi:>7.3f} {p_hi-p_lo:>6.3f}")

    return


def english_ansatz(n, A=0.125, beta=0.484, h=1.393):
    """Ren et al. ansatz f₁(n) = A * n^(β-1) + h"""
    return A * n ** (beta - 1) + h


def ratio_to_english_ansatz(guesses_per_position, min_samples=200):
    """
    Compare Ukrainian upper bound to English ansatz at each position.
    Compute the ratio and derive per-position h estimates.
    """
    print("\n" + "=" * 60)
    print("RATIO TO ENGLISH ANSATZ (Ren et al.)")
    print("=" * 60)

    # Ren et al. parameters: f₁(n) = A * n^(β-1) + h
    # h=1.393 (all data), β=0.484, A≈0.125 (back-computed from h_exp_min=1.407 at n=70)
    # Their final filtered estimate: h_filtered = 1.22
    english_h_final = 1.22

    positions = []
    ukr_uppers = []
    eng_ansatz_vals = []
    ratios = []
    h_estimates = []
    samples = []

    for pos in sorted(guesses_per_position.keys()):
        counts = guesses_per_position[pos]
        if len(counts) < min_samples:
            continue
        upper = compute_upper_bound(counts)
        if upper is None:
            continue
        eng = english_ansatz(pos)
        ratio = upper / eng
        h_est = english_h_final * ratio

        positions.append(pos)
        ukr_uppers.append(upper)
        eng_ansatz_vals.append(eng)
        ratios.append(ratio)
        h_estimates.append(h_est)
        samples.append(len(counts))

    print(f"\n{'Pos':>4} {'n':>5} {'UKR upper':>10} {'ENG ansatz':>11} {'Ratio':>7} {'h est':>7}")
    print("-" * 50)
    for i, pos in enumerate(positions):
        if samples[i] >= 100:
            print(f"{pos:>4} {samples[i]:>5} {ukr_uppers[i]:>10.3f} {eng_ansatz_vals[i]:>11.3f} {ratios[i]:>7.3f} {h_estimates[i]:>7.3f}")

    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    mean_h = np.mean(h_estimates)
    median_h = np.median(h_estimates)

    print(f"\n--- Summary ---")
    print(f"  Mean ratio:   {mean_ratio:.3f} ± {std_ratio:.3f}")
    print(f"  Mean h est:   {mean_h:.3f} bpc")
    print(f"  Median h est: {median_h:.3f} bpc")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Ukrainian upper vs English ansatz
    ax = axes[0]
    ax.plot(positions, ukr_uppers, 'o-', color='blue', label='Ukrainian upper bound', markersize=4)
    ax.plot(positions, eng_ansatz_vals, 's-', color='red', label='English ansatz (Ren et al.)', markersize=4)
    ax.set_xlabel('Position')
    ax.set_ylabel('Upper bound (bpc)')
    ax.set_title('Ukrainian vs English Ansatz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Ratio
    ax = axes[1]
    ax.plot(positions, ratios, 'o-', color='purple', markersize=4)
    ax.axhline(y=mean_ratio, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_ratio:.3f}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Ratio (UKR / ENG)')
    ax.set_title('Ratio to English Ansatz')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Per-position h estimate
    ax = axes[2]
    ax.plot(positions, h_estimates, 'o-', color='green', markersize=4)
    ax.axhline(y=mean_h, color='orange', linestyle='--', label=f'Mean: {mean_h:.3f}')
    ax.axhline(y=median_h, color='red', linestyle='--', label=f'Median: {median_h:.3f}')
    ax.axhline(y=1.22, color='gray', linestyle=':', alpha=0.5, label='English h = 1.22')
    ax.set_xlabel('Position')
    ax.set_ylabel('Estimated h (bpc)')
    ax.set_title('Per-position h Estimate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'graphs' / 'ratio_to_english_ansatz.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved: {output_path}")

    return {
        'positions': positions,
        'ratios': ratios,
        'h_estimates': h_estimates,
        'mean_ratio': mean_ratio,
        'mean_h': mean_h,
        'median_h': median_h,
    }


def main():
    print("=" * 60)
    print("SHANNON'S GUESSING GAME - ENTROPY ANALYSIS")
    print("=" * 60 + "\n")

    users, sentences, experiments, guesses = load_data()
    print_basic_info(users, sentences, experiments, guesses)

    analyze_sentence_usage(sentences, experiments)
    plot_observations_by_position(guesses, experiments)
    plot_first_guess_accuracy(guesses)

    guesses_per_position = compute_guesses_to_correct(guesses)
    positions, upper_bounds, lower_bounds = plot_entropy_by_position(guesses_per_position, max_position=110)

    weighted_regression_extrapolation(positions, upper_bounds, guesses_per_position)
    smoothed_fit_extrapolation(positions, upper_bounds)
    weighted_avg_results = weighted_average_upper_bound(guesses, experiments, guesses_per_position)
    ratio_results = ratio_to_english_ansatz(guesses_per_position, min_samples=1)
    bootstrap_sample_size_convergence(guesses, experiments)
    analyze_by_performance_tier(guesses, experiments, min_samples=200)

    session_results = bootstrap_by_session(guesses, experiments, n_bootstrap=2000, min_samples=50)
    trimming_results = bootstrap_by_trimming(guesses, experiments, n_bootstrap=2000, min_samples=50)
    sentence_results = bootstrap_by_sentence(guesses, experiments, sentences, n_bootstrap=2000, min_samples=50)

    plot_bootstrap_results(session_results, trimming_results, sentence_results)
    plot_character_frequencies()
    bootstrapped_curves = bootstrap_per_position(guesses, experiments, n_bootstrap=2000, min_samples=50)

    exp_pos_counts = precompute_guess_counts_by_experiment(guesses)
    cheater_ids = detect_cheaters_binomial(guesses, exp_pos_counts, p_threshold=0.01)

    exp_avg = {}
    for eid, pc in exp_pos_counts.items():
        if pc and eid not in cheater_ids:
            exp_avg[eid] = sum(pc.values()) / len(pc)
    sorted_exps_clean = sorted(exp_avg.items(), key=lambda x: x[1])

    obs_weighted_results = bootstrap_obs_weighted(
        exp_pos_counts, sorted_exps_clean,
        bottom_trim_pcts=list(range(0, 95, 5)),
        n_bootstrap=2000, min_samples=50,
    )
    plot_trim_sensitivity(obs_weighted_results)


if __name__ == "__main__":
    main()