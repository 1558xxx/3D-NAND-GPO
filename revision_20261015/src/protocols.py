"""Coordinate-only holdout and nested queries; target frequencies are unavailable here."""
import numpy as np

def fixed_test_indices(length, max_budget=9):
    if length < max_budget + 2:
        raise ValueError("Not enough coordinates for a common held-out test set")
    count = min(max(2, int(round(0.2 * length))), length - max_budget)
    # Reserve both tails; never acquire these points at any budget.
    chosen = [1, length - 2]
    if count == 2:
        return np.array(sorted(chosen), dtype=int)
    candidates = np.linspace(2, length - 3, count)[1:-1].round().astype(int)
    for index in candidates:
        if int(index) not in chosen:
            chosen.append(int(index))
        if len(chosen) == count:
            break
    return np.array(sorted(chosen), dtype=int)

def nested_order(steps, test_indices, source_peak=None):
    steps = np.asarray(steps, float)
    available = sorted(set(range(len(steps))) - set(test_indices))
    chosen = [available[0], available[-1]]
    if source_peak is not None:
        nearest = min(available, key=lambda i: (abs(steps[i] - source_peak), i))
        if nearest not in chosen:
            chosen.append(nearest)
    while len(chosen) < len(available):
        left = [i for i in available if i not in chosen]
        index = max(left, key=lambda i: (min(abs(steps[i] - steps[j]) for j in chosen), -i))
        chosen.append(index)
    return np.array(chosen, dtype=int)

def oracle_order(steps, frequencies, test_indices):
    # Diagnostic only: this function explicitly uses hidden target information.
    return nested_order(steps, test_indices, float(steps[np.argmax(frequencies)]))
