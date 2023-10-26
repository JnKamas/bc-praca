import numpy as np
import time

import reader

## numpy solution that takes roughly 1.7 seconds to perform a simulation
## 10 000 simulations takes roughly 5 hours

def random_choices(probabilities, num_choices):
    if num_choices <= 0:
        return {}

    # Check if the probabilities sum to 1. If not, normalize them.
    total_prob = sum(probabilities.values())
    if total_prob != 1:
        probabilities = {key: prob / total_prob for key, prob in probabilities.items()}

    keys, probs = zip(*probabilities.items())

    choices = np.random.choice(keys, num_choices, p=probs)

    results = {key: np.count_nonzero(choices == key) for key in set(choices)}
    sorted_results = {k: results[k] for k in sorted(results.keys())}
    return sorted_results

if __name__ == "__main__":
    num_choices = 4388872
    votes_counts, party_names = reader.reader()
    probabilities = {x: y for x, y, z in votes_counts}
    probabilities["No valid vote"] = num_choices - sum(probabilities.values())
    start_time = time.time()
    sorted_results = random_choices(probabilities, num_choices)
    print(time.time() - start_time)
    for x, y in sorted_results.items():
        print(f'{x} \t {y}')
