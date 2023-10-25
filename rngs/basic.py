import random
import time

# simplest solution that takes roughly 10 second to perform a simulation
# 10 000 simulations is roughly 3 days


def random_choices(probabilities, num_choices):
    if num_choices <= 0:
        return {}

    # Check if the probabilities sum to 1. If not, normalize them.
    total_prob = sum(probabilities.values())
    if total_prob != 1:
        probabilities = {key: prob / total_prob for key, prob in probabilities.items()}

    choices = []
    for _ in range(num_choices):
        choice = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
        choices.append(choice)

    results = {key: choices.count(key) for key in set(choices)}
    sorted_results = {k: results[k] for k in sorted(results.keys())}
    return sorted_results


if __name__ == "__main__":
    probabilities = {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}
    num_choices = 5000000
    start_time = time.time()
    sorted_results = random_choices(probabilities, num_choices)
    print(time.time() - start_time)
    print(sorted_results)
