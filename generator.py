from np_rng import *

for i in range(100):
    num_choices = 4388872
    probabilities = reader.reader(1)
    probabilities["No valid vote"] = num_choices - sum(probabilities.values())
    start_time = time.time()
    sorted_results = random_choices(probabilities, num_choices)
    print(time.time() - start_time)