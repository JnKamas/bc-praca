# This file contains by encapsulated backend to make programming all the simulations easier
# Political party number 0 is artificial - represents non-voters and invalid votes

import random
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Apportionment:

    def __init__(self, num_seats, voters, link=None):
        self.num_seats = num_seats
        self.voters = voters
        self.subject_votes = {}
        self.subject_names = {}
        self.treshold = lambda: 5
        if link: self.read_votes_from_csv(link) # else insert data manually
        self.probabilities = None 
        if link: self.generate_probabilities() # else generate probs manually
        self.boxes = None
    
    def __str__(self):
        return (
            f"""Apportionment(\n
                num_seats={self.num_seats},\n
                subject_votes={self.subject_votes},\n
                voters={self.voters}\n
            )"""
        )


    def copy(self):
        cpy = Apportionment(num_seats=self.num_seats, voters=self.voters)
        cpy.subject_votes = self.subject_votes.copy()
        cpy.subject_names = self.subject_names
        cpy.treshold = self.treshold
        cpy.probabilities = self.probabilities
        cpy.boxes = self.boxes
        return cpy

    def generate_probabilities(self):
        # requires numeric improvement (probably logarithms instead)
        total_prob = sum(self.subject_votes.values())
        if total_prob != 1:
            self.probabilities = {key: prob / total_prob for key, prob in self.subject_votes.items()}

    def read_votes_from_csv(self, link):
        with open(link, 'r', encoding='utf-8') as csvfile:
            data = csv.reader(csvfile)
            
            # Skip header
            next(data)

            for row in data:
                if len(row) >= 3:
                    subject_number = int(row[0].strip())
                    subject_name = row[1].strip()
                    valid_votes = int(row[2].strip())
                    self.subject_votes[subject_number] = valid_votes
                    self.subject_names[subject_number] = subject_name
    
    def counted_votes(self):
        return{x : y for x, y in self.subject_votes.items() if (((y * 100) / (sum(self.subject_votes.values()) - self.subject_votes[0])) > self.treshold() and x != 0)}


    def slovak_apportionment(self):

        def get_top_x_indexes(numbers, x):
            if x >= len(numbers):
                return list(range(len(numbers)))
            
            indexes_with_values = list(enumerate(numbers))
            indexes_with_values.sort(key=lambda x: (-x[1], random.random()))
            
            top_x_indexes = [index for index, _ in indexes_with_values[:x]]
            
            return top_x_indexes

        counted_votes = self.counted_votes()

        sum_counted_votes = sum(counted_votes.values())
        republic_number = round(sum_counted_votes / (self.num_seats + 1))
        
        seats_given = [int(x / republic_number) for x in counted_votes.values()]
        division_remainders = [x / republic_number - int(x / republic_number) for x in counted_votes.values()]
        if sum(seats_given) > 150:
            # this requires more testing
            seats_given[division_remainders.index(min(division_remainders))] -= 1
        else:    
            for x in get_top_x_indexes(division_remainders, self.num_seats - sum(seats_given)):
                seats_given[x] += 1
        return {x: y for x, y in zip(counted_votes.keys(), seats_given)} 


    def hagenbach_bischoff_apportionment(self):
        results = {}
        counted_votes = self.counted_votes()  # Assuming this is a method to count votes
        sorted_subject_votes = sorted(counted_votes.items(), key=lambda x: x[1], reverse=True)
        allocated_seats = {subject_number: 0 for subject_number in self.subject_votes}
        
        for _ in range(self.num_seats):  # Assuming self.num_seats represents the total seats available
            subject_number, votes = max(
                sorted_subject_votes,
                key=lambda x: x[1] / (allocated_seats[x[0]] + 1)
            )
            allocated_seats[subject_number] += 1
            results[subject_number] = allocated_seats[subject_number]
        
        return results


    def dhont_apportionment(self):
        results = {}
        counted_votes = self.counted_votes()
        sorted_subject_votes = sorted(counted_votes.items(), key=lambda x: x[1], reverse=True)
        allocated_seats = {subject_number: 0 for subject_number in self.subject_votes}
        for _ in range(self.num_seats):
            subject_number, votes = max(sorted_subject_votes, key=lambda x: x[1] / (allocated_seats[x[0]] + 1))
            allocated_seats[subject_number] += 1
            results[subject_number] = allocated_seats[subject_number]
        return results


    def divide_seats(self, method):
        if method == "slovak":
            return self.slovak_apportionment()
        elif method == "d'hont":
            return self.dhont_apportionment()
        elif method == "hagenbach bischoff":
            return self.hagenbach_bischoff_apportionment()
        else:
            print("Invalid method choice. Please choose 'slovak', 'd'hont' or 'hagenbach bischoff'.")

    def generate_additional_votes(self, count):
        keys, probs = zip(*self.probabilities.items())
        choices = np.random.choice(keys, count, p=probs)
        results = {key: np.count_nonzero(choices == key) for key in set(choices)}
        return results

    def basic_simulation(self): ## obsolete
        choices = []
        for _ in range(self.voters):
            choice = random.choices(list(self.probabilities.keys()), list(self.probabilities.values()))[0]
            choices.append(choice)

        results = {key: choices.count(key) for key in set(choices)}
        sorted_results = {k: results[k] for k in sorted(results.keys())}
        return sorted_results

    def numpy_simulation(self):
        keys, probs = zip(*self.probabilities.items())

        choices = np.random.choice(keys, self.voters, p=probs)

        results = {key: np.count_nonzero(choices == key) for key in set(choices)}
        sorted_results = {k: results[k] for k in sorted(results.keys())}
        return sorted_results

    def boxes_simulation(self):
        keys, probs = zip(*self.probabilities.items())

        if self.boxes == None:
            self.boxes = [np.random.choice(keys, 100000, p=probs) for _ in range(1000)]

        choices = np.random.choice(keys, self.voters % 100000, p=probs)
        for _ in range(int(self.voters / 100000)):
            rand_num = np.random.randint(1000)
            addition = self.boxes[rand_num]
            choices = np.concatenate((choices, addition))

        results = {key: np.count_nonzero(choices == key) for key in set(choices)}
        sorted_results = {k: results[k] for k in sorted(results.keys())}
        return sorted_results

    def advanced_simulation(self):
        # Implement advanced method logic here
        print('Advanced simulation method implemented.')
        pass



    def simulate_results(self, method):
        if method == "basic":
            return self.basic_simulation()
        elif method == "numpy":
            return self.numpy_simulation()
        elif method == "boxes":
            return self.boxes_simulation()
        elif method == "advanced":
            return self.advanced_simulation()
        else:
            print("Invalid option. Please choose 'basic,' 'numpy,' 'boxes,' or 'advanced'.")



    def iterated_simulate(self, method, file, nit=10, group_size=10):

        print("Initializing simulation...")
        start_time = time.time()

        columns = ['interation_number', 'party_number', 'samples', 'diff']

        with open(file, 'w', newline='', encoding='utf-8') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for i in range(nit):
                print(f'{i+1} / {nit}')

                results = self.simulate_results(method)

                default_ap = Apportionment(self.num_seats, self.voters)
                default_ap.subject_votes = results.copy()
                main_seats_vector = self.dictionary_to_vector(default_ap.divide_seats('slovak'))

            
                # NESTED LOOP TO TEST CHANGES
                ap = Apportionment(self.num_seats, self.voters - group_size)
                ap.subject_votes = results.copy()
                ap.generate_probabilities()
                # print(ap)

                for size in range(group_size, 0, -1):
                    for party in self.subject_names.keys():
                        apx = ap.copy()
                        # print(apx)
                        try: apx.subject_votes[party] += size
                        except KeyError: apx.subject_votes[party] = size
                        seats_vector = self.dictionary_to_vector(apx.divide_seats('slovak'))
                        apx.subject_votes[party] -= size

                        distance = compare_vectors(main_seats_vector, seats_vector)
                        
                        new_data = {'interation_number': i+1, 'party_number': party, 'samples' : size, 'diff' : distance}
                        writer.writerow(new_data)
                        # Append a record
                        # return_df1.append(new_data)

                    for index, x in self.generate_additional_votes(1).items():
                        try: apx.subject_votes[index] += x
                        except KeyError: apx.subject_votes[index] = x
                    
        print(f'''Simulation finished. Detailed results in file {file}\nTime: {time.time() - start_time} seconds.''')

    def dictionary_to_vector(self, input_dict):
        max_index = len(self.subject_votes.keys())
        result_vector = [0] * max_index

        for key, value in input_dict.items():
            if key != 0:
                index = int(key)
                if 1 <= index <= max_index:
                    result_vector[index - 1] = value

        return result_vector
    
def compare_vectors(first, second):
    diff = 0
    for i in range(len(first)):
        diff += abs(first[i] - second[i])
    return diff


def get_votes():
    voters = 1000
    num_seats = 150
    link='NRSR2023_clean.csv'
    ap = Apportionment(num_seats, voters, link=link) 
    return ap.subject_votes

def raw2visualisable(input_file, size, weighted=True, only_electable=False, subjects=26):
    '''
    This method provides a transformation of .csv file containing generated data to a properly averaged form.
    The data is transformed from tens GB to few MB.
    The created files are then used to create a visualisation and they are stored in github repo.
    '''

    chunksize = 130000000 #TODO adapt to subjects and group size
    all_xdfs = []
    
    # adaptation of weights to chosen averaging method
    weights = get_votes()
    valid_votes = sum(weights.values()) - weights[0]
    for key in weights.keys():
        if only_electable and weights[key] / valid_votes < 0.03: # electable is consdidered from 3% even for coalitions to simplify
            weights[key] = 0
        elif not weighted:
            weights[key] = 1  


    # iteration through all records
    # it basically puts averages together, correctly
    for chunk in pd.read_csv("./raw_data/"+input_file, chunksize=chunksize):
        chunk['weight'] = chunk['party_number'].map(weights)
        xdf = chunk.groupby('samples').apply(lambda x: np.average(x['diff'], weights=x['weight'])).reset_index(name='diff')
  
        all_xdfs.append(xdf)

    # second iteration of the algorith ensures pseudo O(logn) space efficiency
    result_df = pd.concat(all_xdfs, axis=0, ignore_index=True)
    export_df = result_df.groupby('samples').apply(lambda x: np.average(x['diff'])).reset_index(name='diff')

    # export to a file
    file_prefix = "" if weighted else "un"
    if only_electable: file_prefix = "electable-" + file_prefix
    export_df.to_csv(f"./vis_data/{file_prefix}weighted-vis-{input_file}", index=False)
    print(f"{input_file} done")

if __name__ == "__main__":
    import time
    start = time.time()
    raw2visualisable("100k-large.csv", 100000)
    print(time.time() - start)
