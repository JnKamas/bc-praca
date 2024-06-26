# This file contains by encapsulated backend to make programming all the simulations easier
# Political party number 0 is artificial - represents non-voters and invalid votes

# libraries
import random
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3

# votelib
import decimal
import votelib.candidate
import votelib.evaluate.core
import votelib.evaluate.threshold
import votelib.evaluate.proportional

# my files
import constants


class Apportionment:

    def __init__(self, num_seats, voters, link=None, treshold=lambda x:5):
        self.num_seats = num_seats
        self.voters = voters
        self.subject_votes = {}
        self.subject_names = {}
        self.subject_names_inv = {}
        self.treshold = treshold
        if link: self.read_votes_from_csv(link) # else insert data manually
        self.probabilities = None 
        if link: self.generate_probabilities() # else generate probabilities manually
        self.boxes = None
    
    def __str__(self):
        return (
            f"""Apportionment(\n
                num_seats={self.num_seats},\n
                subject_votes={self.subject_votes},\n
                voters={self.voters}\n
            )"""
        )

    # copies instance of Apportionment class
    def copy(self):
        cpy = Apportionment(num_seats=self.num_seats, voters=self.voters)
        cpy.subject_votes = self.subject_votes.copy()
        cpy.subject_names = self.subject_names
        cpy.subject_names_inv = self.subject_names_inv
        cpy.treshold = self.treshold
        cpy.probabilities = self.probabilities
        cpy.boxes = self.boxes
        return cpy

    # generates probabilities based on real data votes
    def generate_probabilities(self):
        total_prob = sum(self.subject_votes.values())
        if total_prob != 1:
            self.probabilities = {key: prob / total_prob for key, prob in self.subject_votes.items()}

    # reads data from a file, link is a path to file
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
        self.subject_names_inv = invert_dict(self.subject_names)
    
    # calculates number of counted votes - votes used in apportionment
    def counted_votes(self):
        return{x : y for x, y in self.subject_votes.items() if (((y * 100) / (sum(self.subject_votes.values()) - self.subject_votes[0])) > self.treshold(x) and x != 0)}

    # method for modifield hagenbach-bischoff apportionment, used in Slovakia
    def slovak_apportionment(self):
        # This method unlike other apportionment methods, was programmed by the author of the thesis, since few adaptations were required to make it work correctly. 
        
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
            seats_given[division_remainders.index(min(division_remainders))] -= 1
        else:    
            for x in get_top_x_indexes(division_remainders, self.num_seats - sum(seats_given)):
                seats_given[x] += 1
        return {x: y for x, y in zip(counted_votes.keys(), seats_given)} 

    # method for hagenbach-bischoff apportionment
    def hagenbach_bischoff_apportionment(self):
        # credits: https://github.com/simberaj/votelib/blob/master/docs/examples/sk_nr_2020.ipynb
        core_evaluator = votelib.evaluate.proportional.LargestRemainder(
            'hagenbach_bischoff'
        )

        standard_elim = votelib.evaluate.threshold.RelativeThreshold(
            decimal.Decimal('.05'), accept_equal=True
        )
        mem_2_3_elim = votelib.evaluate.threshold.RelativeThreshold(
            decimal.Decimal('.07'), accept_equal=True
        )
        mem_4plus_elim = votelib.evaluate.threshold.RelativeThreshold(
            decimal.Decimal('.1'), accept_equal=True
        )
        preselector = votelib.evaluate.threshold.CoalitionMemberBracketer(
            {1: standard_elim, 2: mem_2_3_elim, 3: mem_2_3_elim},
            default=mem_4plus_elim
        )

        evaluator = votelib.evaluate.core.FixedSeatCount(
            votelib.evaluate.core.Conditioned(preselector, core_evaluator), 150
        )

        votes = {
            votelib.candidate.PoliticalParty(self.subject_names[x]): y 
            for x, y in self.subject_votes.items()
            if int(x) != 0
        }

        evaluated = evaluator.evaluate(votes)
        return {self.subject_names_inv[party.name] : mandates for party, mandates in evaluated.items()}

    # method for d'hondt apportionment
    def dhondt_apportionment(self):
        # credits: https://github.com/simberaj/votelib/blob/master/docs/examples/sk_nr_2020.ipynb
        core_evaluator = votelib.evaluate.proportional.HighestAverages(
            'd_hondt'
        )

        standard_elim = votelib.evaluate.threshold.RelativeThreshold(
            decimal.Decimal('.05'), accept_equal=True
        )
        mem_2_3_elim = votelib.evaluate.threshold.RelativeThreshold(
            decimal.Decimal('.07'), accept_equal=True
        )
        mem_4plus_elim = votelib.evaluate.threshold.RelativeThreshold(
            decimal.Decimal('.1'), accept_equal=True
        )
        preselector = votelib.evaluate.threshold.CoalitionMemberBracketer(
            {1: standard_elim, 2: mem_2_3_elim, 3: mem_2_3_elim},
            default=mem_4plus_elim
        )

        evaluator = votelib.evaluate.core.FixedSeatCount(
            votelib.evaluate.core.Conditioned(preselector, core_evaluator), 150
        )

        votes = {
            votelib.candidate.PoliticalParty(self.subject_names[x]): y 
            for x, y in self.subject_votes.items()
            if int(x) != 0
        }

        evaluated = evaluator.evaluate(votes)
        return {self.subject_names_inv[party.name] : mandates for party, mandates in evaluated.items()}


    # unified method for dividing seats
    def divide_seats(self, method):
        if method == "slovak":
            return self.slovak_apportionment()
        elif method == "d_hondt":
            return self.dhondt_apportionment()
        elif method == "hagenbach_bischoff":
            return self.hagenbach_bischoff_apportionment()
        else:
            print("Invalid method choice. Please choose 'slovak', 'd_hondt' or 'hagenbach_bischoff'.")

    # method that generates random votes
    def generate_additional_votes(self, count):
        keys, probs = zip(*self.probabilities.items())
        choices = np.random.choice(keys, count, p=probs)
        results = {key: np.count_nonzero(choices == key) for key in set(choices)}
        return results

    # this method generates v votes as defined in initialisation
    def simulate_results(self):
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

    # this method performs simulation and exports the result into a .csv file
    def iterated_simulate(self, file, year, nit=10, group_size=10, divide="slovak", coalition=False, multi=False):

        print("Initializing simulation...")
        start_time = time.time()

        columns = ['iteration_number', 'party_number', 'samples', 'diff']

        if divide == "all":
            columns = ['iteration_number', 'party_number', 'samples', 'diff', 'apport']

        if multi:
            columns = ['iteration_number', 'party_number', 'party2_number', 'samples', 'diff', "proportion"]


        with open(file, 'w', newline='', encoding='utf-8') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            for i in range(nit):
                print(f'{i+1} / {nit}')

                results = self.simulate_results()

                default_ap = Apportionment(self.num_seats, self.voters, treshold=self.treshold)
                default_ap.subject_names = self.subject_names
                default_ap.subject_names_inv = self.subject_names_inv

                default_ap.subject_votes = results.copy()
                main_seats_vector = self.dictionary_to_vector(default_ap.divide_seats(divide if divide != "all" else "slovak"))
            
                # NESTED LOOP TO TEST CHANGES
                ap = Apportionment(self.num_seats, self.voters - group_size, treshold=self.treshold)
                ap.subject_votes = results.copy()
                ap.subject_names = self.subject_names
                ap.subject_names_inv = self.subject_names_inv
                ap.generate_probabilities()


                # parties with at least 3% - used to optimize multichoice in group
                likely_electable = []
                for x, y in self.subject_votes.items():
                    if ((y * 100) / (sum(self.subject_votes.values()) - self.subject_votes[0])) > 3 and x != 0:
                        likely_electable.append(x)

                for size in range(group_size, 0, -1):
                    if multi:
                        for party in likely_electable:
                            for party2 in likely_electable:
                                if party == party2: continue
                                for prop in [90, 80, 70, 60, 50]: 
                                    size1 = int(size * (prop / 100))
                                    size2 = size - size1
                                    apx = ap.copy()

                                    try: apx.subject_votes[party] += size1
                                    except KeyError: apx.subject_votes[party] = size1
                                    try: apx.subject_votes[party2] += size2
                                    except KeyError: apx.subject_votes[party2] = size2
                                    seats_vector = self.dictionary_to_vector(apx.divide_seats(divide))
                                    apx.subject_votes[party] -= size1
                                    apx.subject_votes[party2] -= size2

                                    distance = compare_vectors(main_seats_vector, seats_vector, year, coalition)

                                    new_data = {'iteration_number': i+1, 'party_number': party, 'party2_number' : party2, 'samples' : size, 'diff' : distance, 'proportion' : prop}
                                    writer.writerow(new_data)
                    elif divide == "all":
                        for party in self.subject_names.keys():
                            apx = ap.copy()
                            try: apx.subject_votes[party] += size
                            except KeyError: apx.subject_votes[party] = size
                            for apport in ['slovak', 'hagenbach_bischoff', 'd_hondt']:
                                seats_vector = self.dictionary_to_vector(apx.divide_seats(apport))
                                distance = compare_vectors(main_seats_vector, seats_vector, year, coalition)
                                new_data = {'iteration_number': i+1, 'party_number': party, 'samples' : size, 'diff' : distance, 'apport' : apport}
                                writer.writerow(new_data)
                            apx.subject_votes[party] -= size                
                    else:
                        for party in self.subject_names.keys():
                            apx = ap.copy()
                            try: apx.subject_votes[party] += size
                            except KeyError: apx.subject_votes[party] = size
                            seats_vector = self.dictionary_to_vector(apx.divide_seats(divide))
                            apx.subject_votes[party] -= size

                            distance = compare_vectors(main_seats_vector, seats_vector, year, coalition)

                            new_data = {'iteration_number': i+1, 'party_number': party, 'samples' : size, 'diff' : distance}
                            writer.writerow(new_data)
                    

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
    
def compare_vectors(first, second, year, coalition):
    if coalition:
        # in this case it will return total mandates of real coalition that created government after those elections
        mandates = {i : second[i-1] for i in range(1, len(second) + 1)}
        return sum(mandates[x] for x in constants.winning_coalition[year])
    diff = 0
    for i in range(len(first)):
        diff += abs(first[i] - second[i])
    return diff

def invert_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

def get_votes(year):
    voters = 1000
    num_seats = 150 
    link=f'./real_data/NRSR{year}_clean.csv'
    ap = Apportionment(num_seats, voters, link=link) 
    return ap.subject_votes

def raw2visualisable(input_file, weighted=True, only_electable=False, neglected=[], year=2023, multi=False, suff=None):
    '''
    This method provides a transformation of .csv file containing generated data to a properly averaged form.
    The data is transformed from tens GB to few MB.
    The created files are then used to create a visualisation and they are stored in github repo.
    '''
    if multi == True and weighted == True:
        raise NotImplementedError
    '''
    for multi, weighted is NOT IMPLEMENTED
    '''
    chunksize = constants.subjects[year] * 1000000
    all_xdfs = []
    
    # adaptation of weights to chosen averaging method
    weights = get_votes(year)
    valid_votes = sum(weights.values()) - weights[0]
    for key in weights.keys():
        if (only_electable and (weights[key] / valid_votes < 0.03)) or int(key) in neglected: # electable is consdidered from 3% even for coalitions to simplify
            weights[key] = 0
        elif not weighted:
            weights[key] = 1  

    # iteration through all records
    # it averages by chunks, and then the total averages are created as averages by individual chunks
    for chunk in pd.read_csv("./raw_data/"+input_file, chunksize=chunksize):
        chunk['weight'] = chunk['party_number'].map(weights)
        if multi:
            xdf = chunk.groupby(['samples', 'proportion']).apply(lambda x: np.average(x['diff'], weights=x['weight'])).reset_index(name='diff')
        else:
            xdf = chunk.groupby('samples').apply(lambda x: np.average(x['diff'], weights=x['weight'])).reset_index(name='diff')
  
        all_xdfs.append(xdf)

    result_df = pd.concat(all_xdfs, axis=0, ignore_index=True)

    if multi:
        export_df = result_df.groupby(['samples', 'proportion']).apply(lambda x: np.average(x['diff'])).reset_index(name='diff')
    else:
        export_df = result_df.groupby(['samples']).apply(lambda x: np.average(x['diff'])).reset_index(name='diff')

    # export to a file
    file_prefix = "" if weighted else "un"
    if only_electable: file_prefix = "electable-" + file_prefix
    export_df.to_csv(f"./vis_data/{file_prefix}weighted-vis-{input_file}{suff}", index=False)
    print(f"{input_file} done")


# example how to generate synthetic data (500 iteratiosn and million voters results in roughly 6GB and take about 3-4 hours)
if __name__ == "__main__":
    # Simulation parameters
    voters = 1000000
    num_seats = 150
    nit = 3
    group_size = int(0.03 * voters)
    link='./real_data/NRSR2023_clean.csv'
    file='1m-2023-demo.csv'

    ap = Apportionment(num_seats, voters, link=link) 
    print("No of votes from source:", sum(ap.subject_votes.values()))
    print("Considered votes:", ap.voters)
    print("No. of seats:", num_seats)

    ap.iterated_simulate(file, 2023, nit=nit, group_size=group_size, divide="slovak")
