import random
import time
import csv
import numpy as np

class Apportionment:

    def __init__(self, num_seats, total_voters, treshold=lambda: 5):
        self.num_seats = num_seats
        self.total_voters = total_voters
        self.active_voters = None
        self.subject_votes = {}
        self.subject_names = {}
        self.treshold = treshold
        self.boxes = None

    def read_votes_from_csv(self, link):
        self.active_voters = 0

        with open(link, 'r', encoding='utf-8') as csvfile:
            data = csv.reader(csvfile)
            
            # Skip header
            next(data)

            for row in data:
                if len(row) >= 3:
                    subject_number = row[0].strip()
                    subject_name = row[1].strip()
                    valid_votes = int(row[2].strip())
                    self.subject_votes[subject_number] = valid_votes
                    self.subject_names[subject_number] = subject_name
                    self.active_voters += valid_votes
    
    def counted_votes(self):
        if self.active_voters == None:
            self.active_voters = sum(self.subject_votes.values())
        return{x : y for x, y in self.subject_votes.items() if ((y * 100) / self.active_voters) > self.treshold()}


    def slovak_apportionment(self): ## returns dictionary subject_number : seats

        def get_top_x_indexes(numbers, x):
            if x >= len(numbers):
                return list(range(len(numbers)))
            
            indexes_with_values = list(enumerate(numbers))
            indexes_with_values.sort(key=lambda x: (-x[1], random.random()))
            
            top_x_indexes = [index for index, _ in indexes_with_values[:x]]
            
            return top_x_indexes

        counted_votes = self.counted_votes()

        sum_counted_votes = sum(counted_votes.values())
        republic_number = round(sum_counted_votes / (num_seats + 1))
        
        seats_given = [int(x / republic_number) for x in counted_votes.values()]
        division_remainders = [x / republic_number - int(x / republic_number) for x in counted_votes.values()]
        if sum(seats_given) > 150:
            # this requires more testing
            seats_given[division_remainders.index(min(division_remainders))] -= 1
        else:    
            for x in get_top_x_indexes(division_remainders, num_seats - sum(seats_given)):
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


    def basic_simulation(self):
        probabilities = {x: y for x, y in self.subject_votes.items()}
        probabilities["No valid vote"] = self.total_voters - self.active_voters

        # Check if the probabilities sum to 1. If not, normalize them.
        total_prob = sum(probabilities.values())
        if total_prob != 1:
            probabilities = {key: prob / total_prob for key, prob in probabilities.items()}

        choices = []
        for _ in range(self.total_voters):
            choice = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
            choices.append(choice)

        results = {key: choices.count(key) for key in set(choices)}
        sorted_results = {k: results[k] for k in sorted(results.keys())}
        return sorted_results


    def numpy_simulation(self):
        probabilities = {x: y for x, y in self.subject_votes.items()}
        probabilities["No valid vote"] = self.total_voters - self.active_voters

        # Check if the probabilities sum to 1. If not, normalize them.
        total_prob = sum(probabilities.values())
        if total_prob != 1:
            probabilities = {key: prob / total_prob for key, prob in probabilities.items()}

        keys, probs = zip(*probabilities.items())

        choices = np.random.choice(keys, self.total_voters, p=probs)

        results = {key: np.count_nonzero(choices == key) for key in set(choices)}
        sorted_results = {k: results[k] for k in sorted(results.keys())}
        return sorted_results

    def boxes_simulation(self):
        probabilities = {x: y for x, y in self.subject_votes.items()}
        probabilities["No valid vote"] = self.total_voters - self.active_voters

        # Check if the probabilities sum to 1. If not, normalize them.
        total_prob = sum(probabilities.values())
        if total_prob != 1:
            probabilities = {key: prob / total_prob for key, prob in probabilities.items()}

        keys, probs = zip(*probabilities.items())

        if self.boxes == None:
            self.boxes = [np.random.choice(keys, 100000, p=probs) for _ in range(10)]

        choices = np.random.choice(keys, self.total_voters % 100000, p=probs)
        print(type(choices[0]))
        for _ in range(int(self.total_voters / 100000)):
            rand_num = np.random.randint(10)
            addition = self.boxes[rand_num]
            print(choices, addition)
            choices = np.concatenate((choices, addition))
        print(choices, len(choices))

        results = {key: np.count_nonzero(choices == key) for key in set(choices)}
        sorted_results = {k: results[k] for k in sorted(results.keys())}
        return sorted_results

    def advanced_simulation(self):
        # Implement advanced method logic here
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

    def iterated_simulate(self, method, file, num_simulations=10):
        print("Initializing simulation...")
        start_time = time.time()

        with open(file, 'w', newline='', encoding='utf-8') as csvfile:
            
            fieldnames = ['Time', 'No valid vote']
            fieldnames.extend(self.subject_names.values())
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with open("ap" + file, 'w', newline='', encoding='utf-8') as seatsfile:
                
                seatsfieldnames = self.subject_names.values()

                seatswriter = csv.DictWriter(seatsfile, fieldnames=seatsfieldnames)
                seatswriter.writeheader()


                for i in range(num_simulations):
                    results = self.simulate_results(method)
                    simulation_time = time.time() - start_time

                    row_data = {'Time': simulation_time}
                    for xx, yy in results.items():
                        try:
                            row_data[self.subject_names[xx]] = yy
                        except KeyError:
                            row_data['No valid vote'] = yy
                    writer.writerow(row_data)
                    print(f'{i+1} / {num_simulations}')

                    ap = Apportionment(self.num_seats, self.total_voters)
                    del results['No valid vote']
                    ap.subject_votes = {self.subject_names[x] : y for x, y in results.items()}
                    export = ap.divide_seats('slovak')
                    seatswriter.writerow(export)

        print(f'''Simulation finished. Results in file {file}\nTime: {simulation_time} seconds.''')

if __name__ == "__main__":

    total_voters = 4388872
    num_seats = 150
    votes = {}

    ap = Apportionment(num_seats, total_voters) # -> TODO higher tresholds for coalitions
    #necessary for proper testing of 2020 elections
    ap.read_votes_from_csv('NRSR2023_SK_tab03a.csv')
    # print(ap.subject_votes)
    # print(ap.subject_names)
    # print(ap.divide_seats("slovak"))
    ### FROM THERE
    rex = ap.divide_seats("hagenbach bischoff")
    print(sum(rex.values()))
    ll = {ap.subject_names[x]: y  for x, y in rex.items()}
    for xx, yy in ll.items(): print(f'{yy} \t {xx}')
    ### TO THERE 
    ### I want to encapsulate as __str__ or something like that
    ap.iterated_simulate('boxes', 'test.csv')