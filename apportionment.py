import random
import time
import csv

class Apportionment:

    def __init__(self, num_seats, total_voters):
        self._num_seats = num_seats
        self._total_voters = total_voters
        self._active_voters = -1
        self._subject_votes = {}
        self._subject_names = {}


    def get_num_seats(self):
        return self._num_seats

    def get_subject_votes(self):
        return self._subject_votes

    def get_total_voters(self):
        return self._total_voters

    def get_active_voters(self):
        return self._active_voters
    
    def get_subject_names(self):
        return self._subject_names

    def read_votes_from_csv(self, link):
        self._active_voters = 0

        with open(link, 'r', encoding='utf-8') as csvfile:
            data = csv.reader(csvfile)
            
            # Skip header
            next(data)

            for row in data:
                if len(row) >= 3:
                    subject_number = row[0].strip()
                    subject_name = row[1].strip()
                    valid_votes = int(row[2].strip())
                    self._subject_votes[subject_number] = valid_votes
                    self._subject_names[subject_number] = subject_name
                    self._active_voters += valid_votes
    
    def _counted_votes(self):
        return{x : y for x, y in self._subject_votes.items() if ((y * 100) / self._active_voters) > 5} # -> TODO higher tresholds for coalitions


    def _slovak_apportionment(self): ## returns dictionary subject_number : seats

        def get_top_x_indexes(numbers, x):
            if x >= len(numbers):
                return list(range(len(numbers)))
            
            indexes_with_values = list(enumerate(numbers))
            indexes_with_values.sort(key=lambda x: (-x[1], random.random()))
            
            top_x_indexes = [index for index, _ in indexes_with_values[:x]]
            
            return top_x_indexes

        counted_votes = self._counted_votes()

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

    def _dhont_apportionment(self):
        results = {}
        
        counted_votes = self._counted_votes()

        sorted_subject_votes = sorted(counted_votes.items(), key=lambda x: x[1], reverse=True)
        
        allocated_seats = {subject_number: 0 for subject_number in self._subject_votes}
        
        for _ in range(self._num_seats):
            subject_number, votes = max(sorted_subject_votes, key=lambda x: x[1] / (allocated_seats[x[0]] + 1))
            
            allocated_seats[subject_number] += 1
            
            results[subject_number] = allocated_seats[subject_number]
        
        return results


    def divide_seats(self, method):
        if method == "slovak":
            return self._slovak_apportionment()
        elif method == "d'hont":
            return self._dhont_apportionment()
        else:
            print("Invalid method choice. Please choose 'slovak' or 'd'hont'.")


    def _basic_simulation(self):
        # Implement basic method logic here
        pass

    def _numpy_simulation(self):
        # Implement numpy method logic here
        pass

    def _boxes_simulation(self):
        # Implement boxes method logic here
        pass

    def _advanced_simulation(self):
        # Implement advanced method logic here
        pass

    def simulate_results(self, method):
        if method == "basic":
            return self._basic_simulation()
        elif method == "numpy":
            return self._basic_simulation()
        elif method == "boxes":
            return self._basic_simulation()
        elif method == "advanced":
            return self._basic_simulation()
        else:
            print("Invalid option. Please choose 'basic,' 'numpy,' 'boxes,' or 'advanced'.")


if __name__ == "__main__":

    total_voters = 4388872
    num_seats = 150
    votes = {}

    ap = Apportionment(num_seats, total_voters)
    ap.read_votes_from_csv('NRSR2023_SK_tab03a.csv')
    # print(ap.get_subject_votes())
    # print(ap.get_subject_names())
    # print(ap.divide_seats("slovak"))
    rex = ap.divide_seats("d'hont")
    print(sum(rex.values()))
    ll = {ap.get_subject_names()[x]: y  for x, y in rex.items()}
    for xx, yy in ll.items(): print(f'{yy} \t {xx}')