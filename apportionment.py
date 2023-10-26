import random
import time
import csv

class Apportionment:

    def __init__(self, num_seats, total_voters):
        self._num_seats = num_seats
        self._total_voters = total_voters
        self._subject_votes = {}
        self._subject_names = {}


    def get_num_seats(self):
        return self._num_seats

    def get_subject_votes(self):
        return self._subject_votes

    def get_total_voters(self):
        return self._total_voters
    
    def get_subject_names(self):
        return self._subject_names

    def read_votes_from_csv(self, link):

        with open(link, 'r', encoding='utf-8') as csvfile:
            data = csv.reader(csvfile)
            
            # Skip header
            next(data)

            for row in data:
                if len(row) >= 3:
                    subject_number = row[0].strip()
                    subject_name = row[1].strip()
                    valid_votes = int(row[2].strip())
                    self._subject_votes[subject_number] =  valid_votes
                    self._subject_names[subject_number] = subject_name
    

    def _slovak_apportionment(self):
        # Implement Slovak method logic here
        pass

    def _dhont_apportionment(self):
        # Implement D'Hondt method logic here
        pass

    def divide_seats(self, method):
        if method == "slovak":
            self._slovak_apportionment()
        elif method == "d'hont":
            self._dhont_apportionment()
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
    print(ap.get_subject_votes())
    print(ap.get_subject_names())