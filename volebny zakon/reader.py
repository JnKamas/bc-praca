import csv

def reader(variant=0):
    votes_counts = []
    party_names = {}

    names_votes = {}

    with open('volebny zakon/NRSR2023_SK_tab03a.csv', 'r', encoding='utf-8') as csvfile:
        data = csv.reader(csvfile)
        
        next(data)

        for row in data:
            if len(row) >= 3:
                subject_number = row[0].strip()
                subject_name = row[1].strip()
                valid_votes = int(row[2].strip())
                valid_votes_percval = float(row[3].strip())
                votes_counts.append([subject_number, valid_votes, valid_votes_percval])
                party_names[subject_number] = subject_name
                names_votes[subject_name] = valid_votes
    if variant == 0:
        return votes_counts, party_names
    return names_votes

if __name__ == "__main__":
    print(reader())






