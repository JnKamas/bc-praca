import csv

def reader():
    votes_counts = []
    party_names = {}

    with open('volebny zakon/NRSR2023_SK_tab03a.csv', 'r', encoding='utf-8') as csvfile:
        data = csv.reader(csvfile)
        
        next(data)

        for row in data:
            if len(row) >= 3:
                first_column = row[0].strip()
                second_column = row[1].strip()
                third_column = int(row[2].strip())
                forth_column = float(row[3].strip())
                votes_counts.append([first_column, third_column, forth_column])
                party_names[first_column] = second_column
    print([[x[0], x[1]] for x in votes_counts])
    return votes_counts, party_names

if __name__ == "__main__":
    print(reader())






