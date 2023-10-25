import random

def get_top_x_indexes(numbers, x):
    if x >= len(numbers):
        return list(range(len(numbers)))
    
    indexes_with_values = list(enumerate(numbers))
    indexes_with_values.sort(key=lambda x: (-x[1], random.random()))
    
    top_x_indexes = [index for index, _ in indexes_with_values[:x]]
    
    return top_x_indexes

def division_sk(votes, names, seats):
    counted_votes = {x: y for x, y, z in votes if z >= 5}

    all_votes = sum(counted_votes.values())
    republic_number = round(all_votes / (seats + 1))
    
    seats_given = [int(x / republic_number) for x in counted_votes.values()]
    division_remainders = [x / republic_number - int(x / republic_number) for x in counted_votes.values()]
    if sum(seats_given) > 150:
        # this requires more testing
        seats_given[division_remainders.index(min(division_remainders))] -= 1
    else:    
        for x in get_top_x_indexes(division_remainders, seats - sum(seats_given)):
            seats_given[x] += 1
    return {x: y for x, y in zip(counted_votes.keys(), seats_given)}    
    # return {names[x]: y for x, y in zip(counted_votes.keys(), seats_given)}    



if __name__ == "__main__":
    import reader

    votes, names = reader.reader()
    seats = 150
    res = division_sk(votes, names, seats)
    print(sum(res.values()))
    for x, y in res.items():
        print(f'{x} \t {y}')