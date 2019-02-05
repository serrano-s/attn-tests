import sys

test_data_file = sys.argv[1]
take_first_n = int(sys.argv[2])


list_to_take = []
counter = 0
with open(test_data_file, 'r') as f:
    for line in f:
        if counter == 0:
            first_line = line
        elif counter <= take_first_n:
            list_to_take.append(line)
        else:
            break
        counter += 1


with open(test_data_file[:test_data_file.rfind('.')] + '_first' + str(take_first_n) + '.tsv', 'w') as f:
    f.write(first_line)
    for line in list_to_take:
        f.write(line)
