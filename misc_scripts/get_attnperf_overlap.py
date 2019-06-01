# all label filenames should have same length. Each line should consist of either a 1 or a 0.
label_filenames = []

list_of_lists = []
with open(label_filenames[0], 'r') as f:
    for line in f:
        line = line.strip()
        if line != '':
            list_of_lists.append([int(line)])

for i in range(1, len(label_filenames)):
    with open(label_filenames[i], 'r') as f:
        counter = 0
        for line in f:
            line = line.strip()
            if line != '':
                list_of_lists[counter].append(int(line))
                counter += 1

results_tally = [0] * (len(list_of_lists[0]) + 1)

for label_list in list_of_lists:
    s = sum(label_list)
    results_tally[s] += 1

total_num_instances = sum(results_tally)

print()
for i in range(len(results_tally)):
    print("Num instances with a totaled label of " + str(i) + " across all models: " +
          str(results_tally[i]) + " (" + str(100 * results_tally[i] / total_num_instances) + "%)")
print()
