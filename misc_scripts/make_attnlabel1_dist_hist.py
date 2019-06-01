import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import fabs
matplotlib.use('Agg')

from tqdm import tqdm
from math import ceil

filepath = '/homes/gws/sofias6/models/imdb-hanrnn-attnperf/imdb_test_reallabel_guessedlabel.csv'
output_filename = '/homes/gws/sofias6/attn-tests/imgs/data_descriptions/bad_attn_performance_dist/imdb-hanrnn-test-guessedperformance'

if filepath.endswith('.csv'):
    is_csv = True
else:
    is_csv = False

list_of_lens = []
counter = 0
with open(filepath, 'r') as f:
    if is_csv:
        f.readline()
    for line in tqdm(f):
        line = line.strip()
        if is_csv:
            line = line[line.index(',') + 1:]
        if line == '1':
            list_of_lens.append(counter)
        counter += 1

print("Collected " + str(len(list_of_lens)) + " indices corresponding to 1s")

fig = plt.figure()

plt.title(output_filename[output_filename.rfind('/') + 1:] + ' dist of 1 labels')

bin_width = ceil(counter / 50)
plt.hist(list_of_lens, bins=[i * bin_width for i in range(ceil(counter / bin_width) + 1)])
plt.savefig(output_filename)

plt.close(fig)

print("Successfully made figure.")
print("Saved figure to " + output_filename)
