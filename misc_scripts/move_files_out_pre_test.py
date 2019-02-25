import os
import shutil
import sys
from glob import glob

dir = sys.argv[1]
if not dir.endswith('/'):
    dir += '/'
new_dir = dir + 'old_test_results/'
os.makedirs(new_dir)

fnames_to_move = [
    'dec_flip_stats.csv',
    'first_vs_second.csv',
    'grad_based_stats.csv',
    'rand_nontop_decflipjs.csv',
    'rand_sample_stats.csv'
]

for fname in fnames_to_move:
    shutil.move(dir + fname, new_dir + fname)

paths_starting_with__sentence = list(glob(dir + '_sentence_attention*'))
if len(paths_starting_with__sentence) > 0:
    inner_dir = dir + '_sentence_attention_corresponding_vects/'
    new_inner_dir = new_dir + '_sentence_attention_corresponding_vects/'
else:
    inner_dir = dir + '_word_attention_corresponding_vects/'
    new_inner_dir = new_dir + '_word_attention_corresponding_vects/'

os.makedirs(new_inner_dir)

inner_fnames_to_move = [
    'gradients/',
    'next_available_counter.txt'
]

for fname in inner_fnames_to_move:
    shutil.move(inner_dir + fname, new_inner_dir + fname)

print("Done moving files into " + new_dir)
