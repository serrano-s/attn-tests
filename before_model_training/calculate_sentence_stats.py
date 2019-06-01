import sys
import os
from tqdm import tqdm
from allennlp.data.dataset_readers.text_classification.textcat import TextCatReader
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter
import numpy as np

filenames_to_use = []
for i in range(1, len(sys.argv)):
    assert os.path.isfile(sys.argv[i])
    filenames_to_use.append(sys.argv[i])

write_to_filename = filenames_to_use[0]
write_to_filename = write_to_filename[:write_to_filename.rfind('_')]
if '/' in write_to_filename:
    write_to_filename = write_to_filename[write_to_filename.rfind('/') + 1:]
write_to_filename = write_to_filename + "_sentstats.txt"
print("Will write results to " + write_to_filename)

def get_nth_field_in_line(line, ind):
    counter = 0
    while counter < ind:
        line = line[line.index('\t') + 1:]
        counter += 1
    if line.rfind('\t') == -1:
        # this is the last field, so just remove the trailing newline
        return line[:-1]
    else:
        # return the remaining line up to the next tab
        return line[:line.index('\t')]

def get_info_about_data_len_distribution(filepaths):
    numsents_maxnumtokens = []
    for filepath in filepaths:
        first_line = True
        with open(filepath, 'r') as f:
            for line in f:
                if first_line:
                    # find which fields are num_sentences and max_num_tokens_in_sentence
                    temp_line = line
                    num_sents_field_ind = 0
                    while not (temp_line.startswith('num_sentences\t') or temp_line.startswith('num_sentences\n')):
                        temp_line = temp_line[temp_line.index('\t') + 1:]
                        num_sents_field_ind += 1
                    temp_line = line
                    max_num_tokens_field_ind = 0
                    while not (temp_line.startswith('max_num_tokens_in_sentence\t') or
                               temp_line.startswith('max_num_tokens_in_sentence\n')):
                        temp_line = temp_line[temp_line.index('\t') + 1:]
                        max_num_tokens_field_ind += 1
                    first_line = False
                else:
                    if line.strip() == '':
                        continue
                    num_sents = int(get_nth_field_in_line(line, num_sents_field_ind))
                    max_num_tokens = int(get_nth_field_in_line(line, max_num_tokens_field_ind))
                    numsents_maxnumtokens.append((num_sents, max_num_tokens))

    num_sentences = [tup[0] for tup in numsents_maxnumtokens]
    return num_sentences


all_lengths = get_info_about_data_len_distribution(filenames_to_use)

arr_of_lengths = np.array(all_lengths)
m = np.mean(arr_of_lengths)
sd = np.std(arr_of_lengths)
with open(write_to_filename, 'w') as f:
    f.write('Mean: ' + str(m) + '\n')
    f.write('SD:   ' + str(sd) + '\n')

print("Done calculating sentence stats.")
