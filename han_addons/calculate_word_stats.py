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
write_to_filename = write_to_filename + "_wordstats.txt"
print("Will write results to " + write_to_filename)

class InstanceLenGenerator:
    def __init__(self, allennlp_formatted_reader, filepaths):
        self.allennlp_formatted_reader = allennlp_formatted_reader
        self.filepaths = filepaths

    def __iter__(self):
        for filepath in self.filepaths:
            for instance in tqdm(self.allennlp_formatted_reader._read(file_path=filepath)):
                instance_as_text_field = instance.fields['tokens']
                yield len(instance_as_text_field.tokens)


allennlp_reader = TextCatReader(word_tokenizer=WordTokenizer(word_filter=PassThroughWordFilter()),
                                segment_sentences=False)
len_generator = InstanceLenGenerator(allennlp_reader, filenames_to_use)
all_lengths = []
for length in iter(len_generator):
    all_lengths.append(length)

arr_of_lengths = np.array(all_lengths)
m = np.mean(arr_of_lengths)
sd = np.std(arr_of_lengths)
with open(write_to_filename, 'w') as f:
    f.write('Mean: ' + str(m) + '\n')
    f.write('SD:   ' + str(sd) + '\n')

print("Done calculating word stats.")
