from allennlp.data.dataset_readers.text_classification.textcat import TextCatReader
from gensim.models import Word2Vec
import datetime
import os
import numpy as np
import h5py
from glob import glob
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN as unk_token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter
from allennlp.data import Vocabulary
from tqdm import tqdm
import resource
import argparse

filepaths_of_data_to_train_on = ['/homes/gws/sofias6/data/amazon_train.tsv',
                                 '/homes/gws/sofias6/data/amazon_dev.tsv']
dir_to_save_vocab_in = '/homes/gws/sofias6/vocabs/amazon-lowercase-vocab-30its/'
allocate_extra_memory = False
write_temp_file_of_all_provided_data_with_set_vocab = True
lowercase_all_tokens = True
embedding_file_tag = "_embeddings"
min_word_count_to_avoid_unking = 5
iterations_for_training_word_embeddings = 30
word_embedding_dimension = 200
vocabword_ind_file_ending = "_indstowords.txt"
vocabword_ind_not_numbered_file_ending = "_indstowords_nonums.txt"
label_ind_file_ending = "_indstolabels.txt"
label_ind_not_numbered_file_ending = "_indstolabels_nonums.txt"
lowercase_tag = "_lowercase"
if lowercase_all_tokens:
    embedding_file_tag = lowercase_tag + embedding_file_tag
    vocabword_ind_file_ending = lowercase_tag + vocabword_ind_file_ending
    vocabword_ind_not_numbered_file_ending = lowercase_tag + vocabword_ind_not_numbered_file_ending


class GensimSentenceIterator:
    def __init__(self, allennlp_formatted_reader, filepaths, valid_vocab_words:dict=None, total_num_sentences=None):
        self.allennlp_formatted_reader = allennlp_formatted_reader
        self.filepaths = filepaths
        self.valid_vocab_words = valid_vocab_words
        self.total_num_sentences = total_num_sentences

    def __iter__(self):
        for filepath in self.filepaths:
            if self.total_num_sentences is None:
                for instance in tqdm(self.allennlp_formatted_reader._read(file_path=filepath)):
                    list_of_sentences = instance.fields['tokens'].field_list
                    for sentence_as_text_field in list_of_sentences:
                        list_of_tokens = sentence_as_text_field.tokens
                        if lowercase_all_tokens:
                            if self.valid_vocab_words is None:
                                list_of_str_tokens = [token.text.lower() for token in list_of_tokens]
                            else:
                                # we know which words shouldn't be unked, so unk the rest
                                list_of_str_tokens = [token.text.lower() if (token.text.lower() in
                                                                             self.valid_vocab_words)
                                                      else unk_token for token in list_of_tokens]
                        else:
                            if self.valid_vocab_words is None:
                                list_of_str_tokens = [token.text for token in list_of_tokens]
                            else:
                                # we know which words shouldn't be unked, so unk the rest
                                list_of_str_tokens = [token.text if (token.text in self.valid_vocab_words) else
                                                      unk_token
                                                      for token in list_of_tokens]
                        yield list_of_str_tokens
            else:
                for instance in tqdm(self.allennlp_formatted_reader._read(file_path=filepath),
                                     total=self.total_num_sentences):
                    list_of_sentences = instance.fields['tokens'].field_list
                    for sentence_as_text_field in list_of_sentences:
                        list_of_tokens = sentence_as_text_field.tokens
                        if lowercase_all_tokens:
                            if self.valid_vocab_words is None:
                                list_of_str_tokens = [token.text.lower() for token in list_of_tokens]
                            else:
                                # we know which words shouldn't be unked, so unk the rest
                                list_of_str_tokens = [token.text.lower() if (token.text.lower() in
                                                                             self.valid_vocab_words)
                                                      else unk_token for token in list_of_tokens]
                        else:
                            if self.valid_vocab_words is None:
                                list_of_str_tokens = [token.text for token in list_of_tokens]
                            else:
                                # we know which words shouldn't be unked, so unk the rest
                                list_of_str_tokens = [token.text if (token.text in self.valid_vocab_words) else
                                                      unk_token
                                                      for token in list_of_tokens]
                        yield list_of_str_tokens


class SimpleIterator:
    def __init__(self, filepaths, total_num_sentences=None):
        self.filepaths = filepaths
        self.total_num_sentences = total_num_sentences

    def __iter__(self):
        for filepath in self.filepaths:
            with open(filepath, 'r') as f:
                if self.total_num_sentences is not None:
                    for line in tqdm(f, total=self.total_num_sentences):
                        if line.strip() == '':
                            continue
                        yield line[:-1].split(' ')  # get rid of the newline at the end
                else:
                    for line in tqdm(f):
                        if line.strip() == '':
                            continue
                        yield line[:-1].split(' ')  # get rid of the newline at the end


class LabelIterator:
    def __init__(self, allennlp_formatted_reader, filepaths):
        self.allennlp_formatted_reader = allennlp_formatted_reader
        self.filepaths = filepaths

    def __iter__(self):
        for filepath in self.filepaths:
            for instance in self.allennlp_formatted_reader.read(file_path=filepath):
                yield instance.fields['label'].label


def make_ind_to_word_mapping_file(wordcounts, threshold_for_inclusion_in_vocab, filename, filename_wo_nums):
    print("Starting to construct vocabulary from wordcounts at " + str(datetime.datetime.now()))
    num_actual_words_in_vocab = 0
    words_to_inds = {}
    f_nonums = open(filename_wo_nums, "w")
    with open(filename, "w") as f:
        for word in wordcounts.keys():
            if wordcounts[word] >= threshold_for_inclusion_in_vocab:
                num_actual_words_in_vocab += 1
                f.write(str(num_actual_words_in_vocab) + ":" + word + "\n")
                f_nonums.write(word + "\n")
                words_to_inds[word] = num_actual_words_in_vocab
        unk_index = num_actual_words_in_vocab + 1
        f.write(str(unk_index) + ":" + unk_token + "\n")
        f_nonums.write(unk_token + "\n")
    f_nonums.close()
    return words_to_inds, unk_index


def make_label_ind_files(label_iterator):
    next_unassigned_label = 0
    cur_label_dict = {}
    for label in iter(label_iterator):
        try:
            cur_label_dict[label]
        except:
            cur_label_dict[label] = next_unassigned_label
            next_unassigned_label += 1
    all_labels_are_numbers = True
    numerical_labels_so_far = []
    for label in cur_label_dict.keys():
        try:
            float_version = float(label)
            numerical_labels_so_far.append((float_version, label))
        except:
            all_labels_are_numbers = False
            break
    if all_labels_are_numbers:
        # then use one of the indexings that would arise from sorting the labels in numerical order
        all_labels_in_order = [tup[1] for tup in sorted(numerical_labels_so_far, key=(lambda x: x[0]), reverse=False)]
    else:
        # then just use the ordering from cur_label_dict
        all_labels_in_order = [None] * next_unassigned_label
        for label in cur_label_dict.keys():
            all_labels_in_order[cur_label_dict[label]] = label
        assert None not in all_labels_in_order

    first_data_filepath = filepaths_of_data_to_train_on[0]
    filename = first_data_filepath[:first_data_filepath.rfind('.')] + label_ind_file_ending
    filename_wo_nums = first_data_filepath[:first_data_filepath.rfind('.')] + label_ind_not_numbered_file_ending
    f_nonums = open(filename_wo_nums, "w")
    with open(filename, "w") as f:
        for label_ind in range(len(all_labels_in_order)):
            str_label = all_labels_in_order[label_ind]
            f.write(str(label_ind) + ":" + str_label + "\n")
            f_nonums.write(str_label + "\n")
    f_nonums.close()


def get_vocab_and_collect_num_sentences(sentence_iterator, first_data_filepath):
    wordcounts = {}
    total_num_sentences = 0
    for sentence in iter(sentence_iterator):
        total_num_sentences += 1
        for word in sentence:
            try:
                wordcounts[word] += 1
            except:
                wordcounts[word] = 1
    print("Collected wordcounts, now starting on vocab creation.")
    filename_wo_nums = first_data_filepath[:first_data_filepath.rfind('.')] + vocabword_ind_not_numbered_file_ending
    words_to_inds, num_words_in_vocab_including_unk = \
        make_ind_to_word_mapping_file(wordcounts, min_word_count_to_avoid_unking,
                                      first_data_filepath[:first_data_filepath.rfind('.')] +
                                      vocabword_ind_file_ending,
                                      filename_wo_nums)
    print("Finished creating vocabulary.")
    return words_to_inds, num_words_in_vocab_including_unk, total_num_sentences, filename_wo_nums


def make_temp_file_and_iterate_over_that_instead(sentence_iterator):
    print("Starting to make temp file containing tokenized sentences...")
    temp_filename = filepaths_of_data_to_train_on[0]
    temp_filename = temp_filename + ".temp"
    with open(temp_filename, 'w') as f:
        for sentence in iter(sentence_iterator):
            f.write(" ".join(sentence) + "\n")
    new_sentence_iterator = SimpleIterator([temp_filename])
    print("Done making that file.")
    return new_sentence_iterator, temp_filename


def get_word2vec_and_vocab_part_3():
    first_data_filepath = filepaths_of_data_to_train_on[0]
    vocab_filename_nonums = first_data_filepath[:first_data_filepath.rfind('.')] + \
                            vocabword_ind_not_numbered_file_ending
    num_words_in_vocab_including_unk = 0
    with open(vocab_filename_nonums, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            num_words_in_vocab_including_unk += 1
    print(str(num_words_in_vocab_including_unk) + " words in vocab, including unk token.")
    if write_temp_file_of_all_provided_data_with_set_vocab:
        temp_data_filename = first_data_filepath + ".temp"
        sentence_iterator = SimpleIterator([temp_data_filename])
        total_num_sentences = -1
        with open(temp_data_filename, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                total_num_sentences += 1
    else:
        allennlp_reader = TextCatReader(word_tokenizer=WordTokenizer(word_filter=PassThroughWordFilter()),
                                        segment_sentences=True)
        sentence_iterator = GensimSentenceIterator(allennlp_reader, filepaths_of_data_to_train_on)
        total_num_sentences = 0
        for sentence in iter(sentence_iterator):
            total_num_sentences += 1
    sentence_iterator.total_num_sentences = total_num_sentences

    print("Starting to train model.")

    trained_model = Word2Vec(None, iter=iterations_for_training_word_embeddings,
                             min_count=0, size=word_embedding_dimension, workers=4)
    trained_model.build_vocab(sentence_iterator)
    trained_model.train(sentence_iterator, total_examples=total_num_sentences,
                        epochs=iterations_for_training_word_embeddings)
    temp_filename = (first_data_filepath[:first_data_filepath.rfind('.')] + "_tempgensim")
    trained_model.save(temp_filename)

    print("Starting to move trained embeddings into numpy matrix at " + str(datetime.datetime.now()))
    np_embedding_filename = (first_data_filepath[:first_data_filepath.rfind('.')] + embedding_file_tag + ".npy")
    num_vocab_words = num_words_in_vocab_including_unk - 1
    embedding_matrix = np.zeros((num_vocab_words + 2, word_embedding_dimension))
    ind_counter = 1
    with open(vocab_filename_nonums, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            line = line[:-1]  # get rid of newline
            embedding_matrix[ind_counter] = trained_model[line]
            ind_counter += 1
    embedding_matrix[num_words_in_vocab_including_unk] = trained_model[unk_token]
    norm_of_embeddings = np.linalg.norm(embedding_matrix, axis=1)
    norm_of_embeddings[norm_of_embeddings == 0] = 1e-13
    embedding_matrix = embedding_matrix / norm_of_embeddings[:, None]
    np.save(np_embedding_filename, embedding_matrix)

    print("Starting to save numpy matrix as hdf5 file containing torch tensor at " + str(datetime.datetime.now()))
    hdf5_filename = (first_data_filepath[:first_data_filepath.rfind('.')] + embedding_file_tag + ".h5")
    with h5py.File(hdf5_filename, "w") as f:
        dset = f.create_dataset("embedding", (num_words_in_vocab_including_unk + 1, word_embedding_dimension),
                                dtype='f')
        dset[...] = embedding_matrix

    print("Removing temporary gensim model files at " + str(datetime.datetime.now()))
    # remove temp gensim model files, now that embedding matrix has been saved
    if os.path.isfile(temp_filename):
        os.remove(temp_filename)
    other_files_to_rm = glob(temp_filename + ".*")
    for fname in other_files_to_rm:
        os.remove(fname)
    if write_temp_file_of_all_provided_data_with_set_vocab:
        if os.path.isfile(temp_data_filename):
            os.remove(temp_data_filename)


def get_word2vec_and_vocab_part_1():
    first_data_filepath = filepaths_of_data_to_train_on[0]
    allennlp_reader = TextCatReader(word_tokenizer=WordTokenizer(word_filter=PassThroughWordFilter()),
                                    segment_sentences=True)
    sentence_iterator = GensimSentenceIterator(allennlp_reader, filepaths_of_data_to_train_on)
    words_to_inds, num_words_in_vocab_including_unk, total_num_sentences, vocab_filename_nonums = \
        get_vocab_and_collect_num_sentences(sentence_iterator, first_data_filepath)
    sentence_iterator.valid_vocab_words = words_to_inds
    if write_temp_file_of_all_provided_data_with_set_vocab:
        sentence_iterator, temp_data_filename = make_temp_file_and_iterate_over_that_instead(sentence_iterator)
        words_to_inds = None

    label_iterator = LabelIterator(allennlp_reader, filepaths_of_data_to_train_on)
    make_label_ind_files(label_iterator)


def get_word2vec_and_vocab_part_2():
    allennlp_reader = TextCatReader(word_tokenizer=WordTokenizer(word_filter=PassThroughWordFilter()),
                                    segment_sentences=True)
    label_iterator = LabelIterator(allennlp_reader, filepaths_of_data_to_train_on)
    make_label_ind_files(label_iterator)


def save_vocab_in_allennlp_format():
    first_data_filepath = filepaths_of_data_to_train_on[0]
    numless_vocab_file = first_data_filepath[:first_data_filepath.rfind('.')] + vocabword_ind_not_numbered_file_ending
    numless_label_file = first_data_filepath[:first_data_filepath.rfind('.')] + label_ind_not_numbered_file_ending

    vocab = Vocabulary()
    vocab.set_from_file(numless_vocab_file, is_padded=True, oov_token=unk_token, namespace='tokens')
    vocab.set_from_file(numless_label_file, is_padded=False, namespace='labels')
    vocab.save_to_files(dir_to_save_vocab_in)


def get_new_limits_for_resource(res):
    original_limits = resource.getrlimit(res)
    new_limits = (original_limits[1], original_limits[1])
    return new_limits


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-part", type=int, required=True,
                        help="Run part 1, 2, or 3 of word2vec-vocab preprocessing",
                        choices=[1, 2, 3])
    args = parser.parse_args()
    if allocate_extra_memory:
        resource.setrlimit(resource.RLIMIT_CORE, get_new_limits_for_resource(resource.RLIMIT_CORE))
        resource.setrlimit(resource.RLIMIT_CPU, get_new_limits_for_resource(resource.RLIMIT_CPU))
        resource.setrlimit(resource.RLIMIT_DATA, get_new_limits_for_resource(resource.RLIMIT_DATA))
        resource.setrlimit(resource.RLIMIT_FSIZE, get_new_limits_for_resource(resource.RLIMIT_FSIZE))
        resource.setrlimit(resource.RLIMIT_MEMLOCK, get_new_limits_for_resource(resource.RLIMIT_MEMLOCK))
        resource.setrlimit(resource.RLIMIT_RSS, get_new_limits_for_resource(resource.RLIMIT_RSS))
        print("Successfully increased script resources.")
    # we split these up because otherwise we use too much memory
    if args.run_part == 1:
        get_word2vec_and_vocab_part_1()
    elif args.run_part == 2:
        get_word2vec_and_vocab_part_2()
    else:
        get_word2vec_and_vocab_part_3()
        save_vocab_in_allennlp_format()


if __name__ == '__main__':
    main()
