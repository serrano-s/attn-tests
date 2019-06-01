"""
For the Yahoo data, I first appended a couple of characters to the end of file 1 (FullOct2007.xml.part1),
and prepended a couple of characters to the beginning of file 2 (FullOct2007.xml.part2)
to make each a complete XML file; although the split happens over a single instance,
it just so happens that all of the relevant fields for that instance fall in just
one of the files, so we can process it from that one and discard it from the other file.
"""

import datetime
import xml.etree.ElementTree as ElementTree
from allennlp.data.dataset_readers.text_classification.textcat import TextCatReader
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter, StopwordFilter
from tqdm import tqdm
from random import shuffle
import pickle
import os
import json
import sys

yahoo_fixed_filename_1 = '/Users/sofias6/Documents/checking-interpretability-methods/data/raw_data/yahoo/Webscope_L6-1/FullOct2007.xml.part1'
yahoo_fixed_filename_2 = '/Users/sofias6/Documents/checking-interpretability-methods/data/raw_data/yahoo/FullOct2007.xml.part2'
yahoo_accepted_label_file = '/Users/sofias6/Documents/checking-interpretability-methods/data/raw_data/yahoo/yahoo_labels.txt'

yahoo_output_full_data_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/yahoo10cat_alldata.tsv'
yahoo_output_train_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/yahoo10cat_train.tsv'
yahoo_output_dev_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/yahoo10cat_dev.tsv'
yahoo_output_test_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/yahoo10cat_test.tsv'

imdb_data_file = '/Users/sofias6/Documents/checking-interpretability-methods/data/raw_data/imdb/data.json'

imdb_output_full_data_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/imdb_alldata.tsv'
imdb_output_train_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/imdb_train.tsv'
imdb_output_dev_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/imdb_dev.tsv'
imdb_output_test_filename = '/Users/sofias6/Documents/checking-interpretability-methods/data/processed/imdb_test.tsv'


def make_pickled_list_of_instances_from_xml(filename, label_field_name, text_field_names, labels_to_care_about,
                                            instance_signifier, pkl_file_to_write_to, instance_id_field_name=None):
    print("Starting to read in data from xml file into python list of instances at " +
          str(datetime.datetime.now()))

    root = ElementTree.parse(filename).getroot()  # this takes quite a while for both yahoo files
    list_of_instances = []
    if instance_id_field_name is None:
        counter = 0
    for instance in tqdm(root.findall('.//' + instance_signifier), desc="Loading in data from xml file"):
        skip_instance = False
        str_label = None
        text = None
        if instance_id_field_name is not None:
            instance_id = None
        for attribute in instance:
            if attribute.tag == label_field_name:
                str_label = attribute.text.strip()
            elif instance_id_field_name is not None and attribute.tag == instance_id_field_name:
                instance_id = int(attribute.text)
            elif attribute.tag in text_field_names:
                if text is None:
                    text = attribute.text
                else:
                    text += ' ' + attribute.text
        if skip_instance:
            continue
        if str_label is None or text is None:
            if instance_id_field_name is None:
                # so that we don't throw our labeling of the data instance off by 1
                counter += 1
            continue
        if str_label not in labels_to_care_about:
            for accepted_label in labels_to_care_about:
                if str_label in accepted_label or accepted_label in str_label:
                    print("Careful: maybe mistakenly rejecting label " + str_label)
            continue
        if instance_id_field_name is not None:
            if instance_id is None:
                continue
        if instance_id_field_name is None:
            list_of_instances.append((counter, str_label, text))
            counter += 1
        else:
            list_of_instances.append((instance_id, str_label, text))
    with open(pkl_file_to_write_to, 'wb') as pickle_f:
        pickle.dump(list_of_instances, pickle_f)
    print("Successfully wrote pickle file " + str(pkl_file_to_write_to))


def get_writeable_text(text):
    return text.replace('\n', ' ').replace('\t', ' ')


def get_pickled_instance_list(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        ret = pickle.load(f)
    return ret


def get_printed_lines(list_of_lines):
    str_in_progress = ''
    for line in list_of_lines:
        str_in_progress += line
    return str_in_progress


def test_opening_file(fname):
    with open(fname, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            else:
                if line.strip() == '':
                    continue
                else:
                    line = line.strip('\n').split('\t')
                    assert len(line) == 3
                    try:
                        int(line[0])
                    except:
                        print("Error: first item in line wasn't an int.")
                        exit(1)


def reformat_file_to_remove_newlines_not_caught_by_replace(fname):
    with open(fname, 'r') as f:
        first_line = None
        list_of_instances = []
        inds_needing_fixing = []
        for line in f:
            if first_line is None:
                first_line = line
            else:
                if line.strip() == '':
                    continue
                line_split = line.strip('\n').split('\t')
                line_is_ok = (len(line_split) == 3)
                try:
                    int(line_split[0])
                except:
                    line_is_ok = False
                if line_is_ok:
                    list_of_instances.append((line_split[0], line_split[1], line_split[2]))
                else:
                    condensed_line = " ".join(line_split)
                    list_of_instances[-1] = (list_of_instances[-1][0], list_of_instances[-1][1],
                                             list_of_instances[-1][2] + ' ' + condensed_line)
                    if len(inds_needing_fixing) == 0 or inds_needing_fixing[-1] != len(list_of_instances) - 1:
                        inds_needing_fixing.append(len(list_of_instances) - 1)

    print("Num lines that still had newlines not caught by replace: " + str(len(inds_needing_fixing)))
    with open(fname, 'w') as f:
        f.write(first_line)
        for instance in list_of_instances:
            f.write(instance[0] + '\t' + instance[1] + '\t' + instance[2] + '\n')


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


def add_info_about_num_sents_and_max_num_tokens_to_data_file(data_filename):
    print("Starting to calculate information about number of sentences and max number of tokens for each " +
          "instance to add to " + data_filename)
    numsents_maxnumtokens = []
    numsents_maxnumtokens_if_not_considering_stop_words = []
    allennlp_formatted_reader = TextCatReader(word_tokenizer=WordTokenizer(word_filter=PassThroughWordFilter()),
                                              segment_sentences=True)
    pass_through_word_filter = PassThroughWordFilter()
    stop_word_filter = StopwordFilter()

    first_line = True
    num_instances_passed = 0
    with open(data_filename, 'r') as f:
        for line in tqdm(f):
            if first_line:
                # find which field is tokens
                temp_line = line
                tokens_field_ind = 0
                while not (temp_line.startswith('tokens\t') or temp_line.startswith('tokens\n')):
                    temp_line = temp_line[temp_line.index('\t') + 1:]
                    tokens_field_ind += 1
                first_line = False
            else:
                if line.strip() == '':
                    continue
                text_field = get_nth_field_in_line(line, tokens_field_ind)

                num_instances_passed += 1
                if len(text_field) == 0:
                    # reader will skip this, so its tuple is (0, 0)
                    numsents_maxnumtokens.append((0, 0))
                    numsents_maxnumtokens_if_not_considering_stop_words.append((0, 0))
                    continue

                allennlp_formatted_reader._word_tokenizer._word_filter = pass_through_word_filter
                instance = allennlp_formatted_reader.text_to_instance(tokens=text_field, category='placeholder')
                if instance is None:
                    # reader will skip this, so its tuple is (0, 0)
                    # and since the stop word version is even MORE restrictive, it would be skipped there too
                    numsents_maxnumtokens.append((0, 0))
                    numsents_maxnumtokens_if_not_considering_stop_words.append((0, 0))
                    continue
                list_of_sentences = instance.fields['tokens'].field_list
                num_sents = len(list_of_sentences)
                max_num_tokens = 0
                for sentence_as_text_field in list_of_sentences:
                    list_of_tokens = sentence_as_text_field.tokens
                    if len(list_of_tokens) > max_num_tokens:
                        max_num_tokens = len(list_of_tokens)
                numsents_maxnumtokens.append((num_sents, max_num_tokens))

                allennlp_formatted_reader._word_tokenizer._word_filter = stop_word_filter
                instance = allennlp_formatted_reader.text_to_instance(tokens=text_field,
                                                                         category='placeholder')
                if instance is None:
                    # reader will skip this, so its tuple is (0, 0)
                    numsents_maxnumtokens_if_not_considering_stop_words.append((0, 0))
                    continue
                list_of_sentences = instance.fields['tokens'].field_list
                num_sents = len(list_of_sentences)
                max_num_tokens = 0
                for sentence_as_text_field in list_of_sentences:
                    list_of_tokens = sentence_as_text_field.tokens
                    if len(list_of_tokens) > max_num_tokens:
                        max_num_tokens = len(list_of_tokens)
                numsents_maxnumtokens_if_not_considering_stop_words.append((num_sents, max_num_tokens))

    assert len(numsents_maxnumtokens) == num_instances_passed
    assert len(numsents_maxnumtokens_if_not_considering_stop_words) == num_instances_passed

    temp_full_filename = data_filename[:data_filename.rfind('.')] + "_temp.tsv"
    old_f = open(data_filename, 'r')
    first_line = True
    instance_counter = 0
    with open(temp_full_filename, 'w') as f:
        for line in old_f:
            if first_line:
                old_line = line[:line.rfind('\n')]

                new_line = (old_line + '\t' +
                            'num_sentences_post_stopword_removal' + '\t' +
                            'max_num_tokens_in_sentence_post_stopword_removal' + '\t' +
                            'num_sentences' + '\t' +
                            'max_num_tokens_in_sentence' + '\n')
                f.write(new_line)
                first_line = False
            else:
                if line.strip() == '':
                    continue
                str_num_sents_no_stopwords = str(
                    numsents_maxnumtokens_if_not_considering_stop_words[instance_counter][0])
                str_max_num_tokens_no_stopwords = str(
                    numsents_maxnumtokens_if_not_considering_stop_words[instance_counter][1])
                str_num_sents = str(numsents_maxnumtokens[instance_counter][0])
                str_max_num_tokens = str(numsents_maxnumtokens[instance_counter][1])
                old_line = line[:line.rfind('\n')]

                new_line = (old_line + '\t' +
                            str_num_sents_no_stopwords + '\t' +
                            str_max_num_tokens_no_stopwords + '\t' +
                            str_num_sents + '\t' +
                            str_max_num_tokens + '\n')

                f.write(new_line)
                instance_counter += 1
    old_f.close()

    if os.path.isfile(data_filename):
        os.remove(data_filename)
    os.rename(temp_full_filename, data_filename)


def make_yahoo_full_data_file():
    labels_to_care_about = []
    with open(yahoo_accepted_label_file, 'r') as f:
        for line in f:
            if line.strip() != '':
                labels_to_care_about.append(line.strip())
    pickle_filename_1 = yahoo_fixed_filename_1 + "_instances.pkl"
    pickle_filename_2 = yahoo_fixed_filename_2 + "_instances.pkl"
    # see note at beginning of file about characters added to ends of yahoo files
    make_pickled_list_of_instances_from_xml(yahoo_fixed_filename_1, 'maincat', ['subject', 'content', 'bestanswer'],
                                            labels_to_care_about, 'document', pickle_filename_1,
                                            instance_id_field_name='uri')
    make_pickled_list_of_instances_from_xml(yahoo_fixed_filename_1, 'maincat', ['subject', 'content', 'bestanswer'],
                                            labels_to_care_about, 'document', pickle_filename_2,
                                            instance_id_field_name='uri')
    # we reload the pickle files after finishing with both xml files because otherwise this python process
    # uses too much memory at once
    first_set_of_instances = get_pickled_instance_list(pickle_filename_1)
    second_set_of_instances = get_pickled_instance_list(pickle_filename_2)
    shuffle(first_set_of_instances)
    shuffle(second_set_of_instances)
    all_data_f = open(yahoo_output_full_data_filename, 'w')
    all_data_f.write('id' + '\t' + 'category' + '\t' + 'tokens' + '\n')
    instances_written_so_far = 0

    for instance in first_set_of_instances:
        str_id = str(instance[0])
        str_label = str(instance[1])
        str_text = get_writeable_text(instance[2])

        all_data_f.write(str_id + '\t' + str_label + '\t' + str_text + '\n')
        instances_written_so_far += 1
    for instance in second_set_of_instances:
        str_id = str(instance[0])
        str_label = str(instance[1])
        str_text = get_writeable_text(instance[2])

        all_data_f.write(str_id + '\t' + str_label + '\t' + str_text + '\n')
        instances_written_so_far += 1
    all_data_f.close()

    print("Total number of instances: " + str(instances_written_so_far))
    os.remove(pickle_filename_1)
    os.remove(pickle_filename_2)
    reformat_file_to_remove_newlines_not_caught_by_replace(yahoo_output_full_data_filename)

    add_info_about_num_sents_and_max_num_tokens_to_data_file(yahoo_output_full_data_filename)


def make_imdb_full_data_file():
    with open(imdb_data_file, 'r') as f:
        data = json.load(f)
    with open(imdb_output_full_data_filename, 'w') as f:
        f.write('id' + '\t' + 'category' + '\t' + 'tokens' + '\n')
        counter = 1
        for instance in data:
            category = str(instance["rating"]).replace('\t', ' ').replace('\n', ' ')
            tokens = str(instance["review"]).replace('\t', ' ').replace('\n', ' ')  # title is not used
            id = str(counter)
            f.write(id + '\t' + category + '\t' + tokens + '\n')
            counter += 1
    reformat_file_to_remove_newlines_not_caught_by_replace(imdb_output_full_data_filename)
    print("Wrote " + str(counter - 1) + " instances to file.")


def main():
    make_yahoo_full_data_file()
    make_imdb_full_data_file()


if __name__ == '__main__':
    add_info_about_num_sents_and_max_num_tokens_to_data_file(sys.argv[1])
