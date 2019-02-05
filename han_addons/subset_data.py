from numpy.random import choice
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_datasets import yahoo_output_full_data_filename
from random import shuffle
from random import random
from make_datasets import yahoo_output_train_filename
from make_datasets import yahoo_output_dev_filename
from make_datasets import yahoo_output_test_filename
from make_datasets import imdb_output_full_data_filename
from make_datasets import imdb_output_train_filename
from make_datasets import imdb_output_dev_filename
from make_datasets import imdb_output_test_filename
from make_datasets import get_nth_field_in_line
from bisect import bisect_right
import numpy as np


def get_writeable_text(text):
    return text.replace('\n', ' ').replace('\t', ' ')


def get_nonzero_len_instance_inds_by_class(data_filename):
    class_inds_dict = {}
    instance_ind = 0
    with open(data_filename, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                temp_line = line
                category_ind = 0
                while not (temp_line.startswith('category\t') or temp_line.startswith('category\n')):
                    temp_line = temp_line[temp_line.index('\t') + 1:]
                    category_ind += 1

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
                category = get_nth_field_in_line(line, category_ind)
                num_sents = int(get_nth_field_in_line(line, num_sents_field_ind))
                max_num_tokens = int(get_nth_field_in_line(line, max_num_tokens_field_ind))
                try:
                    list_to_append_to = class_inds_dict[category]
                except KeyError:
                    class_inds_dict[category] = []
                    list_to_append_to = class_inds_dict[category]
                if num_sents > 0 and max_num_tokens > 0:
                    list_to_append_to.append([instance_ind, num_sents, max_num_tokens])
                instance_ind += 1
    assert len(class_inds_dict) == 10
    print("Iterated over " + str(instance_ind) + " instances.")
    return class_inds_dict


def random_draw_and_remove_from_list(original_list, draw_x):
    total_num_instances = len(original_list)
    original_list = np.array(original_list)
    subset_inds = choice([i for i in range(total_num_instances)], size=draw_x, replace=False)
    binary_mask = np.zeros(total_num_instances, dtype=bool)
    binary_mask[subset_inds] = 1
    chosen = list(original_list[binary_mask])
    not_chosen = list(original_list[np.logical_not(binary_mask)])
    assert len(chosen) + len(not_chosen) == total_num_instances
    return chosen, not_chosen


def write_list_of_instances_to_file(fname, instances, first_line):
    with open(fname, 'w') as f:
        if not first_line.endswith('\n'):
            first_line = first_line + '\n'
        f.write(first_line)
        for instance in instances:
            if not instance.endswith('\n'):
                instance = instance + '\n'
            f.write(instance)
    print("Done writing instances to " + fname)


def make_class_balanced_train_dev_test_sets(full_data_filename, max_num_sents_for_traindev,
                                            max_max_num_tokens_for_traindev, num_test_instances_per_class,
                                            num_traindev_per_class, frac_traindev_that_are_dev,
                                            training_fname, dev_fname, test_fname):
    classes_to_inds = get_nonzero_len_instance_inds_by_class(full_data_filename)
    ind_lists = []
    for class_name in classes_to_inds:
        ind_lists.append(classes_to_inds[class_name])

    # first, choose test inds
    test_inds = []
    remaining_choices = []
    for class_inds in ind_lists:
        chosen, not_chosen = random_draw_and_remove_from_list(class_inds, num_test_instances_per_class)
        test_inds += chosen
        remaining_choices.append(not_chosen)
    test_inds = sorted([i[0] for i in test_inds])

    # now remove any instances from consideration for training or dev that are outside of the given windows
    for remaining_choice_list in remaining_choices:
        for i in range(len(remaining_choice_list) - 1, -1, -1):
            cur_val = remaining_choice_list[i]
            if cur_val[1] > max_num_sents_for_traindev or cur_val[2] > max_max_num_tokens_for_traindev:
                del remaining_choice_list[i]
            else:
                remaining_choice_list[i] = cur_val[0]

    traindev_inds = []
    ind_lists = remaining_choices
    for class_inds in ind_lists:
        chosen, not_chosen = random_draw_and_remove_from_list(class_inds, num_traindev_per_class)
        traindev_inds += chosen

    # now randomly separate traindev into training and dev
    dev_inds, training_inds = random_draw_and_remove_from_list(traindev_inds,
                                                               int(frac_traindev_that_are_dev * len(traindev_inds)))
    dev_inds = sorted(dev_inds)
    training_inds = sorted(training_inds)

    training_instances = []
    dev_instances = []
    test_instances = []
    first_line = None
    instance_counter = 0
    with open(full_data_filename, 'r') as f:
        for line in f:
            if first_line is None:
                first_line = line
            else:
                if line.strip() == '':
                    continue
                if len(training_inds) > 0 and instance_counter == training_inds[0]:
                    training_instances.append(line)
                    del training_inds[0]
                elif len(dev_inds) > 0 and instance_counter == dev_inds[0]:
                    dev_instances.append(line)
                    del dev_inds[0]
                elif len(test_inds) > 0 and instance_counter == test_inds[0]:
                    test_instances.append(line)
                    del test_inds[0]
                instance_counter += 1

    assert len(training_inds) == 0
    assert len(dev_inds) == 0
    assert len(test_inds) == 0

    shuffle(training_instances)
    shuffle(dev_instances)
    shuffle(test_instances)

    print("Collected " + str(len(training_instances)) + " training instances.")
    print("Collected " + str(len(dev_instances)) + " dev instances.")
    print("Collected " + str(len(test_instances)) + " test instances.")

    write_list_of_instances_to_file(training_fname, training_instances, first_line)
    write_list_of_instances_to_file(dev_fname, dev_instances, first_line)
    write_list_of_instances_to_file(test_fname, test_instances, first_line)


def make_train_dev_test_sets(full_data_filename, pct_train_instances, pct_dev_instances,
                             training_fname, dev_fname, test_fname):
    with open(full_data_filename, 'r') as f:
        num_instances = -1
        for line in f:
            if line.strip() == '':
                continue
            num_instances += 1
        print("Found " + str(num_instances) + " instances in total.")

    training_inds = []
    dev_inds = []
    test_inds = []

    for i in range(num_instances):
        decider = random()
        if decider < pct_train_instances:
            training_inds.append(i)
        elif decider < pct_train_instances + pct_dev_instances:
            dev_inds.append(i)
        else:
            test_inds.append(i)

    training_instances = []
    dev_instances = []
    test_instances = []
    first_line = None
    instance_counter = 0
    with open(full_data_filename, 'r') as f:
        for line in f:
            if first_line is None:
                first_line = line
            else:
                if line.strip() == '':
                    continue
                if len(training_inds) > 0 and instance_counter == training_inds[0]:
                    training_instances.append(line)
                    del training_inds[0]
                elif len(dev_inds) > 0 and instance_counter == dev_inds[0]:
                    dev_instances.append(line)
                    del dev_inds[0]
                elif len(test_inds) > 0 and instance_counter == test_inds[0]:
                    test_instances.append(line)
                    del test_inds[0]
                instance_counter += 1

    assert len(training_inds) == 0
    assert len(dev_inds) == 0
    assert len(test_inds) == 0

    shuffle(training_instances)
    shuffle(dev_instances)
    shuffle(test_instances)

    print("Collected " + str(len(training_instances)) + " training instances.")
    print("Collected " + str(len(dev_instances)) + " dev instances.")
    print("Collected " + str(len(test_instances)) + " test instances.")

    write_list_of_instances_to_file(training_fname, training_instances, first_line)
    write_list_of_instances_to_file(dev_fname, dev_instances, first_line)
    write_list_of_instances_to_file(test_fname, test_instances, first_line)


def get_info_about_data_len_distribution(filepaths, base_output_image_dir):
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
    max_num_tokens = [tup[1] for tup in numsents_maxnumtokens]
    fig = plt.figure()
    plt.scatter(num_sentences, max_num_tokens, s=2)
    plt.xlabel("# of sentences in instance")
    plt.ylabel("Max # tokens in instance")
    plt.savefig(base_output_image_dir + "sents_by_maxnumtokens.png", bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    num_sentences = sorted(num_sentences)
    plt.hist(num_sentences, log=True, bins=range(min(num_sentences), max(num_sentences) + 1, 1))
    plt.title("# of sentences in instance")
    plt.savefig(base_output_image_dir + "sents_histogram.png", bbox_inches='tight')
    plt.close(fig)

    decided = False
    while not decided:
        greater_than = int(input("Find % of instances with # of sentences greater than: "))
        i = bisect_right(num_sentences, greater_than)
        if i:
            print(((len(num_sentences) - 1) - i)/len(num_sentences))
        decided = (input("Done? (y/n): ").startswith('y'))

    fig = plt.figure()
    max_num_tokens = sorted(max_num_tokens)
    plt.hist(max_num_tokens, log=True, bins=range(min(max_num_tokens), max(max_num_tokens) + 1, 1))
    plt.title("Max # tokens in instance")
    plt.savefig(base_output_image_dir + "maxnumtokens_histogram.png", bbox_inches='tight')
    plt.close(fig)

    decided = False
    while not decided:
        greater_than = int(input("Find % of instances with max # of tokens greater than: "))
        i = bisect_right(max_num_tokens, greater_than)
        if i:
            print(((len(max_num_tokens) - 1) - i) / len(max_num_tokens))
        decided = (input("Done? (y/n): ").startswith('y'))


def split_file_into_training_and_dev(filename, frac_that_should_be_dev):
    assert filename.endswith("traindev.tsv")
    new_training_name = filename[:filename.rfind("traindev.tsv")] + "train.tsv"
    new_dev_name = filename[:filename.rfind("traindev.tsv")] + "dev.tsv"
    train_f = open(new_training_name, 'w')
    dev_f = open(new_dev_name, 'w')
    num_training = 0
    num_dev = 0
    with open(filename, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                train_f.write(line)
                dev_f.write(line)
                first_line = False
            else:
                if line.strip() == '':
                    continue
                decider = random()
                if decider < frac_that_should_be_dev:
                    dev_f.write(line)
                    num_dev += 1
                else:
                    train_f.write(line)
                    num_training += 1
    train_f.close()
    dev_f.close()
    print("Selected " + str(num_training) + " training instances.")
    print("Selected " + str(num_dev) + " dev instances.")


def get_version_without_outliers(filepaths, max_num_sents, max_num_tokens_in_sent):
    for filepath in filepaths:
        new_filepath = filepath[:filepath.rfind('.tsv')] + "_remoutliers.tsv"
        new_f = open(new_filepath, 'w')
        first_line = True
        counter = 0
        with open(filepath, 'r') as f:
            for line in f:
                if first_line:
                    new_f.write(line)
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
                    if num_sents <= max_num_sents and max_num_tokens <= max_num_tokens_in_sent:
                        new_f.write(line)
                        counter += 1
        print("Wrote " + str(counter) + " instances to file.")
        new_f.close()


def remove_zero_length_instances_from_file(filepath):
    new_filepath = filepath[:filepath.rfind('.tsv')] + "_nozerolens.tsv"
    new_f = open(new_filepath, 'w')
    first_line = True
    counter = 0
    with open(filepath, 'r') as f:
        for line in f:
            if first_line:
                new_f.write(line)
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
                if num_sents > 0 or max_num_tokens > 0:
                    new_f.write(line)
                    counter += 1
    print("Wrote " + str(counter) + " instances to file.")
    new_f.close()


def main():
    get_info_about_data_len_distribution([yahoo_output_full_data_filename], '/Users/sofias6/Downloads/')
    make_class_balanced_train_dev_test_sets(yahoo_output_full_data_filename, 50, 350, 5000, 140000, .1,
                                            yahoo_output_train_filename, yahoo_output_dev_filename,
                                            yahoo_output_test_filename)
    split_file_into_training_and_dev("/Users/sofias6/Downloads/amazon_traindev.tsv", .1)
    split_file_into_training_and_dev("/Users/sofias6/Downloads/yelp_traindev.tsv", .1)
    make_train_dev_test_sets(imdb_output_full_data_filename, .8, .1, imdb_output_train_filename,
                             imdb_output_dev_filename, imdb_output_test_filename)
    get_info_about_data_len_distribution(['/homes/gws/sofias6/data/amazon_train.tsv'], '/homes/gws/sofias6/')
    get_version_without_outliers(['/homes/gws/sofias6/data/amazon_train.tsv'], 15, 130)
    get_info_about_data_len_distribution(['/homes/gws/sofias6/data/yelp_train.tsv'], '/homes/gws/sofias6/')
    get_version_without_outliers(['/homes/gws/sofias6/data/yelp_train.tsv'], 43, 200)
    get_info_about_data_len_distribution(['/homes/gws/sofias6/data/yahoo10cat_train.tsv'], '/homes/gws/sofias6/')
    get_version_without_outliers(['/homes/gws/sofias6/data/yahoo10cat_train.tsv'], 25, 175)
    remove_zero_length_instances_from_file('/homes/gws/sofias6/data/amazon_test.tsv')


if __name__ == '__main__':
    main()
