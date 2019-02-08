import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import normalize
from allennlp.models.model import Model
from allennlp.common import Params
from allennlp.models import load_archive
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter
from allennlp.data.iterators import BasicIterator, DataIterator
from textcat import TextCatReader
from allennlp.data import Vocabulary
import attn_tests_lib
from allennlp.nn import util
from scipy.misc import logsumexp
from attn_tests_lib import load_attn_dists, load_log_unnormalized_attn_dists
from attn_tests_lib import IntermediateBatchIterator
import json
from plain_model_test import evaluate
from allennlp.common.util import import_submodules
from random import shuffle
import pickle
from copy import deepcopy


import random
random.seed(5)
np.random.seed(5)
torch.manual_seed(5)
# Seed all GPUs with the same seed if available.
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(5)



unchanged_fname = "original_results.csv"
first_v_second_fname = "first_vs_second.csv"
dec_flip_stats_fname = "dec_flip_stats.csv"
rand_results_fname = "rand_sample_stats.csv"
grad_based_stats_fname = "grad_based_stats.csv"
dec_flip_rand_nontop_stats_fname = "rand_nontop_decflipjs.csv"
num_rand_samples_to_take = 5


def load_testing_models_in_eval_mode_from_serialization_dir(s_dir, attn_weight_filename, corr_vector_dir,
                                                            total_num_test_instances, training_config_filename,
                                                            name_of_attn_layer_to_replace="_sentence_attention",
                                                            cuda_device=-1):
    loaded_params = Params.from_file(training_config_filename, "")
    model = Model.load(loaded_params, s_dir, cuda_device=cuda_device)

    original_attn_layer = getattr(model, name_of_attn_layer_to_replace)
    talkative_attn_layer = \
        attn_tests_lib.TalkativeSimpleHanAttention(original_attn_layer, attn_weight_filename, corr_vector_dir,
                                                   total_num_test_instances)
    setattr(model, name_of_attn_layer_to_replace, talkative_attn_layer)
    just_the_classifier = \
        attn_tests_lib.ClassifierFromAttnAndInputVects(model._output_logit)
    if cuda_device >= 0:
        model = model.cuda(device=cuda_device)
        just_the_classifier = just_the_classifier.cuda(device=cuda_device)
    model = model.eval()
    just_the_classifier = just_the_classifier.eval()
    return model, just_the_classifier


def set_up_inorder_test_loader(data_file, batch_size, max_samples_per_batch, vocab_dir_name, model_folder,
                               check_num_iterated=True, loading_han=True):
    # make sure the number of test instances that the iterator will present is equal to the number of test
    # instances that we think we're iterating through
    # first, get that second number: how many test instances do we *think* we should be iterating through?
    print("Calculating the number of instances expected from the test set iterator...")
    num_instances_expected = -1  # to account for first line
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip() != '':
                num_instances_expected += 1

    if not model_folder.endswith('/'):
        model_folder += '/'

    # now construct the iterator
    config = Params.from_file(model_folder + 'config.json')
    reader_params = config.pop('validation_dataset_reader', None)
    if reader_params is None:
        reader_params = config.pop('dataset_reader', None)
    dataset_reader = DatasetReader.from_params(reader_params)

    if loading_han:
        dataset_iterator = BasicIterator(batch_size=batch_size,
                                         maximum_samples_per_batch=("list_num_tokens", max_samples_per_batch))
    else:
        dataset_iterator = BasicIterator(batch_size=batch_size,
                                         maximum_samples_per_batch=("num_tokens", max_samples_per_batch))
    dataset_iterator.index_with(Vocabulary.from_files(vocab_dir_name))

    # now iterate through and make sure the number of instances we pass matches what we expect
    """if check_num_iterated:
        print("Checking how many instances the iterator actually goes through...")
        num_instances_found = 0
        for batch in dataset_iterator._create_batches(dataset_reader._read(data_file), shuffle=False):
            batch.index_instances(dataset_iterator.vocab)
            tokens = batch.as_tensor_dict()['tokens']['tokens']
            num_instances_found += tokens.size(0)

        assert num_instances_expected == num_instances_found, ("From file, expected " + str(num_instances_expected) +
                                                               " test instances, but dataset iterator turned up " +
                                                               str(num_instances_found))
        print("Found correct number of instances in test iterator.")"""
    return dataset_reader, dataset_iterator, num_instances_expected


def get_gradients_wrt_attention(attention_weights, corr_vects, gradient_reporting_classifier,
                                differentiable_function_on_output, base_filename, starting_ind, gpu):
    # make classifier collect gradients
    gradient_reporting_classifier._temp_filename = base_filename + str(starting_ind) + '-' + \
                                                   str(starting_ind + attention_weights.size(0) - 1)
    model_presoftmax_output = gradient_reporting_classifier(corr_vects,
                                                            torch.autograd.Variable(attention_weights,
                                                                                    requires_grad=True))['label_logits']
    thing_to_differentiate = differentiable_function_on_output(model_presoftmax_output)
    thing_to_differentiate.backward()

    # open gradients file, process it
    with open(gradient_reporting_classifier._temp_filename, 'rb') as f:
        corr_grad_for_attn = pickle.load(f)
        corr_grad_for_attn = util.move_to_device(corr_grad_for_attn, gpu)
        assert corr_grad_for_attn.size() == attention_weights.size()

    return corr_grad_for_attn


def set_up_gradient_reporting_classifier(vanilla_classifier):
    reporting_classifier = \
        attn_tests_lib.GradientReportingClassifierFromAttnAndInputVects(
            deepcopy(vanilla_classifier._classification_module))
    reporting_classifier = reporting_classifier.train()
    return reporting_classifier


def exp_highest_val_over_sum_exp_all_vals(output_logits):
    output_logits = torch.exp(output_logits)
    max_vals = torch.max(output_logits, dim=1)[0]
    denoms = torch.sum(output_logits, dim=1)
    return (max_vals / denoms).sum()


def get_gradient_based_stats(classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                             original_output_filename, grad_based_stats_filename):
    assert not os.path.isfile(grad_based_stats_filename)
    batch_iterator = IntermediateBatchIterator(attn_weight_filename, corr_vector_dir, batch_size)
    grad_getting_classifier = set_up_gradient_reporting_classifier(vanilla_classifier=classifier)
    grad_getting_classifier = grad_getting_classifier.cuda(device=gpu)
    next_available_ind = 1
    lists_to_write_to_file = []
    corr_output_yielder = iter(OriginalOutputDistIterator(original_output_filename))
    if not corr_vector_dir.endswith('/'):
        corr_vector_dir += '/'
    if not os.path.isdir(corr_vector_dir):
        os.makedirs(corr_vector_dir)
    base_filename = corr_vector_dir + 'gradients/'
    if not os.path.isdir(base_filename):
        os.makedirs(base_filename)
    base_filename += 'gradient_wrt_attn_weights_'
    for batch_tup in tqdm(batch_iterator, desc="Calculating grad-based decision flip stats"):
        list_of_lens = batch_tup[2]
        original_attn_weights = util.move_to_device(batch_tup[0], gpu)
        original_attn_weights_for_backpropagating = deepcopy(original_attn_weights)
        original_attn_weights_for_backpropagating = util.move_to_device(original_attn_weights_for_backpropagating,
                                                                        gpu)
        corr_vects = util.move_to_device(batch_tup[1], gpu)
        corr_vects_for_later = deepcopy(corr_vects)
        corr_vects_for_later = util.move_to_device(corr_vects_for_later, gpu)

        function_of_presoftmax_output = exp_highest_val_over_sum_exp_all_vals

        corr_grads_tensor = get_gradients_wrt_attention(original_attn_weights_for_backpropagating, corr_vects,
                                                        grad_getting_classifier, function_of_presoftmax_output,
                                                        base_filename, next_available_ind, gpu)

        attn_times_grads = corr_grads_tensor * original_attn_weights

        list_of_original_log_np_arrays = []
        for i in range(len(list_of_lens)):
            list_of_original_log_np_arrays.append(next(corr_output_yielder))
        list_of_original_log_np_arrays = np.array(list_of_original_log_np_arrays)
        original_dists = np.array(list_of_original_log_np_arrays)
        _, list_of_batch_results_plaingrad = \
            get_first_v_second_stats_for_batch(next_available_ind, gpu, original_attn_weights, classifier,
                                               batch_tup[1], batch_tup[2], list_of_original_log_np_arrays,
                                               zero_out_by=corr_grads_tensor.cpu().numpy())
        _, list_of_batch_results_gradmult = \
            get_first_v_second_stats_for_batch(next_available_ind, gpu, original_attn_weights, classifier,
                                               batch_tup[1], batch_tup[2], list_of_original_log_np_arrays,
                                               zero_out_by=attn_times_grads.cpu().numpy())

        inds_in_dec_order, _ = get_inds_of_sorted_attnshaped_vals(corr_grads_tensor, list_of_lens)
        inds_in_dec_order_by_mult, _ = get_inds_of_sorted_attnshaped_vals(attn_times_grads, list_of_lens)

        print_at_end = str(inds_in_dec_order[9])

        # remove highest-gradient element first

        original_labels = torch.from_numpy(np.argmax(original_dists, axis=1))
        original_labels = util.move_to_device(original_labels, gpu)

        corr_instance_inds = torch.from_numpy(np.arange(start=next_available_ind,
                                                        stop=(next_available_ind + len(list_of_lens))))
        corr_instance_inds = util.move_to_device(corr_instance_inds, gpu)

        lists_to_write_from_top = get_dec_flip_info_for_batch(inds_in_dec_order, corr_instance_inds, original_labels,
                                                              corr_vects_for_later, original_attn_weights, classifier,
                                                              print_10_output_for_debugging=False)
        lists_to_write_from_top_mult = get_dec_flip_info_for_batch(inds_in_dec_order_by_mult, corr_instance_inds,
                                                                   original_labels, corr_vects_for_later,
                                                                   original_attn_weights, classifier)

        next_available_ind += len(list_of_lens)
        for i in range(len(list_of_lens)):
            top_list = lists_to_write_from_top[i]
            top_mult_list = lists_to_write_from_top_mult[i]
            top_singlestats_plaingrad = list_of_batch_results_plaingrad[i]
            top_singlestats_gradmult = list_of_batch_results_gradmult[i]
            assert top_list[0] == top_mult_list[0]
            assert top_list[0] == int(top_singlestats_plaingrad[0]), \
                str(top_list[0]) + ', ' + str(top_singlestats_plaingrad[0])
            assert top_list[0] == int(top_singlestats_gradmult[0]), \
                str(top_list[0]) + ', ' + str(top_singlestats_gradmult[0])
            instance_list = [top_list[0],
                             str(list_of_lens[i]),
                             str(top_list[2]),
                             str(top_list[3]),
                             str(top_list[4]),
                             str(top_mult_list[2]),
                             str(top_mult_list[3]),
                             str(top_mult_list[4]),
                             str(top_singlestats_plaingrad[1]),  # kl div highest
                             str(top_singlestats_plaingrad[2]),  # js div highest
                             str(top_singlestats_plaingrad[3]),  # decision flip highest
                             str(top_singlestats_plaingrad[4]),  # kl div 2nd highest
                             str(top_singlestats_plaingrad[5]),  # js div 2nd highest
                             str(top_singlestats_plaingrad[6]),  # decision flip 2nd highest
                             str(top_singlestats_gradmult[1]),
                             str(top_singlestats_gradmult[2]),
                             str(top_singlestats_gradmult[3]),
                             str(top_singlestats_gradmult[4]),
                             str(top_singlestats_gradmult[5]),
                             str(top_singlestats_gradmult[6])]
            lists_to_write_to_file.append(instance_list)

    with open(grad_based_stats_filename, 'w') as f:
        f.write('id,seq_len,needed_to_rem_x_top_items_for_decflip,needed_to_rem_x_top_probmass_for_decflip,' +
                'needed_to_rem_frac_x_top_items_for_decflip,needed_to_rem_x_top_multitems_for_decflip,' +
                'needed_to_rem_x_top_multprobmass_for_decflip,needed_to_rem_frac_x_top_multitems_for_decflip,' +
                'plaingrad_klhighest,plaingrad_jshighest,plaingrad_decfliphighest,plaingrad_kl2ndhighest,' +
                'plaingrad_js2ndhighest,plaingrad_decflip2ndhighest,gradmult_klhighest,gradmult_jshighest,' +
                'gradmult_decfliphighest,gradmult_kl2ndhighest,gradmult_js2ndhighest,gradmult_decflip2ndhighest\n')
        prev_id = -1
        for instance_list in tqdm(lists_to_write_to_file, desc="Writing dec flip stats file"):
            assert prev_id < instance_list[0]
            prev_id = instance_list[0]
            f.write(str(instance_list[0]) + ',' + instance_list[1] + ',' + instance_list[2] + ',' + instance_list[3] +
                    ',' + instance_list[4] + ',' + instance_list[5] + ',' + instance_list[6] + ',' +
                    instance_list[7] + ',' + instance_list[8] + ',' + instance_list[9] + ',' + instance_list[10] +
                    ',' + instance_list[11] + ',' + instance_list[12] + ',' + instance_list[13] +
                    ',' + instance_list[14] + ',' + instance_list[15] + ',' + instance_list[16] +
                    ',' + instance_list[17] + ',' + instance_list[18] + ',' + instance_list[19] + '\n')
    print("Done writing gradient-based decision-flip stats file.")
    print(print_at_end)


def get_batch_size_max_samples_per_batch_from_config_file(config_filename):
    looking_for_max_samples = False
    batch_size = None
    max_samples_per_batch = None
    vocab_dir = None
    with open(config_filename, 'r') as f:
        for line in f:
            if looking_for_max_samples:
                start_counter = 0
                end_counter = len(line)
                while start_counter < end_counter and not line[start_counter].isdigit():
                    start_counter += 1
                if start_counter == end_counter:
                    continue
                while not line[end_counter - 1].isdigit():
                    end_counter -= 1
                max_samples_per_batch = int(line[start_counter: end_counter])
                looking_for_max_samples = False
            elif '"batch_size"' in line:
                start_counter = 0
                end_counter = len(line)
                while not line[start_counter].isdigit():
                    start_counter += 1
                while not line[end_counter - 1].isdigit():
                    end_counter -= 1
                batch_size = int(line[start_counter: end_counter])
            elif '"maximum_samples_per_batch"' in line:
                start_counter = 0
                end_counter = len(line)
                while start_counter < end_counter and not line[start_counter].isdigit():
                    start_counter += 1
                if start_counter == end_counter:
                    looking_for_max_samples = True
                    continue
                while not line[end_counter - 1].isdigit():
                    end_counter -= 1
                max_samples_per_batch = int(line[start_counter: end_counter])
            elif '"directory_path"' in line:
                line = line[line.index(':') + 1:].strip().strip('"').strip("'")
                vocab_dir = line
    return batch_size, max_samples_per_batch, vocab_dir


def get_entropy_of_dists(log_dists, lengths_of_dists):
    entropies = []
    for i in range(log_dists.shape[0]):
        log_dist = log_dists[i, :lengths_of_dists[i]]
        log_dist = log_dist - logsumexp(log_dist)
        exp_dist = np.exp(log_dist)
        total = np.sum(exp_dist)
        assert .98 < total < 1.02, str(exp_dist) + '\n' + str(np.sum(exp_dist)) + "\n" + str(log_dist)
        entropy = -1 * np.sum(log_dist * exp_dist)
        if not entropy > -1e-10:
            print("Calculated an entropy of " + str(entropy))
        entropies.append(entropy)
    return entropies


def get_kl_div_of_dists(log_dists, new_log_dists):
    kl_divs = []
    for i in range(log_dists.shape[0]):
        log_dist = log_dists[i] - logsumexp(log_dists[i])
        new_log_dist = new_log_dists[i] - logsumexp(new_log_dists[i])
        assert .98 < np.sum(np.exp(log_dist)) < 1.02, str(np.exp(log_dist)) + '\n' + str(np.sum(np.exp(log_dist))) + \
                                                      '\n' + str(log_dist)
        assert .98 < np.sum(np.exp(new_log_dist)) < 1.02, str(np.exp(new_log_dist)) + '\n' + \
                                                          str(np.sum(np.exp(new_log_dist))) + '\n' + str(new_log_dist)
        kl_div = (np.exp(new_log_dist) * (new_log_dist - log_dist)).sum()
        if not kl_div > -1e-10:
            print("Calculated a kl div of " + str(kl_div))
        kl_divs.append(kl_div)
    return kl_divs


def get_js_div_of_dists(log_dists, new_log_dists):
    js_divs = []
    for i in range(log_dists.shape[0]):
        p = np.reshape(log_dists[i], (1, log_dists.shape[1]))
        q = np.reshape(new_log_dists[i], (1, log_dists.shape[1]))
        m = np.reshape(logsumexp(np.concatenate([p, q], axis=0), axis=0) - float(np.log(2)), (1, log_dists.shape[1]))
        js_div = (get_kl_div_of_dists(m, p)[0] + get_kl_div_of_dists(m, q)[0]) / 2
        if not js_div > -1e-10:
            print("Calculated a js div of " + str(js_div))
        js_divs.append(js_div)
    return js_divs


def get_two_highest_and_inds(list_to_check):
    highest_val = -1
    highest_ind = -1
    second_highest_val = -1
    second_highest_ind = -1
    lowest_val = 100000
    lowest_ind = -1
    second_lowest_val = 100000
    second_lowest_ind = -1
    if isinstance(list_to_check, list):
        list_len = len(list_to_check)
    else:
        list_len = list_to_check.shape[0]
    for i in range(list_len):
        val = list_to_check[i]
        if val > highest_val:
            second_highest_val = highest_val
            second_highest_ind = highest_ind
            highest_val = val
            highest_ind = i
        elif val > second_highest_val:
            second_highest_val = val
            second_highest_ind = i
        if val < lowest_val:
            second_lowest_val = lowest_val
            second_lowest_ind = lowest_ind
            lowest_val = val
            lowest_ind = i
        elif val < second_lowest_val:
            second_lowest_val = val
            second_lowest_ind = i
    return highest_val, highest_ind, second_highest_val, second_highest_ind, \
           lowest_val, lowest_ind, second_lowest_val, second_lowest_ind


def make_zeroed_out_copies_for_highest_2ndhighest_lowest_2ndlowest(two_dim_arr, other_arr_to_sort_by=None):
    highest_zeroed_out = two_dim_arr.clone()
    second_highest_zeroed_out = two_dim_arr.clone()
    lowest_zeroed_out = two_dim_arr.clone()
    second_lowest_zeroed_out = two_dim_arr.clone()
    for i in range(two_dim_arr.shape[0]):
        if other_arr_to_sort_by is None:
            _, highest_ind, _, second_highest_ind, _, lowest_ind, _, second_lowest_ind = \
                get_two_highest_and_inds(two_dim_arr[i])
        else:
            _, highest_ind, _, second_highest_ind, _, lowest_ind, _, second_lowest_ind = \
                get_two_highest_and_inds(other_arr_to_sort_by[i])
        highest_zeroed_out[i, highest_ind] = 0
        second_highest_zeroed_out[i, second_highest_ind] = 0
        lowest_zeroed_out[i, lowest_ind] = 0
        second_lowest_zeroed_out[i, second_lowest_ind] = 0
    highest_zeroed_out = normalize(highest_zeroed_out, p=1, dim=1)
    second_highest_zeroed_out = normalize(second_highest_zeroed_out, p=1, dim=1)
    lowest_zeroed_out = normalize(lowest_zeroed_out, p=1, dim=1)
    second_lowest_zeroed_out = normalize(second_lowest_zeroed_out, p=1, dim=1)
    return highest_zeroed_out, second_highest_zeroed_out, lowest_zeroed_out, second_lowest_zeroed_out


class OriginalOutputDistIterator:
    def __init__(self, original_output_filename):
        self.original_output_filename = original_output_filename
        self.starting_field_of_output_dist = None
        self.ending_field_plus_1_of_output_dist = None
        with open(original_output_filename, 'r') as f:
            first_line = f.readline().strip('\n')
            fields = first_line.split(',')
            for i in range(len(fields)):
                field = fields[i]
                if field.startswith("log_output_val_") and self.starting_field_of_output_dist is None:
                    self.starting_field_of_output_dist = i
                elif self.starting_field_of_output_dist is not None and not field.startswith("log_output_val_"):
                    self.ending_field_plus_1_of_output_dist = i
            if self.ending_field_plus_1_of_output_dist is None:
                self.ending_field_plus_1_of_output_dist = len(fields)

    def __iter__(self):
        with open(self.original_output_filename, 'r') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                elif line.strip() == '':
                    continue
                else:
                    dist = [float(x) for x in line.strip('\n').split(',')[self.starting_field_of_output_dist:
                                                                          self.ending_field_plus_1_of_output_dist]]
                    yield np.array(dist)


def get_onerun_diff_from_original_stats(new_output_dict, log_original_np_output_arrays):
    logits = new_output_dict["label_logits"].data.cpu().numpy()

    old_decisions = np.argmax(log_original_np_output_arrays, axis=1)
    new_decisions = np.argmax(logits, axis=1)
    decision_flips_arr = np.zeros(old_decisions.shape[0]) - 1
    dec_flipped = (old_decisions != new_decisions)
    decision_flips_arr[dec_flipped] = new_decisions[dec_flipped]

    kl_divs = get_kl_div_of_dists(log_original_np_output_arrays, logits)

    js_divs = get_js_div_of_dists(log_original_np_output_arrays, logits)

    return kl_divs, js_divs, decision_flips_arr


def binary_search_for_first_ind_with_num_geq(arr, x):
    possible_start = 0
    len_of_arr = arr.size(0)
    possible_end_plus_1 = len_of_arr
    check = (possible_start + possible_end_plus_1) // 2
    found = False
    while not found:
        if arr[check] >= x and arr[check - 1] < x:
            found = True
        elif check == len_of_arr - 1:
            check = len_of_arr
            found = True
        elif arr[check] < x:
            possible_start = check + 1
            check = (possible_start + possible_end_plus_1) // 2
        elif arr[check - 1] >= x:
            possible_end_plus_1 = check
            check = (possible_start + possible_end_plus_1) // 2
    return check


def get_inds_of_sorted_attnshaped_vals(original_vals, list_of_lens):
    sorted_attn_inds_increasing_order = []
    sorted_attn_inds_decreasing_order = []
    if (original_vals < 0).sum() > 0:
        original_vals[original_vals == 0] = -1000000  # disadvantage padding in favor of negative gradient vals
    sorted_attn_weights_dec, sorted_corr_inds_dec = original_vals.sort(1, descending=True)
    sorted_corr_inds_dec = sorted_corr_inds_dec.cpu().numpy()
    for i in range(original_vals.size(0)):
        attn_inds_in_dec_order = list(sorted_corr_inds_dec[i, :list_of_lens[i]])
        attn_inds_in_inc_order = attn_inds_in_dec_order.copy()
        attn_inds_in_inc_order.reverse()
        sorted_attn_inds_decreasing_order.append(attn_inds_in_dec_order)
        sorted_attn_inds_increasing_order.append(attn_inds_in_inc_order)
    return sorted_attn_inds_decreasing_order, sorted_attn_inds_increasing_order


def get_randomized_order_copies_of_lists(lists, num_copies):
    num_copies_lists = [[] for i in range(num_copies)]
    for ls in lists:
        for i in range(num_copies):
            new_ls = ls.copy()
            shuffle(new_ls)
            num_copies_lists[i].append(new_ls)
    return num_copies_lists


def get_dec_flip_stats_and_rand(classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                                original_output_filename, dec_flip_stats_filename, rand_results_filename):
    batch_iterator = IntermediateBatchIterator(attn_weight_filename, corr_vector_dir, batch_size)
    next_available_ind = 1

    # figure out whether that's actually next_available_ind or not
    if os.path.isfile(dec_flip_stats_filename):
        last_used_ind = None
        with open(dec_flip_stats_filename, 'r') as f:
            for line in tqdm(f, desc="Determining what next available ind is"):
                line = line.strip().split(',')
                if len(line) > 1:
                    last_used_ind = line[0]
            try:
                last_used_ind = int(last_used_ind)
            except:
                last_used_ind = None
        if last_used_ind is not None:
            next_available_ind = last_used_ind + 1

    corr_output_yielder = iter(OriginalOutputDistIterator(original_output_filename))
    starting_ind_of_batch = 1
    havent_found_start_yet = True
    for batch_tup in tqdm(batch_iterator, desc="Calculating decision flip and rand stats"):
        list_of_lens = batch_tup[2]
        if starting_ind_of_batch + len(list_of_lens) <= next_available_ind:
            starting_ind_of_batch += len(list_of_lens)
            continue  # we've already covered all the instances in this batch
        elif havent_found_start_yet:
            havent_found_start_yet = False
            # could potentially repeat some ids, but at least ensures each instance is correctly labeled
            next_available_ind = starting_ind_of_batch
        starting_ind_of_batch += len(list_of_lens)

        lists_to_write_to_file = []
        rand_result_list = []

        original_attn_weights = util.move_to_device(batch_tup[0], gpu)
        corr_vects = util.move_to_device(batch_tup[1], gpu)

        list_of_original_np_arrays = []
        for i in range(len(list_of_lens)):
            list_of_original_np_arrays.append(next(corr_output_yielder))
        original_dists = np.array(list_of_original_np_arrays)
        original_labels = torch.from_numpy(np.argmax(original_dists, axis=1))
        original_labels = util.move_to_device(original_labels, gpu)

        corr_instance_inds = torch.from_numpy(np.arange(start=next_available_ind,
                                                        stop=(next_available_ind + len(list_of_lens))))
        corr_instance_inds = util.move_to_device(corr_instance_inds, gpu)

        inds_in_dec_order, inds_in_inc_order = get_inds_of_sorted_attnshaped_vals(original_attn_weights, list_of_lens)
        inds_in_rand_orders = get_randomized_order_copies_of_lists(inds_in_dec_order, num_rand_samples_to_take)

        lists_to_write_from_top = get_dec_flip_info_for_batch(inds_in_dec_order, corr_instance_inds, original_labels,
                                                              corr_vects, original_attn_weights, classifier,
                                                              print_10_output_for_debugging=False)
        lists_to_write_from_bottom = get_dec_flip_info_for_batch(inds_in_inc_order, corr_instance_inds, original_labels,
                                                                 corr_vects, original_attn_weights, classifier)
        for i in range(num_rand_samples_to_take):
            rand_results, weight_kl_js = get_dec_flip_info_for_batch(inds_in_rand_orders[i], corr_instance_inds,
                                                                     original_labels, corr_vects,
                                                                     original_attn_weights, classifier,
                                                                    get_attnvals_kljsdivs_for_first_inds=original_dists)
            assert len(rand_results) == len(lists_to_write_from_top)
            for local_inst_ind in range(len(rand_results)):
                w = weight_kl_js[local_inst_ind]
                res = rand_results[local_inst_ind]
                if i == 0:
                    rand_result_list.append([lists_to_write_from_top[local_inst_ind][0],
                                             str(list_of_lens[local_inst_ind]),
                                             str(float(w[0])),  # attn weight randomly pulled out
                                             str(float(w[1])),  # kl div resulting from running that
                                             str(float(w[2])),  # js div resulting from running that
                                             str(res[2]),  # num random weights that had to be zeroed before dec flip
                                             str(res[3]),  # prob mass "" "" "" ""
                                             str(res[4])])  # frac of weights "" "" ""
                else:
                    list_to_add_to = rand_result_list[-1 * (len(rand_results) - local_inst_ind)]
                    assert list_to_add_to[0] == lists_to_write_from_top[local_inst_ind][0]
                    list_to_add_to.append(str(float(w[0])))
                    list_to_add_to.append(str(float(w[1])))
                    list_to_add_to.append(str(float(w[2])))
                    list_to_add_to.append(str(res[2]))
                    list_to_add_to.append(str(res[3]))
                    list_to_add_to.append(str(res[4]))
        next_available_ind += len(list_of_lens)
        for i in range(len(list_of_lens)):
            top_list = lists_to_write_from_top[i]
            bottom_list = lists_to_write_from_bottom[i]
            assert top_list[0] == bottom_list[0] and top_list[1] == bottom_list[1], str(top_list[:2]) + ", " + \
                                                                                    str(bottom_list[:2])
            instance_list = [top_list[0],
                             str(list_of_lens[i]),
                             str(top_list[2]),
                             str(top_list[3]),
                             str(top_list[4]),
                             str(bottom_list[2]),
                             str(bottom_list[3]),
                             str(bottom_list[4])]
            lists_to_write_to_file.append(instance_list)
                    
        if not os.path.isfile(dec_flip_stats_filename):
            need_to_add_header_line = True
        else:
            need_to_add_header_line = False
        with open(dec_flip_stats_filename, 'a') as f:
            if need_to_add_header_line:
                f.write('id,seq_len,needed_to_rem_x_top_items_for_decflip,needed_to_rem_x_top_probmass_for_decflip,' +
                        'needed_to_rem_frac_x_top_items_for_decflip,needed_to_rem_x_bottom_items_for_decflip,' +
                        'needed_to_rem_x_bottom_probmass_for_decflip,needed_to_rem_frac_x_bottom_items_for_decflip\n')
            prev_id = 0
            for instance_list in lists_to_write_to_file:
                assert prev_id < instance_list[0]
                prev_id = instance_list[0]
                f.write(str(instance_list[0]) + ',' + instance_list[1] + ',' + instance_list[2] + ',' + instance_list[3] +
                        ',' + instance_list[4] + ',' + instance_list[5] + ',' + instance_list[6] + ',' +
                        instance_list[7] + '\n')
        if not os.path.isfile(rand_results_filename):
            need_to_add_header_line = True
        else:
            need_to_add_header_line = False
        with open(rand_results_filename, 'a') as f:
            dont_write_header = not need_to_add_header_line
            reordered_pieces = []
            if need_to_add_header_line:
                f.write('id,seq_len,')
            for instance_ind in range(len(rand_result_list)):
                reordered_pieces.append([rand_result_list[instance_ind][0], rand_result_list[instance_ind][1]])

            write_part_of_header_and_reorder('attn_weight_extracted_', 2, f, rand_result_list,
                                             reordered_pieces, dont_write_header=dont_write_header)
            if need_to_add_header_line:
                f.write(',')
            write_part_of_header_and_reorder('kl_div_', 3, f, rand_result_list,
                                             reordered_pieces, dont_write_header=dont_write_header)
            if need_to_add_header_line:
                f.write(',')
            write_part_of_header_and_reorder('js_div_', 4, f, rand_result_list,
                                             reordered_pieces, dont_write_header=dont_write_header)
            if need_to_add_header_line:
                f.write(',')
            write_part_of_header_and_reorder('needed_to_rem_x_items_for_decflip_', 5, f, rand_result_list,
                                             reordered_pieces, dont_write_header=dont_write_header)
            if need_to_add_header_line:
                f.write(',')
            write_part_of_header_and_reorder('needed_to_rem_x_probmass_for_decflip_', 6, f, rand_result_list,
                                             reordered_pieces, dont_write_header=dont_write_header)
            if need_to_add_header_line:
                f.write(',')
            write_part_of_header_and_reorder('needed_to_rem_frac_x_for_decflip_', 7, f, rand_result_list,
                                             reordered_pieces, dont_write_header=dont_write_header)
            if need_to_add_header_line:
                f.write('\n')

            prev_id = 0
            for instance_list in reordered_pieces:
                assert prev_id < instance_list[0]
                prev_id = instance_list[0]
                f.write(str(instance_list[0]) + ',')
                for i in range(1, len(instance_list) - 1):
                    f.write(instance_list[i] + ',')
                f.write(instance_list[-1] + '\n')

    print("Done writing decision-flip stats file.")
    print("Done writing rand-order stats file.")
    

def get_dec_flip_stats_for_rand_nontop(classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                                original_output_filename, dec_flip_rand_nontop_stats_filename):
    batch_iterator = IntermediateBatchIterator(attn_weight_filename, corr_vector_dir, batch_size)
    next_available_ind = 1
    lists_to_write_to_file = []
    corr_output_yielder = iter(OriginalOutputDistIterator(original_output_filename))
    num_neg_kldivs = 0
    num_neg_jsdivs = 0
    num_pos_kldivs = 0
    num_pos_jsdivs = 0
    for batch_tup in tqdm(batch_iterator, desc="Calculating rand nontop decision flip stats"):
        list_of_lens = batch_tup[2]
        original_attn_weights = util.move_to_device(batch_tup[0], gpu)
        corr_vects = util.move_to_device(batch_tup[1], gpu)

        list_of_original_np_arrays = []
        for i in range(len(list_of_lens)):
            list_of_original_np_arrays.append(next(corr_output_yielder))
        original_dists = np.array(list_of_original_np_arrays)
        original_labels = torch.from_numpy(np.argmax(original_dists, axis=1))
        original_labels = util.move_to_device(original_labels, gpu)

        corr_instance_inds = torch.from_numpy(np.arange(start=next_available_ind,
                                                        stop=(next_available_ind + len(list_of_lens))))
        corr_instance_inds = util.move_to_device(corr_instance_inds, gpu)

        inds_in_dec_order, inds_in_inc_order = get_inds_of_sorted_attnshaped_vals(original_attn_weights, list_of_lens)

        rand_nontop_ind_lists = []
        zeroed_weights = []
        for i in range(len(inds_in_dec_order)):
            ind_list = inds_in_dec_order[i]
            if len(ind_list) == 1:
                ind_pulled = 0  # we'll just filter this out during processing
            elif len(ind_list) == 2:
                ind_pulled = 1
            else:
                ind_pulled = random.randint(1, len(ind_list) - 1)
            rand_nontop_ind_lists.append([ind_pulled])
            zeroed_weights.append(float(original_attn_weights[i, ind_pulled]))
        lists_to_write, val_kldivs = get_dec_flip_info_for_batch(rand_nontop_ind_lists, corr_instance_inds, original_labels,
                                                              corr_vects, original_attn_weights, classifier,
                                                              print_10_output_for_debugging=False,
                                                     get_attnvals_kljsdivs_for_first_inds=original_dists)
        next_available_ind += len(list_of_lens)
        for i in range(len(list_of_lens)):
            weight_kl_js = val_kldivs[i]
            assert weight_kl_js[0] == zeroed_weights[i]
            removed_one_nontop_rand = lists_to_write[i]
            if float(weight_kl_js[1]) > 0:
                num_pos_kldivs += 1
            else:
                num_neg_kldivs += 1
            if float(weight_kl_js[2]) > 0:
                num_pos_jsdivs += 1
            else:
                num_neg_jsdivs += 1
            instance_list = [removed_one_nontop_rand[0],
                             str(list_of_lens[i]),
                             str(removed_one_nontop_rand[2]),
                             str(zeroed_weights[i]),
                             str(float(weight_kl_js[1])),
                             str(float(weight_kl_js[2]))]
            lists_to_write_to_file.append(instance_list)
    print("Num neg KL divs: " + str(num_neg_kldivs))
    print("Num pos KL divs: " + str(num_pos_kldivs))
    print("Num neg JS divs: " + str(num_neg_jsdivs))
    print("Num pos JS divs: " + str(num_pos_jsdivs))

    with open(dec_flip_rand_nontop_stats_filename, 'w') as f:
        f.write('id,seq_len,not_negone_if_rand_caused_decflip,zeroed_weight,rand_kl_div,rand_js_div\n')
        prev_id = -1
        for instance_list in tqdm(lists_to_write_to_file, desc="Writing dec flip stats file"):
            assert prev_id < instance_list[0]
            prev_id = instance_list[0]
            f.write(str(instance_list[0]) + ',' + instance_list[1] + ',' + instance_list[2] + ',' + instance_list[3] +
                    ',' + instance_list[4] + ',' + instance_list[5] + '\n')
    print("Done writing rand nontop stats file.")

    
def write_part_of_header_and_reorder(header_piece, ind_offset, f, rand_result_list, reordered_pieces, is_end=False, dont_write_header=False):
    if not dont_write_header:
        for i in range(num_rand_samples_to_take):
            f.write(header_piece + str(i))
            if (not is_end) and i < num_rand_samples_to_take - 1:
                f.write(',')
    for instance_ind in range(len(rand_result_list)):
        cur_list = reordered_pieces[instance_ind]
        original_list = rand_result_list[instance_ind]
        cur_list += [original_list[6 * j + ind_offset] for j in range(num_rand_samples_to_take)]
        reordered_pieces[instance_ind] = cur_list


def get_dec_flip_info_for_batch(sorted_attn_inds, corr_instance_inds, original_labels, corr_vects,
                                original_attn_weights, classifier, get_attnvals_kljsdivs_for_first_inds=None,
                                print_10_output_for_debugging=False):
    lists_to_write_to_file = []
    done = False
    new_attn_weights = original_attn_weights.clone()
    lens_of_attn_dists = [len(attn_inds) for attn_inds in sorted_attn_inds]
    original_attn_weights_zeroed = [[] for i in range(len(sorted_attn_inds))]
    if get_attnvals_kljsdivs_for_first_inds is not None:
        val_kldivs = [[] for i in range(len(sorted_attn_inds))]
    first_time = True
    while not done:
        cur_num_instances_remaining = len(lens_of_attn_dists)
        # all sorted_attn_inds lists here will have len > 0 and represent instances we want to run again
        for instance_ind in range(cur_num_instances_remaining):
            ind_list = sorted_attn_inds[instance_ind]
            index_to_pull = ind_list[0]
            original_attn_weights_zeroed[instance_ind].append(
                original_attn_weights[instance_ind, index_to_pull])
            if first_time and get_attnvals_kljsdivs_for_first_inds is not None:
                val_kldivs[instance_ind].append(original_attn_weights[instance_ind, index_to_pull])
            new_attn_weights[instance_ind, sorted_attn_inds[instance_ind][0]] = 0
            del sorted_attn_inds[instance_ind][0]

        new_attn_weights = normalize(new_attn_weights, p=1, dim=1)
        mod_output = classifier(corr_vects, new_attn_weights)["label_logits"].data
        assert mod_output.size(0) == corr_instance_inds.shape[0], str(mod_output.size()) + ', ' + \
                                                                  str(corr_instance_inds.shape[0])
        if print_10_output_for_debugging:
            for i in range(corr_instance_inds.shape[0]):
                if corr_instance_inds[i] == 10:
                    ind_to_print = i
                    break
                elif i == corr_instance_inds.shape[0] - 1:
                    ind_to_print = None
            if ind_to_print is None:
                print("Finished processing instance 10 already; no further output for it")
            else:
                print(mod_output.cpu().numpy()[ind_to_print])
            input()
        if first_time and get_attnvals_kljsdivs_for_first_inds is not None:
            old_log_dists = get_attnvals_kljsdivs_for_first_inds
            new_log_dists = mod_output.cpu().numpy()
            kl_divs = get_kl_div_of_dists(old_log_dists, new_log_dists)
            js_divs = get_js_div_of_dists(old_log_dists, new_log_dists)
            for i in range(cur_num_instances_remaining):
                val_kldivs[i].append(kl_divs[i])
                val_kldivs[i].append(js_divs[i])
        _, new_labels = torch.max(mod_output, 1)

        stayed_same_mask = (new_labels == original_labels)

        for i in range(cur_num_instances_remaining - 1, -1, -1):
            if len(sorted_attn_inds[i]) == 0 and new_labels[i] == original_labels[i]:
                # this is one of the instances that doesn't need any input
                list_to_write = [int(corr_instance_inds[i]),
                                 int(lens_of_attn_dists[i]),
                                 -1,
                                 -1,
                                 -1]
                lists_to_write_to_file.append(list_to_write)
                stayed_same_mask[i] = 0
                del lens_of_attn_dists[i]
                del original_attn_weights_zeroed[i]
                del sorted_attn_inds[i]
            elif new_labels[i] != original_labels[i]:
                zeroed_weights = original_attn_weights_zeroed[i]
                list_to_write = [int(corr_instance_inds[i]),
                                 int(lens_of_attn_dists[i]),
                                 int(len(zeroed_weights)),
                                 float(sum(zeroed_weights)),
                                 float(len(zeroed_weights) / lens_of_attn_dists[i])]
                lists_to_write_to_file.append(list_to_write)
                del lens_of_attn_dists[i]
                del original_attn_weights_zeroed[i]
                del sorted_attn_inds[i]

        corr_instance_inds = corr_instance_inds[stayed_same_mask]
        corr_vects = corr_vects[stayed_same_mask]
        new_attn_weights = new_attn_weights[stayed_same_mask]
        original_attn_weights = original_attn_weights[stayed_same_mask]
        original_labels = original_labels[stayed_same_mask]
        if len(lens_of_attn_dists) == 0:
            done = True
        first_time = False

    lists_to_write_to_file = sorted(lists_to_write_to_file, key=(lambda x: x[0]), reverse=False)
    if get_attnvals_kljsdivs_for_first_inds is not None:
        return lists_to_write_to_file, val_kldivs
    else:
        return lists_to_write_to_file


def get_first_v_second_stats_for_batch(next_available_ind, gpu, original_attn_weights, classifier,
                                       corresponding_vects, list_of_lens, list_of_original_log_np_arrays,
                                       zero_out_by=None):
    attn_zero_highest, attn_zero_2ndhighest, attn_zero_lowest, attn_zero_2ndlowest = \
        make_zeroed_out_copies_for_highest_2ndhighest_lowest_2ndlowest(original_attn_weights,
                                                                       other_arr_to_sort_by=zero_out_by)
    attn_zero_highest, attn_zero_2ndhighest, attn_zero_lowest, attn_zero_2ndlowest = \
        Variable(attn_zero_highest), Variable(attn_zero_2ndhighest), Variable(attn_zero_lowest), \
        Variable(attn_zero_2ndlowest)

    attn_zero_highest = util.move_to_device(attn_zero_highest, gpu)
    attn_zero_2ndhighest = util.move_to_device(attn_zero_2ndhighest, gpu)
    attn_zero_lowest = util.move_to_device(attn_zero_lowest, gpu)
    attn_zero_2ndlowest = util.move_to_device(attn_zero_2ndlowest, gpu)
    corresponding_vects = util.move_to_device(corresponding_vects, gpu)

    output_dict_zero_highest = classifier(corresponding_vects, attn_zero_highest)
    output_dict_zero_second_highest = classifier(corresponding_vects, attn_zero_2ndhighest)
    output_dict_zero_lowest = classifier(corresponding_vects, attn_zero_lowest)
    output_dict_zero_second_lowest = classifier(corresponding_vects, attn_zero_2ndlowest)
    kl_divs_1, js_divs_1, decision_flips_arr_1 = \
        get_onerun_diff_from_original_stats(output_dict_zero_highest, list_of_original_log_np_arrays)
    kl_divs_2, js_divs_2, decision_flips_arr_2 = \
        get_onerun_diff_from_original_stats(output_dict_zero_second_highest, list_of_original_log_np_arrays)
    kl_divs_1_low, js_divs_1_low, decision_flips_arr_1_low = \
        get_onerun_diff_from_original_stats(output_dict_zero_lowest, list_of_original_log_np_arrays)
    kl_divs_2_low, js_divs_2_low, decision_flips_arr_2_low = \
        get_onerun_diff_from_original_stats(output_dict_zero_second_lowest, list_of_original_log_np_arrays)

    batch_results_lists = []
    for i in range(len(list_of_lens)):
        list_to_append = [str(next_available_ind),
                          str(kl_divs_1[i]),
                          str(js_divs_1[i]),
                          str(decision_flips_arr_1[i]),
                          str(kl_divs_2[i]),
                          str(js_divs_2[i]),
                          str(decision_flips_arr_2[i]),
                          str(kl_divs_1_low[i]),
                          str(js_divs_1_low[i]),
                          str(decision_flips_arr_1_low[i]),
                          str(kl_divs_2_low[i]),
                          str(js_divs_2_low[i]),
                          str(decision_flips_arr_2_low[i])]
        batch_results_lists.append(list_to_append)
        next_available_ind += 1
    return next_available_ind, batch_results_lists



def get_first_v_second_stats(classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                             original_output_filename, first_v_second_filename):
    batch_iterator = IntermediateBatchIterator(attn_weight_filename, corr_vector_dir, batch_size)
    corr_output_yielder = iter(OriginalOutputDistIterator(original_output_filename))
    next_available_ind = 1
    lists_to_write_to_file = []
    for batch_tup in batch_iterator:
        original_attn_weights = batch_tup[0]
        list_of_lens = batch_tup[2]

        list_of_original_log_np_arrays = []
        for i in range(len(list_of_lens)):
            list_of_original_log_np_arrays.append(next(corr_output_yielder))
        list_of_original_log_np_arrays = np.array(list_of_original_log_np_arrays)

        next_available_ind, list_of_batch_results = \
            get_first_v_second_stats_for_batch(next_available_ind, gpu, original_attn_weights, classifier,
                                               batch_tup[1], batch_tup[2], list_of_original_log_np_arrays)
        for batch_results_list in list_of_batch_results:
            lists_to_write_to_file.append(batch_results_list)

    with open(first_v_second_filename, 'w') as f:
        f.write('id,kl_div_zero_highest,js_div_zero_highest,dec_flip_highest,kl_div_zero_2ndhighest,' +
                'js_div_zero_2ndhighest,dec_flip_2ndhighest,kl_div_zero_lowest,js_div_zero_lowest,dec_flip_lowest,' +
                'kl_div_zero_2ndlowest,js_div_zero_2ndlowest,dec_flip_2ndlowest\n')
        for instance in tqdm(lists_to_write_to_file, desc="Writing 1st-v-2nd file"):
            f.write(instance[0] + ',' + instance[1] + ',' + instance[2] + ',' + instance[3] + ',' + instance[4] +
                    ',' + instance[5] + ',' + instance[6] + ',' + instance[7] + ',' + instance[8] + ',' +
                    instance[9] + ',' + instance[10] + ',' + instance[11] + ',' + instance[12] + '\n')
    print("Done making first-vs-second output file.")


def do_unchanged_run_and_collect_results(model, test_iterator, test_data, gpu, attn_weight_filename,
                                         unchanged_results_filename, test_data_file):
    test_generator = test_iterator(test_data._read(test_data_file),
                                   num_epochs=1,
                                   shuffle=False)
    num_test_batches = test_iterator.get_num_batches(test_data)
    test_generator_tqdm = tqdm(test_generator, total=num_test_batches, desc="Collecting output stats")
    list_of_instance_infos = []
    num_output_classes = None
    next_available_instance_ind = 1  # indexing starts at 1
    stop_at_10 = False
    instance_inds_passed = 0
    for batch in test_generator_tqdm:
        batch = util.move_to_device(batch, gpu)

        output_dict = model(tokens=batch['tokens'])
        output_log_dists = output_dict["label_logits"].data.cpu().numpy()
        instance_inds_passed += output_log_dists.shape[0]
        if stop_at_10 and instance_inds_passed >= 10:
            num_from_end_to_backtrack = instance_inds_passed - 10
            print(output_log_dists[output_log_dists.shape[0] - 1 - num_from_end_to_backtrack])
            exit(1)
        if num_output_classes is None:
            num_output_classes = output_log_dists.shape[1]
        else:
            assert num_output_classes == output_log_dists.shape[1]
        original_labels_guessed = np.argmax(output_log_dists, axis=1)
        actual_labels = batch["label"].data.cpu().numpy()
        output_dist_entropies = get_entropy_of_dists(output_log_dists,
                                                     [output_log_dists.shape[1]] * output_log_dists.shape[0])
        for i in range(actual_labels.shape[0]):
            instance_info = [next_available_instance_ind,
                             actual_labels[i],
                             original_labels_guessed[i],
                             output_log_dists[i],  # this is a 1-d array
                             output_dist_entropies[i]]
            list_of_instance_infos.append(instance_info)
            next_available_instance_ind += 1

    log_attn_dists, corr_inds = load_log_unnormalized_attn_dists(attn_weight_filename)
    attn_dists, _ = load_attn_dists(attn_weight_filename)
    total_num_instances = next_available_instance_ind - 1
    for i in tqdm(range(len(attn_dists)), total=total_num_instances, desc="Collecting corresponding attn stats"):
        corr_ind = corr_inds[i]
        attn_vals = attn_dists[i]
        log_attn_vals = log_attn_dists[i]
        attn_entropy = get_entropy_of_dists(np.array([log_attn_vals]), [len(log_attn_vals)])[0]
        _, ind_of_highest, _, ind_of_second_highest, _, ind_of_lowest, _, ind_of_second_lowest = \
            get_two_highest_and_inds(log_attn_vals)
        ratio_of_2nd_to_1st = np.exp(log_attn_vals[ind_of_second_highest] - log_attn_vals[ind_of_highest])
        ratio_of_2nd_to_1st_low = np.exp(log_attn_vals[ind_of_second_lowest] - log_attn_vals[ind_of_lowest])
        entropy_of_1st_2nd_dist = get_entropy_of_dists(np.array([[log_attn_vals[ind_of_highest],
                                                                  log_attn_vals[ind_of_second_highest]]]), [2])[0]
        entropy_of_1st_2nd_dist_low = get_entropy_of_dists(np.array([[log_attn_vals[ind_of_lowest],
                                                                      log_attn_vals[ind_of_second_lowest]]]), [2])[0]
        ratio_of_last_to_1st = np.exp(log_attn_vals[ind_of_lowest] - log_attn_vals[ind_of_highest])
        entropy_of_last_1st_dist = get_entropy_of_dists(np.array([[log_attn_vals[ind_of_lowest],
                                                                   log_attn_vals[ind_of_highest]]]), [2])[0]
        assert list_of_instance_infos[corr_ind - 1][0] == corr_ind, str(list_of_instance_infos[corr_ind - 1][0]) + \
                                                                    ", " + str(corr_ind)
        list_to_add_to = list_of_instance_infos[corr_ind - 1]
        list_to_add_to.append(attn_entropy)
        list_to_add_to.append(ratio_of_2nd_to_1st)
        list_to_add_to.append(entropy_of_1st_2nd_dist)
        list_to_add_to.append(ratio_of_2nd_to_1st_low)
        list_to_add_to.append(entropy_of_1st_2nd_dist_low)
        list_to_add_to.append(ratio_of_last_to_1st)
        list_to_add_to.append(entropy_of_last_1st_dist)

    with open(unchanged_results_filename, 'w') as f:
        f.write("id,actual_label,orig_label_guessed,output_entropy,attn_entropy," +
                "weight1st2nd_ratio_high,weight1st2nd_entropy_high,weight1st2nd_ratio_low,weight1st2nd_entropy_low," +
                "weight1stlast_ratio,weight1stlast_entropy,")
        for i in range(num_output_classes - 1):
            f.write("log_output_val_" + str(i) + ",")
        f.write("log_output_val_" + str(num_output_classes - 1) + "\n")

        for instance_info in tqdm(list_of_instance_infos, desc="Writing original-model stats to file"):
            id = str(instance_info[0])
            actual_label = str(instance_info[1])
            orig_label_guessed = str(instance_info[2])
            output_entropy = str(instance_info[4])
            attn_entropy = str(instance_info[5])
            weight1st2nd_ratio = str(instance_info[6])
            weight1st2nd_entropy = str(instance_info[7])
            weight1st2nd_ratio_low = str(instance_info[8])
            weight1st2nd_entropy_low = str(instance_info[9])
            weight1stlast_ratio = str(instance_info[10])
            weight1stlast_entropy = str(instance_info[11])
            output_vals = instance_info[3]
            f.write(id + ',' + actual_label + ',' + orig_label_guessed + ',' + output_entropy + ',' +
                    attn_entropy + ',' + weight1st2nd_ratio + ',' + weight1st2nd_entropy + ',' +
                    weight1st2nd_ratio_low + ',' + weight1st2nd_entropy_low + ',' + weight1stlast_ratio + ',' +
                    weight1stlast_entropy + ',')
            for i in range(num_output_classes - 1):
                f.write(str(output_vals[i]) + ',')
            f.write(str(output_vals[num_output_classes - 1]) + '\n')
    print("Done running tests on original model.")


def run_tests(s_dir, output_dir, test_data_file, attn_layer_to_replace, attn_weight_filename,
              name_of_layer_to_replace, gpu, loading_han):
    if not s_dir.endswith('/'):
        s_dir += '/'
    training_config_filename = s_dir + "config.json"
    assert os.path.isfile(training_config_filename), "Could not find " + training_config_filename
    batch_size, max_samples_per_batch, vocab_dir = \
        get_batch_size_max_samples_per_batch_from_config_file(training_config_filename)
    #batch_size = 1
    dataset_reader, dataset_iterator, total_num_test_instances = \
        set_up_inorder_test_loader(test_data_file, batch_size, max_samples_per_batch, vocab_dir, s_dir,
                                   loading_han=loading_han)
    if not output_dir.endswith('/'):
        output_dir += '/'
    corr_vector_dir = output_dir + name_of_layer_to_replace + '_corresponding_vects/'
    unchanged_results_filename = output_dir + unchanged_fname
    first_v_second_filename = output_dir + first_v_second_fname
    dec_flip_stats_filename = output_dir + dec_flip_stats_fname
    rand_results_filename = output_dir + rand_results_fname
    grad_based_stats_filename = output_dir + grad_based_stats_fname
    dec_flip_rand_nontop_stats_filename = output_dir + dec_flip_rand_nontop_stats_fname
    model, just_the_classifier = \
        load_testing_models_in_eval_mode_from_serialization_dir(s_dir, attn_weight_filename, corr_vector_dir,
                                                                total_num_test_instances, training_config_filename,
                                                                name_of_attn_layer_to_replace=attn_layer_to_replace,
                                                                cuda_device=gpu)
    do_unchanged_run_and_collect_results(model, dataset_iterator, dataset_reader, gpu, attn_weight_filename,
                                         unchanged_results_filename, test_data_file)
    get_first_v_second_stats(just_the_classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                             unchanged_results_filename, first_v_second_filename)
    get_dec_flip_stats_and_rand(just_the_classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                                unchanged_results_filename, dec_flip_stats_filename, rand_results_filename)
    get_gradient_based_stats(just_the_classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                             unchanged_results_filename, grad_based_stats_filename)
    get_dec_flip_stats_for_rand_nontop(just_the_classifier, attn_weight_filename, corr_vector_dir, batch_size, gpu,
                                       unchanged_results_filename, dec_flip_rand_nontop_stats_filename)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-folder-name", type=str, required=True,
                        help="The local name of the serialization directory used while training the model")
    parser.add_argument("--test-data-file", type=str, required=True,
                        help="The file containing the test data")

    parser.add_argument("--gpu", type=int, required=False, default=-1,
                        help="Which GPU device to run the testing on")
    parser.add_argument("--optional-folder-tag", type=str, required=False, default='',
                        help='Tag to tack onto output folder')
    parser.add_argument("--base-serialized-models-dir", type=str, required=False,
                        default="/homes/gws/sofias6/models/",
                        help="The dir to prepend to --model-folder-name")
    parser.add_argument("--base-data-dir", type=str, required=False,
                        default="/homes/gws/sofias6/data/",
                        help="The dir to prepend to --test-data-file")
    parser.add_argument("--base-output-dir", type=str, required=False,
                        default='/homes/gws/sofias6/attn-test-output/',
                        help="The local name of the output directory for this training run")
    parser.add_argument("--attn-weight-filename", type=str, required=False,
                        default='attn_weights_by_instance.txt',
                        help='The local name of the file that will contain the attn weights')
    args = parser.parse_args()
    
    import_submodules('attn_tests_lib')
    import_submodules('textcat')
    
    is_han = ('-han' in args.model_folder_name)
    if is_han:
        attn_layer_to_replace = "_sentence_attention"
        model_is_han = True
    elif not is_han:
        attn_layer_to_replace = "_word_attention"
        model_is_han = False
    else:
        print("ERROR: haven't yet specified which attn layer to replace if not han.")
        exit(1)
    if not args.base_serialized_models_dir.endswith('/'):
        args.base_serialized_models_dir += '/'
    if not args.base_data_dir.endswith('/'):
        args.base_data_dir += '/'
    if not args.base_output_dir.endswith('/'):
        args.base_output_dir += '/'
    if not args.model_folder_name.endswith('/'):
        args.model_folder_name += '/'
    assert os.path.isdir(args.base_output_dir)
    output_dir = args.base_output_dir + args.model_folder_name
    if args.optional_folder_tag != '':
        output_dir = output_dir[:-1] + '-' + args.optional_folder_tag
    if not output_dir.endswith('/'):
        output_dir += '/'
    args.attn_weight_filename = output_dir + attn_layer_to_replace + '_' + args.attn_weight_filename
    args.model_folder_name = args.base_serialized_models_dir + args.model_folder_name
    args.test_data_file = args.base_data_dir + args.test_data_file
    run_tests(args.model_folder_name, output_dir, args.test_data_file, attn_layer_to_replace,
              args.attn_weight_filename, attn_layer_to_replace, args.gpu, model_is_han)


if __name__ == '__main__':
    main()
