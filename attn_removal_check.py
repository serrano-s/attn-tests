import argparse
import os
import numpy as np
import torch
from allennlp.models.model import Model
from allennlp.common import Params
import attn_tests_lib
from allennlp.nn import util
from attn_tests_lib import IntermediateBatchIterator, SimpleHanAttention
from test_model import set_up_inorder_test_loader, get_batch_size_max_samples_per_batch_from_config_file
from glob import glob
import pickle
from allennlp.common.util import import_submodules


class SingleInstanceGenerator:
    def __init__(self, data_iter, data_reader, data_fname, instance_to_ret):
        self.temp_fname = "temp_data"
        temp_f = open(self.temp_fname, 'w')
        with open(data_fname, 'r') as f:
            temp_f.write(f.readline())
            counter = 1
            for line in f:
                if line.strip() == '':
                    continue
                if counter == instance_to_ret:
                    temp_f.write(line)
                counter += 1
        temp_f.close()
        self.data_iter = data_iter
        self.data_reader = data_reader

    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        try:
            for batch in self.data_iter(self.data_reader._read(self.temp_fname), epoch_num=0, num_epochs=1, shuffle=False):
                yield batch
      	except:
      	    for batch in self.data_iter(self.data_reader._read(self.temp_fname), num_epochs=1, shuffle=False):
		        yield batch


def load_testing_models_in_eval_mode_from_serialization_dir(s_dir,
                                                            training_config_filename,
                                                            name_of_attn_layer_to_replace="_sentence_attention",
                                                            cuda_device=-1):
    loaded_params = Params.from_file(training_config_filename, "")
    model = Model.load(loaded_params, s_dir, cuda_device=cuda_device)

    original_attn_layer = getattr(model, name_of_attn_layer_to_replace)
    talkative_attn_layer = \
        attn_tests_lib.TalkativeSimpleHanAttention(original_attn_layer, "temp_weights", "temp_vects/", 1)
    setattr(model, name_of_attn_layer_to_replace, talkative_attn_layer)
    just_the_classifier = \
        attn_tests_lib.ClassifierFromAttnAndInputVects(model._output_logit)
    if cuda_device >= 0:
        model = model.cuda(device=cuda_device)
        just_the_classifier = just_the_classifier.cuda(device=cuda_device)
    model = model.eval()
    just_the_classifier = just_the_classifier.eval()

    return model, just_the_classifier


def add_single_grad_to_dir(dir_to_place_single_grad_in, index_of_ind_starting_from_1, original_dir_with_grads):
    if not dir_to_place_single_grad_in.endswith('/'):
        dir_to_place_single_grad_in += '/'
    if not os.path.isdir(dir_to_place_single_grad_in):
        os.makedirs(dir_to_place_single_grad_in)

    if not original_dir_with_grads.endswith('/'):
        original_dir_with_grads += '/'
    original_grad_filename_ext = original_dir_with_grads + 'gradient_wrt_attn_weights_'

    starting_index_in_actual_file = index_of_ind_starting_from_1
    while len(glob(original_grad_filename_ext + str(starting_index_in_actual_file) + '-*')) == 0:
        starting_index_in_actual_file -= 1
    possible_filenames = glob(original_grad_filename_ext + str(starting_index_in_actual_file) + '-*')
    assert len(possible_filenames) == 1
    original_filename = possible_filenames[0]
    num_after = original_filename[original_filename.rfind('-') + 1:]
    if '.' in num_after:
        num_after = num_after[:num_after.index('.')]
    num_after = int(num_after)
    assert num_after >= index_of_ind_starting_from_1

    with open(original_filename, 'rb') as f:
        torch_grad_tensor = pickle.load(f)
    assert torch_grad_tensor.size(0) == num_after - starting_index_in_actual_file + 1
    row_ind_to_pull = index_of_ind_starting_from_1 - starting_index_in_actual_file
    torch_tens_to_save = torch_grad_tensor[row_ind_to_pull].view(1, torch_grad_tensor.size(1))
    new_filename = dir_to_place_single_grad_in + 'gradient_wrt_attn_weights_1-1'
    with open(new_filename, 'wb') as f:
        pickle.dump(torch_tens_to_save, f)


def get_attn_vects_origdecision_and_classifier(s_dir, training_config_filename, instance_generator,
                                               actual_instance_ind_counting_from_1,
                                               original_dir_with_grads,
                                               name_of_attn_layer_to_replace="_sentence_attention", cuda_device=-1):
    model, classifier = \
        load_testing_models_in_eval_mode_from_serialization_dir(s_dir, training_config_filename,
                                                            name_of_attn_layer_to_replace=name_of_attn_layer_to_replace,
                                                            cuda_device=cuda_device)
    output_log_dist = None
    for batch in instance_generator:
        assert output_log_dist is None  # should only be one batch
        batch = util.move_to_device(batch, cuda_device)

        output_dict = model(tokens=batch['tokens'])
        output_log_dist = output_dict["label_logits"].data.cpu().numpy()
        assert output_log_dist.shape[0] == 1
        output_log_dist = np.reshape(output_log_dist, output_log_dist.shape[1])
        indexed_tokens = batch['tokens']['tokens']

    # add gradients/ subdirectory of temp_vects containing just the gradients for this training example
    add_single_grad_to_dir("temp_vects/gradients/", actual_instance_ind_counting_from_1, original_dir_with_grads)

    int_results_getter = IntermediateBatchIterator("temp_weights", "temp_vects/", 1, return_log_attn_vals=False,
                                                   also_return_grads=True)
    np_attn_vect = None
    for tup in iter(int_results_getter):
        assert np_attn_vect is None
        np_attn_vect = tup[0].numpy()
        corr_vects = tup[1]
        seq_len = tup[2][0]
        one_dim_np_arr_of_grads = tup[3][0]

    assert np_attn_vect.shape[1] == seq_len
    assert np_attn_vect.shape[0] == 1

    dir_with_original_results = original_dir_with_grads[:original_dir_with_grads.rfind('/gradients')]
    dir_with_original_results = dir_with_original_results[:dir_with_original_results.rfind('/') + 1]
    original_decisions_file = dir_with_original_results + "original_results.csv"
    with open(original_decisions_file, 'r') as f:
        instance_counter = 0
        for line in f:
            if instance_counter == 0:
                num_output_classes = line.count("log_output_val")
            elif instance_counter == actual_instance_ind_counting_from_1:
                # it's in this line
                str_vals = line.strip().split(',')[-1 * num_output_classes:]
                original_output_vals = [float(val) for val in str_vals]
                break
            instance_counter += 1

    return np_attn_vect, corr_vects, output_log_dist, classifier, indexed_tokens, one_dim_np_arr_of_grads, \
           original_output_vals


def load_vocab_dicts(vocab_dir):
    if not vocab_dir.endswith('/'):
        vocab_dir += '/'
    word_file = vocab_dir + 'tokens.txt'
    label_file = vocab_dir + 'labels.txt'
    word_dict = {}
    cur_counter = 1
    with open(word_file, 'r') as f:
        for line in f:
            line = line[:-1]  # get rid of ending newline
            if line == '':
                if word_dict[cur_counter - 1] != '\n':
                    word_dict[cur_counter] = '\n'
                    cur_counter += 1
                continue
            word_dict[cur_counter] = line
            cur_counter += 1
    label_dict = {}
    cur_counter = 0
    with open(label_file, 'r') as f:
        for line in f:
            line = line[:-1]  # get rid of ending newline
            label_dict[cur_counter] = line
            cur_counter += 1
    return word_dict, label_dict


def print_options_get_next_ind_to_zero(zeroed_inds, word_dict, indexed_tokens, np_attn_vect,
                                       one_dim_np_arr_of_grads, is_han):
    print('\nInds still available to zero (fields: ind, attn_weight, grad, grad*attn_weight, corresponding_component):')
    max_attn_weight_remaining = -1000000
    max_attn_weight_remaining_ind = -1
    max_grad_remaining = -1000000
    max_grad_remaining_ind = -1
    max_grad_times_attn_weight_remaining = -1000000
    max_grad_times_attn_weight_remaining_ind = -1
    for i in range(np_attn_vect.shape[1]):
        if i not in zeroed_inds:
            attn = np_attn_vect[0, i]
            grad = one_dim_np_arr_of_grads[i]
            grad_x_attn = grad * attn
            if attn > max_attn_weight_remaining:
                max_attn_weight_remaining = attn
                max_attn_weight_remaining_ind = i
            if grad > max_grad_remaining:
                max_grad_remaining = grad
                max_grad_remaining_ind = i
            if grad_x_attn > max_grad_times_attn_weight_remaining:
                max_grad_times_attn_weight_remaining = grad_x_attn
                max_grad_times_attn_weight_remaining_ind = i
            print(str(i) + "\t" + str(attn) + '\t' + str(grad) + '\t' + str(grad_x_attn) + '\t', end='')
            if is_han:
                for j in range(indexed_tokens.size(2)):
                    cur_word_ind = int(indexed_tokens[0, i, j])
                    if cur_word_ind == 0:
                        break
                    else:
                        print(word_dict[cur_word_ind], end=' ')
            else:
                print(word_dict[int(indexed_tokens[0, i])], end=' ')
            print()
    print("MaxRemAttnInd: " + str(max_attn_weight_remaining_ind) + "\tMaxRemGradInd: " +
          str(max_grad_remaining_ind) + '\tMaxRemGradxAttnInd: ' + str(max_grad_times_attn_weight_remaining_ind))

    next_ind_to_zero = None
    while next_ind_to_zero is None:
        maybe = input("Which ind to zero next? ")
        try:
            next_ind_to_zero = int(maybe)
            if next_ind_to_zero < 0 or next_ind_to_zero >= np_attn_vect.shape[1] or next_ind_to_zero in zeroed_inds:
                print("Response must be one of available ints")
                next_ind_to_zero = None
        except:
            print("Response must be one of available ints")
    zeroed_inds.append(next_ind_to_zero)
    print()
    return next_ind_to_zero


def run_check(s_dir, training_config_filename, data_file, instance_ind_counting_from_1, attn_layer_to_rep, gpu,
              original_gradients_dir, is_han):
    _, _, vocab_dir = \
        get_batch_size_max_samples_per_batch_from_config_file(training_config_filename)
    dataset_reader, dataset_iterator, num_instances_found = \
        set_up_inorder_test_loader(data_file, 1, 5000, vocab_dir, s_dir, check_num_iterated=False)
    instance_generator = SingleInstanceGenerator(dataset_iterator, dataset_reader, data_file,
                                                 instance_ind_counting_from_1)
    np_attn_vect, corr_vects, output_log_dist, classifier, indexed_tokens, one_dim_np_arr_of_grads,\
        original_output_vals = \
        get_attn_vects_origdecision_and_classifier(s_dir, training_config_filename, instance_generator,
                                                   instance_ind_counting_from_1,
                                                   original_gradients_dir,
                                                   name_of_attn_layer_to_replace=attn_layer_to_rep, cuda_device=gpu)
    word_dict, label_dict = load_vocab_dicts(vocab_dir)
    corr_vects = util.move_to_device(corr_vects, gpu)

    print_output_distrib = True
    print("ORIGINAL DECISION: " + str(label_dict[int(np.argmax(output_log_dist, axis=0))]) +
          " (ind " + str(int(np.argmax(output_log_dist, axis=0))) + ')\n')
    if print_output_distrib:
        print("Original output from test runs: " + str(original_output_vals))
        print(output_log_dist)
        input("(Press enter to continue.)")
    zeroed_inds = []
    while True:
        next_to_zero = print_options_get_next_ind_to_zero(zeroed_inds, word_dict, indexed_tokens, np_attn_vect,
                                                          one_dim_np_arr_of_grads, is_han)

        np_attn_vect[0, next_to_zero] = 0
        denom = np.sum(np_attn_vect)
        if denom == 0:
            denom = 1
        np_attn_vect = np_attn_vect / denom

        torch_attn_vect = torch.from_numpy(np_attn_vect)
        torch_attn_vect = util.move_to_device(torch_attn_vect, gpu)
        torch_attn_vect = torch.autograd.Variable(torch_attn_vect)

        corr_vects_this_time = corr_vects.clone()

        output_dict = classifier(corr_vects_this_time, torch_attn_vect)
        if print_output_distrib:
            print(output_dict["label_logits"].cpu().data.numpy()[0])
        _, new_labels = torch.max(output_dict["label_logits"], 1)
        new_label = new_labels.data.cpu().numpy()[0]

        print("HAVE REMOVED " + str(len(zeroed_inds)) + " / " + str(np_attn_vect.shape[1]))
        print("NEW DECISION: " + str(label_dict[new_label]) + " (ind " + str(new_label) + ")\n")

        if len(zeroed_inds) == np_attn_vect.shape[1]:
            break
        cont = input("Continue? (y/n): ")
        if cont.startswith('n'):
            break

    os.remove("temp_data")
    os.remove("temp_weights")
    fnames_to_rem = glob("temp_vects/gradients/*")
    for fname in fnames_to_rem:
        os.remove(fname)
    os.removedirs("temp_vects/gradients/")
    fnames_to_rem = glob("temp_vects/*")
    for fname in fnames_to_rem:
        os.remove(fname)
    os.removedirs("temp_vects/")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-folder-name", type=str, required=True,
                        help="The local name of the serialization directory used while training the model")
    parser.add_argument("--test-data-file", type=str, required=True,
                        help="The file containing the test data")
    parser.add_argument("--instance-num", type=int, required=True,
                        help="The number of the test instance to manually check")

    parser.add_argument("--segment-sentences", type=bool, required=False, default=True,
                        help="Whether the model in --model-folder-name requires segmented sentences")
    parser.add_argument("--name-of-attn-layer", type=str, required=False, default="_sentence_attention",
                        help="The instance variable name of the attn layer to analyze for the model")
    parser.add_argument("--gpu", type=int, required=False, default=-1,
                        help="Which GPU device to run the testing on")
    parser.add_argument("--base-serialized-models-dir", type=str, required=False,
                        default="/homes/gws/sofias6/models/",
                        help="The dir to prepend to --model-folder-name")
    parser.add_argument("--base-data-dir", type=str, required=False,
                        default="/homes/gws/sofias6/data/",
                        help="The dir to prepend to --test-data-file")
    parser.add_argument("--base-attn-test-dir", type=str, required=False,
                        default="/homes/gws/sofias6/attn-test-output/",
                        help="The dir containing the saved model attn test results (for pulling out gradients)")
    args = parser.parse_args()
    import_submodules('textcat')
    import_submodules('attn_tests_lib')
    is_han = ('-han' in args.model_folder_name)
    if is_han:
        attn_layer_to_replace = "_sentence_attention"
    elif not is_han:
        attn_layer_to_replace = "_word_attention"
    else:
        print("ERROR: haven't yet specified which attn layer to replace if not han.")
        exit(1)
    if not args.base_serialized_models_dir.endswith('/'):
        args.base_serialized_models_dir += '/'
    if not args.base_data_dir.endswith('/'):
        args.base_data_dir += '/'
    if not args.model_folder_name.endswith('/'):
        args.model_folder_name += '/'
    if not args.base_attn_test_dir.endswith('/'):
        args.base_attn_test_dir += '/'
    gradients_dir = args.base_attn_test_dir + args.model_folder_name
    gradients_dir += attn_layer_to_replace + "_corresponding_vects/gradients/"
    args.model_folder_name = args.base_serialized_models_dir + args.model_folder_name
    training_config_filename = args.model_folder_name + "config.json"
    args.test_data_file = args.base_data_dir + args.test_data_file

    run_check(args.model_folder_name, training_config_filename, args.test_data_file, args.instance_num,
              attn_layer_to_replace, args.gpu, gradients_dir, is_han)


if __name__ == '__main__':
    main()
