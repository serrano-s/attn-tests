import os
import re
import torch
import random
import argparse
import datetime
import logging
from allennlp.common.util import import_submodules
from random import randint, uniform
from glob import glob
from allennlp.commands.train import create_serialization_dir, logger, datasets_from_params

from allennlp.commands.evaluate import evaluate
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.common.util import prepare_global_logging
from allennlp.common.util import dump_metrics
from allennlp.data import Vocabulary
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer


corresponding_config_files = {'hanrnn': '_han_from_paper.jsonnet',
                              'hanconv': '_han_with_convs.jsonnet'}


def edit_config_file_to_have_gpu(filename, gpu_num):
    temp_filename = filename + ".temp"
    new_f = open(temp_filename, 'w')
    with open(filename, 'r') as f:
        for line in f:
            if '"cuda_device"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not line[checking_ind].isdigit():
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(gpu_num) + line_end
            new_f.write(line)
    new_f.close()
    os.rename(temp_filename, filename)


def set_random_seeds(filename, rand_seed, np_seed, pytorch_seed):
    temp_filename = filename + ".temp"
    new_f = open(temp_filename, 'w')
    with open(filename, 'r') as f:
        for line in f:
            if '"random_seed"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not line[checking_ind].isdigit():
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(rand_seed) + line_end
            elif '"numpy_seed"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not line[checking_ind].isdigit():
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(np_seed) + line_end
            elif '"pytorch_seed"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not line[checking_ind].isdigit():
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(pytorch_seed) + line_end
            new_f.write(line)
    new_f.close()
    os.rename(temp_filename, filename)


def set_step_size_3_dropouts(filename, step_size, drp1, drp2, drp3):
    temp_filename = filename + ".temp"
    new_f = open(temp_filename, 'w')
    with open(filename, 'r') as f:
        for line in f:
            if '"lr"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not (line[checking_ind].isdigit() or line[checking_ind] == '.'):
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(step_size) + line_end
            elif '"pre_sentence_encoder_dropout"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not (line[checking_ind].isdigit() or line[checking_ind] == '.'):
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(drp1) + line_end
            elif '"pre_document_encoder_dropout"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not (line[checking_ind].isdigit() or line[checking_ind] == '.'):
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(drp2) + line_end
            elif '"dropout"' in line:
                line_end = ''
                checking_ind = len(line) - 1
                while not (line[checking_ind].isdigit() or line[checking_ind] == '.'):
                    line_end = line[checking_ind:]
                    checking_ind -= 1
                line = line[:line.index(':')] + ": " + str(drp3) + line_end
            new_f.write(line)
    new_f.close()
    os.rename(temp_filename, filename)


def run_hyperparam_search(config_file, output_dir, num_configs_to_try):
    possible_dropout_endpoints = [.2, .5]
    possible_step_endpoints = [.00005, .001]

    next_available_output_dir = 1
    while os.path.isdir(output_dir + str(next_available_output_dir) + '/'):
        next_available_output_dir += 1
    iterator = None
    all_datasets = None
    initial_rseed = str(datetime.datetime.now())
    rseed = initial_rseed[initial_rseed.rfind('.') + 1:]
    while rseed.startswith('0'):
        rseed = rseed[1:]
    if rseed == '':
        rseed = '1'
    random.seed(int(rseed))
    for i in range(num_configs_to_try // 2):
        new_dropouts = [uniform(possible_dropout_endpoints[0], possible_dropout_endpoints[1]) for i in range(3)]
        new_step_size = uniform(possible_step_endpoints[0], possible_step_endpoints[1])
        set_step_size_3_dropouts(config_file, new_step_size, new_dropouts[0], new_dropouts[1], new_dropouts[2])

        for j in range(2):
            sub_output_dir = output_dir + str(next_available_output_dir) + '/'
            next_available_output_dir += 1

            new_rand_seeds = [randint(1, 1000) for i in range(3)]
            set_random_seeds(config_file, new_rand_seeds[0], new_rand_seeds[1], new_rand_seeds[2])

            print("Starting to train model " + str(2 * i + j + 1) + " from " + config_file)
            iterator, all_datasets = train_model(config_file, sub_output_dir, data_iter=iterator,
                                                 all_datasets=all_datasets)

            with open(sub_output_dir + "random_seeds.txt", 'w') as f:
                f.write(str(new_rand_seeds[0]) + '\n')
                f.write(str(new_rand_seeds[1]) + '\n')
                f.write(str(new_rand_seeds[2]) + '\n')

            # get rid of all model's .th files and .tar.gz file: since this is a hyperparam search,
            # we just want the reported metrics, so no need to take up a bunch of extra memory
            filenames_to_delete = glob(sub_output_dir + '*.tar.gz')
            for filename in filenames_to_delete:
                os.remove(filename)
            filenames_to_delete = glob(sub_output_dir + '*.th')
            for filename in filenames_to_delete:
                os.remove(filename)
            filenames_to_delete = glob(sub_output_dir + 'vocabulary/*')
            for filename in filenames_to_delete:
                os.remove(filename)
            os.rmdir(sub_output_dir + 'vocabulary/')


def train_model(config_file, sub_output_dir, data_iter=None, all_datasets=None):
    params = Params.from_file(config_file, '')

    serialization_dir = sub_output_dir

    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, False)
    prepare_global_logging(serialization_dir, False)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    if isinstance(cuda_device, list):
        for device in cuda_device:
            check_for_gpu(device)
    else:
        check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    if all_datasets is None:
        all_datasets = datasets_from_params(params)
    else:
        params.pop("train_data_path")
        params.pop("validation_data_path")
        params.pop("dataset_reader")
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(
        params.pop("vocabulary", {}),
        (instance for key, dataset in all_datasets.items()
         for instance in dataset
         if key in datasets_for_vocab_creation)
    )

    model = Model.from_params(vocab=vocab, params=params.pop('model'))

    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    intercepted_iter_params = params.pop("iterator")
    if data_iter is None:
        iter_type = intercepted_iter_params.pop("type")
        assert iter_type == "extended_bucket", iter_type
        intercepted_iter_params.params["type"] = "extended_bucket_for_reuse"

        iterator = DataIterator.from_params(intercepted_iter_params)
        iterator.index_with(vocab)
    else:
        iterator = data_iter

    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    trainer = Trainer.from_params(model=model, serialization_dir=serialization_dir, iterator=iterator,
                                  train_data=train_data, validation_data=validation_data, params=trainer_params,
                                  validation_iterator=validation_iterator)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
            best_model, test_data, validation_iterator or iterator,
            cuda_device=trainer._cuda_devices[0]  # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    return iterator, all_datasets


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True,
                        help="The type of model to train",
                        choices=['hanrnn', 'hanconv'])
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Which dataset to train the model on",
                        choices=['amazon', 'yahoo10cat', 'yelp', 'imdb'])
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU to use (can supply -1 if not using GPU)")
    parser.add_argument("--num-configs-to-try", type=int, required=True,
                        help="How many different configs to randomly sample")

    parser.add_argument("--output-dir-base", required=False,
                        default="/homes/gws/sofias6/models/hyperparam_search/",
                        help="Which directory each individual model's serialization directory sits in")
    parser.add_argument("--dir-with-config-files", required=False,
                        default="/homes/gws/sofias6/textcat/attn_tests/configs/",
                        help="Base directory for all config files")
    args = parser.parse_args()
    if not args.output_dir_base.endswith('/'):
        args.output_dir_base += '/'
    if not os.path.isdir(args.output_dir_base):
        os.makedirs(args.output_dir_base)
    if not args.dir_with_config_files.endswith('/'):
        args.dir_with_config_files += '/'
    output_dir = args.output_dir_base + args.dataset_name + '-' + args.model + '/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    import_submodules('attn_tests_lib')  # gives us access to ConvSeq2SeqEncoder, etc.
    config_file = args.dir_with_config_files + args.dataset_name + corresponding_config_files[args.model]
    edit_config_file_to_have_gpu(config_file, args.gpu)

    run_hyperparam_search(config_file, output_dir, args.num_configs_to_try)


if __name__ == '__main__':
    main()
