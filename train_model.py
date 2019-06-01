import argparse
import logging
import os
import re
from glob import glob
from typing import Iterable, NamedTuple
import torch
from default_directories import base_serialized_models_dir
from default_directories import dir_with_config_files as directory_with_config_files
from allennlp_internal_functions import dump_metrics, datasets_from_params, cleanup_global_logging, \
                                        get_frozen_and_tunable_parameter_names

from allennlp.commands.train import train_model_from_file, create_serialization_dir
from allennlp.common.util import import_submodules
import allennlp.nn.util as util
from allennlp.data.instance import Instance
from allennlp.commands.evaluate import evaluate
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging
from allennlp.data import Vocabulary
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


corresponding_config_files = {'hanrnn': '_han_from_paper.jsonnet',
                              'hanconv': '_han_with_convs.jsonnet',
                              'flanrnn': '_flan_with_rnns.jsonnet',
                              'flanconv': '_flan_with_convs.jsonnet',
                              'han_encless': '_han_no_encoders.jsonnet',
                              'flan_encless': '_flan_no_encoders.jsonnet'}


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


def remove_pretrained_embedding_params(params: Params):
    keys = params.keys()
    if 'pretrained_file' in keys:
        del params['pretrained_file']
    for value in params.values():
        if isinstance(value, Params):
            remove_pretrained_embedding_params(value)


def modified_model_load(
          config: Params,
          serialization_dir: str,
          weights_file: str = None,
          cuda_device: int = -1) -> Model:
    """
    Instantiates an already-trained model, based on the experiment
    configuration and some optional overrides.
    """
    weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

    # Load vocabulary from file
    vocab_dir = os.path.join(serialization_dir, 'vocabulary')
    # If the config specifies a vocabulary subclass, we need to use it.
    vocab = Vocabulary.from_files(vocab_dir)

    model_params = config.get('model')

    # The experiment config tells us how to _train_ a model, including where to get pre-trained
    # embeddings from.  We're now _loading_ the model, so those embeddings will already be
    # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
    # want the code to look for it, so we remove it from the parameters here.
    remove_pretrained_embedding_params(model_params)
    model = Model.from_params(vocab=vocab, params=model_params)
    model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
    model.load_state_dict(model_state, strict=False)

    # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
    # in sync with the weights
    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()

    return model


def load_prev_best_model(s_dir):
    if not s_dir.endswith('/'):
        s_dir += '/'
    edit_config_file_to_have_gpu(s_dir + "config.json", -1)
    loaded_prev_params = Params.from_file(os.path.join(s_dir, s_dir + "config.json"), "")
    model = modified_model_load(loaded_prev_params, s_dir, cuda_device=-1)
    return model


def transfer_prev_model_weights_to_new_model(prev_model, new_model):
    params1 = prev_model.named_parameters()
    params2 = new_model.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data)

    return new_model


def train_model_but_load_prev_model_weights(params: Params,
                                            serialization_dir: str,
                                            prev_best_model: Model,
                                            file_friendly_logging: bool = False,
                                            recover: bool = False,
                                            force: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.
    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    if isinstance(cuda_device, list):
        for device in cuda_device:
            check_for_gpu(device)
    else:
        check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets = datasets_from_params(params)
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
    model = transfer_prev_model_weights_to_new_model(prev_best_model, model)

    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
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

    trainer = Trainer.from_params(model=model,
                                  serialization_dir=serialization_dir,
                                  iterator=iterator,
                                  train_data=train_data,
                                  validation_data=validation_data,
                                  params=trainer_params,
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
                cuda_device=trainer._cuda_devices[0] # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    return best_model


def load_model_from_serialization_dir(s_dir, training_config_filename, cuda_device=-1):
    """
    Function not currently in use. This is from back when I was trying to keep each successive
    addition to the model's training in the same serialization directory.
    :param s_dir:
    :param training_config_filename:
    :param cuda_device:
    :return:
    """
    if not s_dir.endswith('/'):
        s_dir += '/'
    if training_config_filename != s_dir + "config.json":
        counter = 1
        new_config_name = s_dir + "config" + str(counter) + ".json"
        while os.path.isfile(new_config_name):
            counter += 1
            new_config_name = s_dir + "config" + str(counter) + ".json"
        os.rename(s_dir + "config.json", new_config_name)
        training_config_changed = True
    else:
        training_config_changed = False
    if training_config_changed and os.path.isfile(s_dir + "random_seeds.txt"):
        new_name = s_dir + "random_seeds" + str(counter) + ".txt"
        os.rename(s_dir + "random_seeds.txt", new_name)

    if not os.path.isfile(s_dir + "random_seeds.txt"):
        # try to make one from the given config file
        rand_seeds = [None, None, None]
        with open(training_config_filename, 'r') as f:
            for line in f:
                if '"random_seed"' in line:
                    line_end = ''
                    checking_ind = len(line) - 1
                    while not line[checking_ind].isdigit():
                        line_end = line[checking_ind:]
                        checking_ind -= 1
                    rand_seed = int(line[line.index(':') + 1: line.rfind(line_end)].strip())
                    rand_seeds[0] = rand_seed
                elif '"numpy_seed"' in line:
                    line_end = ''
                    checking_ind = len(line) - 1
                    while not line[checking_ind].isdigit():
                        line_end = line[checking_ind:]
                        checking_ind -= 1
                    rand_seed = int(line[line.index(':') + 1: line.rfind(line_end)].strip())
                    rand_seeds[1] = rand_seed
                elif '"pytorch_seed"' in line:
                    line_end = ''
                    checking_ind = len(line) - 1
                    while not line[checking_ind].isdigit():
                        line_end = line[checking_ind:]
                        checking_ind -= 1
                    rand_seed = int(line[line.index(':') + 1: line.rfind(line_end)].strip())
                    rand_seeds[2] = rand_seed
        if not (rand_seeds[0] is None and rand_seeds[1] is None and rand_seeds[2] is None):
            with open(s_dir + "random_seeds.txt", 'w') as f:
                if rand_seeds[0] is not None:
                    f.write("random_seed: " + str(rand_seeds[0]) + "\n")
                if rand_seeds[1] is not None:
                    f.write("numpy_seed: " + str(rand_seeds[1]) + "\n")
                if rand_seeds[2] is not None:
                    f.write("pytorch_seed: " + str(rand_seeds[2]) + "\n")

    loaded_params = Params.from_file(training_config_filename, "")
    loaded_params.to_file(os.path.join(s_dir, s_dir + "config.json"))

    cur_optim_params = loaded_params.get("trainer").get("optimizer")
    if not training_config_changed:
        prev_optim_params = cur_optim_params
    else:
        # "new_config_name" sounds like a mistake, but it's "new" because it's the new name for the old file
        prev_params = Params.from_file(new_config_name, "")
        prev_optim_params = prev_params.get("trainer").get("optimizer")

    model = modified_model_load(loaded_params, s_dir, cuda_device=cuda_device)
    return model, loaded_params, prev_optim_params, cur_optim_params


def modified_train_model(serialization_dir,
                         training_config_filename,
                         cuda_device=-1,
                         file_friendly_logging: bool = False) -> Model:
    """
        Function not currently in use. This is from back when I was trying to keep each successive
        addition to the model's training in the same serialization directory.

    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.
    Parameters
    ----------
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    model, params, prev_optimizer_params, cur_optimizer_params = \
        load_model_from_serialization_dir(serialization_dir, training_config_filename, cuda_device=cuda_device)
    prepare_environment(params)

    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    if isinstance(cuda_device, list):
        for device in cuda_device:
            check_for_gpu(device)
    else:
        check_for_gpu(cuda_device)

    all_datasets = datasets_from_params(params)
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

    params.pop('model')

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
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

    list_of_cur_optimizer_param_keys = [key for key in cur_optimizer_params.as_flat_dict().keys()]
    list_of_prev_optimizer_param_keys = [key for key in prev_optimizer_params.as_flat_dict().keys()]
    optimizer_params_match = True
    for key in list_of_cur_optimizer_param_keys:
        if key not in list_of_prev_optimizer_param_keys:
            optimizer_params_match = False
            break
    for key in list_of_prev_optimizer_param_keys:
        if key not in list_of_cur_optimizer_param_keys:
            optimizer_params_match = False
            break
    if not optimizer_params_match:
        # a list of each p is what will be passed to the optimizer constructor while constructing Trainer--
        # adjust if necessary (i.e., if we changed optimizers)
        model_params = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        assert "parameter_groups" not in list_of_cur_optimizer_param_keys, \
            "Current way of dealing with optimizer change doesn't take parameter groups into account"
        assert "parameter_groups" not in list_of_prev_optimizer_param_keys, \
            "Current way of dealing with optimizer change doesn't take parameter groups into account"
        for param_tup in model_params:
            # modify the second element of param_tup in-place (it's a dict) to match the keys specified in
            # cur_optimizer_params
            param_dict = param_tup[1]
            keys_to_del = []
            keys_already_in_dict = []
            try:
                for key in param_dict.keys():
                    if not key in list_of_cur_optimizer_param_keys:
                        keys_to_del.append(key)
                    else:
                        keys_already_in_dict.append(key)
                for key in keys_to_del:
                    del param_dict[key]
                for key_to_have in list_of_cur_optimizer_param_keys:
                    if key_to_have != "type" and key_to_have not in keys_already_in_dict:
                        param_dict[key_to_have] = cur_optimizer_params.get(key_to_have)
            except:
                pass

    trainer = Trainer.from_params(model=model,
                                  serialization_dir=serialization_dir,
                                  iterator=iterator,
                                  train_data=train_data,
                                  validation_data=validation_data,
                                  params=trainer_params,
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

    return best_model


def get_config_filenames_matching(expression):
    matching_filenames = list(glob(expression))
    assert len(matching_filenames) > 0, "No matching config files found for " + str(expression)
    return matching_filenames


def modify_param(param_set_to_modify, name_of_param, val):
    if val is None:
        param_set_to_modify.pop(name_of_param, None)
    else:
        if isinstance(val, Params):
            val = val.as_dict()
        param_set_to_modify.params[name_of_param] = val


def train_seq_models_reuse_iterator(base_filename, output_dir_base, gpu):
    processed_filenames = {}
    base_filename_prefix = base_filename[:base_filename.rfind('.')]
    filename_expression = base_filename_prefix + '*' + base_filename[base_filename.rfind('.'):]
    if os.path.isfile(base_filename):
        params_to_pull_iterator_from = Params.from_file(base_filename, "")
        params_for_copying = Params.from_file(base_filename, "")
    else:
        filename_to_pull = get_config_filenames_matching(filename_expression)[0]
        params_to_pull_iterator_from = Params.from_file(filename_to_pull, "")
        params_for_copying = Params.from_file(filename_to_pull, "")

    all_datasets = datasets_from_params(params_to_pull_iterator_from)
    datasets_for_vocab_creation = set(params_to_pull_iterator_from.pop("datasets_for_vocab_creation", all_datasets))

    vocab = Vocabulary.from_params(
        params_to_pull_iterator_from.pop("vocabulary", {}),
        (instance for key, dataset in all_datasets.items()
         for instance in dataset
         if key in datasets_for_vocab_creation)
    )

    iterator = DataIterator.from_params(params_to_pull_iterator_from.pop("iterator"))
    iterator.index_with(vocab)
    validation_iterator_params = params_to_pull_iterator_from.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    params_to_copy = [
        "validation_iterator",
        "vocabulary",
        "iterator",
        "dataset_reader",
        "datasets_for_vocab_creation",
        "train_data_path",
        "validation_data_path"
    ]
    copied_param_vals = [(param_name, params_for_copying.pop(param_name, None)) for param_name in params_to_copy]

    while True:
        config_filenames = get_config_filenames_matching(filename_expression)
        cur_config_filename = None
        for config_filename in config_filenames:
            if not config_filename in processed_filenames:
                processed_filenames[config_filename] = 0
                cur_config_filename = config_filename
                break

        if cur_config_filename is None:
            break

        edit_config_file_to_have_gpu(cur_config_filename, gpu)

        progressing_ok = True
        try:
            params = Params.from_file(cur_config_filename, "")
        except:
            print("Could not properly read params from " + cur_config_filename + "; skipping.")
            progressing_ok = False

        if progressing_ok:
            try:
                for param_tup in copied_param_vals:
                    modify_param(params, param_tup[0], param_tup[1])
            except:
                print("Something went wrong while modifying params in " + cur_config_filename)
                progressing_ok = False

        if progressing_ok:
            print("Starting to train model from " + cur_config_filename)

        if progressing_ok:
            try:
                cur_config_filename = cur_config_filename[:cur_config_filename.rfind('.')]
                last_letters_to_take = len(cur_config_filename) - len(base_filename_prefix)
                if last_letters_to_take > 0:
                    tag_to_append_to_dir = cur_config_filename[(-1 * last_letters_to_take):]
                else:
                    tag_to_append_to_dir = ''
                serialization_dir = output_dir_base + tag_to_append_to_dir
            except:
                progressing_ok = False
                print("Could not properly assemble a serialization directory")

        if progressing_ok:
            try:
                train_model_given_params_and_iterators(params, serialization_dir, iterator,
                                                       validation_iterator, vocab, all_datasets, params_to_copy)
            except:
                progressing_ok = False
                print("Training model failed for some reason; skipping to next model.")
    print("Done processing all config files.")


class TrainerPieces(NamedTuple):
    """
        We would like to avoid having complex instantiation logic taking place
        in `Trainer.from_params`. This helper class has a `from_params` that
        instantiates a model, loads train (and possibly validation and test) datasets,
        constructs a Vocabulary, creates data iterators, and handles a little bit
        of bookkeeping. If you're creating your own alternative training regime
        you might be able to use this.
        """
    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    params: Params

    def from_params(params: Params, iterator, val_iterator, vocab, all_datasets,
                    serialization_dir: str, recover: bool = False) -> \
            'TrainerPieces':
        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
            get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return TrainerPieces(model, iterator,
                             train_data, validation_data, test_data,
                             val_iterator, trainer_params)


def train_model_given_params_and_iterators(params, serialization_dir, iterator, validation_iterator, vocab,
                                           all_datasets, copied_but_unused_params,
                                           file_friendly_logging: bool = False,
                                           recover: bool = False,
                                           force: bool = False
                                           ):
    """
        Trains the model specified in the given :class:`Params` object, using the data and training
        parameters also specified in that object, and saves the results in ``serialization_dir``.
        Parameters
        ----------
        params : ``Params``
            A parameter object specifying an AllenNLP Experiment.
        serialization_dir : ``str``
            The directory in which to save results and logs.
        file_friendly_logging : ``bool``, optional (default=False)
            If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
            down tqdm's output to only once every 10 seconds.
        recover : ``bool``, optional (default=False)
            If ``True``, we will try to recover a training run from an existing serialization
            directory.  This is only intended for use when something actually crashed during the middle
            of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
        force : ``bool``, optional (default=False)
            If ``True``, we will overwrite the serialization directory if it already exists.
        Returns
        -------
        best_model: ``Model``
            The model with the best epoch weights.
        """
    prepare_environment(params)
    create_serialization_dir(params, serialization_dir, recover, force)
    stdout_handler = prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    for param_name in copied_but_unused_params:
        params.pop(param_name, None)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    assert trainer_type == "default", "Trainer type is given as " + str(trainer_type)
    # Special logic to instantiate backward-compatible trainer.

    pieces = TrainerPieces.from_params(params, iterator, validation_iterator, vocab, all_datasets,
                                       serialization_dir, recover)  # pylint: disable=no-member
    trainer = Trainer.from_params(
        model=pieces.model,
        serialization_dir=serialization_dir,
        iterator=pieces.iterator,
        train_data=pieces.train_dataset,
        validation_data=pieces.validation_dataset,
        params=pieces.params,
        validation_iterator=pieces.validation_iterator)
    evaluation_iterator = pieces.validation_iterator or pieces.iterator
    evaluation_dataset = pieces.test_dataset

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

    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(trainer.model, evaluation_dataset, evaluation_iterator,
                                cuda_device=trainer._cuda_devices[0],  # pylint: disable=protected-access,
                                # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
                                batch_weight_key="")

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    if stdout_handler is not None:
        cleanup_global_logging(stdout_handler)

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)
    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True,
                        help="The type of model to train",
                        choices=['hanrnn', 'hanconv', 'flanrnn', 'flanconv', 'han_encless', 'flan_encless'])
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Which dataset to train the model on",
                        choices=['amazon', 'yahoo10cat', 'yelp', 'imdb', 'whateverDatasetYouHaveInMind'])
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU to use (can supply -1 if not using GPU)")

    parser.add_argument("--optional-model-tag", type=str, required=False, default='',
                        help="Optional tag to append to end of model serialization dir")
    parser.add_argument("--train-multiple-models", type=str, required=False, default='False',
                        help="Whether to train multiple models on the same data, reusing the loaded iterator")
    parser.add_argument("--crashed-last-time", type=bool, required=False, default=False,
                        help="Whether we're resuming from a crashed run")
    parser.add_argument("--train-existing-model", type=bool, required=False, default=False,
                        help="Whether we're resuming from a preexisting model")
    parser.add_argument("--continue-on-same-config-file", type=bool, required=False, default=False,
                        help="Whether we're resuming from an interrupted run")
    parser.add_argument("--output-dir-base", required=False,
                        default=base_serialized_models_dir,
                        help="Which directory each individual model's serialization directory sits in")
    parser.add_argument("--dir-with-config-files", required=False, type=str,
                        default=directory_with_config_files,
                        help="Base directory for all config files")
    args = parser.parse_args()
    if not args.output_dir_base.endswith('/'):
        args.output_dir_base += '/'
    if not os.path.isdir(args.output_dir_base):
        os.makedirs(args.output_dir_base)
    if not args.dir_with_config_files.endswith('/'):
        args.dir_with_config_files += '/'
    output_dir = args.output_dir_base + args.dataset_name + '-' + args.model
    if args.optional_model_tag != '':
        output_dir += '-' + args.optional_model_tag
    if not args.continue_on_same_config_file and not args.train_existing_model:
        assert not os.path.isdir(output_dir), "Output dir " + str(output_dir) + " must not already exist."

    # allows config file to reference the module nicknames registered directly above custom class declarations
    import_submodules('attn_tests_lib')
    import_submodules('textcat')

    config_file = args.dir_with_config_files + args.dataset_name + corresponding_config_files[args.model]
    edit_config_file_to_have_gpu(config_file, args.gpu)

    if args.train_multiple_models.lower().startswith('t'):
        train_seq_models_reuse_iterator(config_file, output_dir, args.gpu)
    elif args.continue_on_same_config_file or (not args.train_existing_model):
        print("Starting to train model from " + config_file)
        train_model_from_file(config_file, output_dir, recover=args.continue_on_same_config_file)
    else:  # train existing model
        print("Starting to train model from " + config_file)
        # figure out which output dir we should actually be using-- will have a number tacked on
        # to the end of it. figure out which one.
        assert os.path.isdir(output_dir)
        if output_dir.endswith('/'):
            output_dir = output_dir[:-1]

        original_output_dir = output_dir

        next_available_ind = 2
        while os.path.isdir(original_output_dir + "-" + str(next_available_ind)):
            next_available_ind += 1
        output_dir = original_output_dir + "-" + str(next_available_ind) + '/'

        output_dir_to_load_prev_best_model_from = original_output_dir
        if next_available_ind > 2:
            output_dir_to_load_prev_best_model_from += "-" + str(next_available_ind - 1) + '/'

        # loaded onto cpu
        new_params = Params.from_file(config_file, "")
        prev_best_model = load_prev_best_model(output_dir_to_load_prev_best_model_from)

        return train_model_but_load_prev_model_weights(new_params, output_dir, prev_best_model, False, False, False)


if __name__ == '__main__':
    main()
