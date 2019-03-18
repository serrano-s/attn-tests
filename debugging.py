from typing import Dict, Iterable
import logging
import numpy as np
from scipy.misc import logsumexp

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import namespace_match, prepare_global_logging
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    This method is just copied from AllenNLP.

    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def debug_vocab(parameter_filename: str,
                serialization_dir: str,
                overrides: str = "",
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)

    prepare_global_logging(serialization_dir, file_friendly_logging)

    check_for_gpu(params.get('trainer').get('cuda_device', -1))

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

    vocab = model.vocab
    vocab_namespace_dict = vocab._token_to_index
    vocab_oov_token = vocab._oov_token
    vocab_non_padded_namespaces = vocab._non_padded_namespaces  # this is a set

    vocab_tokens_dict = vocab_namespace_dict['tokens']
    vocab_labels_dict = vocab_namespace_dict['labels']

    print()
    print("Vocab's OOV token: " + vocab_oov_token)
    print("Non-padded namespaces in vocab: " + str(list(vocab_non_padded_namespaces)))
    print()

    print("Number of words in vocab's tokens dict: " + str(len(vocab_tokens_dict)))
    if any(namespace_match(pattern, 'tokens') for pattern in vocab_non_padded_namespaces):
        is_padded = False
    else:
        is_padded = True
    print("tokens will return True for is_padded: " + str(is_padded))
    print("Vocab's OOV token is in its tokens dict (should be True): " + str(vocab_oov_token in vocab_tokens_dict))
    print()

    print("Number of words in vocab's labels dict: " + str(len(vocab_labels_dict)))
    if any(namespace_match(pattern, 'labels') for pattern in vocab_non_padded_namespaces):
        is_padded = False
    else:
        is_padded = True
    print("labels will return True for is_padded: " + str(is_padded))
    print("Vocab's OOV token is in its labels dict (should be False): " + str(vocab_oov_token in vocab_labels_dict))


def load_config_file_print_params(params_fname: str, param_depth: int = -1):
    params = Params.from_file(params_fname)
    print_params_at_depth(params, 1, depth_cap=param_depth)


def print_params_at_depth(item, cur_depth: int, depth_cap: int):
    if depth_cap != -1 and cur_depth > depth_cap:
        return
    tab_prepend = ''
    if cur_depth >= 1:
        tab_prepend = ''.join(['\t'] * (cur_depth - 1))
    if isinstance(item, Params):
        for key in item.keys():
            print(tab_prepend + key, end='')
            if not isinstance(item[key], Params):
                print(" : " + str(item[key]))
            else:
                print()
                print_params_at_depth(item[key], cur_depth + 1, depth_cap)


def calculate_nonlog_kl_divergence(from_dist, to_dist):
    assert len(from_dist) == len(to_dist)
    total = 0
    for i in range(len(from_dist)):
        total += to_dist[i] * np.log(to_dist[i] / from_dist[i])
    return total


def calculate_nonlog_js_divergence(from_dist, to_dist):
    assert len(from_dist) == len(to_dist)
    midpoint_dist = []
    for i in range(len(from_dist)):
        midpoint_dist.append((from_dist[i] + to_dist[i]) / 2)
    return .5 * calculate_nonlog_kl_divergence(midpoint_dist, from_dist) + \
           .5 * calculate_nonlog_kl_divergence(midpoint_dist, to_dist)


def test_divergences():


    """# Calculated a kl div of -0.11142599918864647
    log_dist_from = np.array([[-5.320183, -5.004407, -3.739851, -2.4476514, -1.146145, 0.20984331, 1.1307197, 0.9277696, -0.45606202, -1.5285392]])#np.array([[-1.704001, -0.21752474, 0.46928683, 1.1984, 1.8707601, 0.4779932,
                    #           -1.3708279, -3.6425877, -5.238203, -6.296172]])
    log_dist_to = np.array([[-5.3201814, -5.004406, -3.7398503, -2.4476511, -1.1461449, 0.20984319, 1.1307195, 0.92776954, -0.45606202, -1.5285391]])#np.array([[-1.7040007, -0.21752474, 0.4692869, 1.1984, 1.8707601, 0.47799325,
                  #           -1.3708278, -3.6425874, -5.238202, -6.296171]])

    log_dist_from = np.load("log_arr_from.npy")
    log_dist_to = np.load("log_arr_to.npy")
    log_dist_from = np.reshape(log_dist_from, (1, log_dist_from.shape[0]))
    log_dist_to = np.reshape(log_dist_to, (1, log_dist_to.shape[0]))


    log_dist_from_preexp = log_dist_from[0] - logsumexp(log_dist_from[0])
    dist_from = list(np.exp(log_dist_from_preexp))
    log_dist_to_preexp = log_dist_to[0] - logsumexp(log_dist_to[0])
    dist_to = list(np.exp(log_dist_to_preexp))

    print(log_dist_from_preexp)
    print(log_dist_to_preexp)
    print()"""

    dist_from = [.6, .3, .1]
    dist_to = [.2, .4, .4]
    dist_from_2 = [.7, .1, .2]
    dist_to_2 = [.6, .3, .1]

    print(dist_from)
    print(dist_to)

    print("Actual KL divergence: " + str(calculate_nonlog_kl_divergence(dist_from, dist_to)))
    print("Actual JS divergence: " + str(calculate_nonlog_js_divergence(dist_from, dist_to)))
    print()
    print(dist_from_2)
    print(dist_to_2)
    print("Actual KL divergence: " + str(calculate_nonlog_kl_divergence(dist_from_2, dist_to_2)))
    print("Actual JS divergence: " + str(calculate_nonlog_js_divergence(dist_from_2, dist_to_2)))
    print()

    log_dist_from = np.array([dist_from, dist_from_2])
    log_dist_to = np.array([dist_to, dist_to_2])
    log_dist_from = np.log(log_dist_from)
    log_dist_to = np.log(log_dist_to)

    kl_divs = get_kl_div_of_dists(log_dist_from, log_dist_to)
    js_divs = get_js_div_of_dists(log_dist_from, log_dist_to)

    print("Calculated KL divergence for dist 1: " + str(kl_divs[0]))
    print("Calculated JS divergence for dist 1: " + str(js_divs[0]))
    print("Calculated KL divergence for dist 2: " + str(kl_divs[1]))
    print("Calculated JS divergence for dist 2: " + str(js_divs[1]))
    print()

    log_dist_from += 5
    log_dist_to -= 7

    kl_divs = get_kl_div_of_dists(log_dist_from, log_dist_to)
    js_divs = get_js_div_of_dists(log_dist_from, log_dist_to)

    print("Calculated KL divergence after adjusting log constant for dist 1: " + str(kl_divs[0]))
    print("Calculated JS divergence after adjusting log constant for dist 1: " + str(js_divs[0]))
    print("Calculated KL divergence after adjusting log constant for dist 2: " + str(kl_divs[1]))
    print("Calculated JS divergence after adjusting log constant for dist 2: " + str(js_divs[1]))


if __name__ == '__main__':
    from test_model import get_kl_div_of_dists, get_js_div_of_dists
    test_divergences()
