from typing import Dict, Iterable
import logging

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
