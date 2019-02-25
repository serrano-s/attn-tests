from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from typing import Dict, Any, Iterable, List
import json
import logging
import torch
import sys


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def dump_metrics(file_path: str, metrics: Dict[str, Any], log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
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


def get_frozen_and_tunable_parameter_names(model: torch.nn.Module) -> List:
    frozen_parameter_names = []
    tunable_parameter_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen_parameter_names.append(name)
        else:
            tunable_parameter_names.append(name)
    return [frozen_parameter_names, tunable_parameter_names]


def cleanup_global_logging(stdout_handler: logging.FileHandler) -> None:
    """
    This function closes any open file handles and logs set up by `prepare_global_logging`.
    Parameters
    ----------
    stdout_handler : ``logging.FileHandler``, required.
        The file handler returned from `prepare_global_logging`, attached to the global logger.
    """
    stdout_handler.close()
    logging.getLogger().removeHandler(stdout_handler)

    if isinstance(sys.stdout, TeeLogger):
        sys.stdout = sys.stdout.cleanup()
    if isinstance(sys.stderr, TeeLogger):
        sys.stderr = sys.stderr.cleanup()
