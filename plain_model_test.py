from typing import Dict, Any, Iterable
import argparse
import logging
import json

import torch
import numpy as np

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             label_fname: str) -> Dict[str, Any]:
    _warned_tqdm_ignores_underscores = False
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        label_file = open(label_fname, 'w')
        label_file.write('real_label,guessed_label\n')

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
        total_num_inst = 0
        for batch in generator_tqdm:
            num_inst = batch['tokens']['tokens'].size(0)
            total_num_inst += num_inst
            batch = util.move_to_device(batch, cuda_device)

            output_dict = model(**batch)
            output_matrix = output_dict['label_logits'].data.numpy()
            output_labels = np.argmax(output_matrix, axis=1)
            true_labels = batch['label'].data.numpy()
            assert true_labels.shape[0] == output_labels.shape[0]
            for i in range(true_labels.shape[0]):
                label_file.write(str(int(true_labels[i])) + ',')
                label_file.write(str(int(output_labels[i])) + '\n')

            metrics = model.get_metrics()
            if (not _warned_tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                _warned_tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)


        print("NUM INSTANCES ITERATED OVER: " + str(total_num_inst))
        label_file.close()

        return model.get_metrics(reset=True)
      

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-file', type=str, help='path to the file containing the evaluation data')
    parser.add_argument('--output-file', type=str, help='path to output file')
    parser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')
    parser.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')
    parser.add_argument('--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--include-package', type=str)
    parser.add_argument('--archive-file', type=str)
    args = parser.parse_args()
    
    if '/' in args.weights_file:
        label_file = args.weights_file[:args.weights_file.rfind('/') + 1]
    else:
        label_file = ''
    label_file += (args.input_file[args.input_file.rfind('/') + 1: args.input_file.rfind('.')] if '/' in args.input_file else
                   args.input_file[:args.input_file.rfind('.')])
    label_file += '_reallabel_guessedlabel.csv'
    print("Will write labels to " + label_file)
    print("Evaluating on " + args.input_file)
    print("Archive file being used is " + args.archive_file)
    print("Weights file being used is " + args.weights_file)
    print()

    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    import_submodules(args.include_package)
    import_submodules("attn_tests_lib")
    import_submodules("textcat")

    with open(args.overrides, 'r') as f:
        args.overrides = " ".join([l.strip() for l in f.readlines()])
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    metrics = evaluate(model, instances, iterator, args.cuda_device, label_file)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    print('\n' + json.dumps(metrics, indent=4))
    print("Successfully wrote labels to " + label_file)

    output_file = args.output_file
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics


if __name__ == '__main__':
    main()
