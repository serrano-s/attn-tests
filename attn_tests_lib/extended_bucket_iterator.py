import random
from overrides import overrides
from typing import List, Tuple, Iterable, cast, Dict
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
import math


def sort_by_padding_modified(instances: List[Instance],
                             sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                             vocab: Vocabulary,
                             padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = instance.get_padding_lengths()
        padding_lengths["sentences"] = {"num_sentences": len(instance.fields['tokens'].field_list)}
        padding_lengths = cast(Dict[str, Dict[str, float]], padding_lengths)
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]


@DataIterator.register("extended_bucket")
class ExtendedBucketIterator(BucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        super().__init__(sorting_keys, padding_noise=padding_noise, biggest_batch_first=biggest_batch_first,
                         batch_size=batch_size, instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory, cache_instances=cache_instances,
                         track_epoch=track_epoch, maximum_samples_per_batch=maximum_samples_per_batch)
        # look out for [sentences, num_sentences]
        self._change_create_batches = False
        for key in sorting_keys:
            if key[0] == "sentences":
                self._change_create_batches = True

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        if not self._change_create_batches:
            for ret_val in iter(super()._create_batches(instances, shuffle)):
                yield ret_val
        else:
            for ret_val in iter(self._modified_create_batches(instances, shuffle)):
                yield ret_val

    def old_ensure_batch_is_sufficiently_small(self, batch_instances: Iterable[Instance]) -> List[List[Instance]]:
        """
        If self._maximum_samples_per_batch is specified, then split the batch into smaller
        sub-batches if it exceeds the maximum size.
        """
        if self._maximum_samples_per_batch is None:
            return [list(batch_instances)]

        # check if we need to break into smaller chunks
        key, limit = self._maximum_samples_per_batch
        padding_length = -1
        list_batch_instances = list(batch_instances)
        for instance in list_batch_instances:
            if self.vocab is not None:
                # we index here to ensure that shape information is available,
                # as in some cases (with self._maximum_samples_per_batch)
                # we need access to shaping information before batches are constructed)
                instance.index_fields(self.vocab)
            field_lengths = instance.get_padding_lengths()
            for _, lengths in field_lengths.items():
                try:
                    padding_length = max(padding_length,
                                         lengths[key])
                except KeyError:
                    pass

        if padding_length * len(list_batch_instances) > limit:
            # need to shrink
            num_samples = padding_length * len(list_batch_instances)
            num_shrunk_batches = math.ceil(num_samples / float(limit))
            shrunk_batch_size = math.ceil(len(list_batch_instances) / num_shrunk_batches)
            shrunk_batches = []
            start = 0
            while start < len(list_batch_instances):
                end = start + shrunk_batch_size
                shrunk_batches.append(list_batch_instances[start:end])
                start = end
            return shrunk_batches
        else:
            return [list_batch_instances]

    def _modified_create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            instance_list = sort_by_padding_modified(instance_list,
                                                     self._sorting_keys,
                                                     self.vocab,
                                                     self._padding_noise)

            batches = []
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self.old_ensure_batch_is_sufficiently_small(batch_instances):
                    batches.append(Batch(possibly_smaller_batches))

            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches
