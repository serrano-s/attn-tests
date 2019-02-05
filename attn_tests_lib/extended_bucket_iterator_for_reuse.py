import random
from overrides import overrides
from typing import List, Tuple, Iterable, cast, Dict
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary


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


@DataIterator.register("extended_bucket_for_reuse")
class ExtendedBucketIteratorForReuse(BucketIterator):
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
        assert self._change_create_batches
        self.training_sortable_batches = None
        self.val_sortable_batches = None
        self.penultimate_batch_train = None
        self.last_batch_train = None
        self.penultimate_batch_val = None
        self.last_batch_val = None
        self.provide_training_batches = True

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        if not self._change_create_batches:
            for ret_val in iter(super()._create_batches(instances, shuffle)):
                yield ret_val
        else:
            for ret_val in iter(self._modified_create_batches(instances, shuffle)):
                yield ret_val

    def _modified_create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        if self.training_sortable_batches is None:
            self.training_sortable_batches = []
            cur_counter = 0
            for instance_list in self._memory_sized_lists(instances):
                instance_list = sort_by_padding_modified(instance_list,
                                                         self._sorting_keys,
                                                         self.vocab,
                                                         self._padding_noise)

                batches = []
                for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                    for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances):
                        batches.append(Batch(possibly_smaller_batches))

                self.move_to_front = self._biggest_batch_first and len(batches) > 1
                if self.move_to_front:
                    # We'll actually pop the last _two_ batches, because the last one might not be full.
                    self.last_batch_train = batches.pop()
                    self.penultimate_batch_train = batches.pop()

                batch_counter = 0
                batch_list = []
                for batch in batches:
                    batch_list.append((batch_counter, batch))
                    batch_counter += 1
                self.training_sortable_batches.append((cur_counter, batch_list))
                cur_counter += 1
            instance_it = self.training_sortable_batches
            self.provide_training_batches = True
        elif self.val_sortable_batches is None:
            self.val_sortable_batches = []
            cur_counter = 0
            for instance_list in self._memory_sized_lists(instances):
                instance_list = sort_by_padding_modified(instance_list,
                                                         self._sorting_keys,
                                                         self.vocab,
                                                         self._padding_noise)

                batches = []
                for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                    for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances):
                        batches.append(Batch(possibly_smaller_batches))

                self.move_to_front = self._biggest_batch_first and len(batches) > 1
                if self.move_to_front:
                    # We'll actually pop the last _two_ batches, because the last one might not be full.
                    self.last_batch_val = batches.pop()
                    self.penultimate_batch_val = batches.pop()

                batch_counter = 0
                batch_list = []
                for batch in batches:
                    batch_list.append((batch_counter, batch))
                    batch_counter += 1
                self.val_sortable_batches.append((cur_counter, batch_list))
                cur_counter += 1
            instance_it = self.val_sortable_batches
            self.provide_training_batches = False
        elif self.provide_training_batches:
            instance_it = self.training_sortable_batches
        elif not self.provide_training_batches:
            instance_it = self.val_sortable_batches
        for instance_list in instance_it:
            instance_list = instance_list[1]

            batches = instance_list

            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if self.move_to_front:
                if self.provide_training_batches:
                    yield from [self.last_batch_train]
                    yield from [self.penultimate_batch_train]
                else:
                    yield from [self.last_batch_val]
                    yield from [self.penultimate_batch_val]

            for batch in batches:
                yield batch[1]

        if self.provide_training_batches:
            self.training_sortable_batches = sorted(self.training_sortable_batches, key=(lambda x: x[0]))
            for i in range(len(self.training_sortable_batches)):
                self.training_sortable_batches[i] = (self.training_sortable_batches[i][0],
                                                     sorted(self.training_sortable_batches[i][1], key=(lambda x: x[0])))
        else:
            self.val_sortable_batches = sorted(self.val_sortable_batches, key=(lambda x: x[0]))
            for i in range(len(self.val_sortable_batches)):
                self.val_sortable_batches[i] = (self.val_sortable_batches[i][0],
                                                sorted(self.val_sortable_batches[i][1], key=(lambda x: x[0])))

        self.provide_training_batches = not self.provide_training_batches
