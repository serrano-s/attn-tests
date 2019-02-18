import torch
import numpy as np
from glob import glob
import pickle


def get_corr_ind_and_attn_vals_out_of_line(line, return_log_attn_vals=False):
    corr_ind = int(line[:line.index(':')])
    attn_vals = [float(str_val) for str_val in line[line.index(':') + 1:].strip().split(' ')]
    if not return_log_attn_vals:
        attn_vals = np.exp(attn_vals - np.max(attn_vals))
        attn_vals = list(attn_vals / attn_vals.sum())
    return corr_ind, attn_vals


def load_attn_dists(attn_weight_filename):
    attn_dists = []
    corr_inds = []
    with open(attn_weight_filename, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            corr_ind, attn_vals = get_corr_ind_and_attn_vals_out_of_line(line, return_log_attn_vals=False)
            attn_dists.append(attn_vals)
            corr_inds.append(corr_ind)
    return attn_dists, corr_inds


def load_log_unnormalized_attn_dists(attn_weight_filename):
    attn_dists = []
    corr_inds = []
    with open(attn_weight_filename, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            corr_ind, attn_vals = get_corr_ind_and_attn_vals_out_of_line(line, return_log_attn_vals=True)
            attn_dists.append(attn_vals)
            corr_inds.append(corr_ind)
    return attn_dists, corr_inds


class AttentionIterator:
    def __init__(self, attn_weight_filename, return_log_attn_vals=True):
        self.attn_weight_filename = attn_weight_filename
        self.return_log_attn_vals = return_log_attn_vals

    def __iter__(self):
        with open(self.attn_weight_filename, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                corr_ind, attn_vals = \
                    get_corr_ind_and_attn_vals_out_of_line(line, return_log_attn_vals=self.return_log_attn_vals)
                yield attn_vals


class IntermediateBatchIterator:
    def __init__(self, attn_weight_filename, corr_vector_dir, batch_size, return_log_attn_vals=False,
                 also_return_grads=False):
        self.attn_weight_filename = attn_weight_filename
        if not corr_vector_dir.endswith('/'):
            corr_vector_dir += '/'
        if also_return_grads:
            self.grad_dir = corr_vector_dir + 'gradients/'
            self.also_return_grads = True
        else:
            self.grad_dir = None
            self.also_return_grads = False
        self.corr_vector_dir = corr_vector_dir
        self.num_instances = 0
        self.batch_size = batch_size
        self.return_log_attn_vals = return_log_attn_vals
        with open(attn_weight_filename, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                instance_num = int(line[:line.index(':')])
                if instance_num > self.num_instances:
                    self.num_instances = instance_num

    def __iter__(self):
        last_instance_ind_delivered = 0
        next_available_corr_vects = self.get_next_corr_vect_np_array(1)
        if self.also_return_grads:
            next_available_grads = self.get_next_grads_np_array(1)
        seq_lens = []
        list_of_attn_vals = []
        max_seq_len_in_batch = -1
        with open(self.attn_weight_filename, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                corr_ind, attn_vals = \
                    get_corr_ind_and_attn_vals_out_of_line(line, return_log_attn_vals=self.return_log_attn_vals)
                seq_len = len(attn_vals)
                seq_lens.append(seq_len)
                if seq_len > max_seq_len_in_batch:
                    max_seq_len_in_batch = seq_len
                attn_vals = np.array(attn_vals)
                list_of_attn_vals.append(attn_vals)

                if len(seq_lens) >= self.batch_size or corr_ind == self.num_instances:
                    # this is the end of a batch
                    next_available_corr_vects, np_corr_vect = \
                        self.get_np_array_of_corr_vects(next_available_corr_vects, max_seq_len_in_batch,
                                                        len(seq_lens), last_instance_ind_delivered)
                    if self.also_return_grads:
                        next_available_grads, np_corr_grad = self.get_np_array_of_grads(next_available_grads,
                                                                                        max_seq_len_in_batch,
                                                                                        len(seq_lens),
                                                                                        last_instance_ind_delivered)
                    np_attn_mat = self.pad_attn_weights_and_concat(list_of_attn_vals, max_seq_len_in_batch)
                    seq_lens_to_deliver = seq_lens

                    last_instance_ind_delivered += len(seq_lens)
                    seq_lens = []
                    list_of_attn_vals = []
                    max_seq_len_in_batch = -1
                    if self.also_return_grads:
                        yield torch.from_numpy(np_attn_mat).float(), \
                              torch.autograd.Variable(torch.from_numpy(np_corr_vect).float()), \
                              seq_lens_to_deliver, \
                              np_corr_grad
                    else:
                        yield torch.from_numpy(np_attn_mat).float(), \
                              torch.autograd.Variable(torch.from_numpy(np_corr_vect).float()), \
                              seq_lens_to_deliver

    def pad_attn_weights_and_concat(self, list_of_attn_vals, max_seq_len_in_batch):
        # before stacking, adjust dim 1 to equal max_seq_len_in_batch
        for i in range(len(list_of_attn_vals)):
            piece = list_of_attn_vals[i]
            if piece.shape[0] > max_seq_len_in_batch:
                list_of_attn_vals[i] = piece[:max_seq_len_in_batch]
            elif piece.shape[0] < max_seq_len_in_batch:
                list_of_attn_vals[i] = \
                    np.concatenate([piece, np.zeros(max_seq_len_in_batch - piece.shape[0])], axis=0)
        return np.vstack(list_of_attn_vals)

    def get_np_array_of_corr_vects(self, next_available_corr_vects, max_seq_len_in_batch, batch_len,
                                   last_instance_ind_delivered):
        pieces_to_stack = []
        num_instances_included = 0
        while num_instances_included < batch_len:
            num_left_to_include = batch_len - num_instances_included
            if num_left_to_include >= next_available_corr_vects.shape[0]:
                pieces_to_stack.append(next_available_corr_vects)
                last_instance_ind_delivered += next_available_corr_vects.shape[0]
                num_instances_included += next_available_corr_vects.shape[0]
                next_available_corr_vects = self.get_next_corr_vect_np_array(last_instance_ind_delivered + 1)
            else:
                pieces_to_stack.append(next_available_corr_vects[:num_left_to_include])
                last_instance_ind_delivered += num_left_to_include
                num_instances_included += num_left_to_include
                next_available_corr_vects = next_available_corr_vects[num_left_to_include:]

        # before stacking, adjust dim 1 to equal max_seq_len_in_batch
        for i in range(len(pieces_to_stack)):
            piece = pieces_to_stack[i]
            if piece.shape[1] > max_seq_len_in_batch:
                pieces_to_stack[i] = piece[:, :max_seq_len_in_batch, :]
            elif piece.shape[1] < max_seq_len_in_batch:
                pieces_to_stack[i] = \
                    np.concatenate([piece, np.zeros((piece.shape[0], max_seq_len_in_batch - piece.shape[1],
                                                     piece.shape[2]), dtype=float)], axis=1)

        np_corr_vect = np.concatenate(pieces_to_stack, axis=0)
        return next_available_corr_vects, np_corr_vect

    def get_np_array_of_grads(self, next_available_grads, max_seq_len_in_batch, batch_len,
                              last_instance_ind_delivered):
        pieces_to_stack = []
        num_instances_included = 0
        while num_instances_included < batch_len:
            num_left_to_include = batch_len - num_instances_included
            if num_left_to_include >= next_available_grads.shape[0]:
                pieces_to_stack.append(next_available_grads)
                last_instance_ind_delivered += next_available_grads.shape[0]
                num_instances_included += next_available_grads.shape[0]
                next_available_grads = self.get_next_grads_np_array(last_instance_ind_delivered + 1)
            else:
                pieces_to_stack.append(next_available_grads[:num_left_to_include])
                last_instance_ind_delivered += num_left_to_include
                num_instances_included += num_left_to_include
                next_available_grads = next_available_grads[num_left_to_include:]

        # before stacking, adjust dim 1 to equal max_seq_len_in_batch
        for i in range(len(pieces_to_stack)):
            piece = pieces_to_stack[i]
            if piece.shape[1] > max_seq_len_in_batch:
                pieces_to_stack[i] = piece[:, :max_seq_len_in_batch]
            elif piece.shape[1] < max_seq_len_in_batch:
                pieces_to_stack[i] = \
                    np.concatenate([piece, np.zeros((piece.shape[0],
                                                     max_seq_len_in_batch - piece.shape[1]), dtype=float)], axis=1)

        np_corr_vect = np.concatenate(pieces_to_stack, axis=0)
        return next_available_grads, np_corr_vect


    def get_next_corr_vect_np_array(self, starting_ind):
        if starting_ind > self.num_instances:
            return None
        possible_next_filenames = glob(self.corr_vector_dir + str(starting_ind) + '-*')
        assert len(possible_next_filenames) == 1, "Query: " + self.corr_vector_dir + str(starting_ind) + '-' + '\n' + \
                                                  "Found: " + str(possible_next_filenames)
        return np.load(possible_next_filenames[0])

    def get_next_grads_np_array(self, starting_ind):
        if starting_ind > self.num_instances:
            return None
        possible_next_filenames = glob(self.grad_dir + 'gradient_wrt_attn_weights_' + str(starting_ind) + '-*')
        assert len(possible_next_filenames) == 1, "Query: " + self.corr_vector_dir + str(starting_ind) + '-' + '\n' + \
                                                  "Found: " + str(possible_next_filenames)
        with open(possible_next_filenames[0], 'rb') as f:
            torch_var_tensor = pickle.load(f)
        np_array = torch_var_tensor.data.cpu().numpy()
        return np_array
