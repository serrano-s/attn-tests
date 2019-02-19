import argparse
import numpy as np
from test_model import get_entropy_of_dists
from test_model import first_v_second_fname
from test_model import dec_flip_stats_fname
from test_model import rand_results_fname  # this one also has variable-length fields now *shrug*
from test_model import unchanged_fname  # index this one last because its fields are of variable length
from test_model import grad_based_stats_fname
from test_model import dec_flip_rand_nontop_stats_fname
from test_model import attn_div_from_unif_fname
from statsmodels.sandbox.stats.runs import mcnemar
import matplotlib.pyplot as plt
import matplotlib
from math import fabs
matplotlib.use('Agg')
import os


ID = 0
KL_DIV_ZERO_HIGHEST = 1
JS_DIV_ZERO_HIGHEST = 2
DEC_FLIP_ZERO_HIGHEST = 3
KL_DIV_ZERO_2NDHIGHEST = 4
JS_DIV_ZERO_2NDHIGHEST = 5
DEC_FLIP_ZERO_2NDHIGHEST = 6
KL_DIV_ZERO_LOWEST = 7
JS_DIV_ZERO_LOWEST = 8
DEC_FLIP_ZERO_LOWEST = 9
KL_DIV_ZERO_2NDLOWEST = 10
JS_DIV_ZERO_2NDLOWEST = 11
DEC_FLIP_ZERO_2NDLOWEST = 12
# starting dec_flip_stats_fname: skip one here for a repeat id field
ATTN_SEQ_LEN = 14
NEEDED_REM_TOP_X_FOR_DECFLIP = 15
NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP = 16
NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP = 17
NEEDED_REM_BOTTOM_X_FOR_DECFLIP = 18
NEEDED_REM_BOTTOM_PROBMASS_FOR_DECFLIP = 19
NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP = 20
# starting rand_stats_fname: skip one here for a repeat id field
ATTN_SEQ_LEN_DUPLICATE_FOR_TESTING = 22
EXTRACTED_SINGLE_ATTN_WEIGHT_START = 23
EXTRACTED_SINGLE_ATTN_WEIGHT_END = None
EXTRACTED_SINGLE_WEIGHT_KL_START = None
EXTRACTED_SINGLE_WEIGHT_KL_END = None
EXTRACTED_SINGLE_WEIGHT_JS_START = None
EXTRACTED_SINGLE_WEIGHT_JS_END = None
NEEDED_REM_RAND_X_FOR_DECFLIP_START = None
NEEDED_REM_RAND_X_FOR_DECFLIP_END = None
NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START = None
NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_END = None
NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START = None
NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END = None
# starting unchanged_fname fields: skip one here for a repeat id field
ACTUAL_LABEL = 2
ORIG_LABEL_GUESSED = 3
OUTPUT_ENTROPY = 4
ATTN_ENTROPY = 5
WEIGHT_1ST2ND_RATIO = 6
WEIGHT_1ST2ND_ENTROPY = 7
WEIGHT_LAST2NDLAST_RATIO = 8
WEIGHT_LAST2NDLAST_ENTROPY = 9
WEIGHT_1STLAST_RATIO = 10
WEIGHT_1STLAST_ENTROPY = 11
STARTING_IND_OF_OUTPUT_CLASSES = 12
LAST_IND_OF_OUTPUT_CLASSES = None
# starting grad_stats file
ID_DUPLICATE = 1
ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING = 2
NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD = 3
NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD = 4
NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD = 5
NEEDED_REM_TOP_X_FOR_DECFLIP_GRADMULT = 6
NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRADMULT = 7
NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT = 8
KL_DIV_ZERO_HIGHESTGRAD = 9
JS_DIV_ZERO_HIGHESTGRAD = 10
DEC_FLIP_ZERO_HIGHESTGRAD = 11
KL_DIV_ZERO_2NDHIGHESTGRAD = 12
JS_DIV_ZERO_2NDHIGHESTGRAD = 13
DEC_FLIP_ZERO_2NDHIGHESTGRAD = 14
KL_DIV_ZERO_HIGHESTGRADMULT = 15
JS_DIV_ZERO_HIGHESTGRADMULT = 16
DEC_FLIP_ZERO_HIGHESTGRADMULT = 17
KL_DIV_ZERO_2NDHIGHESTGRADMULT = 18
JS_DIV_ZERO_2NDHIGHESTGRADMULT = 19
DEC_FLIP_ZERO_2NDHIGHESTGRADMULT = 20
# starting nontop stats file: id,seq_len,not_negone_if_rand_caused_decflip,zeroed_weight,rand_kl_div,rand_js_div
NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE = 3
NONTOP_RAND_ZEROED_WEIGHT = 4
NONTOP_RAND_KL_DIV = 5
NONTOP_RAND_JS_DIV = 6
# starting attn_div_from_unif file: id,seq_len,kl_div_from_unif,js_div_from_unif
ATTN_KL_DIV_FROM_UNIF = 3
ATTN_JS_DIV_FROM_UNIF = 4


image_directory = None
dataset_output_directory = None
data_dir = None


def load_in_data_table(first_v_second_filename, dec_flip_stats_filename, rand_results_filename, unchanged_filename,
                       grad_based_stats_filename, dec_flip_rand_nontop_stats_filename, attn_div_from_unif_filename):
    print("Loading in raw CSV files")
    first_v_second = np.genfromtxt(first_v_second_filename, delimiter=',', skip_header=1)
    dec_flip_stats = np.genfromtxt(dec_flip_stats_filename, delimiter=',', skip_header=1)
    rand_stats = np.genfromtxt(rand_results_filename, delimiter=',', skip_header=1)
    unchanged = np.genfromtxt(unchanged_filename, delimiter=',', skip_header=1)
    grad_stats = np.genfromtxt(grad_based_stats_filename, delimiter=',', skip_header=1)
    nontop_stats = np.genfromtxt(dec_flip_rand_nontop_stats_filename, delimiter=',', skip_header=1)
    attn_div_stats = np.genfromtxt(attn_div_from_unif_filename, delimiter=',', skip_header=1)

    if len(first_v_second.shape) == 1:
        first_v_second = np.reshape(first_v_second, (1, first_v_second.shape[0]))
    if len(dec_flip_stats.shape) == 1:
        dec_flip_stats= np.reshape(dec_flip_stats, (1, dec_flip_stats.shape[0]))
    if len(rand_stats.shape) == 1:
        rand_stats= np.reshape(rand_stats, (1, rand_stats.shape[0]))
    if len(unchanged.shape) == 1:
        unchanged = np.reshape(unchanged, (1, unchanged.shape[0]))
    if len(grad_stats.shape) == 1:
        grad_stats= np.reshape(grad_stats, (1, grad_stats.shape[0]))

    print("Checking that ID tags match up in records from each file before concatenating")
    assert first_v_second.shape[0] == dec_flip_stats.shape[0], str(first_v_second.shape) + ', ' + \
                                                               str(dec_flip_stats.shape)
    assert first_v_second.shape[0] == rand_stats.shape[0], str(first_v_second.shape) + ', ' + \
                                                           str(rand_stats.shape)
    assert first_v_second.shape[0] == unchanged.shape[0], str(first_v_second.shape) + ', ' + \
                                                          str(unchanged.shape)
    assert first_v_second.shape[0] == grad_stats.shape[0], str(first_v_second.shape) + ', ' + \
                                                           str(grad_stats.shape)
    diffs_bet_first_v_second_and_dec_flip_stats = first_v_second[:, 0] - dec_flip_stats[:, 0]
    diffs_bet_first_v_second_and_rand_stats = first_v_second[:, 0] - rand_stats[:, 0]
    diffs_bet_first_v_second_and_unchanged = first_v_second[:, 0] - unchanged[:, 0]
    assert len(np.nonzero(diffs_bet_first_v_second_and_dec_flip_stats)[0]) == 0, \
        "Some inds in first_v_second and dec_flip_stats didn't match"
    assert len(np.nonzero(diffs_bet_first_v_second_and_rand_stats)[0]) == 0, \
        "Some inds in first_v_second and rand_stats didn't match"
    assert len(np.nonzero(diffs_bet_first_v_second_and_unchanged)[0]) == 0, \
        "Some inds in first_v_second and unchanged didn't match"
    assert DEC_FLIP_ZERO_2NDLOWEST == first_v_second.shape[1] - 1
    assert NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP == first_v_second.shape[1] + dec_flip_stats.shape[1] - 1
    global EXTRACTED_SINGLE_ATTN_WEIGHT_END, EXTRACTED_SINGLE_WEIGHT_KL_START, EXTRACTED_SINGLE_WEIGHT_KL_END, \
        EXTRACTED_SINGLE_WEIGHT_JS_START, EXTRACTED_SINGLE_WEIGHT_JS_END, NEEDED_REM_RAND_X_FOR_DECFLIP_START, \
        NEEDED_REM_RAND_X_FOR_DECFLIP_END, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START, \
        NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_END, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START, \
        NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END, ACTUAL_LABEL, ORIG_LABEL_GUESSED, OUTPUT_ENTROPY, ATTN_ENTROPY, \
        WEIGHT_1ST2ND_RATIO, WEIGHT_1ST2ND_ENTROPY, WEIGHT_LAST2NDLAST_RATIO, WEIGHT_LAST2NDLAST_ENTROPY, \
        WEIGHT_1STLAST_RATIO, WEIGHT_1STLAST_ENTROPY, STARTING_IND_OF_OUTPUT_CLASSES, LAST_IND_OF_OUTPUT_CLASSES, \
        ID_DUPLICATE, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD, \
        NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD, \
        NEEDED_REM_TOP_X_FOR_DECFLIP_GRADMULT, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRADMULT, \
        NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT, KL_DIV_ZERO_HIGHESTGRAD, JS_DIV_ZERO_HIGHESTGRAD, \
        DEC_FLIP_ZERO_HIGHESTGRAD, KL_DIV_ZERO_2NDHIGHESTGRAD, JS_DIV_ZERO_2NDHIGHESTGRAD, \
        DEC_FLIP_ZERO_2NDHIGHESTGRAD, KL_DIV_ZERO_HIGHESTGRADMULT, JS_DIV_ZERO_HIGHESTGRADMULT, \
        DEC_FLIP_ZERO_HIGHESTGRADMULT, KL_DIV_ZERO_2NDHIGHESTGRADMULT, JS_DIV_ZERO_2NDHIGHESTGRADMULT, \
        DEC_FLIP_ZERO_2NDHIGHESTGRADMULT, NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE, NONTOP_RAND_ZEROED_WEIGHT, \
        NONTOP_RAND_KL_DIV, NONTOP_RAND_JS_DIV, ATTN_KL_DIV_FROM_UNIF, ATTN_JS_DIV_FROM_UNIF
    if LAST_IND_OF_OUTPUT_CLASSES is None:
        num_rand_ind_orders_sampled = (rand_stats.shape[1] - 2) // 6
        assert num_rand_ind_orders_sampled == (rand_stats.shape[1] - 2) / 6, "Didn't divide evenly: rand_stats.shape[1] was " + str(rand_stats.shape[1])
        EXTRACTED_SINGLE_ATTN_WEIGHT_END = EXTRACTED_SINGLE_ATTN_WEIGHT_START + num_rand_ind_orders_sampled - 1
        EXTRACTED_SINGLE_WEIGHT_KL_START = EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1
        EXTRACTED_SINGLE_WEIGHT_KL_END = EXTRACTED_SINGLE_WEIGHT_KL_START + num_rand_ind_orders_sampled - 1
        EXTRACTED_SINGLE_WEIGHT_JS_START = EXTRACTED_SINGLE_WEIGHT_KL_END + 1
        EXTRACTED_SINGLE_WEIGHT_JS_END = EXTRACTED_SINGLE_WEIGHT_JS_START + num_rand_ind_orders_sampled - 1
        NEEDED_REM_RAND_X_FOR_DECFLIP_START = EXTRACTED_SINGLE_WEIGHT_JS_END + 1
        NEEDED_REM_RAND_X_FOR_DECFLIP_END = NEEDED_REM_RAND_X_FOR_DECFLIP_START + num_rand_ind_orders_sampled - 1
        NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START = NEEDED_REM_RAND_X_FOR_DECFLIP_END + 1
        NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_END = NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START + \
                                                   num_rand_ind_orders_sampled - 1
        NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START = NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_END + 1
        NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END = NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START + num_rand_ind_orders_sampled - 1
        assert NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END == first_v_second.shape[1] + dec_flip_stats.shape[1] + \
               rand_stats.shape[1] - 1
        ACTUAL_LABEL += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        ORIG_LABEL_GUESSED += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        OUTPUT_ENTROPY += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        ATTN_ENTROPY += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        WEIGHT_1ST2ND_RATIO += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        WEIGHT_1ST2ND_ENTROPY += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        WEIGHT_LAST2NDLAST_RATIO += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        WEIGHT_LAST2NDLAST_ENTROPY += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        WEIGHT_1STLAST_RATIO += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        WEIGHT_1STLAST_ENTROPY += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        STARTING_IND_OF_OUTPUT_CLASSES += NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END
        LAST_IND_OF_OUTPUT_CLASSES = first_v_second.shape[1] + dec_flip_stats.shape[1] + rand_stats.shape[1] + \
                                     unchanged.shape[1] - 1
        ID_DUPLICATE += LAST_IND_OF_OUTPUT_CLASSES
        ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING = ID_DUPLICATE + 1
        NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD = ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING + 1
        NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD = NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD + 1
        NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD = NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD + 1
        NEEDED_REM_TOP_X_FOR_DECFLIP_GRADMULT = NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD + 1
        NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRADMULT = NEEDED_REM_TOP_X_FOR_DECFLIP_GRADMULT + 1
        NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT = NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRADMULT + 1
        KL_DIV_ZERO_HIGHESTGRAD = NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT + 1
        JS_DIV_ZERO_HIGHESTGRAD = KL_DIV_ZERO_HIGHESTGRAD + 1
        DEC_FLIP_ZERO_HIGHESTGRAD = JS_DIV_ZERO_HIGHESTGRAD + 1
        KL_DIV_ZERO_2NDHIGHESTGRAD = DEC_FLIP_ZERO_HIGHESTGRAD + 1
        JS_DIV_ZERO_2NDHIGHESTGRAD = KL_DIV_ZERO_2NDHIGHESTGRAD + 1
        DEC_FLIP_ZERO_2NDHIGHESTGRAD = JS_DIV_ZERO_2NDHIGHESTGRAD + 1
        KL_DIV_ZERO_HIGHESTGRADMULT = DEC_FLIP_ZERO_2NDHIGHESTGRAD + 1
        JS_DIV_ZERO_HIGHESTGRADMULT = KL_DIV_ZERO_HIGHESTGRADMULT + 1
        DEC_FLIP_ZERO_HIGHESTGRADMULT = JS_DIV_ZERO_HIGHESTGRADMULT + 1
        KL_DIV_ZERO_2NDHIGHESTGRADMULT = DEC_FLIP_ZERO_HIGHESTGRADMULT + 1
        JS_DIV_ZERO_2NDHIGHESTGRADMULT = KL_DIV_ZERO_2NDHIGHESTGRADMULT + 1
        DEC_FLIP_ZERO_2NDHIGHESTGRADMULT = JS_DIV_ZERO_2NDHIGHESTGRADMULT + 1

        NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE = NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE + DEC_FLIP_ZERO_2NDHIGHESTGRADMULT
        NONTOP_RAND_ZEROED_WEIGHT = NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE + 1
        NONTOP_RAND_KL_DIV = NONTOP_RAND_ZEROED_WEIGHT + 1
        NONTOP_RAND_JS_DIV = NONTOP_RAND_KL_DIV + 1

        ATTN_KL_DIV_FROM_UNIF = ATTN_KL_DIV_FROM_UNIF + NONTOP_RAND_JS_DIV
        ATTN_JS_DIV_FROM_UNIF = ATTN_KL_DIV_FROM_UNIF + 1

    print("Found " + str(LAST_IND_OF_OUTPUT_CLASSES - STARTING_IND_OF_OUTPUT_CLASSES + 1) + " different output classes")
    print("Starting to concatenate data into one table")
    data_table = np.concatenate([first_v_second, dec_flip_stats, rand_stats, unchanged, grad_stats, nontop_stats,
                                 attn_div_stats],
                                axis=1)
    assert len(np.nonzero(data_table[:, ATTN_SEQ_LEN] - data_table[:, ATTN_SEQ_LEN_DUPLICATE_FOR_TESTING])[0]) == 0
    assert len(np.nonzero(data_table[:, ATTN_SEQ_LEN] - data_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING])[0]) == 0
    assert len(np.nonzero(data_table[:, ID] - data_table[:, ID_DUPLICATE])[0]) == 0
    assert not np.any(data_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] > 1)
    assert not np.any(data_table[:, NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP] > 1)
    assert not np.any(data_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] > 1)
    assert not np.any(data_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT] > 1)
    assert data_table.shape[1] - 1 == ATTN_JS_DIV_FROM_UNIF
    print()
    return data_table


def get_test_accuracy(table):
    return np.sum(table[:, ACTUAL_LABEL] == table[:, ORIG_LABEL_GUESSED]) / table.shape[0]


def run_mcnemars_test(lower_list, higher_list, tag):
    chi_square_stat, p_val = mcnemar(lower_list, y=higher_list)
    print("McNemar's results for " + tag + ": chi-square stat = " + str(chi_square_stat) + ", p-val = " + str(p_val))
    return chi_square_stat, p_val


def report_frac_for_model(which, table):
    if which == 'from_top':
        vals = table[np.logical_and(table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] != -1,
                                    table[:, ATTN_SEQ_LEN] > 1)][:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP]
        print("Removing from top:                             ", end='')
    elif which == 'from_bottom':
        vals = table[np.logical_and(table[:, NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP] != -1,
                                    table[:, ATTN_SEQ_LEN] > 1)][:, NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP]
        print("Removing from bottom:                          ", end='')
    elif which == 'avg_random':
        vals = table[np.logical_and(table[:, NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP] != -1,
                                    table[:, ATTN_SEQ_LEN] > 1)][:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START:
                                                                    NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END + 1]
        vals = np.reshape(vals, vals.shape[0] * vals.shape[1])
        print("Removing in a random order:                    ", end='')
    elif which == 'from_top_grad':
        vals = table[np.logical_and(table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] != -1,
                                    table[:, ATTN_SEQ_LEN] > 1)][:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD]
        print("Removing from top **GRADIENTS**:               ", end='')
    elif which == 'from_top_grad_mult':
        vals = table[np.logical_and(table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT] != -1,
                                    table[:, ATTN_SEQ_LEN] > 1)][:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADMULT]
        print("Removing from top **GRADIENTS * ATTNWEIGHTS**: ", end='')
    elif which == 'from_top_probmass':
        vals = table[np.logical_and(table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP] != -1,
                                    table[:, ATTN_SEQ_LEN] > 1)][:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP]
        print("Removing from top, av probmass removed:        ", end='')
    m = np.mean(vals)
    sd = np.std(vals)
    print("Needed to remove mean of " + str(m) + " (std dev " + str(sd) + ")")
    return m, sd


def print_label_distrib_for_rows(rows, tag):
    actual_labels = rows[:, ACTUAL_LABEL]
    guessed_labels = np.argmax(rows[:, STARTING_IND_OF_OUTPUT_CLASSES: LAST_IND_OF_OUTPUT_CLASSES + 1], axis=1)
    actual_label_summary = []
    guessed_label_summary = []
    for possible_label in range(LAST_IND_OF_OUTPUT_CLASSES - STARTING_IND_OF_OUTPUT_CLASSES + 1):
        num_of_that_actual_label = np.sum(actual_labels == possible_label)
        if num_of_that_actual_label > 0:
            actual_label_summary.append(str(possible_label) + ": " + str(num_of_that_actual_label))
        num_of_that_guessed_label = np.sum(guessed_labels == possible_label)
        if num_of_that_guessed_label > 0:
            guessed_label_summary.append(str(possible_label) + ": " + str(num_of_that_guessed_label))
    print("FOR " + str(tag) + " ROWS:")
    print("\tActual labels:\n\t", end='')
    print('   '.join(actual_label_summary))
    print("\tGuessed labels:\n\t", end='')
    print('   '.join(guessed_label_summary))


def get_default_class_info(table):
    print("NEVER-FLIPPED ANALYSIS:")
    default_class_rows_by_grad = table[table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] < 0]
    default_class_rows = table[table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] < 0]
    default_class_rows_from_bottom = table[table[:, NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP] < 0]
    default_class_rows_at_least_one_random = table[np.any(table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START:
                                                                   NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END + 1] < 0,
                                                          axis=1)]
    default_class_rows_for_random = table[table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START] < 0]
    every_never_change_instance = np.union1d(np.union1d(default_class_rows[:, 0],
                                                        default_class_rows_from_bottom[:, 0]),
                                             default_class_rows_at_least_one_random[:, 0])
    print("Removing by grad, there are " + str(default_class_rows_by_grad.shape[0]) + " instances in this category.")
    print("Removing from top, there are " + str(default_class_rows.shape[0]) + " instances in this category.")
    print("Removing in a (1) random order, there are " + str(default_class_rows_for_random.shape[0]) + " instances in this category.")
    print("Removing in a random order, there are " + str(default_class_rows_at_least_one_random.shape[0]) +
          " intances where at least one of the " +
          str(NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END - NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START + 1) +
          " random runs never flipped.")
    print("Removing from bottom, there are " + str(default_class_rows_from_bottom.shape[0]) +
          " instances in this category.")
    print("Found " + str(every_never_change_instance.shape[0]) + " instances in any of those categories.")
    from_top_not_in_from_bottom = np.setdiff1d(default_class_rows[:, 0], default_class_rows_from_bottom[:, 0]).shape[0]
    from_bottom_not_in_from_top = np.setdiff1d(default_class_rows_from_bottom[:, 0], default_class_rows[:, 0]).shape[0]
    print(str(from_top_not_in_from_bottom) + " of those from-top instances aren't also from-bottom instances.")
    print(str(from_bottom_not_in_from_top) + " of those from-bottom instances aren't also from-top instances.")
    from_rand_not_in_from_top = np.setdiff1d(default_class_rows_at_least_one_random[:, 0], default_class_rows[:, 0])
    from_rand_not_in_from_bottom = np.setdiff1d(default_class_rows_at_least_one_random[:, 0],
                                                default_class_rows_from_bottom[:, 0])
    print(str(from_rand_not_in_from_bottom.shape[0]) +
          " of those at-least-one-rand instances aren't also from-bottom instances.")
    print(str(from_rand_not_in_from_top.shape[0]) +
          " of those at-least-one-rand instances aren't also from-top instances.")

    output_log_dists = default_class_rows[:, STARTING_IND_OF_OUTPUT_CLASSES: LAST_IND_OF_OUTPUT_CLASSES + 1]
    if default_class_rows.shape[0] > 0:
        default_class = default_class_rows[0, ORIG_LABEL_GUESSED]
        print(str(default_class_rows.shape[0]) + ' / ' + str(np.sum(table[:, ORIG_LABEL_GUESSED] == default_class)) + ' (' +
              str(default_class_rows.shape[0] / np.sum(table[:, ORIG_LABEL_GUESSED] == default_class)) +
              '%) of instances originally guessed as label ' + str(default_class) + " never flipped from top")
    else:
        print("There were NO default class rows.")
    def_output_entropies = get_entropy_of_dists(output_log_dists,
                                                [LAST_IND_OF_OUTPUT_CLASSES - STARTING_IND_OF_OUTPUT_CLASSES + 1] *
                                                 output_log_dists.shape[0])
    output_log_dists_from_bottom = default_class_rows_from_bottom[:,
                                   STARTING_IND_OF_OUTPUT_CLASSES: LAST_IND_OF_OUTPUT_CLASSES + 1]
    def_output_entropies_bot = get_entropy_of_dists(output_log_dists_from_bottom,
                                                    [LAST_IND_OF_OUTPUT_CLASSES - STARTING_IND_OF_OUTPUT_CLASSES + 1] *
                                                     output_log_dists_from_bottom.shape[0])
    print_label_distrib_for_rows(default_class_rows, tag="NEVER-FLIPPED-FROM-TOP")
    print_label_distrib_for_rows(default_class_rows_from_bottom, tag="NEVER-FLIPPED-FROM-BOTTOM")
    non_default_class_rows = table[table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] > 0]
    assert default_class_rows.shape[0] + non_default_class_rows.shape[0] == table.shape[0], "Shouldn't be any 0s"
    output_log_dists = non_default_class_rows[:, STARTING_IND_OF_OUTPUT_CLASSES: LAST_IND_OF_OUTPUT_CLASSES + 1]
    nondef_output_entropies = get_entropy_of_dists(output_log_dists,
                                                   [LAST_IND_OF_OUTPUT_CLASSES - STARTING_IND_OF_OUTPUT_CLASSES + 1] *
                                                    output_log_dists.shape[0])
    if len(def_output_entropies) > 0:
        m = np.mean(def_output_entropies)
        sd = np.std(def_output_entropies)
        print("Never-flipped from-top output distrib entropies have mean " + str(m) + " and std dev " + str(sd) + ".")
    if len(def_output_entropies_bot) > 0:
        m = np.mean(def_output_entropies_bot)
        sd = np.std(def_output_entropies_bot)
        print("Never-flipped from-bottom output distrib entropies have mean " + str(m) + " and std dev " +
              str(sd) + ".")
    if len(nondef_output_entropies) > 0:
        m = np.mean(nondef_output_entropies)
        sd = np.std(nondef_output_entropies)
        print("All other output distrib entropies have mean " + str(m) + " and std dev " + str(sd) + ".")
    print()


def make_and_save_hist(data, fname, title='', bin_size=None, have_left_bin_edge_at=None, num_bins=None,
                       make_log_scale=False):
    fig = plt.figure()
    assert not (bin_size is not None and num_bins is not None)
    if bin_size is not None:
        bins = [i for i in np.arange(min(data), max(data) + bin_size, bin_size)]
    else:
        if num_bins is None:
            num_bins = 20  # just assume this is fine
        bin_size = (max(data) - min(data)) * 1.01 // num_bins
        bins = [i for i in np.arange(min(data), max(data) + bin_size, bin_size)]
    if have_left_bin_edge_at is not None:
        # find the closest bin edge, and adjust so it's at have_left_bin_edge_at
        if len(bins) > 1:
            closest = None
            for left_bin_edge in bins[:-1]:
                if closest is None:
                    closest = left_bin_edge
                elif fabs(left_bin_edge - have_left_bin_edge_at) < fabs(closest - have_left_bin_edge_at):
                    closest = left_bin_edge
            val_to_add = have_left_bin_edge_at - closest
            bins = [i + val_to_add for i in bins]
        else:
            if bins[0] < have_left_bin_edge_at:
                while have_left_bin_edge_at > bins[0]:
                    have_left_bin_edge_at -= bin_size
                bins[0] = have_left_bin_edge_at
            elif bins[0] >= have_left_bin_edge_at:
                while have_left_bin_edge_at + bin_size <= bins[0]:
                    have_left_bin_edge_at += bin_size
                bins[0] = have_left_bin_edge_at
    plt.hist(data, bins=bins, log=make_log_scale)
    expand_window_by = (max(data) - min(data)) / 20
    plt.xlim(min(data) - expand_window_by, max(data) + expand_window_by)
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def make_and_save_plot(data_x, data_y, fname, title='', x_title='', y_title='', marker_size=2):
    if isinstance(data_x, list):
        len_x = len(data_x)
    else:
        assert len(data_x.shape) == 1
        len_x = data_x.shape[0]
    if isinstance(data_y, list):
        len_y = len(data_y)
    else:
        assert len(data_y.shape) == 1
        len_y = data_y.shape[0]
    assert len_x == len_y
    fig = plt.figure()
    plt.scatter(data_x, data_y, s=marker_size)
    expand_x_window_by = (max(data_x) - min(data_x)) / 20
    plt.xlim(min(data_x) - expand_x_window_by, max(data_x) + expand_x_window_by)
    expand_y_window_by = (max(data_y) - min(data_y)) / 20
    plt.ylim(min(data_y) - expand_y_window_by, max(data_y) + expand_y_window_by)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def get_np_arr_of_one_attn_weight_per_instance(ind, is_han, data_directory,
                                               ind_corresponds_to_weight_sorted_in_dec_order=True):
    weights = []
    if is_han:
        filename = data_directory + '_sentence_attention_attn_weights_by_instance.txt'
    else:
        filename = data_directory + '_word_attention_attn_weights_by_instance.txt'
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            line = line[line.index(':') + 1:].strip().split(' ')
            log_unnormalized_attn_weights = [float(x) for x in line]
            min_val = min(log_unnormalized_attn_weights)
            log_unnormalized_attn_weights = [x - min_val for x in log_unnormalized_attn_weights]
            unnormalized_attn_weights = np.exp(log_unnormalized_attn_weights)
            attn_weights = list(unnormalized_attn_weights / np.sum(unnormalized_attn_weights))
            if ind_corresponds_to_weight_sorted_in_dec_order:
                attn_weights = sorted(attn_weights, reverse=True)
            try:
                weight_to_append = attn_weights[ind]
            except IndexError:
                weight_to_append = -1
            weights.append(weight_to_append)
    return np.array(weights)


def make_kljs_correlation_plot(comparing_to_remhighest, col_to_compare_to_divergence, col_to_compare_to_corr_weight,
                               tag_for_comparison: str, table, is_han, use_js: bool=True):
    if comparing_to_remhighest:
        if use_js:
            div_after_extreme_removed = table[:, JS_DIV_ZERO_HIGHEST]
        elif not use_js:
            div_after_extreme_removed = table[:, KL_DIV_ZERO_HIGHEST]
        corr_weights = get_np_arr_of_one_attn_weight_per_instance(0, is_han, data_dir)
    elif not comparing_to_remhighest:
        if use_js:
            div_after_extreme_removed = table[:, JS_DIV_ZERO_LOWEST]
        elif not use_js:
            div_after_extreme_removed = table[:, KL_DIV_ZERO_LOWEST]
        corr_weights = get_np_arr_of_one_attn_weight_per_instance(-1, is_han, data_dir)
    filename = image_directory + ("js" if use_js else "kl") + "divdiff-" + \
               ("remhighest" if comparing_to_remhighest else "remlowest") + "-" + tag_for_comparison + \
               "_weightdiff-" + ("highest" if comparing_to_remhighest else "lowest") + "-" + tag_for_comparison + \
               ".png"
    if comparing_to_remhighest:
        x_title = "Highest weight - " + tag_for_comparison
        y_title = ("JS" if use_js else "KL") + " div w/o highest - " + ("JS" if use_js else "KL") + " div w/o " + \
                  tag_for_comparison
        title = "(" + x_title + ") vs (" + y_title + ")"
        make_and_save_plot(corr_weights - col_to_compare_to_corr_weight,
                           div_after_extreme_removed - col_to_compare_to_divergence,
                           filename, title, x_title=x_title, y_title=y_title)
    else:
        x_title = tag_for_comparison + " - lowest weight"
        y_title = ("JS" if use_js else "KL") + " div w/o " + tag_for_comparison + " - " + \
                  ("JS" if use_js else "KL") + " div w/o lowest"
        title = "(" + x_title + ") vs (" + y_title + ")"
        make_and_save_plot(col_to_compare_to_corr_weight - corr_weights,
                           col_to_compare_to_divergence - div_after_extreme_removed,
                           filename, title, x_title=x_title, y_title=y_title)


def zoomed_out_vs_random_tests(table, is_han):
    avg_rand = table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START: NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_END + 1]
    assert np.all(avg_rand != 0)
    avg_rand = np.clip(avg_rand, a_min=0, a_max=None)
    num_actual_dec_flips = np.sum(avg_rand > 0, axis=1)
    num_actual_dec_flips = np.clip(num_actual_dec_flips, a_min=1, a_max=None)
    avg_rand = np.divide(np.sum(avg_rand, axis=1), num_actual_dec_flips)
    avg_rand[avg_rand == 0] = -1

    rem_from_bottom = table[:, NEEDED_REM_BOTTOM_FRAC_X_FOR_DECFLIP]
    rem_from_top = table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP]

    from_bottom_mask = np.logical_and(rem_from_bottom != -1, avg_rand != -1)
    run_mcnemars_test(avg_rand[from_bottom_mask], rem_from_bottom[from_bottom_mask], tag="from-bottom vs avg-rand")
    from_top_mask = np.logical_and(rem_from_top != -1, avg_rand != -1)
    run_mcnemars_test(rem_from_top[from_top_mask], avg_rand[from_top_mask], tag="from-top vs avg-rand")

    avg_weight_sampled_for_extraction = np.sum(table[:, EXTRACTED_SINGLE_ATTN_WEIGHT_START:
                                                        EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1], axis=1) / \
                                        (EXTRACTED_SINGLE_ATTN_WEIGHT_END - EXTRACTED_SINGLE_ATTN_WEIGHT_START + 1)
    avg_random_js = np.sum(table[:, EXTRACTED_SINGLE_WEIGHT_JS_START: EXTRACTED_SINGLE_WEIGHT_JS_END + 1], axis=1) / \
                    (EXTRACTED_SINGLE_WEIGHT_JS_END - EXTRACTED_SINGLE_WEIGHT_JS_START + 1)

    """make_kljs_correlation_plot(True, avg_random_js, avg_weight_sampled_for_extraction, "(avg) random weight",
                               table, is_han)
    make_kljs_correlation_plot(False, avg_random_js, avg_weight_sampled_for_extraction, "(avg) random weight",
                               table, is_han)"""

    #print(np.concatenate([np.reshape(avg_rand[from_bottom_mask], (np.sum(from_bottom_mask), 1)),
    #               np.reshape(rem_from_bottom[from_bottom_mask], (np.sum(from_bottom_mask), 1))], axis=1)[:40])


def coarse_grained_single_attn_weight_tests(is_higher, table):
    if is_higher:
        extreme_flipped_the_decision = table[:, NEEDED_REM_TOP_X_FOR_DECFLIP]
    else:
        extreme_flipped_the_decision = table[:, NEEDED_REM_BOTTOM_X_FOR_DECFLIP]
    extreme_flipped_the_decision = (extreme_flipped_the_decision != -1)
    avg_rand_flipped_the_decision = (table[:, NEEDED_REM_RAND_X_FOR_DECFLIP_START:
                                              NEEDED_REM_RAND_X_FOR_DECFLIP_END + 1] == 1).astype('float')
    avg_rand_flipped_the_decision = np.sum(avg_rand_flipped_the_decision, axis=1) / \
                                    (NEEDED_REM_RAND_X_FOR_DECFLIP_END - NEEDED_REM_RAND_X_FOR_DECFLIP_START + 1)
    if not is_higher:
        run_mcnemars_test(extreme_flipped_the_decision, avg_rand_flipped_the_decision,
                          "lower has fewer decflips than rand")
    else:
        run_mcnemars_test(avg_rand_flipped_the_decision, extreme_flipped_the_decision,
                          "higher has more decflips than rand")


def fine_grained_single_attn_weight_tests(is_higher, table):
    if is_higher:
        high_js_div = table[:, JS_DIV_ZERO_HIGHEST]
        low_js_div = table[:, JS_DIV_ZERO_2NDHIGHEST]
        high_flip_outcomes = (table[:, DEC_FLIP_ZERO_HIGHEST] != -1)
        low_flip_outcomes = (table[:, DEC_FLIP_ZERO_2NDHIGHEST] != -1)
    else:
        high_js_div = table[:, JS_DIV_ZERO_2NDLOWEST]
        low_js_div = table[:, JS_DIV_ZERO_LOWEST]
        high_flip_outcomes = (table[:, DEC_FLIP_ZERO_2NDLOWEST] != -1)
        low_flip_outcomes = (table[:, DEC_FLIP_ZERO_LOWEST] != -1)
    run_mcnemars_test(low_js_div, high_js_div, ("rem-highest js div higher than rem-2ndhighest js div" if is_higher
                                                else "rem-lowest js div lower than rem-2ndlowest js div"))
    both_flipped = np.sum(np.logical_and(high_flip_outcomes == 1, low_flip_outcomes == 1))
    neither_flipped = np.sum(np.logical_and(high_flip_outcomes == 0, low_flip_outcomes == 0))
    only_higher_flipped = np.sum(np.logical_and(high_flip_outcomes == 1, low_flip_outcomes == 0))
    only_lower_flipped = np.sum(np.logical_and(high_flip_outcomes == 0, low_flip_outcomes == 1))
    print("BothFlipped: " + str(both_flipped) + "   NeitherFlipped: " + str(neither_flipped) +
          "   OnlyHigherFlipped: " + str(only_higher_flipped) + "   OnlyLowerFlipped: " + str(only_lower_flipped))


def test_needed_rem_lots_vs_not(table):
    rem_from_top = table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP]
    mean_needed_rem = np.mean(rem_from_top)
    needed_rem_lots = table[rem_from_top >= mean_needed_rem]
    needed_rem_little = table[rem_from_top < mean_needed_rem]

    lots_m = np.mean(needed_rem_lots[:, ATTN_ENTROPY])
    lots_sd = np.std(needed_rem_lots[:, ATTN_ENTROPY])
    print("For needed-to-remove-lots, mean attn entropy was " + str(lots_m) + " (std dev " + str(lots_sd) + ")")

    little_m = np.mean(needed_rem_little[:, ATTN_ENTROPY])
    little_sd = np.std(needed_rem_little[:, ATTN_ENTROPY])
    print("For needed-to-remove-little, mean attn entropy was " + str(little_m) + " (std dev " + str(little_sd) + ")")


def get_vsrand_2x2_decflip_jointdist(table, is_han):
    table = table[table[:, ATTN_SEQ_LEN] > 1]  # get rid of seqs of length 1 for this
    rand_flipped_decision = (table[:, NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE] != -1)
    top_flipped_decision = (table[:, DEC_FLIP_ZERO_HIGHEST] != -1)
    print_2x2_decflip_jointdist(rand_flipped_decision, top_flipped_decision, "Top vs. rand nontop weight", table)


def get_vs2nd_2x2_decflip_jointdist(table):
    table = table[table[:, ATTN_SEQ_LEN] > 1]  # get rid of seqs of length 1 for this
    second_flipped_decision = (table[:, DEC_FLIP_ZERO_2NDHIGHEST] != -1)
    top_flipped_decision = (table[:, DEC_FLIP_ZERO_HIGHEST] != -1)
    print_2x2_decflip_jointdist(second_flipped_decision, top_flipped_decision, "Top vs. 2nd-highest", table)


def print_2x2_decflip_jointdist(vs_flipped_decision, top_flipped_decision, label, table):
    both_flipped = np.sum(np.logical_and(vs_flipped_decision, top_flipped_decision))
    neither_flipped = np.sum(np.logical_and(np.logical_not(vs_flipped_decision),
                                            np.logical_not(top_flipped_decision)))
    only_top_flipped = np.sum(np.logical_and(np.logical_not(vs_flipped_decision),
                                             top_flipped_decision))
    only_rand_flipped = np.sum(np.logical_and(vs_flipped_decision,
                                              np.logical_not(top_flipped_decision)))
    assert both_flipped + neither_flipped + only_top_flipped + only_rand_flipped == table.shape[0]
    print(label + ":  bothflipped:" + str(both_flipped / table.shape[0]) +
          '\tneitherflipped:' + str(neither_flipped / table.shape[0]) + '\tonlytopflipped:' +
          str(only_top_flipped / table.shape[0]) + '\tonlyotherflipped:' + str(only_rand_flipped / table.shape[0]))
    
    
def write_grad_labels_to_file(rows_where_grad_more_efficient, model_folder_name, top_level_data_folder):
    if not top_level_data_folder.endswith('/'):
        top_level_data_folder += '/'
    output_file = top_level_data_folder
    just_the_model = model_folder_name[:-1]
    just_the_model = just_the_model[just_the_model.rfind('/') + 1:]
    just_the_model = just_the_model[just_the_model.index('-') + 1:]
    just_the_dataset_name = model_folder_name[:-1]
    if '/' in just_the_dataset_name:
        just_the_dataset_name = just_the_dataset_name[just_the_dataset_name.rfind('/') + 1:]
    if '-' in just_the_dataset_name:
        just_the_dataset_name = just_the_dataset_name[:just_the_dataset_name.index('-')]
        if just_the_dataset_name == '':
            dataset_name = 'dataset'
        else:
            dataset_name = just_the_dataset_name
    else:
        dataset_name = 'dataset'
    if just_the_model.endswith('-train'):
        output_file += dataset_name + '_train'
        just_the_model = just_the_model[:just_the_model.rfind('-')]
    elif just_the_model.endswith('-dev'):
        output_file += dataset_name + '_dev'
        just_the_model = just_the_model[:just_the_model.rfind('-')]
    else:
        output_file += dataset_name + '_test'
    output_file += '_attnperformancelabels_' + just_the_model + '.txt'
    with open(output_file, 'w') as f:
        for i in range(rows_where_grad_more_efficient.shape[0]):
            if rows_where_grad_more_efficient[i] == 0:
                f.write('0\n')
            elif rows_where_grad_more_efficient[i] == 1:
                f.write('1\n')
            else:
                print("ERROR: found value " + str(rows_where_grad_more_efficient[i]) + " in rows_where_grad_more_efficient")
    print("Successfully wrote attn performance labels to " + output_file)
    
    
def softmax(arr, axis):
    arr = arr - np.expand_dims(np.max(arr, axis=axis), axis)
    arr = np.exp(arr)
    ax_sum = np.expand_dims(np.sum(arr, axis=axis), axis)
    p = arr / ax_sum
    return p
    
    
def compare_outputs_by_grad(table, model_folder_name):
    table = table[table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] > 0]
    table = table[table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] > 0]
    table = table[table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] < table[:, ATTN_SEQ_LEN]]
    table = table[table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] < table[:, ATTN_SEQ_LEN]]

    grad_more_efficient = table[table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] < table[:, NEEDED_REM_TOP_X_FOR_DECFLIP]]
    gme_presoftmax_outputs = grad_more_efficient[:, STARTING_IND_OF_OUTPUT_CLASSES:LAST_IND_OF_OUTPUT_CLASSES + 1]
    assert gme_presoftmax_outputs.shape[1] == 2
    should_not_all_be_1 = np.sum(gme_presoftmax_outputs, axis=1)
    all_really_close_to_1 = True
    for i in range(should_not_all_be_1.shape[0]):
        if should_not_all_be_1[i] < .98 or should_not_all_be_1[i] > 1.02:
            all_really_close_to_1 = False
            break
    assert not all_really_close_to_1
    gme_outputs = softmax(gme_presoftmax_outputs, axis=1)
    gme_output_diffs = np.absolute(gme_outputs[:, 1] - gme_outputs[:, 0])

    grad_less_efficient = table[table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] > table[:, NEEDED_REM_TOP_X_FOR_DECFLIP]]
    gle_presoftmax_outputs = grad_less_efficient[:, STARTING_IND_OF_OUTPUT_CLASSES:LAST_IND_OF_OUTPUT_CLASSES + 1]
    assert gle_presoftmax_outputs.shape[1] == 2
    should_not_all_be_1 = np.sum(gle_presoftmax_outputs, axis=1)
    all_really_close_to_1 = True
    for i in range(should_not_all_be_1.shape[0]):
        if should_not_all_be_1[i] < .98 or should_not_all_be_1[i] > 1.02:
            all_really_close_to_1 = False
            break
    assert not all_really_close_to_1
    gle_outputs = softmax(gle_presoftmax_outputs, axis=1)
    gle_output_diffs = np.absolute(gle_outputs[:, 1] - gle_outputs[:, 0])

    if not model_folder_name.endswith('/'):
        model_folder_name += '/'
    gme_filename = model_folder_name + "grad_more_efficient_outputclass_absvaldiffs.txt"
    gle_filename = model_folder_name + "grad_less_efficient_outputclass_absvaldiffs.txt"
    with open(gme_filename, 'w') as f:
        for i in range(gme_output_diffs.shape[0]):
            f.write(str(float(gme_output_diffs[i])) + '\n')
    with open(gle_filename, 'w') as f:
        for i in range(gle_output_diffs.shape[0]):
            f.write(str(float(gle_output_diffs[i])) + '\n')
    
    
def compare_outputs_by_attndecflip(table, model_folder_name):
    never_changed_mask = np.logical_or(table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] == -1,
                                       table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] == table[:, ATTN_SEQ_LEN])

    never_changed = table[never_changed_mask]
    nc_presoftmax_outputs = never_changed[:, STARTING_IND_OF_OUTPUT_CLASSES:LAST_IND_OF_OUTPUT_CLASSES + 1]
    assert nc_presoftmax_outputs.shape[1] == 2
    should_not_all_be_1 = np.sum(nc_presoftmax_outputs, axis=1)
    all_really_close_to_1 = True
    for i in range(should_not_all_be_1.shape[0]):
        if should_not_all_be_1[i] < .98 or should_not_all_be_1[i] > 1.02:
            all_really_close_to_1 = False
            break
    assert not all_really_close_to_1
    nc_outputs = softmax(nc_presoftmax_outputs, axis=1)
    nc_output_diffs = np.absolute(nc_outputs[:, 1] - nc_outputs[:, 0])

    changed = table[np.logical_not(never_changed_mask)]
    dc_presoftmax_outputs = changed[:, STARTING_IND_OF_OUTPUT_CLASSES:LAST_IND_OF_OUTPUT_CLASSES + 1]
    assert dc_presoftmax_outputs.shape[1] == 2
    should_not_all_be_1 = np.sum(dc_presoftmax_outputs, axis=1)
    all_really_close_to_1 = True
    for i in range(should_not_all_be_1.shape[0]):
        if should_not_all_be_1[i] < .98 or should_not_all_be_1[i] > 1.02:
            all_really_close_to_1 = False
            break
    assert not all_really_close_to_1
    dc_outputs = softmax(dc_presoftmax_outputs, axis=1)
    dc_output_diffs = np.absolute(dc_outputs[:, 1] - dc_outputs[:, 0])

    if not model_folder_name.endswith('/'):
        model_folder_name += '/'
    nc_filename = model_folder_name + "neverchangedbyattn_outputclass_absvaldiffs.txt"
    dc_filename = model_folder_name + "changedbyattn_outputclass_absvaldiffs.txt"
    with open(nc_filename, 'w') as f:
        for i in range(nc_output_diffs.shape[0]):
            f.write(str(float(nc_output_diffs[i])) + '\n')
    with open(dc_filename, 'w') as f:
        for i in range(dc_output_diffs.shape[0]):
            f.write(str(float(dc_output_diffs[i])) + '\n')


def main(constrain_to_guessed_label=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-folder-name", type=str, required=True,
                        help="The local name of the directories associated with the model")
    parser.add_argument("--write-attnperf-labels", type=str, required=False, default="True")
    parser.add_argument("--base-output-dir", type=str, required=False,
                        default='/homes/gws/sofias6/attn-test-output/',
                        help="The name of the top-level output directory containing other output directories")
    parser.add_argument("--base-images-dir", type=str, required=False,
                        default='imgs/',
                        help="The directory in which to store any created plots or histograms")
    parser.add_argument("--top-level-data-dir", type=str, required=False,
                        default='/homes/gws/sofias6/data/',
                        help='Top level dir containing data (some info will be written there)')
    args = parser.parse_args()
    if args.write_attnperf_labels.lower().startswith('t'):
        write_attnperf = True
    else:
        write_attnperf = False
    base_output_dir = args.base_output_dir
    if not base_output_dir.endswith('/'):
        base_output_dir += '/'
    model_folder_name = args.model_folder_name
    if '-han' in model_folder_name:
        is_han = True
    else:
        is_han = False
    if not model_folder_name.endswith('/'):
        model_folder_name += '/'
    image_dir = args.base_images_dir
    if not image_dir.endswith('/'):
        image_dir += '/'
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    global image_directory, dataset_output_directory
    image_directory = image_dir + model_folder_name
    if not os.path.isdir(image_directory):
        os.makedirs(image_directory)
    dataset_name = model_folder_name[:model_folder_name.index('-')]
    dataset_output_directory = image_dir + dataset_name + '/'
    if not os.path.isdir(dataset_output_directory):
        os.makedirs(dataset_output_directory)
    file_dir = base_output_dir + model_folder_name
    global first_v_second_fname, dec_flip_stats_fname, rand_results_fname, unchanged_fname, data_dir, \
        grad_based_stats_fname, dec_flip_rand_nontop_stats_fname, attn_div_from_unif_fname
    data_dir = file_dir
    first_v_second_fname = file_dir + first_v_second_fname
    dec_flip_stats_fname = file_dir + dec_flip_stats_fname
    rand_results_fname = file_dir + rand_results_fname
    unchanged_fname = file_dir + unchanged_fname
    grad_based_stats_fname = file_dir + grad_based_stats_fname
    dec_flip_rand_nontop_stats_fname = file_dir + dec_flip_rand_nontop_stats_fname
    attn_div_from_unif_fname = file_dir + attn_div_from_unif_fname
    table = load_in_data_table(first_v_second_fname, dec_flip_stats_fname, rand_results_fname, unchanged_fname,
                               grad_based_stats_fname, dec_flip_rand_nontop_stats_fname, attn_div_from_unif_fname)
    if constrain_to_guessed_label is not None:
        table = table[table[:, ORIG_LABEL_GUESSED] == constrain_to_guessed_label]
        
    compare_outputs_by_grad(table, file_dir)
    compare_outputs_by_attndecflip(table, file_dir)

    # make sure that test results didn't get garbled-- do a couple of quick tests
    if table.shape[0] > 10:
        assert int(np.sum(table[:, DEC_FLIP_ZERO_2NDHIGHESTGRADMULT] != table[:, DEC_FLIP_ZERO_2NDHIGHEST])) > 0
        assert int(np.sum(table[:, DEC_FLIP_ZERO_2NDHIGHESTGRAD] != table[:, DEC_FLIP_ZERO_2NDHIGHEST])) > 0
        assert int(np.sum(table[:, DEC_FLIP_ZERO_2NDHIGHESTGRADMULT] != table[:, DEC_FLIP_ZERO_2NDHIGHESTGRAD])) > 0

    print("Test set accuracy: " + str(get_test_accuracy(table)))
    print_label_distrib_for_rows(table, "ALL")
    print()
    report_frac_for_model('from_top_grad_mult', table)
    report_frac_for_model('from_top_grad', table)
    report_frac_for_model('from_top', table)
    report_frac_for_model('avg_random', table)
    report_frac_for_model('from_bottom', table)
    report_frac_for_model('from_top_probmass', table)

    table_of_seqs_longer_than_1_singleneg1 = table[np.logical_and(np.logical_or(table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] == -1,
                                                                                table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] == -1),
                                                                  table[:, ATTN_SEQ_LEN] > 1)]
    table_of_seqs_longer_than_1_singleneg1 = table_of_seqs_longer_than_1_singleneg1[np.logical_not(np.logical_and(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP] == -1,
                                                                                                                  table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] == -1))]
    table_of_seqs_longer_than_1_no_neg1s = table[np.logical_and(np.logical_and(table[:, ATTN_SEQ_LEN] > 1,
                                                                               table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] != -1),
                                                                table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] != -1)]
    total_num_seqs_longer_than_1 = table_of_seqs_longer_than_1_singleneg1.shape[0] + \
                                   table_of_seqs_longer_than_1_no_neg1s.shape[0]
    attn_more_efficient = int(np.sum(table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_X_FOR_DECFLIP] <
                                     table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD])) + \
                          int(np.sum(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP] >
                                     table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD]))  # we achieved a decflip with attn, not grad
    assert (int(np.sum(table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] <
                       table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD])) +
            int(np.sum(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] >
                       table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD])) ==
            attn_more_efficient)
    """
    grad_more_efficient = int(np.sum(table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_X_FOR_DECFLIP] >
                                     table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD])) + \
                          int(np.sum(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP] <
                                     table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD]))
    """
    mask_for_seqs_longer_than_1_no_neg1s = (np.logical_and(np.logical_and(table[:, ATTN_SEQ_LEN] > 1,
                                                                          table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] != -1),
                                                           table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] != -1))
    mask_for_seqs_longer_than_1_singleneg1 = (np.logical_and(np.logical_xor(table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] == -1,
                                                                            table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD] == -1),
                                                             table[:, ATTN_SEQ_LEN] > 1))
    rows_where_grad_more_efficient = np.logical_or(np.logical_and(mask_for_seqs_longer_than_1_no_neg1s,
                                                                  table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] >
                                                                  table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD]),
                                                   np.logical_and(mask_for_seqs_longer_than_1_singleneg1,
                                                                  table[:, NEEDED_REM_TOP_X_FOR_DECFLIP] <
                                                                  table[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD]))
    grad_more_efficient = np.sum(rows_where_grad_more_efficient)
    assert (int(np.sum(table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] >
                       table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD])) +
            int(np.sum(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] <
                       table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD])) ==
            grad_more_efficient)
    assert rows_where_grad_more_efficient.shape[0] == table.shape[0]

    if write_attnperf:
        write_grad_labels_to_file(rows_where_grad_more_efficient, model_folder_name, args.top_level_data_dir)
        
    both_same = int(np.sum(table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_X_FOR_DECFLIP] ==
                           table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD])) + \
                int(np.sum(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP] ==
                           table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD]))
    assert (int(np.sum(table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] ==
                       table_of_seqs_longer_than_1_no_neg1s[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD])) +
            int(np.sum(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] ==
                       table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD])) ==
            both_same)
    assert attn_more_efficient + grad_more_efficient + both_same == total_num_seqs_longer_than_1
    assert np.all(table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP] !=
                  table_of_seqs_longer_than_1_singleneg1[:, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD])
    print("Found " + str(table_of_seqs_longer_than_1_singleneg1.shape[0]) +
          " cases where exactly one of either attn or grad never flipped the decision; leaving those in for analysis.")
    print("REMOVING FROM ANALYSIS the " +
          str(np.sum(table[:, ATTN_SEQ_LEN] == 1)) + " cases with an attention length of 1")
    temp_table = table[table[:, ATTN_SEQ_LEN] > 1]
    print("REMOVING FROM ANALYSIS the " +
          str(np.sum(np.logical_and(temp_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] == -1,
                                    temp_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] == -1))) +
          " OTHER cases where neither attn nor grad EVER flipped the decision")
    print("Cases where rem_by_highest_attn more efficient than rem_by_highest_grad: ", end='')
    print(str(attn_more_efficient) + "\t (" + str(attn_more_efficient/total_num_seqs_longer_than_1) +
          " of cases with attn len > 1)")
    print("Cases where rem_by_highest_attn less efficient than rem_by_highest_grad: ", end='')
    print(str(grad_more_efficient) + "\t (" + str(grad_more_efficient / total_num_seqs_longer_than_1) +
          " of cases with attn len > 1)")
    print("Cases where both equally efficient: ", end='')
    print(str(both_same) + "\t (" + str( both_same / total_num_seqs_longer_than_1) + " of cases with attn len > 1)")
    get_vsrand_2x2_decflip_jointdist(table, is_han)
    get_vs2nd_2x2_decflip_jointdist(table)
    print()
    get_default_class_info(table)
    zoomed_out_vs_random_tests(table, is_han)

    make_and_save_hist(table[:, ATTN_SEQ_LEN], dataset_output_directory + "num_sents_test_docs.png",
                       title="Number of sentences in test docs",
                       have_left_bin_edge_at=0, bin_size=1, make_log_scale=True)
    coarse_grained_single_attn_weight_tests(True, table)
    fine_grained_single_attn_weight_tests(True, table)
    print()
    coarse_grained_single_attn_weight_tests(False, table)
    fine_grained_single_attn_weight_tests(False, table)
    print()
    test_needed_rem_lots_vs_not(table)

    print("needed to remove:   " + str(table[90:100, NEEDED_REM_TOP_X_FOR_DECFLIP]) + " (removing from highest)")
    print("needed to remove:   " + str(table[90:100, NEEDED_REM_TOP_X_FOR_DECFLIP_GRAD]) +
          " (removing from highest grad)")
    print("sequence lens:      " + str(table[90:100, ATTN_SEQ_LEN]))
    print("original decisions: " + str(table[90:100, ORIG_LABEL_GUESSED]))


def test_js_divs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-folder-name", type=str, required=True,
                        help="The local name of the directories associated with the model")
    parser.add_argument("--base-output-dir", type=str, required=False,
                        default='/homes/gws/sofias6/attn-test-output/',
                        help="The name of the top-level output directory containing other output directories")
    parser.add_argument("--base-images-dir", type=str, required=False,
                        default='imgs/',
                        help="The directory in which to store any created plots or histograms")
    args = parser.parse_args()
    base_output_dir = args.base_output_dir
    if not base_output_dir.endswith('/'):
        base_output_dir += '/'
    model_folder_name = args.model_folder_name
    if '-han' in model_folder_name:
        is_han = True
    else:
        is_han = False
    if not model_folder_name.endswith('/'):
        model_folder_name += '/'
    image_dir = args.base_images_dir
    if not image_dir.endswith('/'):
        image_dir += '/'
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    global image_directory, dataset_output_directory
    image_directory = image_dir + model_folder_name
    if not os.path.isdir(image_directory):
        os.makedirs(image_directory)
    dataset_name = model_folder_name[:model_folder_name.index('-')]
    dataset_output_directory = image_dir + dataset_name + '/'
    if not os.path.isdir(dataset_output_directory):
        os.makedirs(dataset_output_directory)
    file_dir = base_output_dir + model_folder_name
    global first_v_second_fname, dec_flip_stats_fname, rand_results_fname, unchanged_fname, data_dir, \
        grad_based_stats_fname, dec_flip_rand_nontop_stats_fname, attn_div_from_unif_fname
    data_dir = file_dir
    first_v_second_fname = file_dir + first_v_second_fname
    dec_flip_stats_fname = file_dir + dec_flip_stats_fname
    rand_results_fname = file_dir + rand_results_fname
    unchanged_fname = file_dir + unchanged_fname
    grad_based_stats_fname = file_dir + grad_based_stats_fname
    dec_flip_rand_nontop_stats_fname = file_dir + dec_flip_rand_nontop_stats_fname
    attn_div_from_unif_fname = file_dir + attn_div_from_unif_fname
    table = load_in_data_table(first_v_second_fname, dec_flip_stats_fname, rand_results_fname, unchanged_fname,
                               grad_based_stats_fname, dec_flip_rand_nontop_stats_fname, attn_div_from_unif_fname)

    # make sure that test results didn't get garbled-- do a couple of quick tests
    if table.shape[0] > 10:
        assert int(np.sum(table[:, DEC_FLIP_ZERO_2NDHIGHESTGRADMULT] != table[:, DEC_FLIP_ZERO_2NDHIGHEST])) > 0
        assert int(np.sum(table[:, DEC_FLIP_ZERO_2NDHIGHESTGRAD] != table[:, DEC_FLIP_ZERO_2NDHIGHEST])) > 0
        assert int(np.sum(table[:, DEC_FLIP_ZERO_2NDHIGHESTGRADMULT] != table[:, DEC_FLIP_ZERO_2NDHIGHESTGRAD])) > 0

    mask = (table[:, ATTN_SEQ_LEN] > 1)
    table = table[mask]
    max_weights = get_np_arr_of_one_attn_weight_per_instance(0, is_han, data_dir,
                                               ind_corresponds_to_weight_sorted_in_dec_order=True)[mask]
    high_js_divdiff_mask = ((table[:, JS_DIV_ZERO_HIGHEST] - table[:, NONTOP_RAND_JS_DIV]) > 0.1)
    high_js_divdiff_rows = table[high_js_divdiff_mask]
    max_weights = max_weights[high_js_divdiff_mask]
    print(np.sum((max_weights - high_js_divdiff_rows[:, NONTOP_RAND_ZEROED_WEIGHT]) < 0.05))


if __name__ == '__main__':
    main(constrain_to_guessed_label=None)
