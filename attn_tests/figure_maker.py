import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import seaborn as sns
import pandas as pd
from test_model import first_v_second_fname
from test_model import dec_flip_stats_fname
from test_model import rand_results_fname  # this one also has variable-length fields now *shrug*
from test_model import unchanged_fname  # index this one last because its fields are of variable length
from test_model import grad_based_stats_fname
from test_model import dec_flip_rand_nontop_stats_fname
from process_test_outputs import load_in_data_table, get_np_arr_of_one_attn_weight_per_instance
from math import ceil
from random import random

base_output_dir = '/homes/gws/sofias6/attn-test-output/'

def get_filenames_for_subdir(mid_dir):
    global base_output_dir
    if not base_output_dir.endswith('/'):
        base_output_dir += '/'
    if not mid_dir.endswith('/'):
        mid_dir += '/'
    return base_output_dir + mid_dir + first_v_second_fname, base_output_dir + mid_dir + dec_flip_stats_fname, \
           base_output_dir + mid_dir + rand_results_fname, base_output_dir + mid_dir + unchanged_fname, \
           base_output_dir + mid_dir + grad_based_stats_fname, base_output_dir + mid_dir + dec_flip_rand_nontop_stats_fname

yahoo_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-hanrnn-postattnfix'))
imdb_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('imdb-hanrnn-postattnfix'))
amazon_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('amazon-hanrnn-postattnfix'))
yelp_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('yelp-hanrnn-postattnfix-2'))
yahoo_hanconv_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-hanconv-convfix'))
imdb_hanconv_table = load_in_data_table(*get_filenames_for_subdir('imdb-hanconv-convfix'))
amazon_hanconv_table = load_in_data_table(*get_filenames_for_subdir('amazon-hanconv-convfix'))
yelp_hanconv_table = load_in_data_table(*get_filenames_for_subdir('yelp-hanconv-convfix'))
yahoo_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-flanrnn'))
imdb_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('imdb-flanrnn'))
amazon_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('amazon-flanrnn'))
yelp_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('yelp-flanrnn-moreword2vec'))
yahoo_flanconv_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-flanconv-convfix'))
imdb_flanconv_table = load_in_data_table(*get_filenames_for_subdir('imdb-flanconv-convfix'))
amazon_flanconv_table = load_in_data_table(*get_filenames_for_subdir('amazon-flanconv-convfix'))
yelp_flanconv_table = load_in_data_table(*get_filenames_for_subdir('yelp-flanconv-convfix'))

from process_test_outputs import EXTRACTED_SINGLE_ATTN_WEIGHT_END, EXTRACTED_SINGLE_WEIGHT_KL_START, EXTRACTED_SINGLE_WEIGHT_KL_END, \
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
        DEC_FLIP_ZERO_2NDHIGHESTGRADMULT, EXTRACTED_SINGLE_ATTN_WEIGHT_START, DEC_FLIP_ZERO_HIGHEST, \
        JS_DIV_ZERO_HIGHEST, DEC_FLIP_ZERO_2NDHIGHEST, JS_DIV_ZERO_2NDHIGHEST, NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE, \
        NONTOP_RAND_ZEROED_WEIGHT, NONTOP_RAND_KL_DIV, NONTOP_RAND_JS_DIV, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP, \
        NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP

yahoo_han_mask = yahoo_hanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
imdb_han_mask = imdb_hanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
amazon_han_mask = amazon_hanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
yelp_han_mask = yelp_hanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
yahoo_flan_mask = yahoo_flanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
imdb_flan_mask = imdb_flanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
amazon_flan_mask = amazon_flanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1
yelp_flan_mask = yelp_flanrnn_table[:, ATTN_SEQ_LEN_DUPLICATE2_FOR_TESTING] > 1

yahoo_hanrnn_table = yahoo_hanrnn_table[yahoo_han_mask]
imdb_hanrnn_table = imdb_hanrnn_table[imdb_han_mask]
amazon_hanrnn_table = amazon_hanrnn_table[amazon_han_mask]
yelp_hanrnn_table = yelp_hanrnn_table[yelp_han_mask]
yahoo_hanconv_table = yahoo_hanconv_table[yahoo_han_mask]
imdb_hanconv_table = imdb_hanconv_table[imdb_han_mask]
amazon_hanconv_table = amazon_hanconv_table[amazon_han_mask]
yelp_hanconv_table = yelp_hanconv_table[yelp_han_mask]
yahoo_flanrnn_table = yahoo_flanrnn_table[yahoo_flan_mask]
imdb_flanrnn_table = imdb_flanrnn_table[imdb_flan_mask]
amazon_flanrnn_table = amazon_flanrnn_table[amazon_flan_mask]
yelp_flanrnn_table = yelp_flanrnn_table[yelp_flan_mask]
yahoo_flanconv_table = yahoo_flanconv_table[yahoo_flan_mask]
imdb_flanconv_table = imdb_flanconv_table[imdb_flan_mask]
amazon_flanconv_table = amazon_flanconv_table[amazon_flan_mask]
yelp_flanconv_table = yelp_flanconv_table[yelp_flan_mask]

assert LAST_IND_OF_OUTPUT_CLASSES is not None




try:
    sns.set()
except:
    pass


def make_2x2_2boxplot_set(list1_of_two_vallists_to_boxplot, list2_of_two_vallists_to_boxplot,
                          list3_of_two_vallists_to_boxplot, list4_of_two_vallists_to_boxplot, list_of_colorlabels,
                          list_of_two_color_tuples, labels_for_4_boxplot_sets):
    pass


def make_4_4boxplot_set(list1_of_four_vallists_to_boxplot, list2_of_four_vallists_to_boxplot,
                        list3_of_four_vallists_to_boxplot, list4_of_four_vallists_to_boxplot, list_of_colorlabels,
                        list_of_four_color_tuples, labels_for_4_boxplot_sets):
    pass


def make_kdeplot(tuples_of_title_x_y, filename):
    subplot_title = 'Dataset'
    x_title = "Difference in Attention Weight Magnitudes"
    y_title = "Log Difference in Corresponding JS Divergences\nfrom Original Output"
    list_of_row_dicts = []
    assert len(tuples_of_title_x_y[1]) == len(tuples_of_title_x_y[2])
    for j in range(len(tuples_of_title_x_y)):
        dataset_tup = tuples_of_title_x_y[j]
        assert len(dataset_tup[1]) == len(dataset_tup[2])
        for i in range(len(dataset_tup[1])):
            list_of_row_dicts.append({subplot_title:dataset_tup[0], x_title:dataset_tup[1][i],
                                      y_title:dataset_tup[2][i]})
    data_to_plot = pd.DataFrame(list_of_row_dicts)
    fig = plt.figure()
    g = sns.FacetGrid(data_to_plot, col=subplot_title, hue=None, col_wrap=2)
    g.map(sns.kdeplot, x_title, y_title, cmap = "Blues", shade = True, shade_lowest = False)
    g.set_titles("{col_name}")
    #ax = sns.kdeplot(getattr(all_rows, x_title), getattr(all_rows, y_title),
    #                 cmap = "Blues", shade = True, shade_lowest = False,
    #                 )
    #ax.set_title("Differences in Attention weight ")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def make_2x2_vsrand_decflip_violinplot(tuples_of_title_x_y_noneflipped, filename):
    x_label = "Difference in (average) zeroed attention\nweight magnitude"
    y_label = "Difference in decision flip indicators\ncompared to original output"
    title_of_each_graph = "Dataset"
    list_of_row_dicts = []
    for tup in tuples_of_title_x_y_noneflipped:
        dataset_title = tup[0]
        assert len(tup[1]) == len(tup[2])
        assert len(tup[1]) == tup[3].shape[0]
        for i in range(len(tup[2])):
            list_of_row_dicts.append({title_of_each_graph: dataset_title, x_label: tup[1][i], y_label: str(tup[2][i]),
                                      "No Flips Occurred": ("True" if tup[3][i] else "False")})
    list_of_row_dicts = sorted(list_of_row_dicts, key=(lambda x: 1 if x["No Flips Occurred"] == "True" else 0))
    data_to_plot = pd.DataFrame(list_of_row_dicts)
    g = sns.catplot(kind='violin', x=x_label, y=y_label, col=title_of_each_graph, hue="No Flips Occurred", palette=None,
                       data=data_to_plot, col_wrap=2, legend=True, split=True, inner=None, orient="h")
    plt.savefig(filename, bbox_inches='tight')


def make_2x2_vsrand_decflip_stackplot(tuples_of_title_x_y_noneflipped, filename, bin_size, vs_avg=True):
    if vs_avg:
        x_label = "Difference in (average) zeroed attention\nweight magnitude"
    else:
        x_label = "Difference in zeroed attention\nweight magnitude"
    if vs_avg:
        y_label = ["Highest didn't flip, 5/5 others flipped",
                   "Highest didn't flip, 4/5 others flipped",
                   "Highest didn't flip, 3/5 others flipped",
                   "Highest didn't flip, 2/5 others flipped",
                   "Highest didn't flip, 1/5 others flipped",
                   "Highest didn't flip, 0/5 others flipped",
                   "Highest flipped, 5/5 others flipped",
                   "Highest flipped, 4/5 others flipped",
                   "Highest flipped, 3/5 others flipped",
                   "Highest flipped, 2/5 others flipped",
                   "Highest flipped, 1/5 others flipped",
                   "Highest flipped, 0/5 others flipped"]
    else:
        y_label = ["Highest didn't flip, 2nd highest flipped",
                   "",
                   "",
                   "",
                   "",
                   "Neither flipped",
                   "Both flipped",
                   "",
                   "",
                   "",
                   "",
                   "Highest flipped, 2nd highest didn't flip"]
    title_of_each_graph = "Dataset"
    list_of_row_dicts = []
    how_many_bins = ceil(1.0 / bin_size)
    for dataset_ind in range(len(tuples_of_title_x_y_noneflipped)):
        title = tuples_of_title_x_y_noneflipped[dataset_ind][0]
        x = np.array(tuples_of_title_x_y_noneflipped[dataset_ind][1])
        y = np.array(tuples_of_title_x_y_noneflipped[dataset_ind][2])
        noneflipped = tuples_of_title_x_y_noneflipped[dataset_ind][3]
        assert np.sum(y[noneflipped] == 0) == np.sum(noneflipped), str(y[noneflipped]) + ', ' + str(title)
        list_of_binleft_countdist = []
        for i in range(how_many_bins):
            bin_left_edge = i * bin_size
            bin_right_edge = bin_left_edge + bin_size
            mask = np.logical_and(x >= bin_left_edge, x < bin_right_edge)
            available_y = y[mask]
            available_nonflipped = noneflipped[mask]
            list_of_counts = [0 for j in range(12)]
            for j in range(5):
                left_edge = -1.1 + .2 * j
                right_edge = left_edge + .2
                list_of_counts[j] = np.sum(np.logical_and(available_y > left_edge,
                                                          available_y < right_edge))
            list_of_counts[5] = np.sum(available_nonflipped)
            list_of_counts[6] = np.sum(np.logical_and(available_y > -.1, available_y < .1)) - list_of_counts[5]
            for j in range(5):
                left_edge = .1 + .2 * j
                right_edge = left_edge + .2
                list_of_counts[7 + j] = np.sum(np.logical_and(available_y > left_edge,
                                                              available_y < right_edge))
            np_arr_for_bin = np.array(list_of_counts)
            assert np.all(np_arr_for_bin >= 0), str(np_arr_for_bin) + ', ' + str(title)
            assert np.sum(mask) == np.sum(np_arr_for_bin), str(np.sum(mask)) + ", " + str(np_arr_for_bin)
            if np.sum(np_arr_for_bin) != 0:
                denom = np.sum(np_arr_for_bin)
                list_of_binleft_countdist.append((bin_left_edge, np_arr_for_bin / denom))
            elif len(list_of_binleft_countdist) > 0:
                list_of_binleft_countdist.append((bin_left_edge, None))
        while list_of_binleft_countdist[-1][1] is None:
            list_of_binleft_countdist = list_of_binleft_countdist[:-1]

        # now go through and interpolate for any remaining Nones in the middle
        list_of_inds_in_need_of_interpolation = []
        for i in range(len(list_of_binleft_countdist)):
            if list_of_binleft_countdist[i][1] is not None:
                if len(list_of_inds_in_need_of_interpolation) > 0:
                    # interpolate
                    endpoint_on_right = list_of_binleft_countdist[i][1]
                    step_sizes = [(endpoint_on_right[k] - most_recent_nonnone_on_left[k]) /
                                  (len(list_of_inds_in_need_of_interpolation) + 1) for k in range(12)]
                    for j in range(len(list_of_inds_in_need_of_interpolation)):
                        ind = list_of_inds_in_need_of_interpolation[j]
                        new_array = np.array([most_recent_nonnone_on_left[k] + step_sizes[k] * (j + 1)
                                              for k in range(12)])
                        assert np.all(new_array) >= 0
                        assert .99 < np.sum(new_array) < 1.01, str(np.sum(new_array)) + ', ' + str(new_array)
                        list_of_binleft_countdist[ind] = (list_of_binleft_countdist[ind][0], new_array)
                    list_of_inds_in_need_of_interpolation = []
                else:
                    most_recent_nonnone_on_left = list_of_binleft_countdist[i][1]
            else:
                list_of_inds_in_need_of_interpolation.append(i)

        tuples_of_title_x_y_noneflipped[dataset_ind] = [title, [tup[0] for tup in list_of_binleft_countdist],
                                                        [tup[1] for tup in list_of_binleft_countdist]]

    # each entry of tuples_of_title_x_y_noneflipped now in format title, list_of_bin_edges, list_of_corr_dists




    for tup in tuples_of_title_x_y_noneflipped:
        dataset_title = tup[0]
        assert len(tup[1]) == len(tup[2])
        avg_pct_noflip = 0
        for i in range(len(tup[1])):
            avg_pct_noflip += tup[2][i][5]
            list_of_row_dicts.append({title_of_each_graph: dataset_title, x_label: tup[1][i],
                                      y_label[0]: tup[2][i][0], y_label[1]: tup[2][i][1],
                                      y_label[2]: tup[2][i][2], y_label[3]: tup[2][i][3],
                                      y_label[4]: tup[2][i][4], y_label[5]: tup[2][i][5],
                                      y_label[6]: tup[2][i][6], y_label[7]: tup[2][i][7],
                                      y_label[8]: tup[2][i][8], y_label[9]: tup[2][i][9],
                                      y_label[10]: tup[2][i][10], y_label[11]: tup[2][i][11]})
        print("Avg pct no-flip: " + str(avg_pct_noflip / len(tup[1])))
    data_to_plot = pd.DataFrame(list_of_row_dicts)
    g = sns.FacetGrid(data_to_plot, col=title_of_each_graph, hue=None, col_wrap=2)
    #g = g.map(plt.stackplot, x_label, y_label[0], y_label[1], y_label[2], y_label[3], y_label[4], y_label[5],
    #          y_label[6], y_label[7], y_label[8], y_label[9], y_label[10], y_label[11], labels=y_label, colors=None)
    #g = g.set_titles("{col_name}")
    data_to_plot = data_to_plot[data_to_plot[title_of_each_graph] == 'Yahoo']

    if len(tuples_of_title_x_y_noneflipped) == 1:
        subplot_args = [111]
    elif len(tuples_of_title_x_y_noneflipped) == 2:
        subplot_args = [121, 122]
    else:
        subplot_args = [221, 222, 223, 224]
    subplot_args = [221, 222, 223, 224]


    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(12, 9))

    for i in range(len(tuples_of_title_x_y_noneflipped)):

        axs[i // 2, i % 2].set_title("Dataset = " + tuples_of_title_x_y_noneflipped[i][0])
        sample_data = tuples_of_title_x_y_noneflipped[i]
        ys = np.concatenate([np.reshape(arr, (12, 1)) for arr in sample_data[2]], axis=1)
        ys = np.concatenate([ys[:5], ys[6:]], axis=0)
        axs[i // 2, i % 2].stackplot(sample_data[1], ys, labels=y_label, colors=['#F8A102', '#FF9900', '#FFB60D', '#FFD5A6',
                                                                  '#FFF1E0', '#D6D6D6', '#E0F6FC', '#B3EBFC',
                                                                  '#90CDFC', '#3AAEFC', '#85B7FF'])

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def make_fracremoved_boxplots(filename, list_of_ordering_names, dataset_orderingfracdistribs, model,
                              y_axis_title="Fraction of Original Attention\nWeights Removed"):
    if "Fraction" in y_axis_title:
        plot_title = "Fractions of Original Attended Items Removed Before First Decision " +\
                     "Flip Occurred: " + model
        palette = {"Random": "#2589CC", "Attention": "#80D4FF", "Gradient": "#D4F3FF"}
    else:
        plot_title = "Probability Masses of Original Attention Distributions Removed Before First Decision " +\
                     "Flip Occurred: " + model
        palette = {"Random": "#A970FF", "Attention": "#EEA8FF", "Gradient": "#FFE0FF"}
    title_of_each_graph = "Dataset"
    list_of_row_dicts = []
    for tup in dataset_orderingfracdistribs:
        for i in range(len(list_of_ordering_names)):
            ordering_name = list_of_ordering_names[i]
            tup_ordering_list = tup[i + 1]
            for data_point in tup_ordering_list:
                row_dict = {title_of_each_graph: tup[0],
                            "AllTheSameInCol": "filler",
                            "Ranking Scheme": ordering_name,
                            y_axis_title: data_point}
                list_of_row_dicts.append(row_dict)
    data_to_plot = pd.DataFrame(list_of_row_dicts)

    fig = plt.figure(figsize=(12, 2))

    ax = sns.boxplot(x=title_of_each_graph, y=y_axis_title,
                hue="Ranking Scheme", palette=palette,
                data=data_to_plot)
    ax.set_title(plot_title)
    ax.legend_.remove()
    sns.despine(offset=20, trim=True)

    if 'conv' not in model:
        plt.legend(loc='lower left', prop={'size': 10})
    else:
        plt.legend(loc='upper right', prop={'size': 10})
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def make_hists(filename, dataset_xval_tups,
                              y_axis_title="Fraction of Original Attention\nWeights Removed"):
    title_of_each_graph = "Dataset"
    list_of_row_dicts = []
    for tup in dataset_xval_tups:
        x_vals = tup[1]
        try:
            for i in range(x_vals.shape[0]):
                row_dict = {title_of_each_graph: tup[0],
                            "AllTheSameInCol": "filler",
                            "Difference in Attention Weights": x_vals[i]}
                list_of_row_dicts.append(row_dict)
        except:
            print(x_vals)
            print(x_vals.shape)
    data_to_plot = pd.DataFrame(list_of_row_dicts)

    fig = plt.figure()

    g = sns.FacetGrid(data_to_plot, col=title_of_each_graph, sharey=False, col_wrap=2)
    g.map(sns.distplot, "Difference in Attention Weights", kde=False, rug=False)

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def make_2x2_regression_set(tuples_of_title_x_y, filename, y_label="Log difference in JS divergences\n" +
                                                                   "from original output distribution", vs_avg=True):
    if vs_avg:
        x_label = "Difference in (average) zeroed attention\nweight magnitude"
    else:
        x_label = "Difference in zeroed attention weight magnitude"
    if not vs_avg:
        y_label = "Log difference in JS divergences from\noriginal output distribution"
    title_of_each_graph = "Dataset"
    list_of_row_dicts = []
    if y_label == "Differences in JS divergences":
        pass
    for tup in tuples_of_title_x_y:
        assert len(tup[1]) == len(tup[2])
        for i in range(len(tup[1])):
            if True or "flip" in y_label or tup[2][i] < 2.5:
                list_of_row_dicts.append({title_of_each_graph: tup[0], x_label: tup[1][i], y_label: tup[2][i],
                                          "AllTheSameInCol": "filler"})
            else:
                print("Excluding a data point for " + y_label)
    data_to_plot = pd.DataFrame(list_of_row_dicts)
    marker_format_dict = {'alpha': 0.15, 's': 2}
    if 'flip' in y_label:
        marker_format_dict["s"] = 10
    g = sns.lmplot(x=x_label, y=y_label, col=title_of_each_graph, hue="AllTheSameInCol", palette=None,
                   data = data_to_plot, col_wrap = 2, legend=False,
                   scatter_kws=marker_format_dict, sharey=False, sharex=False)
    print("Saving file to " + filename)
    plt.savefig(filename, bbox_inches='tight')


def test_plots():
    tuples_of_title_x_y = []
    tuples_of_title_x_y.append(("Amazon", [0, .1, .2, .3, .4, .5], [.1, .3, .5, .2, .4, .7]))
    tuples_of_title_x_y.append(("Yelp", [0, .1, .2, .3, .4, .5], [.1, .3, .5, .2, .4, .7]))
    tuples_of_title_x_y.append(("Yahoo", [0, .1, .2, .3, .4, .5], [.1, .3, .5, .2, .4, .7]))
    tuples_of_title_x_y.append(("IMDB", [0, .1, .2, .3, .4, .5], [.1, .3, .5, .2, .4, .7]))
    if not os.path.isdir("testimagedir/"):
        os.makedirs("testimagedir/")
    filename = "testimagedir/regression"
    make_2x2_regression_set(tuples_of_title_x_y, filename)
    filename = "testimagedir/regression2"
    make_2x2_regression_set(tuples_of_title_x_y, filename)


def dec_table_size_to_keep_x_pct_of_data(x, table):
    rand_ind_list = []
    for i in range(table.shape[0]):
        if random() < x:
            rand_ind_list.append(1)
        else:
            rand_ind_list.append(0)
    np_ind_mask = np.array(rand_ind_list, dtype=bool)
    return table[np_ind_mask]


def make_boxplots(model_tag, yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag=''):
    if model_tag.endswith('/'):
        model_tag = model_tag[:-1]
    if not yahoo_tag.endswith('/'):
        yahoo_tag += '/'
    if len(yahoo_tag) > 1:
        yahoo_tag = '-' + yahoo_tag
    if not imdb_tag.endswith('/'):
        imdb_tag += '/'
    if len(imdb_tag) > 1:
        imdb_tag = '-' + imdb_tag
    if not amazon_tag.endswith('/'):
        amazon_tag += '/'
    if len(amazon_tag) > 1:
        amazon_tag = '-' + amazon_tag
    if not yelp_tag.endswith('/'):
        yelp_tag += '/'
    if len(yelp_tag) > 1:
        yelp_tag = '-' + yelp_tag
    print("Starting to make fraction-removed boxplot")
    if model_tag.startswith('hanconv'):
        yahoo_table = yahoo_hanconv_table
        imdb_table = imdb_hanconv_table
        amazon_table = amazon_hanconv_table
        yelp_table = yelp_hanconv_table
        model = 'hanconv'
        model_name = 'HANconvs'
        is_han = True
    elif model_tag.startswith('hanrnn'):
        yahoo_table = yahoo_hanrnn_table
        imdb_table = imdb_hanrnn_table
        amazon_table = amazon_hanrnn_table
        yelp_table = yelp_hanrnn_table
        model = 'hanrnn'
        model_name = 'HANrnns'
        is_han = True
    elif model_tag.startswith('flanconv'):
        yahoo_table = yahoo_flanconv_table
        imdb_table = imdb_flanconv_table
        amazon_table = amazon_flanconv_table
        yelp_table = yelp_flanconv_table
        model = 'flanconv'
        model_name = 'FLANconvs'
        is_han = False
    elif model_tag.startswith('flanrnn'):
        yahoo_table = yahoo_flanrnn_table
        imdb_table = imdb_flanrnn_table
        amazon_table = amazon_flanrnn_table
        yelp_table = yelp_flanrnn_table
        model_name = 'FLANrnns'
        model = 'flanrnn'
        is_han = False
    list_of_ordering_names = ["Random", "Attention", "Gradient"]
    yahoo_table = dec_table_size_to_keep_x_pct_of_data(1, yahoo_table)
    imdb_table = dec_table_size_to_keep_x_pct_of_data(1, imdb_table)
    amazon_table = dec_table_size_to_keep_x_pct_of_data(1, amazon_table)
    yelp_table = dec_table_size_to_keep_x_pct_of_data(1, yelp_table)
    yahoo_model_tup = ("Yahoo",
                       yahoo_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START][yahoo_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START] != -1],
                       yahoo_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP][yahoo_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] != -1],
                       yahoo_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD][yahoo_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] != -1])
    imdb_model_tup = ("IMDB",
                       imdb_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START][imdb_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START] != -1],
                       imdb_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP][imdb_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] != -1],
                       imdb_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD][imdb_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] != -1])
    amazon_model_tup = ("Amazon",
                       amazon_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START][amazon_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START] != -1],
                       amazon_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP][amazon_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] != -1],
                       amazon_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD][amazon_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] != -1])
    yelp_model_tup = ("Yelp",
                       yelp_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START][yelp_table[:, NEEDED_REM_RAND_FRAC_X_FOR_DECFLIP_START] != -1],
                       yelp_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP][yelp_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP] != -1],
                       yelp_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD][yelp_table[:, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRAD] != -1])
    make_fracremoved_boxplots(model_tag + "_fracremoved_boxplots", list_of_ordering_names,
                              [yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup], model_name)


def make_probmass_boxplots(model_tag, yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag=''):
    if model_tag.endswith('/'):
        model_tag = model_tag[:-1]
    if not yahoo_tag.endswith('/'):
        yahoo_tag += '/'
    if len(yahoo_tag) > 1:
        yahoo_tag = '-' + yahoo_tag
    if not imdb_tag.endswith('/'):
        imdb_tag += '/'
    if len(imdb_tag) > 1:
        imdb_tag = '-' + imdb_tag
    if not amazon_tag.endswith('/'):
        amazon_tag += '/'
    if len(amazon_tag) > 1:
        amazon_tag = '-' + amazon_tag
    if not yelp_tag.endswith('/'):
        yelp_tag += '/'
    if len(yelp_tag) > 1:
        yelp_tag = '-' + yelp_tag
    print("Starting to make probmass-removed boxplot")
    if model_tag.startswith('hanconv'):
        yahoo_table = yahoo_hanconv_table
        imdb_table = imdb_hanconv_table
        amazon_table = amazon_hanconv_table
        yelp_table = yelp_hanconv_table
        model_name = 'HANconvs'
        model = 'hanconv'
        is_han = True
    elif model_tag.startswith('hanrnn'):
        yahoo_table = yahoo_hanrnn_table
        imdb_table = imdb_hanrnn_table
        amazon_table = amazon_hanrnn_table
        yelp_table = yelp_hanrnn_table
        model = 'hanrnn'
        model_name = "HANrnns"
        is_han = True
    elif model_tag.startswith('flanconv'):
        yahoo_table = yahoo_flanconv_table
        imdb_table = imdb_flanconv_table
        amazon_table = amazon_flanconv_table
        yelp_table = yelp_flanconv_table
        model = 'flanconv'
        model_name = 'FLANconvs'
        is_han = False
    elif model_tag.startswith('flanrnn'):
        yahoo_table = yahoo_flanrnn_table
        imdb_table = imdb_flanrnn_table
        amazon_table = amazon_flanrnn_table
        yelp_table = yelp_flanrnn_table
        model = 'flanrnn'
        model_name = 'FLANrnns'
        is_han = False
    list_of_ordering_names = ["Random", "Attention", "Gradient"]
    yahoo_table = dec_table_size_to_keep_x_pct_of_data(1, yahoo_table)
    imdb_table = dec_table_size_to_keep_x_pct_of_data(1, imdb_table)
    amazon_table = dec_table_size_to_keep_x_pct_of_data(1, amazon_table)
    yelp_table = dec_table_size_to_keep_x_pct_of_data(1, yelp_table)
    yahoo_model_tup = ("Yahoo",
                       yahoo_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START][yahoo_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START] != -1],
                       yahoo_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP][yahoo_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP] != -1],
                       yahoo_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD][yahoo_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD] != -1])
    imdb_model_tup = ("IMDB",
                       imdb_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START][imdb_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START] != -1],
                       imdb_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP][imdb_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP] != -1],
                       imdb_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD][imdb_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD] != -1])
    amazon_model_tup = ("Amazon",
                       amazon_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START][amazon_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START] != -1],
                       amazon_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP][amazon_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP] != -1],
                       amazon_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD][amazon_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD] != -1])
    yelp_model_tup = ("Yelp",
                       yelp_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START][yelp_table[:, NEEDED_REM_RAND_PROBMASS_FOR_DECFLIP_START] != -1],
                       yelp_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP][yelp_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP] != -1],
                       yelp_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD][yelp_table[:, NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRAD] != -1])
    make_fracremoved_boxplots(model_tag + "_probmassremoved_boxplots", list_of_ordering_names,
                              [yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup], model_name,
                              y_axis_title="Probability Mass of Original\nAttention Dist Removed")


def make_decflip_regression_plot(model_tag):
    print("Starting to make a decision flip plot")
    if model_tag.startswith('hanconv'):
        yahoo_table = yahoo_hanconv_table
        imdb_table = imdb_hanconv_table
        amazon_table = amazon_hanconv_table
        yelp_table = yelp_hanconv_table
        model = 'hanconv'
        is_han = True
    elif model_tag.startswith('hanrnn'):
        yahoo_table = yahoo_hanrnn_table
        imdb_table = imdb_hanrnn_table
        amazon_table = amazon_hanrnn_table
        yelp_table = yelp_hanrnn_table
        model = 'hanrnn'
        is_han = True
    elif model_tag.startswith('flanconv'):
        yahoo_table = yahoo_flanconv_table
        imdb_table = imdb_flanconv_table
        amazon_table = amazon_flanconv_table
        yelp_table = yelp_flanconv_table
        model = 'flanconv'
        is_han = False
    elif model_tag.startswith('flanrnn'):
        yahoo_table = yahoo_flanrnn_table
        imdb_table = imdb_flanrnn_table
        amazon_table = amazon_flanrnn_table
        yelp_table = yelp_flanrnn_table
        model = 'flanrnn'
        is_han = False
    if is_han:
        yahoo_mask = yahoo_han_mask
        imdb_mask = imdb_han_mask
        amazon_mask = amazon_han_mask
        yelp_mask = yelp_han_mask
    else:
        yahoo_mask = yahoo_flan_mask
        imdb_mask = imdb_flan_mask
        amazon_mask = amazon_flan_mask
        yelp_mask = yelp_flan_mask
    yahoo_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yahoo10cat-' + model_tag)[yahoo_mask]
    yahoo_model_tup = make_rand_decflip_model_tup(yahoo_highestattnweights, yahoo_table, "Yahoo")

    imdb_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'imdb-' + model_tag)[imdb_mask]
    imdb_model_tup = make_rand_decflip_model_tup(imdb_highestattnweights, imdb_table, "IMDB")

    amazon_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'amazon-' + model_tag)[amazon_mask]
    amazon_model_tup = make_rand_decflip_model_tup(amazon_highestattnweights, amazon_table, "Amazon")

    yelp_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yelp-' + model_tag)[yelp_mask]
    yelp_model_tup = make_rand_decflip_model_tup(yelp_highestattnweights, yelp_table, "Yelp")

    #make_2x2_vsrand_decflip_violinplot([yahoo_hanrnn_tup, imdb_hanrnn_tup], "differences_in_decflips_" + model)
    make_2x2_vsrand_decflip_stackplot([yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup],
                                      "differences_in_decflips_" + model, .05)
    print("Finished making a decision flip plot")


def make_decflip_regression_plot_vs2ndhighest(model_tag, yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag=''):
    if model_tag.endswith('/'):
        model_tag = model_tag[:-1]
    if not yahoo_tag.endswith('/'):
        yahoo_tag += '/'
    if len(yahoo_tag) > 1:
        yahoo_tag = '-' + yahoo_tag
    if not imdb_tag.endswith('/'):
        imdb_tag += '/'
    if len(imdb_tag) > 1:
        imdb_tag = '-' + imdb_tag
    if not amazon_tag.endswith('/'):
        amazon_tag += '/'
    if len(amazon_tag) > 1:
        amazon_tag = '-' + amazon_tag
    if not yelp_tag.endswith('/'):
        yelp_tag += '/'
    if len(yelp_tag) > 1:
        yelp_tag = '-' + yelp_tag
    print("Starting to make a decision flip plot")
    if model_tag.startswith('hanconv'):
        yahoo_table = yahoo_hanconv_table
        imdb_table = imdb_hanconv_table
        amazon_table = amazon_hanconv_table
        yelp_table = yelp_hanconv_table
        model = 'hanconv'
        is_han = True
    elif model_tag.startswith('hanrnn'):
        yahoo_table = yahoo_hanrnn_table
        imdb_table = imdb_hanrnn_table
        amazon_table = amazon_hanrnn_table
        yelp_table = yelp_hanrnn_table
        model = 'hanrnn'
        is_han = True
    elif model_tag.startswith('flanconv'):
        yahoo_table = yahoo_flanconv_table
        imdb_table = imdb_flanconv_table
        amazon_table = amazon_flanconv_table
        yelp_table = yelp_flanconv_table
        model = 'flanconv'
        is_han = False
    elif model_tag.startswith('flanrnn'):
        yahoo_table = yahoo_flanrnn_table
        imdb_table = imdb_flanrnn_table
        amazon_table = amazon_flanrnn_table
        yelp_table = yelp_flanrnn_table
        model = 'flanrnn'
        is_han = False
    if is_han:
        yahoo_mask = yahoo_han_mask
        imdb_mask = imdb_han_mask
        amazon_mask = amazon_han_mask
        yelp_mask = yelp_han_mask
    else:
        yahoo_mask = yahoo_flan_mask
        imdb_mask = imdb_flan_mask
        amazon_mask = amazon_flan_mask
        yelp_mask = yelp_flan_mask
    yahoo_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yahoo10cat-' + model_tag +
                                                   yahoo_tag)[yahoo_mask]
    yahoo_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'yahoo10cat-' + model_tag +
                                                   yahoo_tag)[yahoo_mask]
    yahoo_model_tup = make_vs2nd_decflip_model_tup(yahoo_highestattnweights, yahoo_2ndhighestattnweights, yahoo_table,
                                                  "Yahoo")
    imdb_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'imdb-' + model_tag +
                                                   imdb_tag)[imdb_mask]
    imdb_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'imdb-' + model_tag +
                                                   imdb_tag)[imdb_mask]
    imdb_model_tup = make_vs2nd_decflip_model_tup(imdb_highestattnweights, imdb_2ndhighestattnweights, imdb_table,
                                                  "IMDB")

    amazon_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'amazon-' + model_tag +
                                                   amazon_tag)[amazon_mask]
    amazon_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'amazon-' + model_tag +
                                                   amazon_tag)[amazon_mask]
    amazon_model_tup = make_vs2nd_decflip_model_tup(amazon_highestattnweights, amazon_2ndhighestattnweights, amazon_table,
                                                   "Amazon")
    yelp_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yelp-' + model_tag +
                                                   yelp_tag)[yelp_mask]
    yelp_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'yelp-' + model_tag +
                                                   yelp_tag)[yelp_mask]
    yelp_model_tup = make_vs2nd_decflip_model_tup(yelp_highestattnweights, yelp_2ndhighestattnweights, yelp_table,
                                                  "Yelp")

    #make_2x2_vsrand_decflip_violinplot([yahoo_hanrnn_tup, imdb_hanrnn_tup], "differences_in_decflips_" + model)
    make_2x2_vsrand_decflip_stackplot([yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup], "differences_in_decflips_vs2nd_" + model, .05,
                                      vs_avg=False)
    print("Finished making a decision flip plot")


def make_vs2nd_decflip_model_tup(highestattnweights, secondhighestattnweights, table, dataset_title):
    model_tup = (dataset_title,
                      list(highestattnweights -
                           secondhighestattnweights),
                      list((table[:, DEC_FLIP_ZERO_HIGHEST] != -1).astype(float) -
                           (table[:, DEC_FLIP_ZERO_2NDHIGHEST] != -1).astype(float)),
                      np.logical_and(table[:, DEC_FLIP_ZERO_HIGHEST] == -1,
                                     table[:, DEC_FLIP_ZERO_2NDHIGHEST] == -1)
                      )
    return model_tup
    num_points_to_plot = 500
    inds_to_pick = [(True if random() < num_points_to_plot / len(model_tup[1]) else False) for
                    i in range(len(model_tup[1]))]
    modified_list_1 = []
    modified_list_2 = []
    for i in range(len(inds_to_pick)):
        if inds_to_pick[i]:
            modified_list_1.append(model_tup[1][i])
            modified_list_2.append(model_tup[2][i])
    modified_arr_3 = model_tup[3][np.array(inds_to_pick, dtype=bool)]
    return (model_tup[0], modified_list_1, modified_list_2, modified_arr_3)


def make_rand_decflip_model_tup(highestattnweights, table, dataset_title):
    sampled_weights_that_were_max = (highestattnweights[:, None] ==
                                     table[:, EXTRACTED_SINGLE_ATTN_WEIGHT_START:
                                                    EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1])
    sampled_weights_that_werent_max = np.logical_not(sampled_weights_that_were_max).astype(float)
    assert np.sum(sampled_weights_that_werent_max > 1) == 0
    num_sampled_weights_that_were_max = np.sum(sampled_weights_that_were_max, axis=1)
    all_sampled_weights_are_max = (num_sampled_weights_that_were_max ==
                                   (EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1 - EXTRACTED_SINGLE_ATTN_WEIGHT_START))
    table = table[np.logical_not(all_sampled_weights_are_max)]
    highestattnweights = highestattnweights[np.logical_not(all_sampled_weights_are_max)]
    denom_array = np.sum(sampled_weights_that_werent_max, axis=1)
    assert np.sum(denom_array < 0) == 0
    model_tup = (dataset_title,
                       list(highestattnweights -
                            np.divide((np.sum(np.multiply(table[:, EXTRACTED_SINGLE_ATTN_WEIGHT_START:
                                                                         EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1],
                                                          sampled_weights_that_werent_max), axis=1)),
                                      denom_array)),
                       list((table[:, DEC_FLIP_ZERO_HIGHEST] != -1).astype(float) -
                            np.divide(np.sum(np.multiply(table[:, NEEDED_REM_RAND_X_FOR_DECFLIP_START:
                                                                        NEEDED_REM_RAND_X_FOR_DECFLIP_END + 1] == 1,
                                                         sampled_weights_that_werent_max), axis=1),
                                      denom_array)),
                       np.logical_and(table[:, DEC_FLIP_ZERO_HIGHEST] == -1,
                                      (np.sum(table[:, NEEDED_REM_RAND_X_FOR_DECFLIP_START:
                                                             NEEDED_REM_RAND_X_FOR_DECFLIP_END + 1] == 1, axis=1) == 0))
                       )
    return model_tup


def make_vs2nd_jsdiv_model_tup(highestattnweights, secondhighestattnweights, table, dataset_title):
    model_tup = (dataset_title,
                       list(highestattnweights -
                            secondhighestattnweights),
                       list((table[:, JS_DIV_ZERO_HIGHEST]).astype(float) -
                            (table[:, JS_DIV_ZERO_2NDHIGHEST]).astype(float)))
    num_points_to_plot = 500
    inds_to_pick = [(True if random() < num_points_to_plot / len(model_tup[1]) else False) for
                    i in range(len(model_tup[1]))]
    modified_list_1 = []
    modified_list_2 = []
    for i in range(len(inds_to_pick)):
        if inds_to_pick[i]:
            modified_list_1.append(model_tup[1][i])
            modified_list_2.append(model_tup[2][i])
    return (model_tup[0], modified_list_1, modified_list_2)


def make_rand_jsdiv_model_tup(highestattnweights, table, dataset_title):
    model_tup = (dataset_title,
                       list(highestattnweights -
                            table[:, NONTOP_RAND_ZEROED_WEIGHT]),
                       list(table[:, JS_DIV_ZERO_HIGHEST] -
                            table[:, NONTOP_RAND_JS_DIV]))
    return model_tup


def make_rand_jsdiv_model_tup_sample5(highestattnweights, table, dataset_title):
    sampled_weights_that_were_max = (highestattnweights[:, None] ==
                                     table[:, EXTRACTED_SINGLE_ATTN_WEIGHT_START:
                                                    EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1])
    sampled_weights_that_werent_max = np.logical_not(sampled_weights_that_were_max).astype(float)
    num_sampled_weights_that_were_max = np.sum(sampled_weights_that_were_max, axis=1)
    all_sampled_weights_are_max = (num_sampled_weights_that_were_max ==
                                   (EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1 - EXTRACTED_SINGLE_ATTN_WEIGHT_START))
    table = table[np.logical_not(all_sampled_weights_are_max)]
    highestattnweights = highestattnweights[np.logical_not(all_sampled_weights_are_max)]
    model_tup = (dataset_title,
                       list(highestattnweights -
                            np.divide((np.sum(np.multiply(table[:, EXTRACTED_SINGLE_ATTN_WEIGHT_START:
                                                                         EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1],
                                                          sampled_weights_that_werent_max), axis=1)),
                                      (-1 * num_sampled_weights_that_were_max + EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1 -
                                       EXTRACTED_SINGLE_ATTN_WEIGHT_START))),
                       list(np.log(table[:, JS_DIV_ZERO_HIGHEST]) -
                            np.divide(np.sum(np.multiply(np.log(table[:, EXTRACTED_SINGLE_WEIGHT_JS_START:
                                                                               EXTRACTED_SINGLE_WEIGHT_JS_END + 1]),
                                                         sampled_weights_that_werent_max), axis=1),
                                      (-1 * num_sampled_weights_that_were_max +
                                       EXTRACTED_SINGLE_ATTN_WEIGHT_END + 1 - EXTRACTED_SINGLE_ATTN_WEIGHT_START))))
    return model_tup


def make_jsdiv_regression_plot(model_tag, sample_x, yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag=''):
    if model_tag.endswith('/'):
        model_tag = model_tag[:-1]
    if not yahoo_tag.endswith('/'):
        yahoo_tag += '/'
    if len(yahoo_tag) > 1:
        yahoo_tag = '-' + yahoo_tag
    if not imdb_tag.endswith('/'):
        imdb_tag += '/'
    if len(imdb_tag) > 1:
        imdb_tag = '-' + imdb_tag
    if not amazon_tag.endswith('/'):
        amazon_tag += '/'
    if len(amazon_tag) > 1:
        amazon_tag = '-' + amazon_tag
    if not yelp_tag.endswith('/'):
        yelp_tag += '/'
    if len(yelp_tag) > 1:
        yelp_tag = '-' + yelp_tag
    print("Starting to make a JS div plot")
    if model_tag.startswith('hanconv'):
        yahoo_table = yahoo_hanconv_table
        imdb_table = imdb_hanconv_table
        amazon_table = amazon_hanconv_table
        yelp_table = yelp_hanconv_table
        model = 'hanconv'
        is_han = True
    elif model_tag.startswith('hanrnn'):
        yahoo_table = yahoo_hanrnn_table
        imdb_table = imdb_hanrnn_table
        amazon_table = amazon_hanrnn_table
        yelp_table = yelp_hanrnn_table
        model = 'hanrnn'
        is_han = True
    elif model_tag.startswith('flanconv'):
        yahoo_table = yahoo_flanconv_table
        imdb_table = imdb_flanconv_table
        amazon_table = amazon_flanconv_table
        yelp_table = yelp_flanconv_table
        model = 'flanconv'
        is_han = False
    elif model_tag.startswith('flanrnn'):
        yahoo_table = yahoo_flanrnn_table
        imdb_table = imdb_flanrnn_table
        amazon_table = amazon_flanrnn_table
        yelp_table = yelp_flanrnn_table
        model = 'flanrnn'
        is_han = False
    if is_han:
        yahoo_mask = yahoo_han_mask
        imdb_mask = imdb_han_mask
        amazon_mask = amazon_han_mask
        yelp_mask = yelp_han_mask
    else:
        yahoo_mask = yahoo_flan_mask
        imdb_mask = imdb_flan_mask
        amazon_mask = amazon_flan_mask
        yelp_mask = yelp_flan_mask
    yahoo_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yahoo10cat-' + model_tag +
                                                   yahoo_tag)[yahoo_mask]
    yahoo_model_tup = make_rand_jsdiv_model_tup(yahoo_highestattnweights, yahoo_table, "Yahoo")
    #yahoo_low, yahoo_high = ask_where_to_split_attndiff_plots(yahoo_model_tup)
    imdb_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'imdb-' + model_tag +
                                                   imdb_tag)[imdb_mask]
    imdb_model_tup = make_rand_jsdiv_model_tup(imdb_highestattnweights, imdb_table, "IMDB")
    #imdb_low, imdb_high = ask_where_to_split_attndiff_plots(imdb_model_tup)

    amazon_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'amazon-' + model_tag +
                                                   amazon_tag)[amazon_mask]
    amazon_model_tup = make_rand_jsdiv_model_tup(amazon_highestattnweights, amazon_table, "Amazon")
    #amazon_low, amazon_high = ask_where_to_split_attndiff_plots(amazon_model_tup)
    yelp_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yelp-' + model_tag +
                                                   yelp_tag)[yelp_mask]
    yelp_model_tup = make_rand_jsdiv_model_tup(yelp_highestattnweights, yelp_table, "Yelp")
    #yelp_low, yelp_high = ask_where_to_split_attndiff_plots(yelp_model_tup)

    yahoo_model_tup, yahoo_neg_x = convert_to_log_and_print_how_many_were_originally_negative(yahoo_model_tup,
                                                                                 model_tag)
    imdb_model_tup, imdb_neg_x = convert_to_log_and_print_how_many_were_originally_negative(imdb_model_tup,
                                                                                 model_tag)
    amazon_model_tup, az_neg_x = convert_to_log_and_print_how_many_were_originally_negative(amazon_model_tup,
                                                                                 model_tag)
    yelp_model_tup, yelp_neg_x = convert_to_log_and_print_how_many_were_originally_negative(yelp_model_tup,
                                                                                 model_tag)

    yahoo_model_tup = sample_x_datapoints_per_tenth_on_xaxis(yahoo_model_tup, sample_x)
    imdb_model_tup = sample_x_datapoints_per_tenth_on_xaxis(imdb_model_tup, sample_x)
    amazon_model_tup = sample_x_datapoints_per_tenth_on_xaxis(amazon_model_tup, sample_x)
    yelp_model_tup = sample_x_datapoints_per_tenth_on_xaxis(yelp_model_tup, sample_x)

    """yahoo_model_tup = wonkysample_so_xs_are_roughly_uniform_at_random(yahoo_model_tup, sample_x)
    imdb_model_tup = wonkysample_so_xs_are_roughly_uniform_at_random(imdb_model_tup, sample_x)
    amazon_model_tup = wonkysample_so_xs_are_roughly_uniform_at_random(amazon_model_tup, sample_x)
    yelp_model_tup = wonkysample_so_xs_are_roughly_uniform_at_random(yelp_model_tup, sample_x)"""

    #make_kdeplot([yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup], "differences_in_jsdivs_kde_" + model)
    make_hists("neg_jsdivdiff_xvals_" + model, [("Yahoo", yahoo_neg_x), ("IMDB", imdb_neg_x), ("Amazon", az_neg_x), ("Yelp", yelp_neg_x)])

    make_2x2_regression_set([yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup],
                            "differences_in_jsdivs_" + model, vs_avg=False)
    print("Finished making a JS div plot")


def convert_to_log_and_print_how_many_were_originally_negative(js_model_tup, model_label_to_print_with_report):
    convert_to_log = js_model_tup[2]
    if isinstance(convert_to_log, list):
        convert_to_log = np.array(convert_to_log)
        change_back_to_list = True
    else:
        change_back_to_list = False
    subtract_before_logging = np.min(convert_to_log) - .01
    x_vals_going_with_ys = np.array(js_model_tup[1])[convert_to_log < 0]
    assert len(x_vals_going_with_ys.shape) > 0 and x_vals_going_with_ys.shape[0] > 0, x_vals_going_with_ys
    print(js_model_tup[0] + " " + model_label_to_print_with_report + ": " + str(np.sum(convert_to_log < 0)) +
          " out of " + str(convert_to_log.shape[0]) + " (" + str(np.sum(convert_to_log < 0) / convert_to_log.shape[0]) + ") JSdivdiffs were negative")
    new_y_vals = np.log(convert_to_log - subtract_before_logging)
    if change_back_to_list:
        new_y_vals = list(new_y_vals)
    assert len(js_model_tup) == 3
    return [js_model_tup[0], js_model_tup[1], new_y_vals], x_vals_going_with_ys


def wonkysample_so_xs_are_roughly_uniform_at_random(js_model_tup, sample_x):
    xs = js_model_tup[1]
    if isinstance(xs, list):
        xs = np.array(xs)
        change_back_to_list = True
    else:
        change_back_to_list = False
    lower_bound = np.min(xs)
    upper_bound = np.max(xs)
    inds_lowest_to_highest = np.argsort(xs)
    for j in range(inds_lowest_to_highest.shape[0]):
        if inds_lowest_to_highest[j] == 0:
            ind_to_check = j
            break
    xs = xs[inds_lowest_to_highest]
    corr_ys = np.array(js_model_tup[2])[inds_lowest_to_highest]
    make_histogram_of_x_vals(xs)
    exit(1)


def make_histogram_of_x_vals(x_vals):
    fig = plt.figure(figsize=(12, 2))

    ax = sns.distplot(x_vals, kde=False, rug=True)

    plt.savefig('test_hist', bbox_inches='tight')
    plt.close(fig)



def subsample_so_xs_are_roughly_uniform_at_random(js_model_tup, sample_x):
    xs = js_model_tup[1]
    if isinstance(xs, list):
        xs = np.array(xs)
        change_back_to_list = True
    else:
        change_back_to_list = False
    lower_bound = np.min(xs)
    upper_bound = np.max(xs)
    inds_lowest_to_highest = np.argsort(xs)
    xs = xs[inds_lowest_to_highest]
    corr_ys = np.array(js_model_tup[2])[inds_lowest_to_highest]
    pseudo_xs = np.random.uniform(lower_bound, upper_bound, size=(sample_x,))
    np.sort(pseudo_xs)
    # now go through and greedily mark all values as we find vals that are closest to each pseudo_x
    bool_arr = np.array([0] * xs.shape[0], dtype=bool)
    stack_of_unchosen_val_inds = [0]
    next_full_arr_ind_to_right = 1
    for ind in range(pseudo_xs.shape[0]):
        quit = False
        figured_out_what_to_do_with_ind = False
        while not figured_out_what_to_do_with_ind:
            if pseudo_xs[ind] > xs[next_full_arr_ind_to_right]:
                stack_of_unchosen_val_inds.insert(0, next_full_arr_ind_to_right)
                next_full_arr_ind_to_right += 1
            elif xs[next_full_arr_ind_to_right] == pseudo_xs[ind]:
                figured_out_what_to_do_with_ind = True
                bool_arr[next_full_arr_ind_to_right] = True
                next_full_arr_ind_to_right += 1
            else:
                figured_out_what_to_do_with_ind =True
                # the ind actually does appear on the right, so now's the time to choose what we pull
                if len(stack_of_unchosen_val_inds) == 0:
                    bool_arr[next_full_arr_ind_to_right] = True
                    next_full_arr_ind_to_right += 1
                else:
                    # we have two choices. pick the closer one.
                    if pseudo_xs[ind] - xs[stack_of_unchosen_val_inds[0]] < xs[next_full_arr_ind_to_right] - pseudo_xs[ind]:
                        # closer to previous val
                        bool_arr[stack_of_unchosen_val_inds[0]] = True
                        del stack_of_unchosen_val_inds[0]
                    else:
                        bool_arr[next_full_arr_ind_to_right] = True
                        next_full_arr_ind_to_right += 1
            if next_full_arr_ind_to_right >= xs.shape[0]:
                quit = True
                if len(stack_of_unchosen_val_inds) > 0:
                    bool_arr[stack_of_unchosen_val_inds[0]] = True
                figured_out_what_to_do_with_ind = True
        if quit:
            break
    new_x_vals = xs[bool_arr]
    new_y_vals = corr_ys[bool_arr]
    if change_back_to_list:
        return [js_model_tup[0], list(new_x_vals), list(new_y_vals)]
    else:
        return [js_model_tup[0], new_x_vals, new_y_vals]


def sample_x_datapoints_per_tenth_on_xaxis(model_tup, x):
    x_axis_vals = model_tup[1]
    if isinstance(x_axis_vals, list):
        x_axis_vals = np.array(x_axis_vals)
    lists_of_lists_being_built = []
    for i in range(1, len(model_tup)):
        lists_of_lists_being_built.append([])
    num_bins = 200
    for i in range(num_bins):
        left_val_inclusive = i / num_bins
        right_val_exclusive = left_val_inclusive + (1.0 / num_bins)
        true_where_is_option = np.logical_and(x_axis_vals >= left_val_inclusive, x_axis_vals < right_val_exclusive)
        possible_inds = np.nonzero(true_where_is_option)[0]
        mask_to_take = np.array([0] * true_where_is_option.shape[0], dtype=bool)
        if possible_inds.shape[0] < x:
            mask_to_take[possible_inds] = 1
        else:
            mask_to_take[np.random.choice(possible_inds, size=x)] = 1
        for j in range(1, len(model_tup)):
            if isinstance(model_tup[j], list):
                arr_to_sample = np.array(model_tup[j])
            else:
                arr_to_sample = model_tup[j]
            lists_of_lists_being_built[j - 1] += list(arr_to_sample[mask_to_take])
    print("Pulled " + str(len(lists_of_lists_being_built[0])) + " datapoints")
    new_model_tup = [model_tup[0]]
    for i in range(1, len(model_tup)):
        if isinstance(model_tup[i], list):
            new_model_tup.append(lists_of_lists_being_built[i - 1])
        else:
            new_model_tup.append(np.array(lists_of_lists_being_built[i - 1]))
    return new_model_tup


def ask_where_to_split_attndiff_plots(model_tup):
    np_set_of_attndiffs = np.array(model_tup[1])
    done = False
    while not done:
        val = input("Find pct of diffs that fall above: ")
        try:
            val = float(val)
        except:
            continue
        print(str(np.sum(np_set_of_attndiffs > val) / np_set_of_attndiffs.shape[0]))
        keep_going = input("Keep exploring? (y/n) ")
        if keep_going.startswith('n'):
            done = True
            break
    have_splitting_val = False
    while not have_splitting_val:
        val = input("Splitting val? ")
        try:
            val = float(val)
            have_splitting_val = True
        except:
            continue
    # now go through and create two different model_tups, one of values that fall above and other of vals
    # that fall below
    new_low_tuple = [model_tup[0]]
    new_high_tuple = [model_tup[0]]
    for component_ind in range(1, len(model_tup)):
        comp = model_tup[component_ind]
        if component_ind == 1:
            if isinstance(comp, list):
                val_included_in_low_key = []
                new_low_list = []
                new_high_list = []
                for item in comp:
                    if item <= val:
                        val_included_in_low_key.append(True)
                        new_low_list.append(item)
                    else:
                        val_included_in_low_key.append(False)
                        new_high_list.append(item)
            else:  # assume it's a numpy array
                val_included_in_low_key = list(comp <= val)
                new_low_list = comp[comp <= val]
                new_high_list = comp[comp > val]
            new_low_tuple.append(new_low_list)
            new_high_tuple.append(new_high_list)
        else:  # assume it's a numpy array
            if isinstance(comp, list):
                new_low_list = []
                new_high_list = []
                for i in range(len(val_included_in_low_key)):
                    key = val_included_in_low_key[i]
                    if key:
                        new_low_list.append(comp[i])
                    else:
                        new_high_list.append(comp[i])
            else:
                np_mask = np.array(val_included_in_low_key, dtype=bool)
                new_low_list = comp[np_mask]
                new_high_list = comp[np.logical_not(np_mask)]
            new_low_tuple.append(new_low_list)
            new_high_tuple.append(new_high_list)
    return new_low_tuple, new_high_tuple


def make_jsdiv_regression_plot_vs2ndhighest(model_tag):
    print("Starting to make a JS div plot")
    if model_tag.startswith('hanconv'):
        yahoo_table = yahoo_hanconv_table
        imdb_table = imdb_hanconv_table
        amazon_table = amazon_hanconv_table
        yelp_table = yelp_hanconv_table
        model = 'hanconv'
        is_han = True
    elif model_tag.startswith('hanrnn'):
        yahoo_table = yahoo_hanrnn_table
        imdb_table = imdb_hanrnn_table
        amazon_table = amazon_hanrnn_table
        yelp_table = yelp_hanrnn_table
        model = 'hanrnn'
        is_han = True
    elif model_tag.startswith('flanconv'):
        yahoo_table = yahoo_flanconv_table
        imdb_table = imdb_flanconv_table
        amazon_table = amazon_flanconv_table
        yelp_table = yelp_flanconv_table
        model = 'flanconv'
        is_han = False
    elif model_tag.startswith('flanrnn'):
        yahoo_table = yahoo_flanrnn_table
        imdb_table = imdb_flanrnn_table
        amazon_table = amazon_flanrnn_table
        yelp_table = yelp_flanrnn_table
        model = 'flanrnn'
        is_han = False
    if is_han:
        yahoo_mask = yahoo_han_mask
        imdb_mask = imdb_han_mask
        amazon_mask = amazon_han_mask
        yelp_mask = yelp_han_mask
    else:
        yahoo_mask = yahoo_flan_mask
        imdb_mask = imdb_flan_mask
        amazon_mask = amazon_flan_mask
        yelp_mask = yelp_flan_mask
    yahoo_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yahoo10cat-' + model_tag)[yahoo_mask]
    yahoo_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'yahoo10cat-' + model_tag)[yahoo_mask]
    yahoo_model_tup = make_vs2nd_jsdiv_model_tup(yahoo_highestattnweights, yahoo_2ndhighestattnweights, yahoo_table,
                                                "Yahoo")
    imdb_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'imdb-' + model_tag)[imdb_mask]
    imdb_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'imdb-' + model_tag)[imdb_mask]
    imdb_model_tup = make_vs2nd_jsdiv_model_tup(imdb_highestattnweights, imdb_2ndhighestattnweights, imdb_table,
                                                "IMDB")

    amazon_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'amazon-' + model_tag)[amazon_mask]
    amazon_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'amazon-' + model_tag)[amazon_mask]
    amazon_model_tup = make_vs2nd_jsdiv_model_tup(amazon_highestattnweights, amazon_2ndhighestattnweights, amazon_table,
                                                 "Amazon")
    yelp_highestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(0, is_han, base_output_dir + 'yelp-' + model_tag)[yelp_mask]
    yelp_2ndhighestattnweights = \
        get_np_arr_of_one_attn_weight_per_instance(1, is_han, base_output_dir + 'yelp-' + model_tag)[yelp_mask]
    yelp_model_tup = make_vs2nd_jsdiv_model_tup(yelp_highestattnweights, yelp_2ndhighestattnweights, yelp_table,
                                                "Yelp")

    make_2x2_regression_set([yahoo_model_tup, imdb_model_tup, amazon_model_tup, yelp_model_tup],
                            "differences_in_jsdivs_vs2nd_" + model, vs_avg=False)
    print("Finished making a JS div plot")


def main():
    sample_x = 10
    assert yahoo_hanrnn_table.shape[1] == imdb_hanrnn_table.shape[1]

    make_probmass_boxplots('hanrnn', yahoo_tag='postattnfix', imdb_tag='postattnfix',
                            amazon_tag='postattnfix', yelp_tag='postattnfix-2')
    make_probmass_boxplots('hanconv', yahoo_tag='convfix', imdb_tag='convfix',
                  amazon_tag='convfix', yelp_tag='convfix')
    make_probmass_boxplots('flanrnn', yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag='moreword2vec')
    make_probmass_boxplots('flanconv', yahoo_tag='convfix', imdb_tag='convfix',
                  amazon_tag='convfix', yelp_tag='convfix')

    make_boxplots('hanrnn', yahoo_tag='postattnfix', imdb_tag='postattnfix',
                            amazon_tag='postattnfix', yelp_tag='postattnfix-2')
    make_boxplots('hanconv', yahoo_tag='convfix', imdb_tag='convfix',
                  amazon_tag='convfix', yelp_tag='convfix')
    make_boxplots('flanrnn', yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag='moreword2vec')
    make_boxplots('flanconv', yahoo_tag='convfix', imdb_tag='convfix',
                                              amazon_tag='convfix', yelp_tag='convfix')

    make_decflip_regression_plot_vs2ndhighest('hanrnn', yahoo_tag='postattnfix', imdb_tag='postattnfix',
                                              amazon_tag='postattnfix', yelp_tag='postattnfix-2')
    make_decflip_regression_plot_vs2ndhighest('hanconv', yahoo_tag='convfix', imdb_tag='convfix',
                                              amazon_tag='convfix', yelp_tag='convfix')
    make_decflip_regression_plot_vs2ndhighest('flanrnn', yahoo_tag='', imdb_tag='', amazon_tag='',
                                              yelp_tag='moreword2vec')
    make_decflip_regression_plot_vs2ndhighest('flanconv', yahoo_tag='convfix', imdb_tag='convfix',
                  amazon_tag='convfix', yelp_tag='convfix')

    make_jsdiv_regression_plot('hanrnn', sample_x, yahoo_tag='postattnfix', imdb_tag='postattnfix', amazon_tag='postattnfix',
                               yelp_tag='postattnfix-2')
    make_jsdiv_regression_plot('hanconv', sample_x, yahoo_tag='convfix', imdb_tag='convfix',
                                              amazon_tag='convfix', yelp_tag='convfix')
    make_jsdiv_regression_plot('flanrnn', sample_x, yahoo_tag='', imdb_tag='', amazon_tag='', yelp_tag='moreword2vec')
    make_jsdiv_regression_plot('flanconv', sample_x, yahoo_tag='convfix', imdb_tag='convfix',
                  amazon_tag='convfix', yelp_tag='convfix')


if __name__ == '__main__':
    main()
