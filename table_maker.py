from default_directories import tex_files_dir
from default_directories import base_output_dir
import os
from process_test_outputs import load_in_data_table
import numpy as np
from test_model import first_v_second_fname
from test_model import dec_flip_stats_fname
from test_model import rand_results_fname  # this one also has variable-length fields now *shrug*
from test_model import unchanged_fname  # index this one last because its fields are of variable length
from test_model import grad_based_stats_fname
try:
    from test_model import dec_flip_rand_nontop_stats_fname
except:
    from test_model import dec_flip_rand_nontopbyattn_stats_fname as dec_flip_rand_nontop_stats_fname
from test_model import attn_div_from_unif_fname
from test_model import gradsignmult_based_stats_fname
from test_model import dec_flip_rand_nontopbygrad_stats_fname
from test_model import dec_flip_rand_nontopbygradmult_stats_fname
from test_model import dec_flip_rand_nontopbygradsignmult_stats_fname


def get_filenames_for_subdir(mid_dir):
    global base_output_dir
    if not base_output_dir.endswith('/'):
        base_output_dir += '/'
    if not mid_dir.endswith('/'):
        mid_dir += '/'
    return base_output_dir + mid_dir + first_v_second_fname, base_output_dir + mid_dir + dec_flip_stats_fname, \
           base_output_dir + mid_dir + rand_results_fname, base_output_dir + mid_dir + unchanged_fname, \
           base_output_dir + mid_dir + grad_based_stats_fname, base_output_dir + mid_dir + dec_flip_rand_nontop_stats_fname, \
           base_output_dir + mid_dir + attn_div_from_unif_fname, \
           base_output_dir + mid_dir + gradsignmult_based_stats_fname, \
           base_output_dir + mid_dir + dec_flip_rand_nontopbygrad_stats_fname, \
           base_output_dir + mid_dir + dec_flip_rand_nontopbygradmult_stats_fname, \
           base_output_dir + mid_dir + dec_flip_rand_nontopbygradsignmult_stats_fname


yahoo_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-hanrnn-postattnfix'))
imdb_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('imdb-hanrnn-postattnfix'))
amazon_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('amazon-hanrnn-fiveclassround2-4'))
yelp_hanrnn_table = load_in_data_table(*get_filenames_for_subdir('yelp-hanrnn-fiveclassround2-5smallerstep'))
yahoo_hanconv_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-hanconv-convfix'))
imdb_hanconv_table = load_in_data_table(*get_filenames_for_subdir('imdb-hanconv-convfix'))
amazon_hanconv_table = load_in_data_table(*get_filenames_for_subdir('amazon-hanconv-fiveclass'))
yelp_hanconv_table = load_in_data_table(*get_filenames_for_subdir('yelp-hanconv-fiveclass'))
yahoo_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-flanrnn'))
imdb_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('imdb-flanrnn'))
amazon_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('amazon-flanrnn-fiveclass'))
yelp_flanrnn_table = load_in_data_table(*get_filenames_for_subdir('yelp-flanrnn-fiveclass'))
yahoo_flanconv_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-flanconv-convfix'))
imdb_flanconv_table = load_in_data_table(*get_filenames_for_subdir('imdb-flanconv-convfix'))
amazon_flanconv_table = load_in_data_table(*get_filenames_for_subdir('amazon-flanconv-fiveclass'))
yelp_flanconv_table = load_in_data_table(*get_filenames_for_subdir('yelp-flanconv-fiveclass'))
yahoo_flanencless_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-flan_encless'))
imdb_flanencless_table = load_in_data_table(*get_filenames_for_subdir('imdb-flan_encless'))
amazon_flanencless_table = load_in_data_table(*get_filenames_for_subdir('amazon-flan_encless'))
yelp_flanencless_table = load_in_data_table(*get_filenames_for_subdir('yelp-flan_encless'))
yahoo_hanencless_table = load_in_data_table(*get_filenames_for_subdir('yahoo10cat-han_encless'))
imdb_hanencless_table = load_in_data_table(*get_filenames_for_subdir('imdb-han_encless'))
amazon_hanencless_table = load_in_data_table(*get_filenames_for_subdir('amazon-han_encless'))
yelp_hanencless_table = load_in_data_table(*get_filenames_for_subdir('yelp-han_encless'))


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
        DEC_FLIP_ZERO_2NDHIGHESTGRADMULT, NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE, NONTOP_RAND_ZEROED_WEIGHT, \
        NONTOP_RAND_KL_DIV, NONTOP_RAND_JS_DIV, ATTN_KL_DIV_FROM_UNIF, ATTN_JS_DIV_FROM_UNIF, \
        ID_DUPLICATE_2, ATTN_SEQ_LEN_DUPLICATE3_FOR_TESTING, NEEDED_REM_TOP_X_FOR_DECFLIP_GRADSIGNMULT, \
        NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP_GRADSIGNMULT, NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP_GRADSIGNMULT, \
        KL_DIV_ZERO_HIGHESTGRADSIGNMULT, JS_DIV_ZERO_HIGHESTGRADSIGNMULT, DEC_FLIP_ZERO_HIGHESTGRADSIGNMULT, \
        KL_DIV_ZERO_2NDHIGHESTGRADSIGNMULT, JS_DIV_ZERO_2NDHIGHESTGRADSIGNMULT, DEC_FLIP_ZERO_2NDHIGHESTGRADSIGNMULT, \
        NONTOPBYGRAD_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE, NONTOPBYGRAD_RAND_ZEROED_WEIGHT, NONTOPBYGRAD_RAND_KL_DIV, \
        NONTOPBYGRAD_RAND_JS_DIV, NONTOPBYGRADMULT_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE, \
        NONTOPBYGRADMULT_RAND_ZEROED_WEIGHT, NONTOPBYGRADMULT_RAND_KL_DIV, NONTOPBYGRADMULT_RAND_JS_DIV, \
        NONTOPBYGRADSIGNMULT_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE, NONTOPBYGRADSIGNMULT_RAND_ZEROED_WEIGHT, \
        NONTOPBYGRADSIGNMULT_RAND_KL_DIV, NONTOPBYGRADSIGNMULT_RAND_JS_DIV, ATTN_SEQ_LEN, \
        NEEDED_REM_TOP_FRAC_X_FOR_DECFLIP, NEEDED_REM_TOP_X_FOR_DECFLIP, ID, \
        NEEDED_REM_TOP_PROBMASS_FOR_DECFLIP, EXTRACTED_SINGLE_ATTN_WEIGHT_START, JS_DIV_ZERO_HIGHEST, \
        JS_DIV_ZERO_2NDHIGHEST, DEC_FLIP_ZERO_HIGHEST, DEC_FLIP_ZERO_2NDHIGHEST

assert LAST_IND_OF_OUTPUT_CLASSES is not None


if not os.path.isdir(tex_files_dir):
    os.makedirs(tex_files_dir)
if not tex_files_dir.endswith('/'):
    tex_files_dir += '/'


def get_rounded_string_of_num(num):
    return "{:.1f}".format(round(100 * num, 1))


def generate_table_tex_file(caption, yahoo_both_onlytop_onlyrand_neither, imdb_both_onlytop_onlyrand_neither,
                            amazon_both_onlytop_onlyrand_neither, yelp_both_onlytop_onlyrand_neither,
                            vertical_text, tex_label, filename):
    # numbers for each dataset should sum to 1
    str_to_ret = '\subfloat['
    str_to_ret += caption
    str_to_ret += r"""]{
\begin{tabular}{clccllcc}
\multicolumn{1}{l}{}                &      & \multicolumn{6}{c}{\textbf{Remove random: Decision flip?}}                                                                                                                                                                                                                  \\
                                          &                           & \multicolumn{2}{c}{Yahoo}                                                                       &  &                           & \multicolumn{2}{c}{IMDB}                                                                        \\
                                          &                           & Yes                                           & No                                           &  &                           & Yes                                           & No                                           \\ \cline{3-4} \cline{7-8} 
                                          & \multicolumn{1}{r|}{Yes} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(yahoo_both_onlytop_onlyrand_neither[0])
    str_to_ret += '}                         & \multicolumn{1}{c|}{\cellcolor[HTML]{DAE8FC}'
    str_to_ret += get_rounded_string_of_num(yahoo_both_onlytop_onlyrand_neither[1])
    str_to_ret += r"""} &  & \multicolumn{1}{r|}{Yes} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(imdb_both_onlytop_onlyrand_neither[0])
    str_to_ret += r"""}                         & \multicolumn{1}{c|}{\cellcolor[HTML]{DAE8FC}"""
    str_to_ret += get_rounded_string_of_num(imdb_both_onlytop_onlyrand_neither[1])
    str_to_ret += r"""} \\ \cline{3-4} \cline{7-8} 
                                          & \multicolumn{1}{r|}{No} & \multicolumn{1}{c|}{\cellcolor[HTML]{F8A102}"""
    str_to_ret += get_rounded_string_of_num(yahoo_both_onlytop_onlyrand_neither[2])
    str_to_ret += r"""} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(yahoo_both_onlytop_onlyrand_neither[3])
    str_to_ret += r"""}                         &  & \multicolumn{1}{r|}{No} & \multicolumn{1}{c|}{\cellcolor[HTML]{F8A102}"""
    str_to_ret += get_rounded_string_of_num(imdb_both_onlytop_onlyrand_neither[2])
    str_to_ret += r"""} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(imdb_both_onlytop_onlyrand_neither[3])
    str_to_ret += r"""}                         \\ \cline{3-4} \cline{7-8} 
                                          &                           & \multicolumn{1}{l}{}                           & \multicolumn{1}{l}{}                           &  &                           & \multicolumn{1}{l}{}                           & \multicolumn{1}{l}{}                           \\
                                          &                           & \multicolumn{2}{c}{Amazon}                                                                      &  &                           & \multicolumn{2}{c}{Yelp}                                                                        \\
                                          &                           & Yes                                           & No                                           &  &                           & Yes                                           & No                                           \\ \cline{3-4} \cline{7-8} 
                                          & \multicolumn{1}{r|}{Yes} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(amazon_both_onlytop_onlyrand_neither[0])
    str_to_ret += r"""}                         & \multicolumn{1}{c|}{\cellcolor[HTML]{DAE8FC}"""
    str_to_ret += get_rounded_string_of_num(amazon_both_onlytop_onlyrand_neither[1])
    str_to_ret += r"""} &  & \multicolumn{1}{r|}{Yes} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(yelp_both_onlytop_onlyrand_neither[0])
    str_to_ret += r"""}                         & \multicolumn{1}{c|}{\cellcolor[HTML]{DAE8FC}"""
    str_to_ret += get_rounded_string_of_num(yelp_both_onlytop_onlyrand_neither[1])
    str_to_ret += r"""} \\ \cline{3-4} \cline{7-8} 
\multirow{-9}{*}{\rotatebox[origin=c]{90}{\textbf{"""
    str_to_ret += vertical_text
    str_to_ret += r"""}}} & \multicolumn{1}{r|}{No} & \multicolumn{1}{c|}{\cellcolor[HTML]{F8A102}"""
    str_to_ret += get_rounded_string_of_num(amazon_both_onlytop_onlyrand_neither[2])
    str_to_ret += r"""} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(amazon_both_onlytop_onlyrand_neither[3])
    str_to_ret += r"""}                         &  & \multicolumn{1}{r|}{No} & \multicolumn{1}{c|}{\cellcolor[HTML]{F8A102}"""
    str_to_ret += get_rounded_string_of_num(yelp_both_onlytop_onlyrand_neither[2])
    str_to_ret += r"""} & \multicolumn{1}{c|}{"""
    str_to_ret += get_rounded_string_of_num(yelp_both_onlytop_onlyrand_neither[3])
    str_to_ret += r"""}                         \\ \cline{3-4} \cline{7-8} 
\label{"""
    str_to_ret += tex_label
    str_to_ret += r"""}
\end{tabular}
}"""
    with open(filename, 'w') as f:
        f.write(str_to_ret + '\n')
    print("Wrote " + filename)


def get_vsrand_2x2_decflip_jointdist_as_returned_vals(table, nonrand_column, corresponding_rand_column):
    table = table[table[:, ATTN_SEQ_LEN] > 1]  # get rid of seqs of length 1 for this
    rand_flipped_decision = (table[:, corresponding_rand_column] != -1)
    top_flipped_decision = (table[:, nonrand_column] != -1)
    return get_2x2_decflip_jointdist(rand_flipped_decision, top_flipped_decision, table)


def get_2x2_decflip_jointdist(vs_flipped_decision, top_flipped_decision, table):
    both_flipped = np.sum(np.logical_and(vs_flipped_decision, top_flipped_decision))
    neither_flipped = np.sum(np.logical_and(np.logical_not(vs_flipped_decision),
                                            np.logical_not(top_flipped_decision)))
    only_top_flipped = np.sum(np.logical_and(np.logical_not(vs_flipped_decision),
                                             top_flipped_decision))
    only_rand_flipped = np.sum(np.logical_and(vs_flipped_decision,
                                              np.logical_not(top_flipped_decision)))
    assert both_flipped + neither_flipped + only_top_flipped + only_rand_flipped == table.shape[0]
    return [both_flipped / table.shape[0], only_top_flipped / table.shape[0],
            only_rand_flipped / table.shape[0], neither_flipped / table.shape[0]]


def make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, model_name):
    table_attribute_name = model_name.lower()[:-1]
    if table_attribute_name.endswith('noenc'):
        table_attribute_name = table_attribute_name[:table_attribute_name.rfind('noenc')] + 'encless'
    filename = tex_files_dir + 'decflip_rand_' + filename_suffix + '_sub_' + model_name.lower()[:-1] + '.tex'
    yahoo_table = globals()['yahoo_' + table_attribute_name + '_table']
    yahoo_vals = get_vsrand_2x2_decflip_jointdist_as_returned_vals(yahoo_table, i_star_col, rand_col)
    imdb_table = globals()['imdb_' + table_attribute_name + '_table']
    imdb_vals = get_vsrand_2x2_decflip_jointdist_as_returned_vals(imdb_table, i_star_col, rand_col)
    amazon_table = globals()['amazon_' + table_attribute_name + '_table']
    amazon_vals = get_vsrand_2x2_decflip_jointdist_as_returned_vals(amazon_table, i_star_col, rand_col)
    yelp_table = globals()['yelp_' + table_attribute_name + '_table']
    yelp_vals = get_vsrand_2x2_decflip_jointdist_as_returned_vals(yelp_table, i_star_col, rand_col)
    generate_table_tex_file(model_name, yahoo_vals, imdb_vals,
                            amazon_vals, yelp_vals,
                            vertical_label,
                            model_name.lower()[:-1] + '-rand-decflip-table-' + filename_suffix, filename)


def make_set_of_six_tables(i_star_col, rand_col, vertical_label, filename_suffix):
    make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, 'HANrnns')
    make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, 'HANconvs')
    make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, 'HANnoencs')
    make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, 'FLANrnns')
    make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, 'FLANconvs')
    make_single_table(i_star_col, rand_col, vertical_label, filename_suffix, 'FLANnoencs')


def make_tables():
    make_set_of_six_tables(DEC_FLIP_ZERO_HIGHEST, NONTOP_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE,
                           r'Remove $i^\ast$: Decision flip?', 'attn')
    make_set_of_six_tables(DEC_FLIP_ZERO_HIGHESTGRAD, NONTOPBYGRAD_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE,
                           r'Remove $i^\ast_g$: Decision flip?', 'grad')
    make_set_of_six_tables(DEC_FLIP_ZERO_HIGHESTGRADMULT, NONTOPBYGRADMULT_RAND_CAUSED_DECFLIP_IF_NOT_NEGONE,
                           r'Remove $i^\ast_p$: Decision flip?', 'gradmult')


if __name__ == '__main__':
    make_tables()
