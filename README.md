# Attention Tests

Required to run code:

- python3 (I used python 3.6.3)
- allennlp and all its dependencies
- numpy
- matplotlib
- seaborn

This repo contains code to train and test various models for different
interpretability-related criteria.

When you download the repo, open
`default_directories.py` and set the directories in there to the
locations you'd like to use. With the exception of `base_data_dir` and
`dir_with_config_files`, all other directories will be created by
the scripts that use them when necessary; those two are the only ones
that need to be set up separately.

Suppose that you wanted to train and test a new model from scratch.
Here's the pipeline you'd need to follow:

## Preparing the dataset

If this is the first time you've used this dataset in this
repository, there are a few preprocessing steps that need to happen
first. (If you're training a model on a dataset you've already
set up for use in this repository, you can skip this section.)

First, your data needs to be in a format that the code is
designed to work with. The code expects each file of data to
have one document's information per line, with the different
fields tab-separated. (Therefore, all tabs or newlines in the
original documents need to be replaced as part of the preprocessing.)
The only two fields that the code counts on (though there can
be more) are called `category` and `tokens`, and can appear in
any order (there can also be other fields in the data files; they
will just be ignored). `category`, which is the label for the
document, can be an arbitrary string; it will be automatically
mapped to an int later. `tokens` is just the text of the document
without newlines or tabs.

To get your data into that format, you can either process it yourself,
or reuse some of the code that I wrote in `make_datasets.py` in
`before_model_training/`.
(I haven't gone back and looked at that code in a long time,
though, so use at your own risk.)

Then split up your data into train, dev, and test as three
separate files, each with the same format described above.
(I think I used `subset_data.py` in `before_model_training/`
for this, but again, it's been
a while; you might be better off just doing this on your own.)

Once your data is in the correct format, you need to make
pretrained embeddings and a corresponding allennlp vocabulary
for your dataset.
In `before_model_training/`, open
`get_vocab_and_initialize_word2vec_embeddings_from_dataset.py`
and edit the parameters in lines 23-29 to the desired
values. Then, run
`get_vocab_and_initialize_word2vec_embeddings_from_dataset.sh`
(the bash script, not the python script). We run the bash
script because it splits up the python script into three
separate stages, which, if run as part of the same call to the
python script, crash for memory-related reasons on larger
datasets.

## Training a model

Say you wanted to train a HAN with an RNN encoder on the IMDB
dataset. Then, provided you were on a machine with gpus and
wanted to use gpu 0, you could run the command

```
python3 train_model.py --model hanrnn --dataset-name imdb --gpu 0
```

Options for different models are listed in line 759 of `train_model.py`.
(If you want to run off a gpu, supply -1 as the value for `--gpu`.) The `--model` and `--dataset-name` parameters are used to fetch
the correct config file from the configs directory specified in
`default_directories.py`. Let `{file_ending}` be the file ending
that `--model` (in this case, `hanrnn`) maps to in `corresponding_config_files` (line 30 of `train_model.py`).
Then the config filename that `train_model.py` will look for will be
```
{dir_with_config_files}/{--dataset-name}_{file ending}
```

That's the config file that you need to set up; see example config
files in `configs/` for reference.

More optional parameters are listed in lines 755-781 of `train_model.py`.
A particularly useful one is `--optional-model-tag`; allennlp
doesn't allow a model folder to be overwritten, so if you had
already trained a HANrnn for the IMDB dataset and wanted to train
another one, you could add an optional model tag that `train_model.py`
would append to the new model's specific directory, thus
differentiating it from the old model's directory.

## Extracting information about attention from that model

Once you've got a trained model that you'd like to analyze,
the next step is to run a lot of tests on it and write the
results to a bunch of .csv files in a test-result directory created for this model. This is handled automatically
by `test_model.py`. For the example that we described above
with an IMDB HANrnn, the generated model directory would be
`{base_serialized_models_dir}/imdb-hanrnn`. Assuming you wanted
to get information about how this model's attention works
on all the instances in the data file `{base_data_dir}/imdb_test.tsv`,
you could run the command

```
python3 test_model.py --model-folder-name imdb-hanrnn --test-data-file imdb_test.tsv --gpu 0 --optional-folder-tag testdata
```

(`test_model.py` reads in the same parent directory names as `train_model.py` does,
so those aren't provided in the command.) This will write a bunch
of .csv files to a directory (that this script creates) called
`{base_output_dir}/imdb-hanrnn-testdata/`.
Once again, there are optional
commands listed in lines 1481-1506 of `test_model.py`.
`--optional-folder-tag` is what allows you to have separate test
result directories for the same model on, say, both its test and dev
sets. (If we'd left it off, our results directory would have been `{base_output_dir}/imdb-hanrnn/`.)

## Exploring the test results

After all the .csv files produced in the previous step are created,
you can either work with them yourself, or use scripts provided
here to analyze them.

If you want a summary of some test statistics in text form,
`process_test_outputs.py` is the file you want. To run it for our
IMDB HANrnn, we would run

```
python3 process_test_outputs.py --model-folder-name imdb-hanrnn-testdata
```

Optional arguments are listed in lines 986-997 of `process_test_outputs.py`.

If we instead want to generate figures, you can use `figure_making/figure_maker.py` to
generate figures. This script is MUCH more rough around the edges--
there's a lot of stuff in it that you'd probably need to modify for
use on your generated test results (hard-coded model tags, the expectation
that one of each model has been generated, etc.). But it's here in case it's
helpful.

The same goes for `figure_making/table_maker.py`, which generates a bunch
of LaTeX tables looking at differences in single-weight
decision flips; it's probably not going to be useful outside of
the specific setting I used it in.
