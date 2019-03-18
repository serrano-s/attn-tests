#!/usr/bin/env bash

python3 test_model.py --test-data-file yelp_test.tsv --model-folder-name yelp-flanrnn-fiveclass --gpu 0
python3 test_model.py --test-data-file yelp_test.tsv --model-folder-name yelp-hanconv-fiveclass --gpu 0

