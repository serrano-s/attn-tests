#!/usr/bin/env bash

python3 test_model.py --test-data-file yelp_test.tsv --model-folder-name yelp-hanrnn-fiveclassround2-5smallerstep --gpu 2
python3 test_model.py --test-data-file yelp_test.tsv --model-folder-name yelp-flanconv-fiveclass --gpu 2
python3 test_model.py --test-data-file amazon_test.tsv --model-folder-name amazon-hanrnn-fiveclassround2-4 --gpu 2