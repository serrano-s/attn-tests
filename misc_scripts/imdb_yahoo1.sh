#!/usr/bin/env bash

python3 test_model_postoriginal.py --model-folder-name imdb-hanconv-convfix --test-data-file imdb_test.tsv --gpu 2
python3 test_model_postoriginal.py --model-folder-name imdb-hanrnn-postattnfix --test-data-file imdb_test.tsv --gpu 2
python3 test_model_postoriginal.py --model-folder-name yahoo10cat-flanconv-convfix --test-data-file yahoo10cat_test.tsv --gpu 2
python3 test_model_postoriginal.py --model-folder-name yahoo10cat-flanrnn --test-data-file yahoo10cat_test.tsv --gpu 2