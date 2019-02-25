#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name imdb-hanconv-convfix --test-data-file imdb_test.tsv --gpu 0
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name imdb-hanrnn-postattnfix --test-data-file imdb_test.tsv --gpu 0
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name yahoo10cat-hanconv-convfix --test-data-file yahoo10cat_test.tsv --gpu 0
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name yahoo10cat-hanrnn-postattnfix --test-data-file yahoo10cat_test.tsv --gpu 0

CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name imdb-flanconv-convfix --test-data-file imdb_test.tsv --gpu 0
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name imdb-flanrnn --test-data-file imdb_test.tsv --gpu 0
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name yahoo10cat-flanconv-convfix --test-data-file yahoo10cat_test.tsv --gpu 0
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --model-folder-name yahoo10cat-flanrnn --test-data-file yahoo10cat_test.tsv --gpu 0
