{
    "dataset_reader": {
        "type": "textcat",
        "segment_sentences": true,
        "word_tokenizer": "word",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
  "datasets_for_vocab_creation": ["train"],  // this assumes you want to create the vocabulary from the training set-- feel free to ask me if you want to do something else
  "train_data_path": "/homes/gws/sofias6/data/imdb_train.tsv",  // replace this with your training data file, in the format listed at the top of textcat.TextCatReader (in textcat_reader.py)
  "validation_data_path": "/homes/gws/sofias6/data/imdb_dev.tsv",  // replace this with your dev data file, in the format listed at the top of textcat.TextCatReader (in textcat_reader.py)
    "model": {
        "type": "han",
        "pre_sentence_encoder_dropout": 0.4,
        "pre_document_encoder_dropout": 0.4,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
                    "embedding_dim": 50,
                    "trainable": true
                }
            }
        },
        "sentence_encoder": {
           "type": "gru",
           "num_layers": 1,
           "bidirectional": true,
	       "input_size": 50,
           "hidden_size": 50,
        },
         "document_encoder": {
           "type": "gru",
           "num_layers": 1,
           "bidirectional": true,
	       "input_size": 100,
           "hidden_size": 50,
        },
        "word_attention": {
            "type": "intra_sentence_attention",
            "input_dim": 100,
            "combination": "2",
            "similarity_function": {
                "type": "han_paper",
                "input_dim": 100,
                "context_vect_dim": 100
            },
        },
        "sentence_attention": {
            "type": "intra_sentence_attention",
            "input_dim": 100,
            "combination": "2",
            "similarity_function": {
                "type": "han_paper",
                "input_dim": 100,
                "context_vect_dim": 100
            },
        },
        "output_logit": {
            "input_dim": 100,
            "num_layers": 1,
            "hidden_dims": 10,
            "dropout": 0.4,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "extended_bucket",
        "sorting_keys": [["sentences", "num_sentences"], ["tokens", "list_num_tokens"]],
        "batch_size": 64,
        "maximum_samples_per_batch": ["list_num_tokens", 2000],  // confirmed that this affects batch size
        "biggest_batch_first": false
    },
     "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 15,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1,  // swap this to a gpu id on the machine (such as 0 or 1) if you want to use a gpu
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        },
        "shuffle": true
    }
}
