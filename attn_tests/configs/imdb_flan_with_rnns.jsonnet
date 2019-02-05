{
   "random_seed": 370,
   "numpy_seed": 944,
   "pytorch_seed": 972,
   "dataset_reader": {
        "type": "textcat",
        "segment_sentences": false,
        "word_tokenizer": "word",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
  "datasets_for_vocab_creation": [],
  "vocabulary": {
    "directory_path": "/homes/gws/sofias6/vocabs/imdb-lowercase-vocab"
  },
  "train_data_path": "/homes/gws/sofias6/data/imdb_train.tsv",
  "validation_data_path": "/homes/gws/sofias6/data/imdb_dev.tsv",
    "model": {
        "type": "flan",
        "pre_document_encoder_dropout": 0.44446746096594764,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "/homes/gws/sofias6/data/imdb_train_lowercase_embeddings.h5",
                    "embedding_dim": 200,
                    "trainable": true
                }
            }
        },
        "document_encoder": {
           "type": "gru",
           "num_layers": 1,
           "bidirectional": true,
	       "input_size": 200,
           "hidden_size": 50,
        },
        "word_attention": {
            "type": "simple_han_attention",
            "input_dim": 100,
            "context_vector_dim": 100
        },
        "output_logit": {
            "input_dim": 100,
            "num_layers": 1,
            "hidden_dims": 10,
            "dropout": 0.3457355626352195,
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
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 64,
        "maximum_samples_per_batch": ["num_tokens", 20000],  // confirmed that this affects batch size
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
        "cuda_device": 2,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        },
        "shuffle": true
    }
}
