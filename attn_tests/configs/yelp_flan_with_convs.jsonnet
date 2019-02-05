{
    "random_seed": 217,
    "numpy_seed": 735,
    "pytorch_seed": 781,
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
    "directory_path": "/homes/gws/sofias6/vocabs/yelp-lowercase-vocab"
  },
  "train_data_path": "/homes/gws/sofias6/data/yelp_train_remoutliers.tsv",
  "validation_data_path": "/homes/gws/sofias6/data/yelp_dev.tsv",
    "model": {
        "type": "flan",
        "pre_document_encoder_dropout": 0.3633964801928899,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "/homes/gws/sofias6/data/yelp_train_lowercase_embeddings.h5",
                    "embedding_dim": 200,
                    "trainable": true
                }
            }
        },
         "document_encoder": {
           "type": "convolutional_rnn_substitute",
	       "input_size": 200,
           "hidden_size": 100,
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
            "dropout": 0.42604996839375936,
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
        "maximum_samples_per_batch": ["num_tokens", 90000],  // confirmed that this affects batch size
        "biggest_batch_first": false
    },
     "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0005037515992918302
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 15,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 1,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        },
        "shuffle": true
    }
}
