{
    "random_seed": 217,
    "numpy_seed": 735,
    "pytorch_seed": 781,
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
  "datasets_for_vocab_creation": [],
  "vocabulary": {
    "directory_path": "/homes/gws/sofias6/vocabs/yelp-lowercase-vocab-30its"
  },
  "train_data_path": "/homes/gws/sofias6/data/yelp_train_remoutliers.tsv",
  "validation_data_path": "/homes/gws/sofias6/data/yelp_dev.tsv",
    "model": {
        "type": "han",
        "pre_document_encoder_dropout": 0.1,
        "pre_sentence_encoder_dropout": 0.7,
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
        "sentence_encoder": {
           "type": "convolutional_rnn_substitute",
	       "input_size": 200,
           "hidden_size": 100,
        },
         "document_encoder": {
           "type": "convolutional_rnn_substitute",
	       "input_size": 100,
           "hidden_size": 100,
        },
        "word_attention": {
            "type": "simple_han_attention",
            "input_dim": 100,
            "context_vector_dim": 100
        },
        "sentence_attention": {
            "type": "simple_han_attention",
            "input_dim": 100,
            "context_vector_dim": 100
        },
        "output_logit": {
            "input_dim": 100,
            "num_layers": 1,
            "hidden_dims": 5,
            "dropout": 0.7,
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
        "maximum_samples_per_batch": ["list_num_tokens", 6000],  // confirmed that this affects batch size
        "biggest_batch_first": false
    },
     "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 60,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        },
        "shuffle": true
    }
}
