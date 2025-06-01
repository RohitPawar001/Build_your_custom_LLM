model_arguments = {
    "context_length": 1024,    # int: Maximum sequence length the model can process
    "emb_dim": 768,           # int: Embedding dimension (hidden size of the model)
    "vocab_size": 50257,      # int: Size of the vocabulary (number of unique tokens)
    "n_heads": 12,            # int: Number of attention heads in multi-head attention
    "n_layers": 12,           # int: Number of transformer layers/blocks
    "drop_rate": 0.0,         # float: Dropout rate for regularization (0.0 = no dropout)
    "qkv_bias": False         # bool: Whether to use bias in query, key, value projections
}