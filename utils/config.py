"""
Model and training configuration
Based on GPT-2 (124M and 30M parameters)
"""

# Default config (125M model)
vocab_size = 50257  # GPT-2 tokenizer vocabulary size
context_length = 1024  # Maximum sequence length
embedding_dim = 768  # Token embedding dimension
num_heads = 12  # Number of attention heads
num_layers = 12  # Number of transformer blocks
dropout_rate = 0.1  # Dropout probability
qkv_bias = False  # Use bias in QKV projections

# Dataset and DataLoader
stride = context_length
batch_size = 32

# Training
learning_rate = 0.0003
num_epochs = 6


def get_config(model_size: str = "125m") -> dict:
    """
    Get configuration for a specific model size.

    Args:
        model_size: "125m" or "30m"

    Returns:
        Dictionary containing all config values
    """
    configs = {
        "125m": {
            "vocab_size": 50257,
            "context_length": 512,
            "embedding_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dropout_rate": 0.1,
            "qkv_bias": False,
            "batch_size": 32,
            "learning_rate": 0.0003,
            "num_epochs": 6,
        },
        "30m": {
            "vocab_size": 50257,
            "context_length": 512,
            "embedding_dim": 384,
            "num_heads": 6,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "qkv_bias": False,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "num_epochs": 6,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(configs.keys())}")

    return configs[model_size]