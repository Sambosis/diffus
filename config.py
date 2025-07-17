# config.py
"""
This file contains all the hyperparameters and configuration settings for the project.
Centralizing these values makes it easy to experiment and tune the model and training process.
"""

import torch

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data and Tokenizer Configuration ---
# The Hugging Face dataset to use
DATASET_NAME = "Vezora/Tested-143k-Python-Alpaca"
# The name of the column in the dataset that contains the text to be trained on.
DATASET_TEXT_COLUMN = "output"
# The number of text samples to use for training the BPE tokenizer.
# A smaller number speeds up initialization.
TOKENIZER_TRAIN_SAMPLE_SIZE = 40000
# The number of text samples to use for training the diffusion model.
# Set to -1 to use the entire dataset.
DATASET_TRAIN_SAMPLE_SIZE = 100000 # Use 100k samples for a reasonable training time
# The desired vocabulary size for our custom BPE tokenizer
TOKENIZER_VOCAB_SIZE = 16384*4  # 32k tokens, a common size for many NLP tasks
# The file path to save/load the trained tokenizer
TOKENIZER_PATH = "./bpe_tokenizer.json"
# The maximum sequence length for the model. Sentences will be padded or truncated to this length.
MAX_SEQ_LEN = 128


# --- Diffusion Model Configuration ---
# The dimensionality of the token embeddings and the model's hidden states
EMBED_DIM = 512 # d_model
# The number of attention heads in the Transformer's multi-head attention layers
NUM_HEADS = 8
# The number of Transformer encoder layers
NUM_LAYERS = 6
# The dimensionality of the feed-forward network model
DIM_FEEDFORWARD = 2048
# The dropout rate to use in the Transformer model
DROPOUT = 0.1


# --- Diffusion Process Configuration ---
# The total number of diffusion steps (T)
TIMESTEPS = 500
# The starting value of beta for the linear noise schedule
BETA_START = 0.0001
# The ending value of beta for the linear noise schedule
BETA_END = 0.02


# --- Training Configuration ---
# The total number of training epochs
EPOCHS = 250
# The batch size for training
BATCH_SIZE = 1024
# The learning rate for the Adam optimizer
LEARNING_RATE = 2e-4
# Number of workers for the DataLoader
NUM_WORKERS = 8


# --- GUI and Logging Configuration ---
# How often to update the loss plot and stats in the GUI (in training steps)
LOG_INTERVAL = 100
# How often to generate a sample sentence and display it in the GUI (in training steps)
SAMPLE_INTERVAL = 500