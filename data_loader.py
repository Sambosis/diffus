import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import config

# Define the path for the cached tokenizer file from the config module
TOKENIZER_FILE = Path(config.TOKENIZER_PATH)


def get_tokenizer() -> Tokenizer:
    """
    Loads a BPE tokenizer from the path specified in config, or trains a new one
    from the TinyStories dataset if it doesn't exist.

    The tokenizer is trained on a subset of the dataset for efficiency. The size of this
    subset and the vocabulary size are defined in the config module. The trained tokenizer
    is saved to disk for future use.

    After loading or training, the tokenizer is configured to pad and truncate sequences
    to a fixed length (MAX_SEQ_LEN from config).

    Returns:
        Tokenizer: The trained and configured tokenizer instance.
    """
    if TOKENIZER_FILE.exists():
        print(f"Loading existing tokenizer from {TOKENIZER_FILE}...")
        tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    else:
        print(f"Tokenizer not found at '{TOKENIZER_FILE}'. Training a new one...")

        # Initialize a new BPE tokenizer model
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # The trainer is responsible for training the tokenizer model
        trainer = BpeTrainer(
            vocab_size=config.TOKENIZER_VOCAB_SIZE,
            special_tokens=["[UNK]", "[PAD]"]  # [PAD] is crucial for batching
        )

        # Create a Python generator to stream data for tokenizer training.
        # This is memory-efficient as it doesn't load the whole dataset at once.
        def get_training_corpus():
            # Use streaming mode for memory efficiency during tokenizer training
            dataset_stream = load_dataset(config.DATASET_NAME, split="train", streaming=True)
            # Use only a subset of the data as defined in config for faster training
            dataset_subset = dataset_stream.take(config.TOKENIZER_TRAIN_SAMPLE_SIZE)
            for example in dataset_subset:
                # The trainer expects an iterator of text strings
                yield example['text']

        # Train the tokenizer
        print(f"Training tokenizer on {config.TOKENIZER_TRAIN_SAMPLE_SIZE} samples...")
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

        # Ensure the directory exists and save the trained tokenizer
        TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(TOKENIZER_FILE))
        print(f"Tokenizer trained and saved to {TOKENIZER_FILE}")

    # Configure padding and truncation to ensure all sequences have the same length
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("PAD token not found in tokenizer vocabulary! Check trainer special_tokens.")

    tokenizer.enable_padding(
        direction='right',
        pad_id=pad_id,
        pad_token='[PAD]',
        length=config.MAX_SEQ_LEN
    )
    tokenizer.enable_truncation(max_length=config.MAX_SEQ_LEN)

    return tokenizer


class TinyStoriesTextDataset(Dataset):
    """
    PyTorch Dataset for the TinyStories dataset.

    This class wraps the Hugging Face dataset, making it compatible with PyTorch's
    DataLoader. It loads a non-streaming subset of the data (size defined by
    `config.DATASET_TRAIN_SAMPLE_SIZE`) for efficient indexed access required by `__getitem__`.
    Each text sample is tokenized on-the-fly.
    """
    def __init__(self, tokenizer: Tokenizer, split: str = 'train'):
        """
        Args:
            tokenizer (Tokenizer): The tokenizer instance to use for encoding text.
            split (str): The dataset split to use (e.g., 'train').
        """
        super().__init__()
        self.tokenizer = tokenizer

        # Define the slice of the dataset to use for model training.
        # Using a slice like `train[:100000]` loads only that part into memory.
        slice_def = f"{split}"
        if config.DATASET_TRAIN_SAMPLE_SIZE > 0:
            slice_def += f"[:{config.DATASET_TRAIN_SAMPLE_SIZE}]"

        print(f"Loading '{slice_def}' of {config.DATASET_NAME} dataset for model training...")
        self.dataset = load_dataset(config.DATASET_NAME, split=slice_def)
        print(f"Dataset loaded with {len(self.dataset)} samples.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves and tokenizes a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the token IDs under the key 'input_ids'.
                                     This format is compatible with the default collate_fn and
                                     makes batch access in the trainer cleaner (batch['input_ids']).
        """
        text = self.dataset[idx]['text']

        # The tokenizer handles truncation and padding automatically based on the
        # configuration set in get_tokenizer().
        encoded = self.tokenizer.encode(text)

        # Return as a dictionary to make it compatible with the training loop's expectation
        return {'input_ids': torch.tensor(encoded.ids, dtype=torch.long)}


def get_dataloader(tokenizer: Tokenizer, batch_size: int) -> DataLoader:
    """
    Initializes the dataset and creates a PyTorch DataLoader.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for the dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    print("Initializing DataLoader...")
    dataset = TinyStoriesTextDataset(tokenizer=tokenizer, split='train')

    # DataLoader handles batching, shuffling, and multi-threaded data loading.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,  # Speeds up data transfer to the GPU
        drop_last=True    # Drops the last incomplete batch, ensuring all batches have the same size.
    )
    print("DataLoader initialization complete.")
    return dataloader