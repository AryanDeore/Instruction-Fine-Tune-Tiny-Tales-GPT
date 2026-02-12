"""InstructionDataset: Loads curated dataset and tokenizes at initialization."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken
from typing import Dict


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning.

    Loads curated TinyStories-Instruct dataset and tokenizes each example
    at initialization. Concatenates instruction + response + <|endoftext|>.
    """

    def __init__(self, split: str = "train", max_length: int = 512, repo_id: str = "0rn0/tinystories-instruct-balanced"):
        """Initialize dataset.

        Args:
            split: "train" or "validation"
            max_length: Maximum token sequence length (truncate if longer)
            repo_id: HuggingFace dataset repository ID
        """
        self.max_length = max_length
        self.split = split

        # Load tokenizer (GPT-2 tokenizer)
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Load dataset from HuggingFace
        print(f"Loading {split} split from {repo_id}...")
        ds = load_dataset(repo_id, split=split)

        # Tokenize all examples at initialization
        print(f"Tokenizing {len(ds)} examples...")
        self.tokenized_data = []

        for i, example in enumerate(ds):
            if i % 10000 == 0:
                print(f"  Tokenized {i:,} / {len(ds):,}")

            # Concatenate instruction + response + endoftext token
            full_text = example["instruction"] + example["response"] + "<|endoftext|>"

            # Tokenize (allow <|endoftext|> as special token)
            token_ids = self.tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})

            # Truncate to max_length
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            self.tokenized_data.append(torch.tensor(token_ids, dtype=torch.long))

        print(f"✓ Loaded {len(self.tokenized_data)} examples")

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokenized example.

        Args:
            idx: Example index

        Returns:
            Dict with key 'input_ids' containing token IDs tensor
        """
        return {
            "input_ids": self.tokenized_data[idx]
        }


if __name__ == "__main__":
    import sys

    # Test loading and tokenizing
    print("=" * 70)
    print("InstructionDataset Test")
    print("=" * 70)

    # Load dataset
    dataset = InstructionDataset(split="train", max_length=512)

    print(f"\nDataset size: {len(dataset):,} examples")

    # Print first 5 examples with tokenization
    print(f"\nShowing first 5 examples:\n")

    # Load raw dataset to show original text
    raw_ds = load_dataset("0rn0/tinystories-instruct-balanced", split="train")
    tokenizer = tiktoken.get_encoding("gpt2")

    for idx in range(5):
        raw_ex = raw_ds[idx]
        tokenized_ex = dataset[idx]

        full_text = raw_ex["instruction"] + raw_ex["response"] + "<|endoftext|>"
        token_ids = tokenized_ex["input_ids"]

        print(f"\n{'─' * 70}")
        print(f"Example {idx + 1}")
        print(f"{'─' * 70}")

        print(f"\nInstruction:\n{raw_ex['instruction'][:100]}...")
        print(f"\nResponse:\n{raw_ex['response'][:100]}...")
        print(f"\nEnding: {raw_ex['ending']}")

        print(f"\nFull text (first 150 chars):\n{full_text[:150]}...")

        print(f"\nTokenized ({len(token_ids)} tokens):")
        print(f"  Token IDs: {token_ids[:20].tolist()}... (showing first 20)")

        # Decode back to verify
        decoded = tokenizer.decode(token_ids.tolist())
        print(f"\nDecoded (first 150 chars):\n{decoded[:150]}...")

    # Print statistics
    print(f"\n{'─' * 70}")
    print("Dataset Statistics")
    print(f"{'─' * 70}")

    lengths = [len(ex["input_ids"]) for ex in [dataset[i] for i in range(min(1000, len(dataset)))]]
    print(f"Average sequence length: {sum(lengths) / len(lengths):.1f} tokens")
    print(f"Min length: {min(lengths)} tokens")
    print(f"Max length: {max(lengths)} tokens")

    print(f"\n✓ Test complete")
