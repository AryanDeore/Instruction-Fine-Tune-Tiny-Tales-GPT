"""Custom collate function for instruction fine-tuning batches."""

import torch
from typing import List, Dict


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of tokenized examples.

    Pads sequences to batch max length, creates attention masks,
    and shifts targets by 1 for language modeling.

    Args:
        batch: List of dicts with 'input_ids' key (torch.Tensor)

    Returns:
        Dict with:
            - input_ids: padded token IDs (batch_size, seq_len)
            - attention_mask: mask indicating real vs padded tokens (batch_size, seq_len)
            - labels: targets for training, shifted by 1 (batch_size, seq_len)
    """
    # Extract input_ids from batch
    input_ids_list = [ex["input_ids"] for ex in batch]

    # Find max length in batch
    max_length = max(len(ids) for ids in input_ids_list)

    # Pad all sequences to max_length
    padded_input_ids = []
    attention_masks = []

    for input_ids in input_ids_list:
        # Calculate padding needed
        padding_length = max_length - len(input_ids)

        # Pad with 0s
        padded = torch.cat([
            input_ids,
            torch.zeros(padding_length, dtype=torch.long)
        ])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(padding_length, dtype=torch.long)
        ])

        padded_input_ids.append(padded)
        attention_masks.append(attention_mask)

    # Stack into batch tensors
    input_ids = torch.stack(padded_input_ids)  # (batch_size, max_length)
    attention_mask = torch.stack(attention_masks)  # (batch_size, max_length)

    # Create labels: shift input_ids by 1 to the left
    # labels = [input_ids[1], input_ids[2], ..., input_ids[-1], -100]
    labels = input_ids.clone()
    labels = torch.roll(labels, shifts=-1, dims=1)

    # Mark padding positions as -100 (PyTorch's ignore_index for cross-entropy)
    # Set the last position to -100 (shifted out of bounds)
    labels[:, -1] = -100

    # Also mask positions where original had padding
    for i in range(len(batch)):
        padding_start = len(input_ids_list[i])
        labels[i, padding_start:] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    """Test the collate function."""

    print("=" * 70)
    print("Custom Collate Function Test")
    print("=" * 70)

    # Create mock batch with different lengths
    batch = [
        {"input_ids": torch.tensor([1, 2, 3, 4, 5])},
        {"input_ids": torch.tensor([6, 7, 8])},
        {"input_ids": torch.tensor([9, 10, 11, 12, 13, 14])},
    ]

    print("\nInput batch (different lengths):")
    for i, ex in enumerate(batch):
        print(f"  Example {i}: {ex['input_ids'].tolist()}")

    # Collate
    collated = custom_collate_fn(batch)

    print(f"\nCollated batch (max_length={collated['input_ids'].shape[1]}):")
    print(f"\ninput_ids shape: {collated['input_ids'].shape}")
    print(f"{collated['input_ids']}")

    print(f"\nattention_mask shape: {collated['attention_mask'].shape}")
    print(f"{collated['attention_mask']}")

    print(f"\nlabels shape: {collated['labels'].shape}")
    print(f"{collated['labels']}")

    # Verify correctness
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    print("\nExample 0 analysis:")
    print(f"  Original: {batch[0]['input_ids'].tolist()}")
    print(f"  Padded input_ids: {collated['input_ids'][0].tolist()}")
    print(f"  Attention mask: {collated['attention_mask'][0].tolist()}")
    print(f"  Labels (targets): {collated['labels'][0].tolist()}")
    print(f"  Explanation:")
    print(f"    - input_ids[0] → labels[0] shift: 1→2 (next token)")
    print(f"    - input_ids[4] → labels[4] shift: 5→0 (next token is padding)")
    print(f"    - labels[5] = -100 (padding, ignored in loss)")

    print("\n✓ Test complete")
