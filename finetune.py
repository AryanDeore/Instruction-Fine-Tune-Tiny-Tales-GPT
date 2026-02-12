"""Instruction fine-tuning script for GPT-2 model.

Fine-tunes a pretrained GPT-2 model on instruction-following tasks
using the TinyStories instruction dataset with 50-50 happy/sad split.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from data.dataloader import create_dataloader
from models.gpt2 import GPT2
from utils.config import get_config


def load_pretrained_model(checkpoint_path: str, device: str = "cpu") -> GPT2:
    """Load pretrained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        GPT2 instance with loaded weights
    """
    print(f"Loading pretrained model from {checkpoint_path}...")

    # Initialize model with config
    config = get_config("30m")
    model = GPT2(**config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(
        f"✓ Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model


def compute_loss(model_output, labels, ignore_index: int = -100) -> torch.Tensor:
    """Compute language modeling loss.

    Args:
        model_output: Model logits (batch_size, seq_len, vocab_size)
        labels: Target token IDs (batch_size, seq_len), with -100 for ignored positions
        ignore_index: Token ID to ignore in loss computation

    Returns:
        Scalar loss tensor
    """
    batch_size, seq_len, vocab_size = model_output.shape

    # Flatten batch and sequence dimensions
    logits = model_output.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels_flat, ignore_index=ignore_index)
    return loss


def train_epoch(
    model: GPT2,
    train_loader,
    optimizer: AdamW,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch.

    Args:
        model: GPT model to train
        train_loader: Training data loader
        optimizer: Adam optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)

    for batch in pbar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)

        # Compute loss
        loss = compute_loss(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: GPT2,
    val_loader,
    device: str,
) -> float:
    """Evaluate model on validation set.

    Args:
        model: GPT model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = compute_loss(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(
    model: GPT2, optimizer: AdamW, epoch: int, loss: float, checkpoint_dir: str
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"finetune_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    print(f"✓ Saved checkpoint: {checkpoint_path}")


def main(
    pretrained_checkpoint: str = "checkpoints/30M_model.pt",
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 2,
    max_length: int = 512,
    checkpoint_dir: str = "checkpoints/finetune",
    device: Optional[str] = None,
):
    """Main training script.

    Args:
        pretrained_checkpoint: Path to pretrained model checkpoint
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of epochs to train
        max_length: Maximum sequence length
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on ("cpu" or "cuda")
    """
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num epochs: {num_epochs}")
    print(f"Max length: {max_length}")
    print("-" * 70)

    # Load model
    model = load_pretrained_model(pretrained_checkpoint, device=device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        split="train",
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )
    val_loader = create_dataloader(
        split="validation",
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    print("\nOptimizer: AdamW")
    print(f"  Learning rate: {learning_rate}")
    print("  Weight decay: 0.1")
    print("-" * 70)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ Best val loss improved to {val_loss:.4f}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on instruction tasks")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/30M_model.pt",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/finetune", help="Checkpoint directory"
    )
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")

    args = parser.parse_args()

    main(
        pretrained_checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_length=args.max_length,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
