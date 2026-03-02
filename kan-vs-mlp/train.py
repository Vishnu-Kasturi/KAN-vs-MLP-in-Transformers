"""
KAN vs MLP Training Benchmark
==============================
Trains two identical transformers (one with MLP FFN, one with KAN FFN)
on WikiText-2 and logs all metrics for comparison.

Usage:
    python train.py                  # Full run (both models)
    python train.py --model mlp      # Train only MLP variant
    python train.py --model kan      # Train only KAN variant
    python train.py --epochs 5       # Fewer epochs (faster)
    python train.py --device mps     # Use Apple Silicon GPU

Results saved to results.json — use plot_results.py to generate figures.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
import argparse
import os
import sys

from model import SmallTransformer


# ======================================================================
# Dataset
# ======================================================================

class TextDataset(Dataset):
    """Simple character-level dataset from a text corpus.
    
    Character-level keeps things simple (no tokenizer dependency),
    and is sufficient for comparing MLP vs KAN at small scale.
    """
    
    def __init__(self, text, seq_len=128, vocab_size=256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Encode as bytes (0-255) — universal, no tokenizer needed
        self.data = torch.tensor(
            [b for b in text.encode('utf-8', errors='replace')],
            dtype=torch.long
        )
        # Clamp to vocab_size
        self.data = self.data.clamp(0, vocab_size - 1)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target


def load_wikitext2():
    """Load WikiText-2 dataset. Falls back to a synthetic dataset if offline."""
    try:
        from datasets import load_dataset
        print("📥 Loading WikiText-2 from HuggingFace...")
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
        train_text = '\n'.join(ds['train']['text'])
        val_text = '\n'.join(ds['validation']['text'])
        print(f"   Train: {len(train_text):,} chars | Val: {len(val_text):,} chars")
        return train_text, val_text
    except Exception as e:
        print(f"⚠️  Could not load WikiText-2: {e}")
        print("   Using synthetic text for testing...")
        # Generate enough synthetic text to train
        import random
        random.seed(42)
        words = ['the', 'a', 'of', 'to', 'and', 'in', 'is', 'it', 'for', 'that',
                 'was', 'on', 'are', 'with', 'as', 'this', 'be', 'at', 'have', 'from',
                 'or', 'an', 'by', 'not', 'but', 'what', 'all', 'were', 'when', 'we',
                 'there', 'can', 'had', 'has', 'will', 'each', 'about', 'how', 'up',
                 'out', 'them', 'then', 'she', 'many', 'some', 'so', 'these', 'would',
                 'other', 'into', 'more', 'her', 'two', 'like', 'him', 'time', 'very',
                 'make', 'been', 'long', 'after', 'just', 'new', 'also', 'know', 'way']
        train_text = ' '.join(random.choices(words, k=200000))
        val_text = ' '.join(random.choices(words, k=20000))
        return train_text, val_text


# ======================================================================
# Training Loop
# ======================================================================

def train_one_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch. Returns metrics dict."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    start = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(inputs, labels=targets)
        loss.backward()
        
        # Gradient clipping (stabilizes training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=-1)
        mask = targets != -100
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_tokens += mask.sum().item()
        
        # Progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            tps = total_tokens / elapsed
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.3f} | {tps:,.0f} tok/s", flush=True)
    
    elapsed = time.time() - start
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / max(total_tokens, 1)
    ppl = min(math.exp(avg_loss), 1e6)  # cap to avoid inf
    tok_per_sec = total_tokens / elapsed
    
    return {
        'loss': avg_loss,
        'ppl': ppl,
        'accuracy': accuracy * 100,
        'tokens_per_sec': tok_per_sec,
        'time_seconds': elapsed,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation set. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss = model(inputs, labels=targets)
        
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=-1)
        mask = targets != -100
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / max(total_tokens, 1)
    ppl = min(math.exp(avg_loss), 1e6)
    
    return {
        'loss': avg_loss,
        'ppl': ppl,
        'accuracy': accuracy * 100,
    }


import math

def run_experiment(ffn_type, train_text, val_text, args, device):
    """Run full training for one model variant."""
    print(f"\n{'='*60}")
    print(f"  Training: {ffn_type.upper()} FFN")
    print(f"{'='*60}")
    
    # Dataset
    train_ds = TextDataset(train_text, seq_len=args.seq_len, vocab_size=args.vocab_size)
    val_ds = TextDataset(val_text, seq_len=args.seq_len, vocab_size=args.vocab_size)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, drop_last=True)
    
    # Model
    model = SmallTransformer(
        vocab_size=args.vocab_size,
        embedding_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        ffn_type=ffn_type,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"  Parameters: {param_count:,}")
    print(f"  Device:     {device}")
    print(f"  Batches:    {len(train_loader)} train / {len(val_loader)} val")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    
    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # Training loop
    history = {
        'ffn_type': ffn_type,
        'params': param_count,
        'epochs': [],
    }
    
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f})")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        }
        history['epochs'].append(epoch_data)
        
        print(f"  ┌─ Train: Loss={train_metrics['loss']:.3f} | "
              f"PPL={train_metrics['ppl']:.1f} | "
              f"Acc={train_metrics['accuracy']:.1f}% | "
              f"{train_metrics['tokens_per_sec']:,.0f} tok/s")
        print(f"  └─── Val: Loss={val_metrics['loss']:.3f} | "
              f"PPL={val_metrics['ppl']:.1f} | "
              f"Acc={val_metrics['accuracy']:.1f}%")
    
    total_time = time.time() - total_start
    history['total_time_seconds'] = total_time
    
    print(f"\n  ✓ {ffn_type.upper()} training complete in {total_time:.1f}s")
    
    return history


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='KAN vs MLP Transformer Benchmark')
    
    # Model
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='Vocabulary size (256 for byte-level)')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--ffn-dim', type=int, default=512,
                        help='FFN hidden dimension')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # KAN specific
    parser.add_argument('--kan-grid-size', type=int, default=5,
                        help='KAN B-spline grid intervals')
    parser.add_argument('--kan-spline-order', type=int, default=3,
                        help='KAN B-spline order')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Run config
    parser.add_argument('--model', type=str, default='both',
                        choices=['mlp', 'kan', 'both'],
                        help='Which model(s) to train')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not set)')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     KAN vs MLP in Transformers — Research Benchmark     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Device: {str(device):47s} ║")
    print(f"║  Epochs: {args.epochs:<47d} ║")
    print(f"║  Model:  {args.embed_dim}d / {args.num_layers}L / {args.num_heads}H / {args.ffn_dim}ffn{' ':26s} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Load data
    train_text, val_text = load_wikitext2()
    
    # Run experiments
    results = {'config': vars(args), 'experiments': {}}
    
    models_to_train = ['mlp', 'kan'] if args.model == 'both' else [args.model]
    
    for ffn_type in models_to_train:
        # Re-seed for fairness
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        history = run_experiment(ffn_type, train_text, val_text, args, device)
        results['experiments'][ffn_type] = history
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to {output_path}")
    
    # Print summary
    if len(results['experiments']) == 2:
        mlp = results['experiments']['mlp']
        kan = results['experiments']['kan']
        
        mlp_final = mlp['epochs'][-1]
        kan_final = kan['epochs'][-1]
        
        print("\n" + "=" * 60)
        print("  FINAL COMPARISON")
        print("=" * 60)
        print(f"  {'Metric':<25s} {'MLP':>12s} {'KAN':>12s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Parameters':<25s} {mlp['params']:>12,} {kan['params']:>12,}")
        print(f"  {'Val PPL':<25s} {mlp_final['val']['ppl']:>12.1f} {kan_final['val']['ppl']:>12.1f}")
        print(f"  {'Val Accuracy':<25s} {mlp_final['val']['accuracy']:>11.1f}% {kan_final['val']['accuracy']:>11.1f}%")
        print(f"  {'Train Throughput':<25s} {mlp_final['train']['tokens_per_sec']:>10,.0f}/s {kan_final['train']['tokens_per_sec']:>10,.0f}/s")
        print(f"  {'Total Time':<25s} {mlp['total_time_seconds']:>10.1f}s {kan['total_time_seconds']:>10.1f}s")
        print("=" * 60)
    
    print("\n🎨 Run 'python plot_results.py' to generate figures!")


if __name__ == '__main__':
    main()
