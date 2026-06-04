"""
Training script for FRACTAL model on LRA benchmark tasks.

Usage:
    python train.py --task listops --epochs 40
    python train.py --task text --epochs 40
    python train.py --task image --epochs 200
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import FRACTALModel, count_parameters
from lra_datasets import get_dataset, TASK_CONFIGS


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_clip=1.0, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # For continuous input (image), add channel dim
        if x.dtype == torch.float32 and x.dim() == 2:
            x = x.unsqueeze(-1)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples * 100
            lr = optimizer.param_groups[0]['lr']
            print(f"  Step {batch_idx+1}/{len(loader)}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%, lr={lr:.6f}")
    
    return total_loss / total_samples, total_correct / total_samples * 100


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if x.dtype == torch.float32 and x.dim() == 2:
            x = x.unsqueeze(-1)
        
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)
    
    return total_loss / total_samples, total_correct / total_samples * 100


def main():
    parser = argparse.ArgumentParser(description='Train FRACTAL on LRA')
    parser.add_argument('--task', type=str, required=True, choices=['listops', 'text', 'image', 'retrieval'])
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--state_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_frac', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--use_sequential_scan', action='store_true')
    # For quick testing
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Task config
    config = TASK_CONFIGS[args.task]
    epochs = args.epochs or config['epochs']
    batch_size = args.batch_size or config['batch_size']
    
    print(f"\n{'='*60}")
    print(f"Task: {args.task}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Model: d_model={args.d_model}, state_dim={args.state_dim}, n_layers={args.n_layers}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading datasets...")
    train_kwargs = {}
    eval_kwargs = {}
    if args.max_train_samples:
        if args.task == 'listops':
            train_kwargs['n_samples'] = args.max_train_samples
        elif args.task == 'text':
            train_kwargs['max_samples'] = args.max_train_samples
    if args.max_eval_samples:
        if args.task == 'text':
            eval_kwargs['max_samples'] = args.max_eval_samples
    
    train_ds = get_dataset(args.task, split='train', **train_kwargs)
    val_ds = get_dataset(args.task, split='val', **eval_kwargs)
    test_ds = get_dataset(args.task, split='test', **eval_kwargs)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_ds)} samples, {len(test_loader)} batches")
    
    # Build model
    alpha_list = [0, 0, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9]
    
    if config['input_type'] == 'discrete':
        embedding_type = 'embedding'
        model = FRACTALModel(
            d_model=args.d_model,
            state_dim=args.state_dim,
            n_layers=args.n_layers,
            n_classes=config['n_classes'],
            vocab_size=config['vocab_size'],
            max_seq_len=config['seq_len'],
            alpha_list=alpha_list,
            dropout=args.dropout,
            embedding_type='embedding',
        )
    else:
        model = FRACTALModel(
            d_model=args.d_model,
            state_dim=args.state_dim,
            n_layers=args.n_layers,
            n_classes=config['n_classes'],
            max_seq_len=config['seq_len'],
            alpha_list=alpha_list,
            dropout=args.dropout,
            embedding_type='linear',
            input_dim=1,
        )
    
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")
    
    # Optimizer: AdamW with separate LR for SSM params
    # Paper: β1=0.9, β2=0.999
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'Lambda' in name or 'log_dt' in name:
            ssm_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': ssm_params, 'lr': args.lr, 'weight_decay': 0.0},  # No WD for SSM params
        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
    ], betas=(0.9, 0.999))
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # AMP
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"AMP: {use_amp}")
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            grad_clip=args.grad_clip, use_amp=use_amp
        )
        
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        
        elapsed = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.task}_best.pt'))
        
        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s): "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}% "
              f"{'*BEST*' if is_best else ''}")
    
    # Final evaluation on test set with best model
    print(f"\n{'='*60}")
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.task}_best.pt')))
    test_loss, test_acc = evaluate(model, test_loader, device, use_amp=use_amp)
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    
    # Save results
    results = {
        'task': args.task,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'n_params': n_params,
        'epochs': epochs,
        'batch_size': batch_size,
        'd_model': args.d_model,
        'state_dim': args.state_dim,
        'n_layers': args.n_layers,
        'lr': args.lr,
        'history': history,
    }
    
    results_path = os.path.join(args.results_dir, f'{args.task}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return test_acc


if __name__ == '__main__':
    main()
