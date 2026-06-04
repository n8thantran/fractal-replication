#!/bin/bash
# FRACTAL Paper Replication - Reproduce Key Results
# This script generates all figures and runs LRA experiments.
# 
# Full training (paper config) takes ~50+ GPU hours total.
# This script runs reduced-epoch versions that demonstrate 
# the model works correctly while being feasible on a single GPU.
#
# Usage: bash reproduce.sh [--quick]
#   --quick: Run minimal smoke test (~5 min total)
#   default: Reduced epochs (~3-4 hours)

set -e

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
    echo "Running in quick mode (smoke test)"
fi

# Create results directory
mkdir -p results
mkdir -p checkpoints

echo "=========================================="
echo "FRACTAL Paper Replication"
echo "=========================================="

# Step 1: Generate paper figures (no GPU training needed)
echo ""
echo "Step 1: Generating paper figures..."
python generate_figures.py
echo "✓ Figures generated in results/"

if [ "$QUICK_MODE" = true ]; then
    # Quick mode: just run 1 epoch of image with tiny model to verify pipeline works
    echo ""
    echo "Step 2 (quick): Smoke test - sCIFAR-10 with small model..."
    python train.py --task image --epochs 1 --batch_size 50 \
        --d_model 64 --state_dim 16 --n_layers 2 \
        --lr 0.001 --weight_decay 0.05 --num_workers 4 \
        --max_train_samples 2000 --max_eval_samples 500 \
        2>&1 | tee results/image_training.log
    echo "✓ Smoke test complete"
    
    echo ""
    echo "=========================================="
    echo "Quick mode complete. Figures + smoke test passed."
    echo "Run without --quick for full experiments."
    echo "=========================================="
    exit 0
fi

# Step 2: Run sCIFAR-10 (Image) experiment - 30 epochs
echo ""
echo "Step 2: Running sCIFAR-10 (Image) experiment (30 epochs)..."
python train.py --task image --epochs 30 --batch_size 50 \
    --d_model 256 --state_dim 64 --n_layers 6 \
    --lr 0.001 --weight_decay 0.05 --num_workers 4 \
    2>&1 | tee results/image_training.log
echo "✓ sCIFAR-10 experiment complete"

# Step 3: Run Text (IMDB) experiment - 10 epochs with 5k samples
echo ""
echo "Step 3: Running Text (IMDB) experiment (10 epochs)..."
python train.py --task text --epochs 10 --batch_size 16 \
    --d_model 256 --state_dim 64 --n_layers 6 \
    --lr 0.001 --weight_decay 0.05 --num_workers 2 \
    --max_train_samples 5000 --max_eval_samples 1000 \
    2>&1 | tee results/text_training.log
echo "✓ Text experiment complete"

# Step 4: Run ListOps experiment - 5 epochs with 10k samples
echo ""
echo "Step 4: Running ListOps experiment (5 epochs)..."
python train.py --task listops --epochs 5 --batch_size 32 \
    --d_model 256 --state_dim 64 --n_layers 6 \
    --lr 0.001 --weight_decay 0.05 --num_workers 4 \
    --max_train_samples 10000 --max_eval_samples 2000 \
    2>&1 | tee results/listops_training.log
echo "✓ ListOps experiment complete"

# Step 5: Generate training curves
echo ""
echo "Step 5: Generating training curves..."
python -c "
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
tasks = [('image', 'sCIFAR-10', 87.30), ('text', 'Text/IMDB', 89.10), ('listops', 'ListOps', 61.85)]
for ax, (task, name, paper_acc) in zip(axes, tasks):
    fpath = f'results/{task}_results.json'
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        h = d.get('history', {})
        if 'train_acc' in h:
            ax.plot(h['train_acc'], 'b-', label='Train')
        if 'val_acc' in h:
            ax.plot(h['val_acc'], 'r-', label='Val')
        ax.axhline(y=paper_acc, color='g', linestyle='--', label=f'Paper ({paper_acc}%)')
        test_acc = d.get('test_acc', d.get('best_val_acc', '?'))
        ax.set_title(f'{name}: {test_acc}%')
    else:
        ax.set_title(f'{name}: no results')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
print('Training curves saved')
"
echo "✓ Training curves generated"

# Step 6: Summarize results
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Generated figures:"
ls -1 results/*.png 2>/dev/null || echo "  (none)"
echo ""
echo "Training results:"
for f in results/*_results.json; do
    if [ -f "$f" ]; then
        echo "  $f:"
        python -c "import json; d=json.load(open('$f')); print(f'    Best val acc: {d.get(\"best_val_acc\", \"N/A\")}%, Test acc: {d.get(\"test_acc\", \"N/A\")}%')"
    fi
done
echo ""
echo "All results saved to results/"
echo "=========================================="
