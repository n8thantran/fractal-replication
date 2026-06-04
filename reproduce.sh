#!/bin/bash
# FRACTAL Paper Replication - Reproduce Key Results
# This script generates all figures and runs LRA experiments.
# 
# Full training (paper config) takes ~50+ GPU hours total.
# This script runs reduced-epoch versions that demonstrate 
# the model works correctly while being feasible on a single GPU.
#
# Usage: bash reproduce.sh [--quick]
#   --quick: Run minimal epochs for testing (default: reduced epochs)

set -e

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
    echo "Running in quick mode (minimal epochs for testing)"
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

# Step 2: Run sCIFAR-10 (Image) experiment
echo ""
echo "Step 2: Running sCIFAR-10 (Image) experiment..."
if [ "$QUICK_MODE" = true ]; then
    IMAGE_EPOCHS=3
else
    IMAGE_EPOCHS=30
fi
python train.py --task image --epochs $IMAGE_EPOCHS --batch_size 50 \
    --d_model 256 --state_dim 64 --n_layers 6 \
    --lr 0.001 --weight_decay 0.05 --num_workers 4 \
    2>&1 | tee results/image_training.log
echo "✓ sCIFAR-10 experiment complete"

# Step 3: Run ListOps experiment  
echo ""
echo "Step 3: Running ListOps experiment..."
if [ "$QUICK_MODE" = true ]; then
    LISTOPS_EPOCHS=3
    LISTOPS_SAMPLES=5000
else
    LISTOPS_EPOCHS=15
    LISTOPS_SAMPLES=48000
fi
python train.py --task listops --epochs $LISTOPS_EPOCHS --batch_size 32 \
    --d_model 256 --state_dim 64 --n_layers 6 \
    --lr 0.001 --weight_decay 0.05 --num_workers 4 \
    --max_train_samples $LISTOPS_SAMPLES \
    2>&1 | tee results/listops_training.log
echo "✓ ListOps experiment complete"

# Step 4: Run Text (IMDB) experiment
echo ""
echo "Step 4: Running Text (IMDB) experiment..."
if [ "$QUICK_MODE" = true ]; then
    TEXT_EPOCHS=2
    TEXT_SAMPLES=2000
else
    TEXT_EPOCHS=10
    TEXT_SAMPLES=25000
fi
python train.py --task text --epochs $TEXT_EPOCHS --batch_size 16 \
    --d_model 256 --state_dim 64 --n_layers 6 \
    --lr 0.001 --weight_decay 0.05 --num_workers 4 \
    --max_train_samples $TEXT_SAMPLES \
    2>&1 | tee results/text_training.log
echo "✓ Text experiment complete"

# Step 5: Summarize results
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Generated figures:"
ls -la results/*.png 2>/dev/null || echo "  (none)"
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
