#!/bin/bash
# ============================================================================
# Reproduce: Assessing the impact of dimensionality reduction on clustering
# ============================================================================
# This script reproduces the key results from the paper:
#   - Tables A.1-A.4: Synthetic data ARI scores
#   - Tables A.5-A.8: Real-world data ARI scores  
#   - Tables 2-5: Aggregate statistics (win%, avg% change)
#   - Table A.9: Wilcoxon signed-rank test
#   - Figures: Boxplots and heatmaps
#
# Usage:
#   bash reproduce.sh             # Generate tables/figures from cached results (~10 sec)
#   bash reproduce.sh --quick     # Same as above
#   bash reproduce.sh --real      # Re-run real-world experiments (~2-4 hours)
#   bash reproduce.sh --synth     # Re-run synthetic experiments (~1-2 hours)
#   bash reproduce.sh --full      # Re-run everything from scratch (~4-6 hours)
# ============================================================================

set -e

MODE="tables"
if [ "$1" = "--quick" ]; then
    MODE="tables"
elif [ "$1" = "--real" ]; then
    MODE="real"
elif [ "$1" = "--synth" ]; then
    MODE="synth"
elif [ "$1" = "--full" ]; then
    MODE="full"
fi

echo "=========================================="
echo "DR for Clustering - Paper Replication"
echo "Mode: $MODE"
echo "=========================================="

# Create output directories
mkdir -p results/tables results/figures

# Install dependencies if needed
pip install -q numpy scipy scikit-learn pandas matplotlib torch 2>/dev/null || true

if [ "$MODE" = "full" ] || [ "$MODE" = "real" ]; then
    echo ""
    echo "Step 1: Running real-world experiments (20 UCI datasets)..."
    echo "  This runs k-means, AHC, GMM, OPTICS on 20 datasets × 16 conditions."
    python run_remaining_real.py 2>&1 | tail -5
    echo "✓ Real-world experiments complete"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "synth" ]; then
    echo ""
    echo "Step 2: Running synthetic experiments (4 types × 10 repeats)..."
    echo "  This runs Circles, Moons, RSG, Repliclust with noise injection."
    python run_synthetic_v3.py 2>&1 | tail -5
    echo "✓ Synthetic experiments complete"
fi

echo ""
echo "Step 3: Generating all tables and figures from results..."
python generate_results.py 2>&1 | grep -E "(Generated|Table|Figure|All tables)"
echo "✓ Tables and figures generated"

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Tables (in results/tables/):"
echo "  table_A1_synthetic_Circles.csv  - Table A.1: Circles synthetic ARI"
echo "  table_A2_synthetic_Moons.csv    - Table A.2: Moons synthetic ARI"
echo "  table_A3_synthetic_RSG.csv      - Table A.3: RSG synthetic ARI"
echo "  table_A4_synthetic_Repliclust.csv - Table A.4: Repliclust synthetic ARI"
echo "  table_A5_real_kmeans.csv        - Table A.5: Real-world k-means ARI"
echo "  table_A6_real_AHC.csv           - Table A.6: Real-world AHC ARI"
echo "  table_A7_real_GMM.csv           - Table A.7: Real-world GMM ARI"
echo "  table_A8_real_OPTICS.csv        - Table A.8: Real-world OPTICS ARI"
echo "  table_A9_wilcoxon.csv           - Table A.9: Wilcoxon signed-rank test"
echo "  table_2_aggregate_kmeans.csv    - Table 2: k-means aggregate stats"
echo "  table_3_aggregate_AHC.csv       - Table 3: AHC aggregate stats"
echo "  table_4_aggregate_GMM.csv       - Table 4: GMM aggregate stats"
echo "  table_5_aggregate_OPTICS.csv    - Table 5: OPTICS aggregate stats"
echo ""
echo "Figures (in results/figures/):"
echo "  boxplot_k_means_real.png, boxplot_AHC_real.png, etc."
echo "  heatmap_k_means_real.png, heatmap_AHC_real.png, etc."
echo ""
echo "Key finding (k-means, real data):"
cat results/tables/summary.txt | grep -A1 "k-means:" | head -2
cat results/tables/summary.txt | grep "Best DR"  | head -1
echo ""
echo "Paper comparison (k-means No Reduction):"
tail -3 results/tables/paper_comparison.txt
echo ""
echo "=========================================="
echo "All done! See results/ for full outputs."
echo "=========================================="
