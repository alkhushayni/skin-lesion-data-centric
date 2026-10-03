#!/usr/bin/env bash
# Sensitivity analysis (Supplementary Table S5) on ResNet50, all five folds.
# Main-setting values come from results/main; each alternative value is run here.
# Usage: bash run_sensitivity.sh data/HAM10000
set -e
DATA=${1:-data/HAM10000}
COMMON="--data-dir $DATA --cache-dir cache --models resnet50 --budget-models resnet50"
for v in 0.80 0.95; do
  python run_experiments.py $COMMON --strategies sup50 selftrain --out-dir results/sens/st_threshold=$v --set st_threshold=$v
done
for v in 1000 2500; do
  python run_experiments.py $COMMON --strategies diffusion --out-dir results/sens/synth_target_per_class=$v --set synth_target_per_class=$v
done
for v in 3e-05 0.0003; do
  python run_experiments.py $COMMON --strategies baseline --out-dir results/sens/lr=$v --set lr=$v
done
for v in 1.0 3.0; do
  python run_experiments.py $COMMON --strategies focal --out-dir results/sens/focal_gamma=$v --set focal_gamma=$v
done
