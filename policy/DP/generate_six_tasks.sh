#!/bin/bash

echo "Generating six_tasks-demo_clean-300.zarr dataset..."

# 检查是否在正确的目录
if [ ! -f "generate_six_tasks_dataset.py" ]; then
    echo "Error: Please run this script from the RoboTwin/policy/DP directory"
    exit 1
fi

# 检查源数据集是否存在
source_datasets=(
    "data/stack_blocks_three-demo_clean-50.zarr"
    "data/stack_bowls_three-demo_clean-50.zarr"
    "data/blocks_ranking_size-demo_clean-50.zarr"
    "data/blocks_ranking_rgb-demo_clean-50.zarr"
    "data/hanging_mug-demo_clean-50.zarr"
    "data/place_cans_plasticbox-demo_clean-50.zarr"
)

echo "Checking source datasets..."
for dataset in "${source_datasets[@]}"; do
    if [ ! -d "$dataset" ]; then
        echo "Error: Source dataset $dataset not found!"
        echo "Please run process_data.sh for each task first."
        exit 1
    fi
    echo "  ✓ $dataset"
done

echo ""
echo "Running dataset generation..."
python generate_six_tasks_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Dataset generation completed successfully!"
    echo "You can now run: bash train.sh six_tasks demo_clean 300 0 14 0"
else
    echo ""
    echo "Dataset generation failed!"
    exit 1
fi