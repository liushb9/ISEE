#!/bin/bash

# Script to merge all zarr datasets in the data directory
# Usage: bash merge_datasets.sh

echo "Starting dataset merge..."

# Set output path
OUTPUT_PATH="data/dp_1200-integrated_clean-1200.zarr"

# Remove existing output if it exists
if [ -d "$OUTPUT_PATH" ]; then
    echo "Removing existing merged dataset: $OUTPUT_PATH"
    rm -rf "$OUTPUT_PATH"
fi

# Run merge script
echo "Merging datasets..."
python merge_zarr_datasets.py --output_path "$OUTPUT_PATH"

# Verify the result
if [ -d "$OUTPUT_PATH" ]; then
    echo "✅ Merge completed successfully!"
    echo "📊 Verifying merged dataset..."

    python -c "
import zarr
import numpy as np

try:
    zarr_file = zarr.open('$OUTPUT_PATH', 'r')
    episodes = len(zarr_file['meta']['episode_ends'])
    time_steps = zarr_file['data']['action'].shape[0]
    episode_ends_last = zarr_file['meta']['episode_ends'][-1]

    print(f'📈 Episodes: {episodes}')
    print(f'⏱️  Time steps: {time_steps}')
    print(f'🎯 Last episode ends at: {episode_ends_last}')
    print(f'✅ Episode validation: {episode_ends_last} == {time_steps} = {episode_ends_last == time_steps}')

    if episodes == 1200 and time_steps > 400000 and episode_ends_last == time_steps:
        print('✅ Dataset verification passed!')
    else:
        print('⚠️  Dataset may have issues - please check manually')
        print(f'   Expected: 1200 episodes, ~412,000 time steps')
        print(f'   Got: {episodes} episodes, {time_steps} time steps')

except Exception as e:
    print(f'❌ Error verifying dataset: {e}')
    import traceback
    traceback.print_exc()
"

    echo ""
    echo "🚀 Ready for training!"
    echo "Run: bash train.sh dp_1200 integrated_clean 1200 0 16 4,5,6,7"
else
    echo "❌ Merge failed!"
    exit 1
fi
