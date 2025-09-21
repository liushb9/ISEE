# Zarr Dataset Merger for Multi-Task Training

This script merges multiple Zarr datasets into a single dataset for multi-task training with Diffusion Policy.

## Usage

### Option 1: Auto-detect all zarr files
```bash
cd /media/liushengbang/isee/policy/DP
python merge_zarr_datasets.py --output_path data/dp_1200-integrated_clean-1200.zarr
```

### Option 2: Specify specific files
```bash
python merge_zarr_datasets.py \
    --file_list \
        data/blocks_ranking_rgb-integrated_clean-200.zarr \
        data/blocks_ranking_size-integrated_clean-200.zarr \
        data/hanging_mug-integrated_clean-200.zarr \
        data/place_cans_plasticbox-integrated_clean-200.zarr \
        data/stack_blocks_three-integrated_clean-200.zarr \
        data/stack_bowls_three-integrated_clean-200.zarr \
    --output_path data/dp_1200-integrated_clean-1200.zarr
```

## What the Script Does

1. **Auto-detects** all `.zarr` files in the `data/` directory
2. **Concatenates** data along the time dimension:
   - `action`: (N, 16) - Action sequences
   - `action_mask`: (N, 16) - Action validity masks
   - `head_camera`: (N, 3, 240, 320) - Camera images
   - `state`: (N, 16) - State vectors
   - `text_feat`: (1, 1024) - Text features (from first dataset)

3. **Adjusts episode boundaries**:
   - Recalculates `episode_ends` with proper offsets
   - Maintains episode integrity across datasets

4. **Optimizes storage**:
   - Uses Zstd compression
   - Sets appropriate chunk sizes for efficient access

## Output

The merged dataset will contain:
- **1200 episodes** (200 episodes × 6 tasks)
- **~412,410 time steps** (estimated total)
- **16-dimensional actions** (unified through padding)
- **All original camera and state data**

## Training

After merging, use the standard training command:

```bash
# Train with merged dataset
bash train.sh dp_1200 integrated_clean 1200 0 16 4,5,6,7

# Parameters:
# - dp_1200: task name for merged dataset
# - integrated_clean: configuration name
# - 1200: total number of episodes
# - 0: seed
# - 16: action dimension
# - 4,5,6,7: GPU IDs (multi-GPU training)
```

## Expected Data Structure

```
dp_1200-integrated_clean-1200.zarr/
├── data/
│   ├── action: (412410, 16) float32
│   ├── action_mask: (412410, 16) float32
│   ├── head_camera: (412410, 3, 240, 320) uint8
│   ├── state: (412410, 16) float32
│   └── text_feat: (1, 1024) float32
└── meta/
    └── episode_ends: (1200,) int64
```

## Performance Notes

- **Memory usage**: Merging requires ~10-15GB RAM
- **Time**: ~5-10 minutes depending on system
- **Storage**: Output file ~8-12GB compressed

## Troubleshooting

### Common Issues

1. **Memory error**: Reduce chunk sizes or process in batches
2. **File not found**: Check zarr file paths
3. **Dimension mismatch**: Ensure all input files have consistent dimensions

### Verification

After merging, verify the dataset:
```python
import zarr
zarr_file = zarr.open('data/dp_1200-integrated_clean-1200.zarr', 'r')
print(f"Total episodes: {len(zarr_file['meta']['episode_ends'])}")
print(f"Total time steps: {zarr_file['data']['action'].shape[0]}")
```
