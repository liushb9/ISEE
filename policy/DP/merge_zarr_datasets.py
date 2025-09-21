#!/usr/bin/env python3
"""
Merge multiple Zarr datasets into a single dataset for multi-task training.

Usage:
    python merge_zarr_datasets.py --output_path data/dp_1200-integrated_clean-1200.zarr
"""

import argparse
import os
import numpy as np
import zarr
from pathlib import Path
from typing import List


def get_zarr_files(data_dir: str) -> List[str]:
    """Get all zarr files in the data directory."""
    data_path = Path(data_dir)
    zarr_files = []

    # Find all .zarr directories
    for item in data_path.iterdir():
        if item.is_dir() and item.name.endswith('.zarr'):
            zarr_files.append(str(item))

    # Sort by name for consistent ordering
    zarr_files.sort()
    return zarr_files


def merge_zarr_datasets(zarr_paths: List[str], output_path: str):
    """
    Merge multiple zarr datasets into a single dataset.

    Args:
        zarr_paths: List of paths to input zarr files
        output_path: Path to output merged zarr file
    """
    print(f"Merging {len(zarr_paths)} zarr files...")

    # Check if output exists and remove it
    if os.path.exists(output_path):
        print(f"Removing existing output: {output_path}")
        import shutil
        shutil.rmtree(output_path)

    # Open all input zarr files
    zarr_files = []
    for path in zarr_paths:
        try:
            zarr_file = zarr.open(path, 'r')
            zarr_files.append(zarr_file)
            print(f"Opened: {path}")
        except Exception as e:
            print(f"Error opening {path}: {e}")
            continue

    if not zarr_files:
        raise ValueError("No valid zarr files found")

    # Create output zarr file
    output_zarr = zarr.open(output_path, 'w')
    output_data = output_zarr.create_group("data")
    output_meta = output_zarr.create_group("meta")

    # Initialize data arrays
    merged_action = []
    merged_action_mask = []
    merged_head_camera = []
    merged_state = []
    merged_episode_ends = []  # Will be a flat list of all episode end indices

    # Track cumulative offset for episode_ends
    cumulative_offset = 0

    print("Processing datasets...")

    for i, zarr_file in enumerate(zarr_files):
        print(f"Processing dataset {i+1}/{len(zarr_files)}: {zarr_paths[i]}")

        # Read data from current zarr file
        action_data = zarr_file['data']['action'][:]
        action_mask_data = zarr_file['data']['action_mask'][:]
        head_camera_data = zarr_file['data']['head_camera'][:]
        state_data = zarr_file['data']['state'][:]
        episode_ends_data = zarr_file['meta']['episode_ends'][:]

        print(f"  Data shapes: action={action_data.shape}, head_camera={head_camera_data.shape}")

        # Append data
        merged_action.append(action_data)
        merged_action_mask.append(action_mask_data)
        merged_head_camera.append(head_camera_data)
        merged_state.append(state_data)

        # Adjust episode_ends with cumulative offset and append
        # Note: episode_ends stores cumulative indices, so we add the offset
        adjusted_episode_ends = episode_ends_data + cumulative_offset
        merged_episode_ends.extend(adjusted_episode_ends)

        # Update cumulative offset for next file
        cumulative_offset += action_data.shape[0]

        print(f"  Episodes: {len(episode_ends_data)}, Total steps so far: {cumulative_offset}")

    # Concatenate all data
    print("Concatenating data...")
    final_action = np.concatenate(merged_action, axis=0)
    final_action_mask = np.concatenate(merged_action_mask, axis=0)
    final_head_camera = np.concatenate(merged_head_camera, axis=0)
    final_state = np.concatenate(merged_state, axis=0)
    final_episode_ends = np.array(merged_episode_ends)  # Convert list to numpy array

    # Get text features from first file (should be the same for all tasks)
    text_feat = zarr_files[0]['data']['text_feat'][:]

    print(f"Final data shapes:")
    print(f"  action: {final_action.shape}")
    print(f"  action_mask: {final_action_mask.shape}")
    print(f"  head_camera: {final_head_camera.shape}")
    print(f"  state: {final_state.shape}")
    print(f"  episode_ends: {final_episode_ends.shape}")
    print(f"  text_feat: {text_feat.shape}")

    # Verify episode_ends integrity
    print(f"Episode validation:")
    print(f"  Total episodes: {len(final_episode_ends)}")
    print(f"  Last episode ends at: {final_episode_ends[-1]}")
    print(f"  Total time steps: {final_action.shape[0]}")
    print(f"  Validation: {final_episode_ends[-1]} == {final_action.shape[0]} = {final_episode_ends[-1] == final_action.shape[0]}")

    if final_episode_ends[-1] != final_action.shape[0]:
        raise ValueError(f"Episode ends validation failed! Expected {final_action.shape[0]}, got {final_episode_ends[-1]}")

    # Create datasets in output zarr
    print("Creating output zarr datasets...")

    # Data group
    output_data.create_dataset(
        "action",
        data=final_action,
        chunks=(1000, final_action.shape[1]),  # Larger chunks for better performance
        dtype=final_action.dtype,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    )

    output_data.create_dataset(
        "action_mask",
        data=final_action_mask,
        chunks=(1000, final_action_mask.shape[1]),
        dtype=final_action_mask.dtype,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    )

    output_data.create_dataset(
        "head_camera",
        data=final_head_camera,
        chunks=(100, final_head_camera.shape[1], final_head_camera.shape[2], final_head_camera.shape[3]),
        dtype=final_head_camera.dtype,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    )

    output_data.create_dataset(
        "state",
        data=final_state,
        chunks=(1000, final_state.shape[1]),
        dtype=final_state.dtype,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    )

    output_data.create_dataset(
        "text_feat",
        data=text_feat,
        dtype=text_feat.dtype,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    )

    # Meta group
    output_meta.create_dataset(
        "episode_ends",
        data=final_episode_ends,
        dtype=final_episode_ends.dtype,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    )

    print(f"Successfully merged {len(zarr_files)} datasets!")
    print(f"Total episodes: {len(final_episode_ends)}")
    print(f"Total time steps: {final_action.shape[0]}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple Zarr datasets")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing zarr files to merge"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output merged zarr file"
    )
    parser.add_argument(
        "--file_list",
        type=str,
        nargs='*',
        help="Specific zarr files to merge (optional, will auto-detect if not provided)"
    )

    args = parser.parse_args()

    # Get list of zarr files to merge
    if args.file_list:
        zarr_paths = args.file_list
    else:
        zarr_paths = get_zarr_files(args.data_dir)

    print(f"Found {len(zarr_paths)} zarr files:")
    for path in zarr_paths:
        print(f"  {path}")

    # Merge datasets
    merge_zarr_datasets(zarr_paths, args.output_path)


if __name__ == "__main__":
    main()
