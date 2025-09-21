# Heterogeneous Embodiment Support for Diffusion Policy

This document describes the modifications made to support heterogeneous embodiment training using different action dimensions (7D and 8D) with a unified 8D representation via padding and masking.

## Overview

The modifications enable training Diffusion Policy on data from different robot embodiments:
- **7D actions**: 6 joints + 1 gripper (padded to 16D with mask [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
- **8D actions**: 7 joints + 1 gripper (padded to 16D with mask [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
- **14D actions**: Dual arm (6 joints + 1 gripper each) (padded to 16D with mask [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0])
- **16D actions**: Dual arm (7 joints + 1 gripper each) (mask [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

## Key Features

- ✅ **Automatic dimension detection**: Detects action dimensions and applies appropriate padding
- ✅ **Mask-based training**: Uses action masks to ignore padded dimensions during training
- ✅ **Unified representation**: All actions standardized to 8D for batch processing
- ✅ **Backward compatibility**: Maintains original DP architecture and training logic

## Modified Files

### 1. `process_data.py`
**Changes:**
- Added action dimension detection logic
- Implemented padding mechanism for 7D → 8D actions
- Added action_mask generation and storage
- Enhanced logging to show processed data shapes

**Key modifications:**
```python
# Detect action dimension and apply padding
action_dim = joint_state.shape[-1]
if action_dim == 7:  # 6 joints + 1 gripper (single arm)
    # Pad to 16 dimensions, set last 9 dimensions to 0
    padded_action = np.pad(joint_state, (0, 9), mode='constant', constant_values=0)
    action_mask = np.zeros(16, dtype=np.float32)
    action_mask[:7] = 1  # First 7 dimensions are valid (6 joints + 1 gripper)
elif action_dim == 8:  # 7 joints + 1 gripper (single arm)
    # Pad to 16 dimensions, set last 8 dimensions to 0
    padded_action = np.pad(joint_state, (0, 8), mode='constant', constant_values=0)
    action_mask = np.zeros(16, dtype=np.float32)
    action_mask[:8] = 1  # First 8 dimensions are valid (7 joints + 1 gripper)
elif action_dim == 14:  # Dual arm: (6 joints + 1 gripper) * 2 = 14
    # Pad to 16 dimensions, set last 2 dimensions to 0
    padded_action = np.pad(joint_state, (0, 2), mode='constant', constant_values=0)
    action_mask = np.ones(16, dtype=np.float32)
    action_mask[-2:] = 0  # Last 2 dimensions are invalid
elif action_dim == 16:  # Dual arm: (7 joints + 1 gripper) * 2 = 16
    padded_action = joint_state
    action_mask = np.ones(16, dtype=np.float32)
```

### 2. `diffusion_policy/dataset/robot_image_dataset.py`
**Changes:**
- Added "action_mask" to replay buffer keys
- Included action_mask in data processing pipeline
- Added action_mask to normalizer (identity normalization)
- Updated postprocess method to handle action_mask

**Key modifications:**
```python
# In __init__
keys=["head_camera", "state", "action", "text_feat", "action_mask"]

# In get_normalizer
normalizer["action_mask"] = LinearNormalizer.create_identity(dtype=torch.float32)
```

### 3. `diffusion_policy/env_runner/dp_runner.py`
**Changes:**
- Added action_mask handling in get_action method
- Ensures action_mask is passed to policy when available

**Key modifications:**
```python
if "action_mask" in obs_dict:
    obs_dict_input["action_mask"] = obs_dict["action_mask"].unsqueeze(0)
```

### 4. `diffusion_policy/model/vision/multi_image_obs_encoder.py`
**Changes:**
- No direct modifications needed (automatically handles low_dim data including action_mask)
- action_mask is processed as regular low-dimensional input

### 5. Configuration Files

#### `diffusion_policy/config/task/default_task_16.yaml`
```yaml
obs:
  agent_pos:
    shape: [16]
    type: low_dim
  text_feat:
    shape: [1024]
    type: text
  action_mask:        # ← NEW
    shape: [8]
    type: low_dim
action:
  shape: [8]          # ← CHANGED from [16] to [8]
```

#### `diffusion_policy/config/task/default_task_14.yaml`
```yaml
obs:
  agent_pos:
    shape: [14]
    type: low_dim
  text_feat:
    shape: [1024]
    type: text
  action_mask:        # ← NEW
    shape: [8]
    type: low_dim
action:
  shape: [8]          # ← CHANGED from [14] to [8]
```

## Data Processing Workflow

1. **Input Detection**: Automatically detects whether input actions are 7D or 8D
2. **Padding**: Pads 7D actions to 8D by adding zeros
3. **Mask Generation**: Creates binary masks indicating valid dimensions
4. **Storage**: Saves padded actions and masks to Zarr format
5. **Training**: Uses masks during loss computation to ignore padded dimensions

## Usage Instructions

### 1. Data Processing
```bash
# Process data for any task (automatically handles different action dimensions)
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
```

### 2. Training
```bash
# Train with unified 8D action space (no action_dim parameter needed)
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
```

### 3. Evaluation
```bash
# Evaluate as usual
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
```

## Training Considerations

### Mixed Embodiment Training
When training on mixed data from different embodiments:
- ✅ **Batch processing**: All samples unified to 8D
- ✅ **Loss computation**: Only valid dimensions contribute to loss
- ✅ **Gradient flow**: Invalid dimensions receive zero gradients
- ✅ **Inference**: Masks ensure only valid dimensions are executed

### Expected Data Shapes
After processing:
- `head_camera`: `(N, 3, H, W)` - NCHW format
- `state`: `(N, 16)` - Padded state vectors
- `action`: `(N, 16)` - Padded action vectors
- `action_mask`: `(N, 16)` - Binary validity masks
- `text_feat`: `(1, 1024)` - CLIP text features

### Validation
Run the test script to verify modifications:
```bash
cd /media/liushengbang/isee/policy/DP
python test_heterogeneous.py
```

## Future Extensions

### 1. Dynamic Action Dimensions
- Support arbitrary action dimensions (not just 7D/8D)
- Runtime dimension detection and padding

### 2. Advanced Masking
- Learnable masks for better generalization
- Attention-based masking mechanisms

### 3. Embodiment-Aware Features
- Add embodiment identifiers to state representation
- Learn embodiment-specific policies within unified architecture

## Troubleshooting

### Common Issues

1. **CLIP Import Error**: Install CLIP package
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **Dimension Mismatch**: Ensure all tasks use compatible action spaces
   - Currently supports: 7D and 8D actions
   - Extension needed for other dimensions

3. **Memory Issues**: Monitor batch sizes when mixing large datasets
   - Consider gradient accumulation for larger effective batch sizes

### Performance Monitoring

- **Training curves**: Compare convergence across different embodiments
- **Success rates**: Evaluate performance on embodiment-specific test sets
- **Mask utilization**: Monitor which dimensions are most active

## Contact

For questions about these modifications, please refer to the original implementation and documentation.
