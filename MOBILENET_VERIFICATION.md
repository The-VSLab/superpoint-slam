# MobileNetV2 Channel Count Verification

## Issue Summary

A review comment on PR #48 suggested that the comment in `py_superpoint.py` claiming MobileNetV2 outputs 32 channels at the 1/8 resolution point might be incorrect, stating that layer 6 typically outputs 24 channels.

## Investigation Results

After thorough investigation, **the original code and comment are CORRECT**. The confusion stems from a misunderstanding of MobileNetV2's layer structure.

## MobileNetV2 Architecture Breakdown

When using `features[:7]` (layers 0-6) from torchvision's MobileNetV2:

| Layer Index | Block Type | Stride | Resolution | Output Channels |
|-------------|-----------|--------|------------|-----------------|
| 0 | Initial Conv2d | 2 | 1/2 | 32 |
| 1 | InvertedResidual | 1 | 1/2 | 16 |
| 2 | InvertedResidual | 2 | 1/4 | 24 |
| 3 | InvertedResidual | 1 | 1/4 | 24 |
| 4 | InvertedResidual | 2 | 1/8 | **32** |
| 5 | InvertedResidual | 1 | 1/8 | **32** |
| 6 | InvertedResidual | 1 | 1/8 | **32** ✓ |

**Key Finding:** Layer 6 outputs **32 channels** at **1/8 resolution**, not 24 channels.

## Source of Confusion

The confusion likely arose because:
- Layers 2-3 do output **24 channels**, but at **1/4 resolution**
- Layer 6 outputs **32 channels** at **1/8 resolution**

The reviewer may have been thinking of layers 2-3 when commenting about 24 channels.

## Verification

Run the included verification script:

```bash
python3 verify_mobilenet_channels.py
```

This script:
1. Analyzes the MobileNetV2 architecture based on the official torchvision implementation
2. Shows layer-by-layer channel counts and resolutions
3. If PyTorch is installed, performs actual inference to verify output shapes
4. Confirms that `features[:7]` outputs 32 channels at 1/8 resolution

## Changes Made

1. **Added `verify_mobilenet_channels.py`**: A comprehensive verification script that can run with or without PyTorch
2. **Enhanced comments in `py_superpoint.py`**: Added detailed layer-by-layer breakdown to prevent future confusion

## Conclusion

The code in `py_superpoint.py` is correct:
- `self.backbone = nn.Sequential(*list(v2_model.children())[:7])` correctly extracts layers 0-6
- `in_channels = 32` correctly matches the output of layer 6
- The detector and descriptor heads correctly expect 32 input channels

No code changes are needed beyond improved documentation.

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [PyTorch MobileNetV2 Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
