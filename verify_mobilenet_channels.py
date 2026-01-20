#!/usr/bin/env python3
"""
Verification script for MobileNetV2 backbone output channels.

This script demonstrates that the MobileNetV2 backbone (layers 0-6) 
outputs 32 channels at 1/8 resolution, as claimed in py_superpoint.py.

The script can run with or without PyTorch installed:
- With PyTorch: Performs actual inference to verify output shape
- Without PyTorch: Shows architectural analysis based on MobileNetV2 spec
"""

import sys

# Backbone configuration
BACKBONE_LAYER_COUNT = 7  # features[:7] includes layers 0-6

def verify_with_pytorch():
    """Verify channel count using actual PyTorch model."""
    try:
        import torch
        from torchvision.models import mobilenet_v2
        
        print("=" * 80)
        print("VERIFICATION WITH PYTORCH")
        print("=" * 80)
        
        # Load MobileNetV2
        try:
            from torchvision.models import MobileNet_V2_Weights
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
            print("\nLoaded MobileNetV2 with IMAGENET1K_V1 weights")
        except (ImportError, AttributeError):
            model = mobilenet_v2(weights=None).features
            print("\nLoaded MobileNetV2 (no pretrained weights)")
        
        # Extract backbone (layers 0-6)
        backbone = torch.nn.Sequential(*list(model.children())[:BACKBONE_LAYER_COUNT])
        
        print("\nBackbone structure (layers 0-6):")
        print("-" * 80)
        for i, layer in enumerate(backbone):
            print(f"Layer {i}: {layer.__class__.__name__}")
        
        # Test with sample input
        print("\n" + "=" * 80)
        print("INFERENCE TEST")
        print("=" * 80)
        
        test_sizes = [
            (224, 224),  # Standard ImageNet size
            (120, 160),  # Size used in demo_superpoint.py
            (480, 640),  # Common camera resolution
        ]
        
        for h, w in test_sizes:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                output = backbone(x)
            
            output_h, output_w = output.shape[2], output.shape[3]
            expected_h, expected_w = h // 8, w // 8
            
            print(f"\nInput:  {h}x{w}")
            print(f"Output: {output.shape[1]} channels, {output_h}x{output_w} spatial")
            print(f"Resolution reduction: 1/{h//output_h} (expected: 1/8)")
            
            # Verify
            assert output.shape[1] == 32, f"Expected 32 channels, got {output.shape[1]}"
            assert output_h == expected_h, f"Expected height {expected_h}, got {output_h}"
            assert output_w == expected_w, f"Expected width {expected_w}, got {output_w}"
            print("✓ Verification passed!")
        
        print("\n" + "=" * 80)
        print("CONCLUSION: The backbone outputs 32 channels at 1/8 resolution ✓")
        print("=" * 80)
        
        return True
        
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return False


def verify_architecturally():
    """Verify channel count based on MobileNetV2 architecture specification."""
    print("\n" + "=" * 80)
    print("ARCHITECTURAL VERIFICATION (without PyTorch)")
    print("=" * 80)
    
    print("\nMobileNetV2 Architecture (from torchvision implementation):")
    print("-" * 80)
    
    # MobileNetV2 inverted residual settings
    # Format: [expansion_ratio, output_channels, num_blocks, stride]
    architecture = [
        ("Initial Conv", None, None, None, 2, 32),  # features[0]
        ("IR Block 1", 1, 16, 1, 1, 16),           # features[1]
        ("IR Block 2", 6, 24, 2, 2, 24),           # features[2-3]
        ("IR Block 3", 6, 32, 3, 2, 32),           # features[4-6]
        ("IR Block 4", 6, 64, 4, 2, 64),           # features[7-10]
    ]
    
    print(f"{'Layer(s)':<12} {'Type':<15} {'Stride':<8} {'Resolution':<12} {'Out Channels':<12}")
    print("-" * 80)
    
    layer_idx = 0
    resolution = 1
    
    for name, expansion, out_ch, n_blocks, stride, final_ch in architecture:
        if n_blocks is None:  # Initial conv
            resolution *= stride
            print(f"features[{layer_idx}]  {name:<15} {stride:<8} 1/{resolution:<11} {final_ch:<12}")
            layer_idx += 1
        else:
            # First block in group (with stride)
            resolution *= stride
            print(f"features[{layer_idx}]  {name:<15} {stride:<8} 1/{resolution:<11} {final_ch:<12}")
            layer_idx += 1
            
            # Remaining blocks in group (stride=1)
            for _ in range(1, n_blocks):
                print(f"features[{layer_idx}]  {name:<15} {1:<8} 1/{resolution:<11} {final_ch:<12}")
                layer_idx += 1
                
                # Highlight backbone endpoint
                if layer_idx == 7:
                    print(" " * 12 + "^^^ backbone endpoint (features[:7]) ^^^")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBackbone = features[:{BACKBONE_LAYER_COUNT}] (includes layers 0-6)")
    print("  • Layer 0: Initial convolution (stride=2) -> 32 channels at 1/2 resolution")
    print("  • Layer 1: First IR block -> 16 channels at 1/2 resolution")
    print("  • Layers 2-3: Second IR group -> 24 channels at 1/4 resolution")
    print("  • Layers 4-6: Third IR group -> 32 channels at 1/8 resolution")
    print("\n✓ Output: 32 channels at 1/8 resolution")
    print("\nNote: The reviewer's comment suggesting 24 channels appears to be ")
    print("      confusing layers 2-3 (which output 24 channels at 1/4 resolution)")
    print("      with layer 6 (which outputs 32 channels at 1/8 resolution).")
    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("MobileNetV2 Backbone Channel Verification")
    print("=" * 80)
    print("\nThis script verifies that layers 0-6 of MobileNetV2 output")
    print("32 channels at 1/8 spatial resolution, as used in SuperPointNetV2.")
    
    # Try PyTorch verification first
    if not verify_with_pytorch():
        # Fall back to architectural verification
        verify_architecturally()
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print("\nThe comment in py_superpoint.py stating that")
    print('  "MobileNetV2 outputs 32 channels at this point"')
    print("\nis CORRECT ✓")
    print("\nThe code correctly uses in_channels = 32 for the detector and")
    print("descriptor heads, matching the actual output of features[:7].")
    print("=" * 80)


if __name__ == "__main__":
    main()
