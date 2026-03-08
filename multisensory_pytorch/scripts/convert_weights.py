"""
TensorFlow checkpoint → PyTorch state_dict converter.

Converts pretrained TF 1.x checkpoint weights from the original multisensory
codebase to PyTorch format, handling:
  - Tensor transposition (NHWC → NCHW, NDHWC → NCDHW)
  - Variable name mapping (TF scoped names → PyTorch module paths)
  - Batch normalization parameter mapping
"""

import os
import sys
import argparse
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Name mapping: TF variable name → PyTorch parameter name
# ---------------------------------------------------------------------------

def build_shift_name_map():
    """
    Build the name mapping for ShiftNet.

    TF naming convention (shift_net.py):
      im/conv1/weights:0  → image_net.conv1.conv.weight
      im/conv1/BatchNorm/gamma:0 → image_net.bn1.weight
      sf/conv1_1/weights:0 → sound_net.conv1.conv.weight
      joint/merge1/weights:0 → merge.merge_conv1.weight
      joint/logits/weights:0 → logits_conv.weight
    """
    name_map = {}

    # Sound feature network
    sf_layers = [
        ('sf/conv1_1', 'sound_net.conv1.conv', True),
        ('sf/conv1_1/BatchNorm', 'sound_net.bn1', False),
    ]
    # Block2D layers for sound
    for block_name, block_path in [
        ('sf/conv2_1_1', 'sound_net.block2'),
        ('sf/conv3_1_1', 'sound_net.block3'),
        ('sf/conv4_1_1', 'sound_net.block4'),
    ]:
        sf_layers.extend(_block2d_mapping(block_name, block_path))

    # Image feature network
    im_layers = [
        ('im/conv1', 'image_net.conv1.conv', True),
        ('im/conv1/BatchNorm', 'image_net.bn1', False),
    ]
    for block_name, block_path in [
        ('im/conv2_1', 'image_net.block2_1'),
        ('im/conv2_2', 'image_net.block2_2'),
    ]:
        im_layers.extend(_block3d_mapping(block_name, block_path))

    # Merge layers
    merge_layers = [
        ('joint/sf_conv5', 'merge.sf_conv5.conv', True),
        ('joint/sf_conv5/BatchNorm', 'merge.sf_bn5', False),
        ('joint/merge1', 'merge.merge_conv1', True),
        ('joint/merge1/BatchNorm', 'merge.merge_bn1', False),
        ('joint/merge2', 'merge.merge_conv2', True),
        ('joint/merge_bn', 'merge.merge_bn_out', False),
    ]

    # Post-merge blocks
    post_blocks = []
    for block_name, block_path in [
        ('im/conv3_1', 'block3_1'),
        ('im/conv3_2', 'block3_2'),
        ('im/conv4_1', 'block4_1'),
        ('im/conv4_2', 'block4_2'),
        ('im/conv5_1', 'block5_1'),
        ('im/conv5_2', 'block5_2'),
    ]:
        post_blocks.extend(_block3d_mapping(block_name, block_path))

    # Logits
    logits_layers = [
        ('joint/logits', 'logits_conv', True),
    ]

    all_layers = sf_layers + im_layers + merge_layers + post_blocks + logits_layers

    for tf_prefix, pt_prefix, is_conv in all_layers:
        if is_conv:
            name_map[f'{tf_prefix}/weights'] = f'{pt_prefix}.weight'
            name_map[f'{tf_prefix}/biases'] = f'{pt_prefix}.bias'
        else:
            # BatchNorm
            name_map[f'{tf_prefix}/gamma'] = f'{pt_prefix}.weight'
            name_map[f'{tf_prefix}/beta'] = f'{pt_prefix}.bias'
            name_map[f'{tf_prefix}/moving_mean'] = f'{pt_prefix}.running_mean'
            name_map[f'{tf_prefix}/moving_variance'] = f'{pt_prefix}.running_var'

    return name_map


def _block2d_mapping(tf_prefix, pt_prefix):
    """Generate name mappings for a Block2D."""
    mappings = [
        (f'{tf_prefix}/conv1', f'{pt_prefix}.conv1', True),
        (f'{tf_prefix}/conv1/BatchNorm', f'{pt_prefix}.bn1', False),
        (f'{tf_prefix}/conv2', f'{pt_prefix}.conv2', True),
        (f'{tf_prefix}/bn', f'{pt_prefix}.bn_out', False),
    ]
    # Shortcut (if it exists)
    mappings.extend([
        (f'{tf_prefix}/shortcut', f'{pt_prefix}.shortcut.0', True),
        (f'{tf_prefix}/shortcut/BatchNorm', f'{pt_prefix}.shortcut.1', False),
    ])
    return mappings


def _block3d_mapping(tf_prefix, pt_prefix):
    """Generate name mappings for a Block3D."""
    mappings = [
        (f'{tf_prefix}/conv1', f'{pt_prefix}.conv1', True),
        (f'{tf_prefix}/conv1/BatchNorm', f'{pt_prefix}.bn1', False),
        (f'{tf_prefix}/conv2', f'{pt_prefix}.conv2', True),
        (f'{tf_prefix}/bn', f'{pt_prefix}.bn_out', False),
    ]
    mappings.extend([
        (f'{tf_prefix}/shortcut', f'{pt_prefix}.shortcut.0', True),
        (f'{tf_prefix}/shortcut/BatchNorm', f'{pt_prefix}.shortcut.1', False),
    ])
    return mappings


def build_sourcesep_name_map():
    """Build name mapping for SourceSep U-Net."""
    name_map = {}

    # Encoder layers
    for i in range(9):
        tf_prefix = f'gen/conv{i + 1}'
        pt_prefix = f'encoder.layers.{i}'
        name_map[f'{tf_prefix}/weights'] = f'{pt_prefix}.weight'
        name_map[f'{tf_prefix}/biases'] = f'{pt_prefix}.bias'
        pt_bn_prefix = f'encoder.bns.{i}'
        name_map[f'{tf_prefix}/BatchNorm/gamma'] = f'{pt_bn_prefix}.weight'
        name_map[f'{tf_prefix}/BatchNorm/beta'] = f'{pt_bn_prefix}.bias'
        name_map[f'{tf_prefix}/BatchNorm/moving_mean'] = f'{pt_bn_prefix}.running_mean'
        name_map[f'{tf_prefix}/BatchNorm/moving_variance'] = f'{pt_bn_prefix}.running_var'

    # Decoder layers
    for i in range(8):
        tf_prefix = f'gen/deconv{i + 1}'
        pt_prefix = f'decoder.deconvs.{i}'
        name_map[f'{tf_prefix}/weights'] = f'{pt_prefix}.weight'
        name_map[f'{tf_prefix}/biases'] = f'{pt_prefix}.bias'
        pt_bn_prefix = f'decoder.bns.{i}'
        name_map[f'{tf_prefix}/BatchNorm/gamma'] = f'{pt_bn_prefix}.weight'
        name_map[f'{tf_prefix}/BatchNorm/beta'] = f'{pt_bn_prefix}.bias'
        name_map[f'{tf_prefix}/BatchNorm/moving_mean'] = f'{pt_bn_prefix}.running_mean'
        name_map[f'{tf_prefix}/BatchNorm/moving_variance'] = f'{pt_bn_prefix}.running_var'

    # Output heads
    name_map['gen/fg/weights'] = 'fg_head.deconv.weight'
    name_map['gen/fg/biases'] = 'fg_head.deconv.bias'
    name_map['gen/bg/weights'] = 'bg_head.deconv.weight'
    name_map['gen/bg/biases'] = 'bg_head.deconv.bias'

    return name_map


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def transpose_conv_weight(tensor, ndim):
    """
    Transpose convolution weights from TF to PyTorch layout.

    TF 2D conv: (H, W, C_in, C_out) → PT: (C_out, C_in, H, W)
    TF 3D conv: (D, H, W, C_in, C_out) → PT: (C_out, C_in, D, H, W)
    TF 2D deconv: (H, W, C_out, C_in) → PT: (C_in, C_out, H, W)
    """
    if ndim == 4:
        return tensor.transpose(3, 2, 0, 1)
    elif ndim == 5:
        return tensor.transpose(4, 3, 0, 1, 2)
    elif ndim == 2:
        return tensor.transpose(1, 0)
    return tensor


def convert_checkpoint(tf_checkpoint_path, name_map, is_deconv_set=None):
    """
    Read TF checkpoint and create PyTorch state_dict.

    Args:
        tf_checkpoint_path: path to TF checkpoint (without extension)
        name_map: dict mapping TF var names → PT param names
        is_deconv_set: set of TF variable names that are deconvolutions
                       (deconvs have different weight layout)

    Returns:
        state_dict: OrderedDict for PyTorch model
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: tensorflow is required for weight conversion.")
        print("Install with: pip install tensorflow")
        sys.exit(1)

    if is_deconv_set is None:
        is_deconv_set = set()

    reader = tf.train.load_checkpoint(tf_checkpoint_path)
    var_to_shape = reader.get_variable_to_shape_map()

    state_dict = {}
    converted = 0
    skipped = 0

    for tf_name, pt_name in name_map.items():
        # Remove :0 suffix if present
        tf_key = tf_name.replace(':0', '')

        if tf_key not in var_to_shape:
            print(f"  SKIP (not in checkpoint): {tf_key}")
            skipped += 1
            continue

        tensor = reader.get_tensor(tf_key)
        original_shape = tensor.shape

        # Transpose convolution weights
        if 'weight' in pt_name and 'running' not in pt_name:
            if 'BatchNorm' not in tf_key and 'gamma' not in tf_key:
                if tf_key in is_deconv_set:
                    # Deconv: TF (H, W, C_out, C_in) → PT (C_in, C_out, H, W)
                    tensor = transpose_conv_weight(tensor, len(tensor.shape))
                else:
                    # Regular conv
                    tensor = transpose_conv_weight(tensor, len(tensor.shape))

        state_dict[pt_name] = torch.from_numpy(tensor.copy())
        converted += 1

    print(f"\nConverted {converted} variables, skipped {skipped}")
    return state_dict


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert TF checkpoints to PyTorch state_dict"
    )
    parser.add_argument("--tf_checkpoint", type=str, required=True,
                        help="Path to TF checkpoint (e.g., ../results/nets/shift/net.tf-650000)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["shift", "sourcesep"],
                        help="Which model to convert")
    parser.add_argument("--list_vars", action="store_true",
                        help="List TF checkpoint variables and exit")
    args = parser.parse_args()

    if args.list_vars:
        import tensorflow as tf
        reader = tf.train.load_checkpoint(args.tf_checkpoint)
        for name, shape in sorted(reader.get_variable_to_shape_map().items()):
            print(f"  {name}: {shape}")
        return

    print(f"Converting {args.model_type} from {args.tf_checkpoint}")

    if args.model_type == "shift":
        name_map = build_shift_name_map()
        is_deconv = set()
    elif args.model_type == "sourcesep":
        name_map = build_sourcesep_name_map()
        # Mark deconv layers
        is_deconv = {f'gen/deconv{i + 1}/weights' for i in range(8)}
        is_deconv.add('gen/fg/weights')
        is_deconv.add('gen/bg/weights')
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    state_dict = convert_checkpoint(args.tf_checkpoint, name_map, is_deconv)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"Saved PyTorch state_dict to {args.output}")

    # Verify loading
    print("\nVerifying...")
    if args.model_type == "shift":
        from ..models.shift_net import ShiftNet
        from ..utils.params import shift_v1
        model = ShiftNet(shift_v1())
    else:
        from ..models.sourcesep import SourceSepUNet
        from ..utils.params import sep_full
        model = SourceSepUNet(sep_full())

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys ({len(missing)}):")
    for k in missing[:10]:
        print(f"  {k}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    print(f"Unexpected keys ({len(unexpected)}):")
    for k in unexpected[:10]:
        print(f"  {k}")
    if len(unexpected) > 10:
        print(f"  ... and {len(unexpected) - 10} more")

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
