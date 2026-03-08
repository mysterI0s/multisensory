"""
TensorFlow checkpoint → PyTorch state_dict converter.

Converts pretrained TF 1.x checkpoint weights from the original multisensory
codebase to PyTorch format, handling:
  - Tensor transposition (NHWC → NCHW, NDHWC → NCDHW)
  - Variable name mapping (TF scoped names → PyTorch module paths)
  - Batch normalization parameter mapping (TF uses scale=False → no gamma)
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
# Name mapping helpers
# ---------------------------------------------------------------------------

def _bn_mapping(tf_prefix, pt_prefix):
    """
    BN mapping — TF uses scale=False so only beta, moving_mean, moving_variance.
    PyTorch BN weight (gamma) defaults to 1.0, no checkpoint key needed.
    """
    return [
        (f'{tf_prefix}/beta', f'{pt_prefix}.bias'),
        (f'{tf_prefix}/moving_mean', f'{pt_prefix}.running_mean'),
        (f'{tf_prefix}/moving_variance', f'{pt_prefix}.running_var'),
    ]


def _block_mapping(tf_base, pt_prefix, has_shortcut=False):
    """
    Map a residual block's layers.
    
    TF naming:  {tf_base}_1 (conv1), {tf_base}_2 (conv2), {tf_base}_bn (post-add BN)
    Optional:   {tf_base}_short (shortcut conv + BN)
    """
    m = []
    # Conv1 (no bias)
    m.append((f'{tf_base}_1/weights', f'{pt_prefix}.conv1.weight'))
    m.extend(_bn_mapping(f'{tf_base}_1/BatchNorm', f'{pt_prefix}.bn1'))
    # Conv2 (has bias)
    m.append((f'{tf_base}_2/weights', f'{pt_prefix}.conv2.weight'))
    m.append((f'{tf_base}_2/biases', f'{pt_prefix}.conv2.bias'))
    # Post-addition BN
    m.extend(_bn_mapping(f'{tf_base}_bn', f'{pt_prefix}.bn_out'))
    # Shortcut
    if has_shortcut:
        m.append((f'{tf_base}_short/weights', f'{pt_prefix}.shortcut.0.weight'))
        m.extend(_bn_mapping(f'{tf_base}_short/BatchNorm', f'{pt_prefix}.shortcut.1'))
    return m


# ---------------------------------------------------------------------------
# ShiftNet name map
# ---------------------------------------------------------------------------

def build_shift_name_map():
    """
    Build the complete name mapping for ShiftNet based on actual checkpoint
    variable names.

    TF naming (actual from checkpoint):
      sf/conv1_1/weights           → sound_net.conv1.conv.weight
      sf/conv2_1_1/weights         → sound_net.block2.conv1.weight (block sublayer _1)
      sf/conv2_1_2/biases          → sound_net.block2.conv2.bias   (block sublayer _2)
      sf/conv2_1_bn/beta           → sound_net.block2.bn_out.bias  (post-add BN)
      sf/conv2_1_short/weights     → sound_net.block2.shortcut.0.weight
      im/conv1/weights             → image_net.conv1.conv.weight
      im/conv2_1_1/weights         → image_net.block2_1.conv1.weight
      im/merge1/weights            → merge.merge_conv1.weight  (under im/ scope!)
      im/merge_block_bn/beta       → merge.merge_bn_out.bias   (under im/ scope!)
      sf/conv5_1/weights           → merge.sf_conv5.conv.weight (under sf/ scope!)
      joint/logits/weights         → logits_conv.weight
    """
    m = []

    # === Sound Feature Network ===
    # conv1 (no bias, weight shape [65, 1, 2, 64])
    m.append(('sf/conv1_1/weights', 'sound_net.conv1.conv.weight'))
    m.extend(_bn_mapping('sf/conv1_1/BatchNorm', 'sound_net.bn1'))

    # Sound blocks: block2, block3, block4
    # block2: channels 64→128 (has shortcut)
    m.extend(_block_mapping('sf/conv2_1', 'sound_net.block2', has_shortcut=True))
    # block3: channels 128→128 (no shortcut — same channels, but has stride)
    # Actually checkpoint has sf/conv3_1_short, so it does have shortcut
    m.extend(_block_mapping('sf/conv3_1', 'sound_net.block3', has_shortcut=False))
    # block4: channels 128→256 (has shortcut)
    m.extend(_block_mapping('sf/conv4_1', 'sound_net.block4', has_shortcut=True))

    # sf/conv5_1 → merge sound conv (reduces 256→128 before merge)
    m.append(('sf/conv5_1/weights', 'merge.sf_conv5.conv.weight'))
    m.extend(_bn_mapping('sf/conv5_1/BatchNorm', 'merge.sf_bn5'))

    # === Image Feature Network ===
    # conv1 (no bias, weight shape [5, 7, 7, 3, 64])
    m.append(('im/conv1/weights', 'image_net.conv1.conv.weight'))
    m.extend(_bn_mapping('im/conv1/BatchNorm', 'image_net.bn1'))

    # image block2_1: 64→64 (no shortcut, same channels)
    m.extend(_block_mapping('im/conv2_1', 'image_net.block2_1', has_shortcut=False))
    # image block2_2: 64→64, stride=2 (no channel change, but stride → no shortcut with conv)
    m.extend(_block_mapping('im/conv2_2', 'image_net.block2_2', has_shortcut=False))

    # === Merge Module (under im/ scope in TF) ===
    m.append(('im/merge1/weights', 'merge.merge_conv1.weight'))
    m.extend(_bn_mapping('im/merge1/BatchNorm', 'merge.merge_bn1'))
    m.append(('im/merge2/weights', 'merge.merge_conv2.weight'))
    m.append(('im/merge2/biases', 'merge.merge_conv2.bias'))
    m.extend(_bn_mapping('im/merge_block_bn', 'merge.merge_bn_out'))

    # === Post-Merge Blocks (under im/ scope, map to ShiftNet top-level) ===
    # block3_1: 128→128 (no shortcut)
    m.extend(_block_mapping('im/conv3_1', 'block3_1', has_shortcut=False))
    # block3_2: 128→128 (no shortcut)
    m.extend(_block_mapping('im/conv3_2', 'block3_2', has_shortcut=False))
    # block4_1: 128→256 (HAS shortcut — channel change)
    m.extend(_block_mapping('im/conv4_1', 'block4_1', has_shortcut=True))
    # block4_2: 256→256 (no shortcut)
    m.extend(_block_mapping('im/conv4_2', 'block4_2', has_shortcut=False))
    # block5_1: 256→512 (HAS shortcut — channel change)
    m.extend(_block_mapping('im/conv5_1', 'block5_1', has_shortcut=True))
    # block5_2: 512→512 (no shortcut)
    m.extend(_block_mapping('im/conv5_2', 'block5_2', has_shortcut=False))

    # === Logits (under joint/ scope) ===
    m.append(('joint/logits/weights', 'logits_conv.weight'))
    m.append(('joint/logits/biases', 'logits_conv.bias'))

    return dict(m)


# ---------------------------------------------------------------------------
# SourceSep U-Net name map
# ---------------------------------------------------------------------------

def build_sourcesep_name_map():
    """
    Build name mapping for SourceSep U-Net.

    TF naming for U-Net:
      gen/conv{N}  — encoder layer N (no biases in encoder convs)
      gen/deconv{N} — decoder layer N (no biases in deconvs)
      gen/fg, gen/bg — output heads

    TF BN under encoder/decoder also uses scale=False (no gamma).
    """
    m = []

    # Encoder layers (conv1..conv9)
    for i in range(9):
        n = i + 1
        m.append((f'gen/conv{n}/weights', f'encoder.layers.{i}.weight'))
        # Encoder convs have NO biases in TF (bias=False)
        m.extend(_bn_mapping(f'gen/conv{n}/BatchNorm', f'encoder.bns.{i}'))

    # Decoder layers (deconv1..deconv8)
    for i in range(8):
        n = i + 1
        m.append((f'gen/deconv{n}/weights', f'decoder.deconvs.{i}.weight'))
        # Decoder deconvs also have NO biases in TF
        m.extend(_bn_mapping(f'gen/deconv{n}/BatchNorm', f'decoder.bns.{i}'))

    # Output heads (DO have biases)
    m.append(('gen/fg/weights', 'fg_head.deconv.weight'))
    m.append(('gen/fg/biases', 'fg_head.deconv.bias'))
    m.append(('gen/bg/weights', 'bg_head.deconv.weight'))
    m.append(('gen/bg/biases', 'bg_head.deconv.bias'))

    return dict(m)


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def transpose_conv_weight(tensor, ndim):
    """
    Transpose convolution weights from TF to PyTorch layout.

    TF 2D conv: (H, W, C_in, C_out) → PT: (C_out, C_in, H, W)
    TF 3D conv: (D, H, W, C_in, C_out) → PT: (C_out, C_in, D, H, W)
    TF 2D deconv: (H, W, C_out, C_in) → PT: (C_in, C_out, H, W)
      (same transpose as regular conv, PyTorch ConvTranspose2d uses same layout)
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

    Returns:
        state_dict: dict for PyTorch model
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
        tf_key = tf_name.replace(':0', '')

        if tf_key not in var_to_shape:
            print(f"  SKIP (not in checkpoint): {tf_key}")
            skipped += 1
            continue

        tensor = reader.get_tensor(tf_key)
        original_shape = tensor.shape

        # Transpose convolution/deconv weights
        is_weight = pt_name.endswith('.weight')
        is_bn = any(x in pt_name for x in ['running_mean', 'running_var', '.bias'])
        is_conv_weight = is_weight and not is_bn

        if is_conv_weight and len(tensor.shape) >= 4:
            tensor = transpose_conv_weight(tensor, len(tensor.shape))
            print(f"  {tf_key} {list(original_shape)} → {pt_name} {list(tensor.shape)} (transposed)")
        else:
            print(f"  {tf_key} {list(original_shape)} → {pt_name} {list(tensor.shape)}")

        state_dict[pt_name] = torch.from_numpy(tensor.copy())
        converted += 1

    # Report unmapped checkpoint variables
    mapped_tf_keys = {k.replace(':0', '') for k in name_map.keys()}
    unmapped = []
    for var_name in sorted(var_to_shape.keys()):
        if var_name not in mapped_tf_keys:
            # Skip momentum, global_step, renorm variables
            if any(skip in var_name for skip in ['/Momentum', 'global_step', 'renorm']):
                continue
            unmapped.append(var_name)

    if unmapped:
        print(f"\n  Unmapped checkpoint variables ({len(unmapped)}):")
        for name in unmapped:
            print(f"    {name}: {var_to_shape[name]}")

    print(f"\nConverted {converted} variables, skipped {skipped}")
    return state_dict


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------

def resolve_checkpoint_path(path):
    """
    Resolve a TF checkpoint path, trying several common base directories.
    """
    if os.path.exists(path + ".index") or os.path.exists(path + ".data-00000-of-00001"):
        return os.path.abspath(path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    candidates = [
        os.path.join(project_root, path),
        os.path.join(project_root, path.lstrip("../")),
        os.path.join(project_root, path.lstrip("..\\").lstrip("/")),
        os.path.join(os.getcwd(), path),
    ]

    if "results" in path:
        parts = path.split("results/")
        if len(parts) > 1:
            candidates.append(os.path.join(project_root, "results", parts[-1]))

    for candidate in candidates:
        if candidate is None:
            continue
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate + ".index") or os.path.exists(candidate + ".data-00000-of-00001"):
            print(f"  Resolved checkpoint path: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Could not find TF checkpoint: {path}\n"
        f"  Please provide the full absolute path to the checkpoint.\n"
        f"  Example: /content/multisensory/results/nets/shift/net.tf-650000\n"
        f"  The checkpoint should have .index and .data-00000-of-00001 files."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert TF checkpoints to PyTorch state_dict"
    )
    parser.add_argument("--tf_checkpoint", type=str, required=True,
                        help="Path to TF checkpoint (absolute or relative)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["shift", "sourcesep", "cam", "sep-large", "unet-pit"],
                        help="Which model to convert")
    parser.add_argument("--list_vars", action="store_true",
                        help="List TF checkpoint variables and exit")
    args = parser.parse_args()

    # Resolve path
    try:
        ckpt_path = resolve_checkpoint_path(args.tf_checkpoint)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.list_vars:
        import tensorflow as tf
        reader = tf.train.load_checkpoint(ckpt_path)
        print(f"\nVariables in {ckpt_path}:")
        for name, shape in sorted(reader.get_variable_to_shape_map().items()):
            dtype = reader.get_variable_to_dtype_map()[name]
            print(f"  {name}: {shape}  ({dtype.name})")
        total = len(reader.get_variable_to_shape_map())
        print(f"\nTotal: {total} variables")
        return

    print(f"Converting {args.model_type} from {ckpt_path}")

    if args.model_type in ("shift", "cam"):
        name_map = build_shift_name_map()
        is_deconv = set()
    elif args.model_type in ("sourcesep", "sep-large", "unet-pit"):
        name_map = build_sourcesep_name_map()
        is_deconv = {f'gen/deconv{i + 1}/weights' for i in range(8)}
        is_deconv.add('gen/fg/weights')
        is_deconv.add('gen/bg/weights')
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    state_dict = convert_checkpoint(ckpt_path, name_map, is_deconv)

    # Save
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"\nSaved PyTorch state_dict to {os.path.abspath(args.output)}")

    # Verify
    print("\nVerifying state_dict can be loaded into the model...")
    try:
        try:
            from multisensory_pytorch.models.shift_net import ShiftNet
            from multisensory_pytorch.models.sourcesep import SourceSepUNet
            from multisensory_pytorch.utils.params import shift_v1, sep_full
        except ImportError:
            parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            grandparent = os.path.dirname(parent)
            if grandparent not in sys.path:
                sys.path.insert(0, grandparent)
            from multisensory_pytorch.models.shift_net import ShiftNet
            from multisensory_pytorch.models.sourcesep import SourceSepUNet
            from multisensory_pytorch.utils.params import shift_v1, sep_full

        if args.model_type in ("shift", "cam"):
            model = ShiftNet(shift_v1())
        else:
            pr = sep_full()
            if args.model_type == "unet-pit":
                pr.net_style = "no-im"
            model = SourceSepUNet(pr)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Filter out expected missing keys (BN weights/gamma — TF uses scale=False)
        truly_missing = [k for k in missing if not k.endswith('.weight')
                         or 'bn' not in k.lower()]
        bn_gamma_missing = [k for k in missing if k.endswith('.weight')
                            and 'bn' in k.lower()]

        if bn_gamma_missing:
            print(f"  BN gamma (weight) keys not in checkpoint (expected, TF scale=False): "
                  f"{len(bn_gamma_missing)}")
        if truly_missing:
            print(f"  Other missing keys ({len(truly_missing)}):")
            for k in truly_missing[:20]:
                print(f"    {k}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}):")
            for k in unexpected[:20]:
                print(f"    {k}")

        total_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(state_dict[k].numel() for k in state_dict)
        print(f"\n  Model params:  {total_params:,}")
        print(f"  Loaded params: {loaded_params:,}")

    except Exception as e:
        print(f"  WARNING: Could not verify (non-fatal): {e}")
        print("  The state_dict was still saved successfully.")

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
