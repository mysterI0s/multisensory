# Multisensory: TensorFlow → PyTorch Complete Migration Plan

> **Paper**: *Audio-Visual Scene Analysis with Self-Supervised Multisensory Features* (Owens & Efros, 2018)
> **Original framework**: TensorFlow 1.8 (compat v1) with tf-slim
> **Target framework**: PyTorch 2.x with modern best practices

---

# Phase 1 — Repository Architecture Analysis

## 1.1 Architecture Map

```mermaid
graph TD
    subgraph "Model Modules"
        SN[shift_net.py<br/>795 lines<br/>ShiftNet 3D CNN]
        SS[sourcesep.py<br/>1107 lines<br/>Source Separation U-Net]
        VC[videocls.py<br/>419 lines<br/>Video Classification]
    end
    subgraph "Dataset Modules"
        SD[shift_dset.py<br/>354 lines<br/>Shift dataset reader]
        SE[sep_dset.py<br/>400 lines<br/>Separation dataset reader]
    end
    subgraph "Param Modules"
        SP[shift_params.py<br/>259 lines<br/>Shift net params]
        SEP[sep_params.py<br/>148 lines<br/>Separation params]
    end
    subgraph "Utilities"
        TF[tfutil.py<br/>768 lines<br/>TF utilities]
        SR[soundrep.py<br/>99 lines<br/>STFT / Griffin-Lim]
    end
    subgraph "Inference"
        SV[sep_video.py<br/>583 lines<br/>Video inference CLI]
        SX[shift_example.py<br/>33 lines<br/>CAM example]
    end
    subgraph "Support Library"
        UT[aolib/util.py<br/>4176 lines<br/>General utilities]
        IM[aolib/img.py<br/>Image utilities]
        IT[aolib/imtable.py<br/>HTML table display]
        SND[aolib/sound.py<br/>Sound I/O]
        SB[aolib/subband.py<br/>Subband filters]
    end

    SS --> SN
    SS --> SR
    SS --> SE
    SS --> TF
    SN --> SD
    SN --> TF
    VC --> SN
    SV --> SS
    SV --> SN
    SX --> SN
```

## 1.2 Model Architectures

### ShiftNet (Audio-Visual Correspondence Network)

| Component | Details |
|-----------|---------|
| **File** | [shift_net.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py) |
| **Architecture** | Two-stream 3D CNN (ResNet-18 variant) + 2D sound CNN, merged with audio-visual fusion |
| **Purpose** | Self-supervised audio-visual correspondence prediction (binary classification) |
| **Image stream** | `conv1` (5×7×7, stride 2) → max_pool → 4 residual stages (`conv2_1`, `conv2_2`, `conv3_1–2`, `conv4_1–2`, `conv5_1–2`), each with 3D ResNet blocks ([block3](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py#545-582)) |
| **Sound stream** | 1D→2D via [normalize_sfs](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#400-413) → `conv1_1` (65×1, stride 4) → max_pool → 3 ResNet-2D blocks (`conv2_1`, `conv3_1`, `conv4_1`) each (15×1) |
| **Merge** | Fractional max-pooling to align temporal dimensions → 2D conv → tile spatially → concatenate with image features → two 1×1×1 3D convs with residual |
| **Head** | Global average pooling → 1×1×1 conv → sigmoid for binary classification |
| **CAM** | Class Activation Map via reusing the logits conv on the last conv layer |
| **Weight init** | tf-slim `variance_scaling_initializer` |
| **Batch Norm** | slim `batch_norm` with renorm=True, decay=0.9997, epsilon=0.001 |
| **Regularization** | L2 weight decay (1e-5) |

### SourceSep (Audio-Visual Source Separation Network)

| Component | Details |
|-----------|---------|
| **File** | [sourcesep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py) |
| **Architecture** | U-Net operating on spectrograms, conditioned on ShiftNet video features |
| **Input** | Spectrogram (magnitude + phase) from STFT of mixed audio |
| **Encoder** | 9 conv layers (gen/conv1–conv9, kernel 4, LeakyReLU 0.2), strides [1,2] to 2 |
| **Decoder** | 8 deconv layers (gen/deconv1–deconv8) with skip connections from encoder |
| **Video conditioning** | ShiftNet `scales[0–2]` features merged at encoder levels 3, 4, 5 via [merge_level()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py#979-994): GAP over spatial dims → resize → tile → concatenate |
| **Output heads** | Two separate deconv heads: `gen/fg` and `gen/bg`, each producing 2-channel output (spec + phase) |
| **Post-processing** | `tanh()` → unnormalize → pad → STFT → iSTFT → waveform |
| **Loss** | L1 on normalized spectrogram + L1 on normalized phase (weighted), optional PIT loss, optional GAN (discriminator) |
| **Discriminator** | Patch-based 2D CNN (4 conv layers) on spectrograms |

### Video Classification (Fine-tuning Module)

| Component | Details |
|-----------|---------|
| **File** | [videocls.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/videocls.py) |
| **Purpose** | Fine-tune pretrained ShiftNet for action recognition (e.g., UCF-101) |
| **Variants** | [shift](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_params.py#89-175) (uses ShiftNet backbone), `i3d` (uses I3D backbone), `c3d` (uses C3D) |
| **Head** | GAP → 1×1×1 conv → softmax cross-entropy with optional label smoothing |

## 1.3 Training Pipeline

| Aspect | Implementation |
|--------|---------------|
| **Session** | `tf.InteractiveSession` with `ConfigProto(allow_soft_placement=True)` |
| **Data loading** | `tf.TFRecordReader` → `tf.train.string_input_producer` → queue-based batching via `tf.train.shuffle_batch_join` |
| **Multi-GPU** | Manual tower-based gradient averaging ([average_grads](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#76-103) in [tfutil.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py)), `tf.split` across GPUs |
| **Optimizer** | Adam (lr=1e-4, source sep) or Momentum (lr=1e-2, shift net), with step-based LR decay (`gamma^floor(step/step_size)`) |
| **Gradient clipping** | `tf.clip_by_global_norm` (sep: 10.0, shift: 5.0) |
| **Checkpointing** | `tf.train.Saver`, fast (every 1000 steps) and slow (every 10000 steps, max_to_keep=1000) |
| **Summaries** | `tf.summary.merge_all` with `FileWriter` for TensorBoard |
| **Training loop** | Manual `while True` loop with `sess.run([train_op] + loss_ops)` |

## 1.4 Inference Pipeline

| Aspect | Implementation |
|--------|---------------|
| **Wrapper** | [NetClf](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py#730-795) class in each module — builds graph, creates placeholders, restores weights |
| **Placeholders** | `tf.placeholder` for images `[1, T, H, W, 3]` and audio `[1, N, 2]` |
| **Execution** | `sess.run(op, feed_dict)` |
| **Video processing** | FFmpeg subprocess calls to extract frames and audio, then sliding-window processing in [sep_video.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sep_video.py) |

## 1.5 TensorFlow-Specific Constructs

| Construct | Location | Usage |
|-----------|----------|-------|
| `tf.compat.v1` + `disable_v2_behavior()` | All files | TF1 compatibility mode |
| `tf_slim` (`slim.conv2d`, `slim.convolution`, `slim.batch_norm`, etc.) | [shift_net.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py), [sourcesep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py), [videocls.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/videocls.py), [tfutil.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py) | All convolution and batch norm operations |
| `slim.arg_scope` | [shift_net.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py), [sourcesep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py) | Default parameters for conv blocks |
| `tf.train.Saver` / `tf.train.latest_checkpoint` | All Model classes | Checkpoint save/restore |
| `tf.train.Coordinator` / `tf.train.start_queue_runners` | All training code | Queue-based data loading |
| `tf.TFRecordReader` / `tf.train.string_input_producer` | [sep_dset.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sep_dset.py), [shift_dset.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_dset.py) | TFRecord data reading |
| `tf.signal.stft` / `tf.signal.inverse_stft` | [soundrep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/soundrep.py), [sourcesep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py) | Audio STFT/iSTFT |
| `tf.nn.max_pool3d`, `tf.nn.fractional_max_pool` | [shift_net.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py) | Pooling operations |
| `tf.py_func` | [sourcesep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py) | Python callbacks for visualization |
| `tf.profiler` | [sourcesep.py](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py) | Performance profiling |

## 1.6 Pretrained Weights

| Model | Format | Path | Steps |
|-------|--------|------|-------|
| ShiftNet | TF1 checkpoint (`.tf-650000.{data, index, meta}`) | `../results/nets/shift/net.tf-650000` | 650,000 |
| CAM variant | TF1 checkpoint | `../results/nets/cam/net.tf-675000` | 675,000 |
| Source Sep (full) | TF1 checkpoint | `../results/nets/sep/full/net.tf-160000` | 160,000 |
| Source Sep (unet-pit) | TF1 checkpoint | `../results/nets/sep/unet-pit/net.tf-160000` | 160,000 |
| Download URL | ZIP archive | `http://people.eecs.berkeley.edu/~owens/multisensory-nets.zip` | — |

---

# Phase 2 — PyTorch Architecture Design

## 2.1 Proposed Directory Layout

```
multisensory_pytorch/
├── models/
│   ├── __init__.py
│   ├── shift_net.py      # ShiftNet: 3D ResNet + 2D Sound CNN + Merge
│   ├── sourcesep.py      # SourceSep: U-Net + video conditioning
│   ├── discriminator.py  # Patch discriminator for GAN training
│   ├── video_cls.py      # Video classification fine-tuning head
│   └── blocks.py         # Shared residual blocks (Block2D, Block3D)
├── datasets/
│   ├── __init__.py
│   ├── shift_dataset.py  # torch Dataset for shift net
│   ├── sep_dataset.py    # torch Dataset for source separation
│   └── transforms.py     # Audio/video augmentations
├── training/
│   ├── __init__.py
│   ├── train_shift.py    # Training loop for ShiftNet
│   ├── train_sep.py      # Training loop for SourceSep  
│   ├── train_cls.py      # Training loop for video classification
│   └── scheduler.py      # LR scheduling utilities
├── losses/
│   ├── __init__.py
│   ├── separation.py     # L1 spec + phase loss, PIT loss
│   ├── adversarial.py    # GAN loss (sigmoid cross-entropy)
│   └── classification.py # Label smoothing, softmax CE
├── inference/
│   ├── __init__.py
│   ├── sep_video.py      # Video source separation CLI
│   └── shift_example.py  # CAM visualization example
├── utils/
│   ├── __init__.py
│   ├── audio.py          # STFT, iSTFT, Griffin-Lim, normalize_rms
│   ├── video.py          # Frame extraction, heatmap overlay
│   ├── params.py         # Params dataclass (replaces ut.Struct usage)
│   └── misc.py           # Moving average, checkpointing helpers
├── scripts/
│   ├── convert_weights.py # TF checkpoint → PyTorch state_dict
│   └── validate_weights.py # Layer-by-layer weight comparison
├── configs/
│   ├── shift_v1.yaml
│   ├── sep_full.yaml
│   └── sep_unet_pit.yaml
└── requirements.txt
```

## 2.2 Component Mapping Table

| TensorFlow (Current) | PyTorch (Target) | Notes |
|-----------------------|------------------|-------|
| `tf.compat.v1` session-based execution | Eager mode (default) | No graph construction needed |
| `slim.conv2d` / `slim.convolution` | `nn.Conv2d` / `nn.Conv3d` | Explicit `padding` parameter |
| `slim.conv2d_transpose` | `nn.ConvTranspose2d` | `output_padding` may be needed |
| `slim.batch_norm` | `nn.BatchNorm2d` / `nn.BatchNorm3d` | Different momentum convention: TF `decay=0.9997` → PyTorch `momentum=0.0003` |
| `slim.arg_scope` | Constructor defaults / helper functions | No direct equivalent; use [__init__](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sep_video.py#13-18) defaults |
| `slim.l2_regularizer` | [weight_decay](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#20-22) parameter in optimizer | Move from per-layer to optimizer-level |
| `tf.nn.max_pool3d` | `nn.MaxPool3d` | Same semantics |
| `tf.nn.fractional_max_pool` | `nn.FractionalMaxPool2d` | PyTorch has 2D variant; need adaptive pooling or custom for exact match |
| `tf.train.AdamOptimizer` | `torch.optim.Adam` | Default epsilon differs (TF: 1e-8, PyTorch: 1e-8) |
| `tf.train.MomentumOptimizer` | `torch.optim.SGD(momentum=0.9)` | Equivalent |
| `tf.clip_by_global_norm` | `torch.nn.utils.clip_grad_norm_` | Equivalent |
| `tf.train.Saver` | `torch.save` / `torch.load` | state_dict-based |
| `tf.summary.FileWriter` | `torch.utils.tensorboard.SummaryWriter` | Drop-in replacement |
| `tf.TFRecordReader` + queues | `torch.utils.data.Dataset` + `DataLoader` | Much cleaner API |
| `tf.signal.stft` | `torch.stft` | Different default behaviors (window, padding) |
| `tf.signal.inverse_stft` | `torch.istft` | Ensure matching parameters |
| `tf.image.decode_jpeg` | `PIL` / `torchvision.io` | In dataset `__getitem__` |
| `tf.image.resize_images` | `F.interpolate` | Default interpolation mode differs |
| `tf.placeholder` + `sess.run` | Direct function call with tensors | Pythonic eager execution |
| `tf.py_func` | Direct Python callback | No wrapper needed |
| `tf.get_variable` / `tf.variable_scope` | `nn.Module` parameters | Automatic scoping |
| `tf.train.Coordinator` / queue runners | `DataLoader` worker processes | Built-in multiprocessing |

## 2.3 Detailed PyTorch Model Designs

### Block3D (3D Residual Block)

```python
class Block3D(nn.Module):
    """3D ResNet block equivalent to shift_net.block3()"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3),
                 stride=1, rate=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.needs_shortcut = (stride != 1 or in_channels != out_channels)
        
        if self.needs_shortcut:
            if stride != 1 and in_channels == out_channels:
                self.shortcut = nn.MaxPool3d(kernel_size=1, stride=stride)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_channels)
                )
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size,
                               stride=stride, padding='same' if stride==1 else ...,
                               dilation=rate, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size,
                               padding='same', dilation=rate, bias=False)
        # No BN after conv2 — applied after residual addition
        
        if use_bn:
            self.bn_out = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        shortcut = self.shortcut(x) if self.needs_shortcut else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)  # no activation, no BN
        out = shortcut + out
        if self.use_bn:
            out = F.relu(self.bn_out(out))
        else:
            out = F.relu(out)
        return out
```

### ShiftNet

```python
class ShiftNet(nn.Module):
    """Audio-visual correspondence network"""
    def __init__(self, pr):
        super().__init__()
        # Sound feature sub-network (2D convolutions)
        self.sf_conv1 = nn.Conv2d(1, 64, (65, 1), stride=4)
        self.sf_pool = nn.MaxPool2d((4, 1))
        self.sf_block2 = Block2D(64, 128, (15,1), stride=(4,1))
        self.sf_block3 = Block2D(128, 128, (15,1), stride=(4,1))
        self.sf_block4 = Block2D(128, 256, (15,1), stride=(4,1))
        self.sf_conv5 = nn.Conv2d(256, 128, ...)  # merge prep
        
        # Image stream (3D convolutions)
        self.im_conv1 = nn.Conv3d(3, 64, (5,7,7), stride=2, padding=...)
        self.im_pool = nn.MaxPool3d((1,3,3), stride=(1,2,2))
        self.im_block2_1 = Block3D(64, 64, (3,3,3))
        self.im_block2_2 = Block3D(64, 64, (3,3,3), stride=2)
        # ... more blocks through conv5_2
        
        # Merge layers
        self.merge_conv1 = nn.Conv3d(128+64, 512, 1)
        self.merge_conv2 = nn.Conv3d(512, 128, 1)
        self.merge_bn = nn.BatchNorm3d(128)
        
        # Classification head
        self.logits_conv = nn.Conv3d(512, 1, 1)
    
    def forward(self, ims, samples):
        # Sound features
        sf = self.normalize_sfs(samples)
        sf = F.relu(self.sf_bn1(self.sf_conv1(sf)))
        sf = self.sf_pool(sf)
        sf = self.sf_block2(sf)
        sf = self.sf_block3(sf)
        sf = self.sf_block4(sf)
        
        # Image features
        im = self.normalize_ims(ims)
        im = F.relu(self.im_bn1(self.im_conv1(im)))
        im = self.im_pool(im)
        im = self.im_block2_1(im)
        im = self.im_block2_2(im)
        
        # Merge
        net = self.merge(sf, im)
        # ... continue through remaining blocks
        
        last_conv = net
        pooled = net.mean(dim=[2, 3, 4], keepdim=True)
        logits = self.logits_conv(pooled).squeeze(-1).squeeze(-1).squeeze(-1)
        cam = self.logits_conv(last_conv)
        
        return logits, cam, last_conv, im
```

### SourceSepUNet

```python
class SourceSepUNet(nn.Module):
    """U-Net for spectrogram-domain source separation"""
    def __init__(self, pr, shift_net=None):
        super().__init__()
        self.shift_net = shift_net  # Optional video feature extractor
        
        # Encoder (9 layers)
        self.enc1 = nn.Conv2d(2, 64, 4, stride=(1,2), padding=...)
        self.enc2 = nn.Conv2d(64, 128, 4, stride=(1,2), ...)
        # ... through enc9
        
        # Decoder (8 layers)
        self.dec1 = nn.ConvTranspose2d(512, 512, 4, stride=2, ...)
        # ... through dec8
        
        # Output heads
        self.fg_head = nn.ConvTranspose2d(64+64, 2, 4, stride=(1,2), ...)
        self.bg_head = nn.ConvTranspose2d(64+64, 2, 4, stride=(1,2), ...)
    
    def forward(self, ims, samples_trunc, spec_mix, phase_mix):
        # Get video features at multiple scales
        if self.shift_net is not None:
            with torch.no_grad():  # or with grad if fine-tuning
                _, _, _, _, scales, _ = self.shift_net(ims, samples_trunc)
        
        # U-Net encoder with skip connections
        skips = []
        x = torch.cat([normalize_spec(spec_mix), normalize_phase(phase_mix)], dim=1)
        x = F.leaky_relu(self.enc1(x), 0.2); skips.append(x)
        # ... encode
        
        # Merge video features at appropriate levels
        # ... merge_level operations
        
        # U-Net decoder with skip connections
        x = F.relu(torch.cat([x, skips.pop()], dim=1))
        x = self.dec1(x)
        # ... decode
        
        # Output
        fg = self.fg_head(torch.cat([x, skips[0]], dim=1))
        bg = self.bg_head(torch.cat([x, skips[0]], dim=1))
        
        pred_spec_fg, pred_phase_fg, pred_wav_fg = self.process(fg, phase_mix)
        pred_spec_bg, pred_phase_bg, pred_wav_bg = self.process(bg, phase_mix)
        return pred_spec_fg, pred_wav_fg, pred_spec_bg, pred_wav_bg
```

---

# Phase 3 — Weight Migration Strategy

## 3.1 Current Weight Format

- **Format**: TF 1.x checkpoint (`.data-00000-of-00001`, `.index`, `.meta`)
- **Variable naming**: Scoped names like `im/conv1/weights:0`, `sf/conv2_1_1/BatchNorm/gamma:0`
- **Tensor layout**: TF uses **NHWC** (images) and **NDHWC** (3D), PyTorch uses **NCHW** / **NCDHW**

## 3.2 Naming Convention Mapping

| TF Variable Pattern | PyTorch Parameter | Transform |
|---------------------|------------------|-----------|
| `im/conv1/weights:0` | `im_conv1.weight` | Transpose: `[D,H,W,C_in,C_out]` → `[C_out,C_in,D,H,W]` |
| `im/conv1/biases:0` | `im_conv1.bias` | Direct copy |
| `sf/conv1_1/weights:0` | `sf_conv1.weight` | Transpose: `[H,W,C_in,C_out]` → `[C_out,C_in,H,W]` |
| `*/BatchNorm/gamma:0` | `*.bn.weight` | Direct copy |
| `*/BatchNorm/beta:0` | `*.bn.bias` | Direct copy |
| `*/BatchNorm/moving_mean:0` | `*.bn.running_mean` | Direct copy |
| `*/BatchNorm/moving_variance:0` | `*.bn.running_var` | Direct copy |
| `gen/conv1/weights:0` | `enc1.weight` | Transpose |
| `gen/deconv1/weights:0` | `dec1.weight` | Transpose (ConvTranspose kernel layout) |
| `joint/logits/weights:0` | `logits_conv.weight` | Transpose |

## 3.3 Conversion Pipeline

```python
# scripts/convert_weights.py
import tensorflow as tf
import torch
import numpy as np

def convert_tf_to_pytorch(tf_checkpoint_path, pytorch_model, name_map):
    """
    Convert TF1 checkpoint to PyTorch state_dict.
    
    Args:
        tf_checkpoint_path: Path to TF checkpoint (without extension)
        pytorch_model: Instantiated PyTorch nn.Module
        name_map: Dict mapping TF var names to PyTorch param names
    """
    reader = tf.train.load_checkpoint(tf_checkpoint_path)
    var_to_shape = reader.get_variable_to_shape_map()
    
    state_dict = {}
    for tf_name, pt_name in name_map.items():
        if tf_name not in var_to_shape:
            print(f"WARNING: {tf_name} not found in checkpoint")
            continue
        
        tensor = reader.get_tensor(tf_name)
        
        # Apply transformations
        if 'weights' in tf_name and 'BatchNorm' not in tf_name:
            if len(tensor.shape) == 5:
                # 3D conv: TF [D, H, W, C_in, C_out] → PT [C_out, C_in, D, H, W]
                tensor = tensor.transpose(4, 3, 0, 1, 2)
            elif len(tensor.shape) == 4:
                # 2D conv: TF [H, W, C_in, C_out] → PT [C_out, C_in, H, W]
                tensor = tensor.transpose(3, 2, 0, 1)
            elif len(tensor.shape) == 2:
                # FC: TF [in, out] → PT [out, in]
                tensor = tensor.transpose(1, 0)
        
        state_dict[pt_name] = torch.from_numpy(tensor.copy())
    
    # Load into model
    missing, unexpected = pytorch_model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    return pytorch_model
```

## 3.4 Validation Strategy

```python
# scripts/validate_weights.py
def validate_conversion(tf_checkpoint, pt_model, test_input):
    """Layer-by-layer output comparison"""
    # 1. Run TF model
    tf.reset_default_graph()
    # ... build TF graph, restore weights, run inference
    tf_outputs = sess.run([intermediate_ops], feed_dict={...})
    
    # 2. Run PyTorch model  
    pt_model.eval()
    with torch.no_grad():
        pt_outputs = pt_model(test_input)
    
    # 3. Compare
    for name, (tf_out, pt_out) in zip(layer_names, zip(tf_outputs, pt_outputs)):
        tf_arr = np.array(tf_out)
        pt_arr = pt_out.numpy()
        
        # Handle layout differences
        if tf_arr.ndim == 5:  # NDHWC -> NCDHW
            tf_arr = tf_arr.transpose(0, 4, 1, 2, 3)
        elif tf_arr.ndim == 4:  # NHWC -> NCHW
            tf_arr = tf_arr.transpose(0, 3, 1, 2)
        
        max_diff = np.max(np.abs(tf_arr - pt_arr))
        mean_diff = np.mean(np.abs(tf_arr - pt_arr))
        print(f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        
        # Tolerance: BN running stats may have small diffs
        assert max_diff < 1e-4, f"Layer {name} exceeds tolerance!"
```

---

# Phase 4 — Performance Optimization

## 4.1 Current TF Inefficiencies & PyTorch Improvements

| Issue in TF | PyTorch Improvement |
|-------------|---------------------|
| Queue-based data loading with GIL contention | `DataLoader` with `num_workers`, `pin_memory=True`, `persistent_workers=True` |
| Manual multi-GPU gradient averaging | `torch.nn.parallel.DistributedDataParallel` (DDP) |
| Session overhead per `sess.run()` | Eager execution — zero overhead |
| `tf.py_func` for visualization | Direct Python calls, no serialization cost |
| No mixed precision | `torch.amp.autocast` + `GradScaler` |
| No graph compilation | `torch.compile()` for 10–30% speedup |
| Explicit `tf.device` placement | Automatic GPU placement + `.to(device)` |

## 4.2 Recommended Optimizations

### Mixed Precision (AMP)

```python
scaler = torch.amp.GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        outputs = model(batch)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()
```

### torch.compile

```python
# After model creation, before training
model = torch.compile(model, mode='reduce-overhead')
```

### Gradient Checkpointing (for large U-Net)

```python
from torch.utils.checkpoint import checkpoint

class SourceSepUNet(nn.Module):
    def forward(self, ...):
        # Checkpoint encoder layers to reduce memory
        x = checkpoint(self.encoder_block, x, use_reentrant=False)
```

### Efficient DataLoader

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True,
)
```

### Multi-GPU with DDP

```python
# Launch with: torchrun --nproc_per_node=N train.py
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

---

# Phase 5 — Full Migration Roadmap

## Step 1: Isolate and Port Utility Functions

| Item | Detail |
|------|--------|
| **Expected changes** | Create `utils/audio.py` with PyTorch STFT/iSTFT/Griffin-Lim; `utils/params.py` with dataclass-based Params |
| **Key functions** | [stft()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py#887-900), [istft()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py#907-916), [griffin_lim()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/soundrep.py#37-57), [normalize_rms()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#752-756), [normalize_ims()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#380-387), [db_from_amp()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/soundrep.py#76-78), [amp_from_db()](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/soundrep.py#79-81) |
| **Risks** | STFT window defaults differ between TF and PyTorch (`hann_window` vs `None`); frame centering behavior |
| **Validation** | Unit test: generate random audio → compute STFT in both TF and PyTorch → compare magnitude and phase within 1e-5 tolerance |

## Step 2: Port Residual Blocks

| Item | Detail |
|------|--------|
| **Expected changes** | Create `models/blocks.py` with `Block2D` and `Block3D` |
| **Key issue** | TF slim uses `SAME` padding (asymmetric), PyTorch `nn.Conv*d` does not natively support asymmetric padding. Must manually pad or use `padding='same'` (PyTorch ≥1.9) |
| **Risks** | Padding asymmetry with even kernel sizes + stride > 1 (see [conv2d_same](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sourcesep.py#837-841) in tfutil.py); batch norm momentum convention (TF [decay](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#20-22) vs PT `momentum = 1 - decay`) |
| **Validation** | Feed identical random input through TF block and PyTorch block with identical weights → compare output |

## Step 3: Port ShiftNet Model

| Item | Detail |
|------|--------|
| **Expected changes** | Create `models/shift_net.py` with `ShiftNet(nn.Module)`, `SoundFeatureNet`, `ImageFeatureNet`, `MergeModule` |
| **Key issue** | `tf.nn.fractional_max_pool` used in merge layer — PyTorch has `FractionalMaxPool2d` but it may produce different indices. May need to use `F.adaptive_max_pool2d` with matching output size |
| **Risks** | Fractional pooling randomness during training; different padding semantics in 3D convolutions |
| **Validation** | Load converted weights → feed same input → compare logits and CAM outputs against TF model |

## Step 4: Port SourceSep U-Net

| Item | Detail |
|------|--------|
| **Expected changes** | Create `models/sourcesep.py` with `SourceSepUNet(nn.Module)` including encoder, decoder, skip connections, video conditioning |
| **Key issue** | ConvTranspose2d output size ambiguity — TF `conv2d_transpose` infers output shape, PyTorch needs `output_padding` for odd dimensions |
| **Risks** | Skip connection dimension mismatches if padding is slightly different; deconv output sizes |
| **Validation** | Load converted weights → compute spectrogram → run through model → compare predicted wavs |

## Step 5: Port Dataset Pipeline

| Item | Detail |
|------|--------|
| **Expected changes** | Create `datasets/shift_dataset.py` and `datasets/sep_dataset.py` as `torch.utils.data.Dataset` subclasses |
| **Key change** | Replace TFRecord reading with raw file reading (decode JPEG, read WAV) or convert TFRecords to individual files |
| **Option A** | Read TFRecords using `tensorflow` in `__getitem__` (dependency retained) |
| **Option B** | Pre-process TFRecords to individual files (JPEG + WAV) and read natively |
| **Risks** | Data augmentation differences (random crop coordinates, audio augmentation) |
| **Validation** | Compare batch statistics (mean, std) between TF and PyTorch data pipelines |

## Step 6: Port Training Loop

| Item | Detail |
|------|--------|
| **Expected changes** | Create `training/train_shift.py` and `training/train_sep.py` with explicit training loops |
| **Key changes** | Replace `sess.run` with PyTorch forward/backward; replace [average_grads](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#76-103) with DDP; replace manual LR decay with `torch.optim.lr_scheduler.StepLR` |
| **Risks** | Optimizer convergence differences (Adam epsilon, BN momentum) |
| **Validation** | Train for 100 steps on small subset → compare loss curves |

## Step 7: Convert Weights and Validate

| Item | Detail |
|------|--------|
| **Expected changes** | Run `scripts/convert_weights.py` for each model variant |
| **Validation tests** | (1) Layer-by-layer weight comparison; (2) Identical input → output comparison (tolerance < 1e-4); (3) End-to-end source separation on sample video → SDR comparison |

---

# Phase 6 — Manual Implementation Guide

## 6.1 How to Rewrite Models

### General Pattern

Every TF scope-based model becomes a PyTorch `nn.Module`:

```python
# TF (original)
def make_net(ims, sfs, pr, reuse=True, train=True):
    with slim.arg_scope(arg_scope_3d(pr, reuse=reuse, train=train)):
        net = conv3d(ims, 64, [5, 7, 7], scope='im/conv1', stride=2)
        ...

# PyTorch (target)
class ShiftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.im_conv1 = nn.Conv3d(3, 64, (5, 7, 7), stride=2, padding=(...), bias=False)
        self.im_bn1 = nn.BatchNorm3d(64, momentum=0.0003, eps=0.001)
        ...
    
    def forward(self, ims, sfs):
        net = F.relu(self.im_bn1(self.im_conv1(self.normalize_ims(ims))))
        ...
```

Key rules:
1. **Every `slim.conv*d` call** → `nn.Conv*d` in [__init__](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/sep_video.py#13-18), `F.relu(self.bn(self.conv(x)))` in `forward`
2. **Scoped variable names** → Module attribute names
3. **`reuse=True`** → Share the same module instance
4. **[train](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py#465-497) flag** → `model.train()` / `model.eval()` for BN behavior
5. **`slim.arg_scope` defaults** → Set in constructor

### Replicating `SAME` Padding

TF `SAME` padding with stride > 1 adds asymmetric padding. For convolutions where this matters:

```python
def pad_same(x, kernel_size, stride, dims=2):
    """Replicate TF SAME padding for stride > 1"""
    pad_total = [k - 1 for k in kernel_size] if isinstance(kernel_size, tuple) else [kernel_size - 1] * dims
    pad_beg = [p // 2 for p in pad_total]
    pad_end = [p - pb for p, pb in zip(pad_total, pad_beg)]
    # F.pad expects reverse order: (last_dim_beg, last_dim_end, ..., first_dim_beg, first_dim_end)
    padding = []
    for pb, pe in reversed(list(zip(pad_beg, pad_end))):
        padding.extend([pb, pe])
    return F.pad(x, padding)
```

### Replicating Batch Normalization

> [!IMPORTANT]
> TF `decay=0.9997` maps to PyTorch `momentum=1-0.9997=0.0003`. TF BN uses `epsilon=0.001` by default, while PyTorch uses `1e-5`. Both must be set explicitly.

```python
nn.BatchNorm3d(num_features, momentum=0.0003, eps=0.001)
```

## 6.2 How to Rewrite the Training Loop

```python
# TF (original) — sourcesep.py Model.train()
while True:
    step, lr = self.get_step()
    loss_vals = self.sess.run([self.train_op] + loss_ops)[1:]

# PyTorch (target)
model.train()
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    
    ims, samples = batch['ims'].cuda(), batch['samples'].cuda()
    snd = mix_sounds(samples, pr)
    
    pred = model(ims, snd.samples, snd.spec, snd.phase)
    gen_loss = compute_separation_loss(pred, snd, pr)
    
    gen_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), pr.grad_clip)
    optimizer.step()
    scheduler.step()
    
    if step % pr.check_iters == 0:
        torch.save(model.state_dict(), checkpoint_path)
```

## 6.3 How to Migrate Weights

Step-by-step manual process:

```bash
# 1. Install both frameworks
pip install tensorflow torch numpy

# 2. List TF checkpoint variables
python -c "
import tensorflow as tf
reader = tf.train.load_checkpoint('../results/nets/shift/net.tf-650000')
for name, shape in sorted(reader.get_variable_to_shape_map().items()):
    print(f'{name}: {shape}')
"

# 3. Run conversion script
python scripts/convert_weights.py \
    --tf_checkpoint ../results/nets/shift/net.tf-650000 \
    --output shift_net.pt \
    --model_type shift

# 4. Validate
python scripts/validate_weights.py \
    --tf_checkpoint ../results/nets/shift/net.tf-650000 \
    --pt_weights shift_net.pt \
    --test_video ../data/crossfire.mp4
```

## 6.4 How to Test Equivalence

```python
# Generate deterministic test input
torch.manual_seed(42)
np.random.seed(42)

test_ims = np.random.randint(0, 255, (1, 64, 224, 224, 3), dtype=np.uint8)
test_samples = np.random.randn(1, 44100, 2).astype(np.float32) * 0.1

# TF forward pass
tf_logits = tf_model.predict(test_ims, test_samples)

# PT forward pass
pt_ims = torch.from_numpy(test_ims).permute(0, 4, 1, 2, 3).float()  # NDHWC -> NCDHW
pt_samples = torch.from_numpy(test_samples)
pt_model.eval()
with torch.no_grad():
    pt_logits, _, _, _ = pt_model(pt_ims, pt_samples)

# Compare
diff = np.abs(tf_logits - pt_logits.numpy())
assert diff.max() < 1e-4, f"Output mismatch: max diff = {diff.max()}"
print("✅ Outputs match within tolerance")
```

---

# Phase 7 — Risk Analysis

## 7.1 Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Padding behavior** — TF `SAME` adds asymmetric padding with stride > 1 | **High** | High | Implement explicit padding functions matching TF behavior; verify each conv layer output shape |
| **Batch Normalization** — different momentum/epsilon defaults; different running stats update rule | **High** | Medium | Explicitly set `momentum=0.0003`, `eps=0.001`; verify running stats after 100 training steps |
| **Fractional max pooling** — different random pooling regions | **Medium** | High | Use deterministic mode in both; or replace with `F.adaptive_max_pool2d` with exact output size matching |
| **STFT differences** — different windowing, centering, normalization | **High** | Medium | Explicitly set `window_fn=hann`, `center=True`, match `pad_end` behavior; unit test with simple signals |
| **Conv weight layout** — transposition errors during conversion | **High** | Low | Verify each layer's output independently; automate with shape assertions |
| **ConvTranspose output size** — PyTorch may need `output_padding` | **Medium** | Medium | Compute expected output size from TF shapes; pass explicitly |
| **Optimizer convergence** — different Adam implementations (epsilon handling, denominator squaring) | **Medium** | Low | Use same hyperparameters; compare loss curves for first 1000 steps |
| **Broadcasting differences** — subtle shape mismatches | **Low** | Low | Add explicit shape assertions in forward passes |
| **Float precision** — TF defaults to float32, PyTorch defaults to float32, but GPU numerics differ slightly | **Low** | High (but harmless) | Accept tolerance of 1e-4 for single operations, 1e-3 for full model |
| **renorm in BN** — the code patches around it with [SafeBatchNormalization(renorm=False)](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/shift_net.py#23-30) | **Low** | Low | Simply don't enable renorm in PyTorch BN (it's not standard) |

## 7.2 Critical Differences Deep-Dive

### Padding

TF `SAME` padding formula for a given dimension:
```
out_size = ceil(in_size / stride)
pad_total = max((out_size - 1) * stride + kernel_size - in_size, 0)
pad_before = pad_total // 2
pad_after = pad_total - pad_before
```

PyTorch `padding='same'` only works for stride=1. For stride > 1, must use manual `F.pad`.

### Batch Normalization

| Parameter | TensorFlow | PyTorch |
|-----------|-----------|---------|
| Momentum for running stats | [decay](file:///c:/Users/ASUS/Development/Python_Projects/AI/multisensory/src/tfutil.py#20-22) (e.g., 0.9997) | `momentum = 1 - decay` (e.g., 0.0003) |
| Epsilon | 0.001 (as set in code) | 1e-5 (default) — **must override to 0.001** |
| Update rule | `running = decay * running + (1 - decay) * batch` | Same |
| Training mode | `is_training` parameter | `model.train()` / `model.eval()` |

### Adam Optimizer

Both TF and PyTorch compute the same update rule. The key difference is that TF `AdamOptimizer` uses a non-standard epsilon placement by default (in the denominator before sqrt), while PyTorch places it after sqrt. In practice, this rarely matters for convergence.

---

# Phase 8 — Final Deliverables Summary

| Deliverable | Status | Description |
|-------------|--------|-------------|
| **Repository Architecture Analysis** | ✅ Complete | Phase 1 — full map of 11+ source files, model architectures, training/inference pipelines |
| **TF → PyTorch Component Mapping** | ✅ Complete | Phase 2 — table mapping every TF construct to PyTorch equivalent |
| **Full Migration Plan** | ✅ Complete | Phase 5 — 7-step roadmap with risks and validation for each step |
| **Weight Conversion Pipeline** | ✅ Complete | Phase 3 — extraction, transpose, renaming, validation scripts |
| **Optimization Recommendations** | ✅ Complete | Phase 4 — AMP, torch.compile, DDP, efficient DataLoaders |
| **Manual Implementation Guide** | ✅ Complete | Phase 6 — code snippets for models, training loop, weight migration, testing |
| **Validation Strategy** | ✅ Complete | Phase 3.4 + Phase 6.4 — layer-by-layer and end-to-end output comparison |

## Verification Plan

### Automated Tests

Since there are no existing unit tests in this repository, the following tests should be created during migration:

1. **STFT Equivalence Test**: `pytest tests/test_audio.py` — Compare `torch.stft` output vs `tf.signal.stft` for 10 random signals
2. **Block Output Test**: `pytest tests/test_blocks.py` — Compare `Block3D` and `Block2D` outputs with loaded weights
3. **Weight Conversion Test**: `python scripts/validate_weights.py --tf_checkpoint <path> --pt_weights <path>` — Verify all layers within tolerance
4. **End-to-End Test**: `python scripts/validate_e2e.py --video ../data/translator.mp4` — Compare separated audio outputs (SDR, SI-SDR metrics)

### Manual Verification

1. Run `python inference/sep_video.py ../data/translator.mp4 --model full --out ../results_pt/` with converted PyTorch weights and compare audio/video outputs against TF results
2. Run `python inference/shift_example.py` to generate CAM visualization and compare visually with TF output
