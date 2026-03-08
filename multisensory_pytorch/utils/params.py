"""
Parameter configuration system.

Replaces tfutil.Params (which inherits from aolib.util.Struct).
Uses Python dataclass-like pattern for cleaner configuration.
"""

import os
import copy
import numpy as np


class Params:
    """
    Flexible parameter container matching the original TF codebase's Struct-based Params.

    Usage:
        pr = Params(
            train_iters=160000,
            base_lr=1e-4,
            batch_size=6,
            ...
        )
        pr.train_dir  # auto-computed from pr.resdir
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def train_dir(self):
        return _mkdir(os.path.join(self.resdir, "training"))

    @property
    def summary_dir(self):
        return _mkdir(os.path.join(self.resdir, "summary"))

    @property
    def name(self):
        path = self.resdir.rstrip("/").rstrip("\\")
        return os.path.basename(path)

    def copy(self):
        return copy.deepcopy(self)

    def updated(self, other=None, **kwargs):
        new = self.copy()
        if other is not None:
            new.__dict__.update(other.__dict__)
        new.__dict__.update(kwargs)
        return new

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __str__(self):
        items = sorted(self.__dict__.items())
        return "Params(\n" + "\n".join(f"  {k}={v!r}," for k, v in items) + "\n)"

    def __repr__(self):
        return self.__str__()


def _mkdir(path):
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Predefined parameter configurations
# Ported from sep_params.py and shift_params.py
# ---------------------------------------------------------------------------

def sep_base(name=None, num_gpus=1, batch_size=6, vid_dur=None,
             samp_sr=21000.0, resdir=None):
    """Base parameters for source separation (from sep_params.py)."""
    VidDur = 2.135
    if vid_dur is None:
        vid_dur = VidDur

    if resdir is None:
        assert name is not None
        resdir = os.path.abspath(os.path.join("../results/nets/sep", name))

    total_dur = 5.0
    fps = 29.97
    frame_dur = 1.0 / fps

    pr = Params(
        resdir=resdir,
        train_iters=160000,
        step_size=120000,
        opt_method="adam",
        base_lr=1e-4,
        gamma=0.1,
        full_model=False,
        predict_bg=True,
        grad_clip=10.0,
        batch_size=int(batch_size * num_gpus),
        test_batch=10,
        subsample_frames=None,
        weight_decay=1e-5,
        train_list=os.path.join("../data", "celeb-tf-v6-full", "train/tf"),
        val_list=os.path.join("../data", "celeb-tf-v6-full", "val/tf"),
        test_list=os.path.join("../data", "celeb-tf-v6-full", "test/tf"),
        init_type="shift",
        init_path="../results/nets/shift/net.tf-650000",
        net_style="full",
        im_split=False,
        multi_shift=False,
        num_dbs=None,
        im_type="jpeg",
        full_im_dim=256,
        crop_im_dim=224,
        dset_seed=None,
        fps=fps,
        show_videos=False,
        samp_sr=samp_sr,
        vid_dur=vid_dur,
        total_frames=int(total_dur * fps),
        sampled_frames=int(vid_dur * fps),
        full_samples_len=int(total_dur * samp_sr),
        samples_per_frame=samp_sr * frame_dur,
        frame_sample_delta=int(total_dur * fps) / 2,
        fix_frame=False,
        use_3d=True,
        augment_ims=True,
        augment_audio=False,
        dilate=False,
        cam=False,
        do_shift=False,
        variable_frame_count=False,
        use_sound=True,
        bn_last=True,
        l1_weight=1.0,
        phase_weight=0.01,
        gan_weight=0.0,
        use_wav_gan=False,
        log_spec=True,
        spec_min=-100.0,
        spec_max=80.0,
        normalize_rms=True,
        check_iters=1000,
        slow_check_iters=10000,
        print_iters=10,
        summary_iters=10,
        profile_iters=None,
        show_iters=None,
        frame_length_ms=64,
        frame_step_ms=16,
        sample_len=None,
        freq_len=1024,
        augment_rms=False,
        loss_types=["fg-bg"],
        pit_weight=0.0,
        both_videos_in_batch=True,
        bn_scale=True,
        pad_stft=False,
        phase_type="pred",
        alg="sourcesep",
        mono=False,
    )
    pr.spec_len = 128 * int(2 ** np.round(np.log2(vid_dur / float(VidDur))))
    pr.num_samples = int(round(pr.samples_per_frame * pr.sampled_frames))
    return pr


def sep_full(num_gpus=1, vid_dur=None, batch_size=6, **kwargs):
    """Full audio-visual source separation model params."""
    return sep_base("full", num_gpus, vid_dur=vid_dur, batch_size=batch_size, **kwargs)


def sep_unet_pit(num_gpus=1, vid_dur=None, batch_size=24, **kwargs):
    """Audio-only U-Net with PIT loss."""
    VidDur = 2.135
    if vid_dur is None:
        vid_dur = VidDur
    pr = sep_base("unet-pit", num_gpus, vid_dur=vid_dur, batch_size=batch_size, **kwargs)
    pr.net_style = "no-im"
    pr.init_path = None
    pr.loss_types = ["pit"]
    pr.pit_weight = 1.0
    pr.both_videos_in_batch = False
    return pr


def shift_v1(num_gpus=1, shift_dur=4.2):
    """ShiftNet v1 parameters (from shift_params.py)."""
    total_dur = 10.1
    fps = 29.97
    frame_dur = 1.0 / fps
    samp_sr = 21000.0
    spec_sr = 100.0

    pr = Params(
        subsample_frames=None,
        train_iters=100000,
        opt_method="momentum",
        base_lr=1e-2,
        full_model=True,
        grad_clip=5.0,
        skip_notfound=False,
        augment_ims=True,
        init_path=None,
        cam=False,
        batch_size=int(5 * num_gpus),
        test_batch=10,
        shift_dur=shift_dur,
        multipass=False,
        both_examples=True,
        small_augment=False,
        resdir=os.path.abspath("../results/nets/shift"),
        weight_decay=1e-5,
        train_list="../data/audioset-vid-v21/train_tfs.txt",
        test_list="../data/audioset-vid-v21/test_tfs.txt",
        num_dbs=None,
        im_type="jpeg",
        input_type="samples",
        full_im_dim=256,
        full_flow_dim=256,
        crop_im_dim=224,
        sf_pad=int(0.5 * 2 ** 4 * 4),
        use_flow=False,
        renorm=True,
        checkpoint_iters=1000,
        dset_seed=None,
        samp_sr=samp_sr,
        spec_sr=spec_sr,
        fps=fps,
        max_intersection=30 * 2,
        specgram_sr=spec_sr,
        num_mel=64,
        batch_norm=True,
        show_videos=False,
        check_iters=1000,
        decompress_flow=True,
        print_iters=10,
        total_frames=int(total_dur * fps),
        sampled_frames=int(shift_dur * fps),
        full_specgram_samples=int(total_dur * spec_sr),
        full_samples_len=int(total_dur * samp_sr),
        sfs_per_frame=spec_sr * frame_dur,
        samples_per_frame=samp_sr * frame_dur,
        frame_sample_delta=int(total_dur * fps) / 2,
        fix_frame=False,
        use_3d=True,
        augment=False,
        dilate=False,
        do_shift=True,
        variable_frame_count=False,
        momentum_rate=0.9,
        use_sound=True,
        bn_last=True,
        summary_iters=10,
        im_split=True,
        num_splits=2,
        augment_audio=False,
        multi_shift=False,
        model_iter=650000,
        bn_scale=True,
    )
    pr.vid_dur = pr.shift_dur
    pr.num_samples = int(round(pr.samples_per_frame * pr.sampled_frames))
    return pr


def cam_v1(num_gpus=1, shift_dur=4.2):
    """CAM variant of ShiftNet (higher-resolution CAM)."""
    pr = shift_v1(num_gpus, shift_dur)
    pr.cam = True
    pr.init_path = "../results/nets/shift/net.tf-650000"
    pr.resdir = os.path.abspath("../results/nets/cam")
    pr.model_iter = 675000
    return pr
