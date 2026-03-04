import numpy as np
import os
import io
import itertools as itl
from PIL import Image, ImageDraw, ImageFont
from . import util as ut  # Assumes util.py is in the same directory
import scipy.ndimage
import matplotlib.pylab as pylab
import webbrowser

# Handle Pillow version compatibility for Resampling
try:
    from PIL import ImageResampling

    RESAMPLE_FILTER = ImageResampling.LANCZOS
except ImportError:
    # Fallback for older Pillow versions
    RESAMPLE_FILTER = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC


def show(*args, **kwargs):
    import imtable

    return imtable.show(*args, **kwargs)


# --- Conversion Helpers ---


def to_pil(im):
    """Converts a numpy array to a PIL Image."""
    return Image.fromarray(np.uint8(im))


def from_pil(pil):
    """Converts a PIL Image to a numpy array."""
    return np.array(pil)


def to_pylab(a):
    return np.uint8(a)


def rgb_from_gray(img, copy=True):
    """Ensures the image has 3 channels (RGB)."""
    if img.ndim == 3:
        if img.shape[2] == 3:
            return img.copy() if copy else img
        elif img.shape[2] == 4:
            return (img.copy() if copy else img)[..., :3]
        elif img.shape[2] == 1:
            return np.tile(img, (1, 1, 3))
    elif img.ndim == 2:
        return np.tile(img[:, :, np.newaxis], (1, 1, 3))
    raise RuntimeError(f"Cannot convert to RGB. Shape: {img.shape}")


def luminance(im):
    """Converts RGB to Grayscale using standard weights."""
    if im.ndim == 2:
        return im
    return np.uint8(np.round(np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])))


# --- Drawing Functions ---


def draw_on(f, im):
    pil = to_pil(im)
    draw = ImageDraw.Draw(pil)
    f(draw)
    return from_pil(pil)


def color_from_string(s):
    colors = {"r": (255, 0, 0), "g": (0, 255, 0), "b": (0, 0, 255)}
    if s in colors:
        return colors[s]
    else:
        ut.fail(f"unknown color: {s}")


def parse_color(c):
    if isinstance(c, (tuple, list, np.ndarray)):
        return c
    elif isinstance(c, str):
        return color_from_string(c)
    return c


def colors_from_input(color_input, default, n):
    if color_input is None:
        expanded = [default] * n
    elif (
        isinstance(color_input, tuple)
        and len(color_input) == 3
        and all(isinstance(x, int) for x in color_input)
    ):
        expanded = [color_input] * n
    else:
        expanded = color_input
    return [parse_color(c) for c in expanded]


def draw_rects(
    im,
    rects,
    outlines=None,
    fills=None,
    texts=None,
    text_colors=None,
    line_widths=None,
    as_oval=False,
):
    rects = list(rects)
    n = len(rects)
    outlines = colors_from_input(outlines, (0, 0, 255), n)
    text_colors = colors_from_input(text_colors, (255, 255, 255), n)
    fills = colors_from_input(fills, None, n)

    texts = texts if texts is not None else [None] * n
    line_widths = line_widths if line_widths is not None else [None] * n

    def f(draw):
        # Python 3: zip replaces itl.izip
        for (x, y, w, h), outline, fill, text, text_color, lw in zip(
            rects, outlines, fills, texts, text_colors, line_widths
        ):
            coords = (x, y, x + w, y + h)
            if lw is None:
                if as_oval:
                    draw.ellipse(coords, outline=outline, fill=fill)
                else:
                    draw.rectangle(coords, outline=outline, fill=fill)
            else:
                # Custom line width implementation
                d = int(np.ceil(lw / 2))
                draw.rectangle((x - d, y - d, x + w + d, y + d), fill=outline)
                draw.rectangle((x - d, y - d, x + d, y + h + d), fill=outline)
                draw.rectangle((x + w + d, y + h + d, x - d, y + h - d), fill=outline)
                draw.rectangle((x + w + d, y + h + d, x + w - d, y - d), fill=outline)

            if text is not None:
                draw.text((x + 2, y), text, fill=text_color)

    return draw_on(f, im)


def draw_text(im, texts, pts, colors, font_size=None, bold=False):
    im = rgb_from_gray(im)
    colors = colors_from_input(colors, (0, 0, 0), len(texts))

    def f(draw):
        font = None
        if font_size:
            font_choices = [
                (
                    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
                    if bold
                    else "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
                ),
                "/Library/Fonts/PTMono.ttc",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
            ]
            for path in font_choices:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, size=font_size)
                    break
            if font is None:
                font = ImageFont.load_default()

        for pt, text, color in zip(pts, texts, colors):
            draw.text(ut.int_tuple(pt), text, fill=color, font=font)

    return draw_on(f, im)


# --- Geometry & Stacking ---


def resize(im, scale, order=3, hires=False):
    """Modernized resize replacing scipy.misc.imresize."""
    if hires == "auto":
        hires = im.dtype == np.uint8

    if np.ndim(scale) == 0:
        new_scale = [scale, scale]
    elif (scale[0] is None or isinstance(scale[0], int)) and (
        scale[1] is None or isinstance(scale[1], int)
    ):
        if scale[0] is None:
            dims = (int(float(im.shape[0]) / im.shape[1] * scale[1]), scale[1])
        elif scale[1] is None:
            dims = (scale[0], int(float(im.shape[1]) / im.shape[0] * scale[0]))
        else:
            dims = scale[:2]
        new_scale = [float(dims[0]) / im.shape[0], float(dims[1]) / im.shape[1]]
    else:
        new_scale = scale

    if hires:
        new_size = (int(new_scale[1] * im.shape[1]), int(new_scale[0] * im.shape[0]))
        return from_pil(to_pil(im).resize(new_size, RESAMPLE_FILTER))
    else:
        scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)
        return scipy.ndimage.zoom(im, scale_param, order=order)


def hstack_ims(ims, bg_color=(0, 0, 0)):
    if not ims:
        return make(0, 0)
    max_h = max(im.shape[0] for im in ims)
    result = []
    for im in ims:
        frame = make(im.shape[1], max_h, bg_color)
        frame[: im.shape[0], : im.shape[1]] = rgb_from_gray(im)
        result.append(frame)
    return np.hstack(result)


def vstack_ims(ims, bg_color=(0, 0, 0)):
    if not ims:
        return make(0, 0)
    max_w = max(im.shape[1] for im in ims)
    result = []
    for im in ims:
        frame = make(max_w, im.shape[0], bg_color)
        frame[: im.shape[0], : im.shape[1]] = rgb_from_gray(im)
        result.append(frame)
    return np.vstack(result)


def make(w, h, fill=(0, 0, 0)):
    """Creates a solid color image."""
    return np.uint8(np.tile([[fill]], (h, w, 1)))


# --- I/O & Compression ---


def load(im_fname, gray=False):
    with Image.open(im_fname) as img:
        im = from_pil(img)
    if gray:
        return luminance(im)
    return rgb_from_gray(im) if im.ndim == 2 else im


def save(img_fname, a):
    Image.fromarray(np.uint8(a)).save(img_fname, quality=100)


def compress(im, format="png"):
    """Replaces StringIO with BytesIO for binary image data."""
    out = io.BytesIO()
    to_pil(im).save(out, format=format)
    return out.getvalue()


def uncompress(s):
    return from_pil(Image.open(io.BytesIO(s)))


def from_fig(fig=None, size_inches=None):
    """Converts a Matplotlib figure to a numpy array."""
    if fig is None:
        fig = pylab.gcf()
    if size_inches:
        fig.set_size_inches(*size_inches)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # Python 3: np.frombuffer replaces np.fromstring
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    buf = np.roll(buf, 3, axis=2)  # ARGB to RGBA

    # Simple alpha blending for white background
    p = buf[:, :, 3] / 255.0
    return np.array(
        buf[:, :, :3] * p[:, :, np.newaxis] + (1 - p)[:, :, np.newaxis] * 255, "uint8"
    )


# --- Image Filters ---


def blur(im, sigma):
    if im.ndim == 2:
        return scipy.ndimage.gaussian_filter(im, sigma)
    else:
        # Blur each channel separately
        res = [
            scipy.ndimage.gaussian_filter(im[..., i], sigma)[..., np.newaxis]
            for i in range(im.shape[2])
        ]
        return np.concatenate(res, axis=2)


def blit(src, dst, x, y, opt=None):
    """Copies src onto dst at (x,y) with clipping."""
    if opt == "center":
        x -= src.shape[1] // 2
        y -= src.shape[0] // 2
    dx, dy, dw, dh = ut.crop_rect_to_img(
        (int(x), int(y), src.shape[1], src.shape[0]), dst
    )
    sx, sy = dx - int(x), dy - int(y)
    dst[dy : dy + dh, dx : dx + dw] = src[sy : sy + dh, sx : sx + dw]
