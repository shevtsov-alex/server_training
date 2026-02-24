#!/usr/bin/env python3
#export HF_HOME="/workspace/layered/hf_cache/"
import glob, math
import numpy as np
from PIL import Image
from dataclasses import dataclass

from pytoshop.enums import ColorMode, ChannelId, Compression
from pytoshop.user.nested_layers import Image as PsdLayer, nested_layers_to_psd

ALPHA_THRESHOLD = 4

@dataclass(frozen=True)
class CropBox:
    """Describes how a non-transparent region was cropped from the original canvas.

    Stores everything needed to place the processed crop back onto
    a fully-transparent canvas of the original size.
    """
    original_w: int
    original_h: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


def crop_to_content(
    img: Image.Image,
    margin: int = 20,
    min_side: int = 640,
) -> tuple[Image.Image, CropBox]:
    """Crop an RGBA image to the bounding box of its visible pixels plus a margin.

    Pixels with alpha <= ``alpha_threshold`` are treated as transparent
    (noise) -- they are ignored for bounding-box computation and their
    alpha is zeroed out in the returned crop.

    ``margin`` extra transparent pixels are kept around the content on
    every side (clamped to the canvas bounds).

    ``min_side`` guarantees the cropped image is at least this many pixels
    on its shortest side; the crop box is expanded symmetrically when needed.

    Returns the cropped image and a ``CropBox`` that records the original
    canvas size and the padding removed from each side, so the crop can
    later be restored pixel-for-pixel with ``restore_from_crop``.

    Raises ``ValueError`` if no pixels exceed the threshold.
    """
    img = clean_alpha_noise(img)
    rgba = np.array(img.convert("RGBA"))
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    visible = alpha > ALPHA_THRESHOLD

    rows_with_content = np.any(visible, axis=1)
    cols_with_content = np.any(visible, axis=0)

    if not rows_with_content.any():
        raise ValueError("Image is fully transparent -- nothing to crop")

    row_min, row_max = int(np.argmax(rows_with_content)), int(len(rows_with_content) - 1 - np.argmax(rows_with_content[::-1]))
    col_min, col_max = int(np.argmax(cols_with_content)), int(len(cols_with_content) - 1 - np.argmax(cols_with_content[::-1]))

    row_min = max(row_min - margin, 0)
    row_max = min(row_max + margin, h - 1)
    col_min = max(col_min - margin, 0)
    col_max = min(col_max + margin, w - 1)

    crop_h = row_max - row_min + 1
    crop_w = col_max - col_min + 1

    if crop_h < min_side:
        deficit = min_side - crop_h
        expand_top = deficit // 2
        expand_bottom = deficit - expand_top
        row_min = max(row_min - expand_top, 0)
        row_max = min(row_max + expand_bottom, h - 1)
        # redistribute if clamped on one side
        crop_h = row_max - row_min + 1
        if crop_h < min_side:
            row_min = max(row_max - min_side + 1, 0)
            row_max = min(row_min + min_side - 1, h - 1)

    if crop_w < min_side:
        deficit = min_side - crop_w
        expand_left = deficit // 2
        expand_right = deficit - expand_left
        col_min = max(col_min - expand_left, 0)
        col_max = min(col_max + expand_right, w - 1)
        crop_w = col_max - col_min + 1
        if crop_w < min_side:
            col_min = max(col_max - min_side + 1, 0)
            col_max = min(col_min + min_side - 1, w - 1)

    box = CropBox(
        original_w=w,
        original_h=h,
        pad_top=row_min,
        pad_bottom=h - 1 - row_max,
        pad_left=col_min,
        pad_right=w - 1 - col_max,
    )

    rgba[~visible] = 0
    cropped = Image.fromarray(rgba[row_min:row_max + 1, col_min:col_max + 1], mode="RGBA")
    return cropped, box


def clean_alpha_noise(img: Image.Image) -> Image.Image:
    """Zero out pixels whose alpha is below ``threshold``."""
    rgba = np.array(img.convert("RGBA"))
    rgba[rgba[:, :, 3] <= ALPHA_THRESHOLD] = 0
    return Image.fromarray(rgba, mode="RGBA")


def restore_from_crop(img: Image.Image, box: CropBox) -> Image.Image:
    """Place a processed crop back onto a transparent canvas of the original size.

    The returned image has exactly ``(box.original_w, box.original_h)`` dimensions
    with ``img`` positioned at the location described by ``box``.
    """
    canvas = Image.new("RGBA", (box.original_w, box.original_h), (0, 0, 0, 0))
    canvas.paste(img, (box.pad_left, box.pad_top))
    return canvas

def is_empty_layer(img: Image.Image) -> bool:
    rgb = np.array(img.convert("RGB"))
    rgba = np.array(img.convert("RGBA"))

    if rgb.std() == 0 and rgba[:, :, 3].max() == 0:
        return True
    return False

    return np.all(rgba[:, :, 3] == 0)

def pad_rgba_to_multiple_edge(img: Image.Image, multiple: int):
    w, h = img.size
    new_w = math.ceil(w / multiple) * multiple
    new_h = math.ceil(h / multiple) * multiple
    if (new_w, new_h) == (w, h):
        return img, (w, h)

    arr = np.array(img)#img.convert("RGBA"))
    pad_w = new_w - w
    pad_h = new_h - h

    # edge padding = повторяем крайние пиксели
    arr2 = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    out = Image.fromarray(arr2, mode="RGBA")
    return out, (w, h)

def scale_down_image(img: Image.Image, max_size: int):
    w, h = img.size
    original_max_size = max(w, h)
    if original_max_size > max_size:
        scale = max_size / original_max_size
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = w
        new_h = h
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

def shards(pat: str):
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    return files


def create_psd(images: list[Image.Image], output_path: str) -> None:
    """Create a PSD file with each image as a separate layer, preserving the input order.

    The first image becomes the bottom layer (background), the last -- the top layer (foreground).
    Canvas size is auto-derived from the largest layer dimensions.
    Alpha channels are preserved as PSD per-layer transparency.
    """
    if not images:
        raise ValueError("images list must not be empty")

    psd_layers: list[PsdLayer] = []
    for i, img in enumerate(images):
        rgba = np.array(img.convert("RGBA"))
        psd_layers.append(PsdLayer(
            name=f"Layer {i + 1}",
            top=0,
            left=0,
            channels={
                ChannelId.transparency: rgba[:, :, 3],
                ChannelId.red: rgba[:, :, 0],
                ChannelId.green: rgba[:, :, 1],
                ChannelId.blue: rgba[:, :, 2],
            },
            opacity=255,
        ))

    psd = nested_layers_to_psd(
        psd_layers, color_mode=ColorMode.rgb, compression=Compression.rle,
    )

    with open(output_path, 'wb') as f:
        psd.write(f)
    print(f"PSD saved to {output_path} -- {len(images)} layers")