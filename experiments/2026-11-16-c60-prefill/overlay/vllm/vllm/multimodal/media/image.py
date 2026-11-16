# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np
import pybase64
import torch
from blake3 import blake3
from PIL import Image

import vllm.envs as envs
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import (
    check_sparse_tensor_invariants_threadsafe,
    safe_to_dense,
)

from ..image import convert_image_mode, normalize_image, rgba_to_rgb
from .base import MediaIO, MediaWithBytes

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0


_VALIDATED_IMAGE_META_MAX_ENTRIES = 512
_VALIDATED_IMAGE_META_CACHE: OrderedDict[
    tuple[bytes, str | None, tuple[int, int, int], int],
    tuple[tuple[int, int], str, dict[str, object] | None],
] = OrderedDict()
_VALIDATED_IMAGE_META_LOCK = threading.Lock()


class _LazyPILImage(Image.Image):
    """A PIL.Image subclass for previously validated image bytes that exposes
    `.size`, `.width`, `.height`, and `.mode` without decoding pixel data,
    materializing the underlying PIL Image on demand if pixel operations are
    invoked.
    """

    def __init__(
        self,
        raw_bytes: bytes,
        loader: Callable[[bytes], Image.Image],
        size: tuple[int, int],
        mode: str,
    ) -> None:
        super().__init__()
        self._raw_bytes = raw_bytes
        self._loader = loader
        self._size = size
        self._mode = mode
        self._materialized: Image.Image | None = None

    @property
    def is_materialized(self) -> bool:
        return self._materialized is not None

    def _materialize(self) -> Image.Image:
        img = self._materialized
        if img is None:
            img = self._loader(self._raw_bytes)
            self.im = img.im
            self._mode = img.mode
            self._size = img.size
            self.info = img.info
            self.palette = img.palette
            self.readonly = img.readonly
            self._exif = getattr(img, "_exif", None)
            self._materialized = img
        return img

    def getexif(self) -> Image.Exif:
        if self._materialized is not None:
            return self._materialized.getexif()
        return Image.Exif()

    def load(self):
        return self._materialize().load()

    def copy(self) -> Image.Image:
        return self._materialize().copy()

    def convert(self, *args, **kwargs) -> Image.Image:
        return self._materialize().convert(*args, **kwargs)

    def resize(self, *args, **kwargs) -> Image.Image:
        return self._materialize().resize(*args, **kwargs)

    def crop(self, *args, **kwargs) -> Image.Image:
        return self._materialize().crop(*args, **kwargs)

    def draft(self, *args, **kwargs):
        return self._materialize().draft(*args, **kwargs)

    def thumbnail(self, *args, **kwargs) -> None:
        self._materialize().thumbnail(*args, **kwargs)
        self._size = self._materialized.size

    def save(self, *args, **kwargs) -> None:
        self._materialize().save(*args, **kwargs)

    def tobytes(self, *args, **kwargs) -> bytes:
        return self._materialize().tobytes(*args, **kwargs)

    def getdata(self, *args, **kwargs):
        return self._materialize().getdata(*args, **kwargs)

    def getpixel(self, *args, **kwargs):
        return self._materialize().getpixel(*args, **kwargs)

    def putpixel(self, *args, **kwargs) -> None:
        self._materialize().putpixel(*args, **kwargs)

    def split(self):
        return self._materialize().split()

    def getbands(self) -> tuple[str, ...]:
        return Image.getmodebands(self.mode)

    def getbbox(self, *args, **kwargs):
        return self._materialize().getbbox(*args, **kwargs)

    def getextrema(self):
        return self._materialize().getextrema()

    @property
    def __array_interface__(self):
        return self._materialize().__array_interface__


class ImageMediaIO(MediaIO[Image.Image]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self, image_mode: str | None = "RGB", **kwargs) -> None:
        super().__init__()

        # Target mode for loaded images; `None` keeps the original mode
        # (i.e. no conversion, alpha channel is preserved as-is).
        self.image_mode = image_mode
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if self.image_mode is None or image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(
                image, self.image_mode, self.rgba_background_color
            )

    @staticmethod
    def _max_pixel_bounds(w: int, h: int, max_pixels: int) -> tuple[int, int]:
        scale = math.sqrt(max_pixels / (w * h))
        return max(1, int(math.floor(w * scale))), max(1, int(math.floor(h * scale)))

    def _decode_and_convert_bytes(
        self, data: bytes
    ) -> tuple[Image.Image, dict[str, object] | None]:
        try:
            image = Image.open(BytesIO(data))
            w, h = image.size
            max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
            downscaled = False
            if max_pixels > 0 and w * h > max_pixels:
                image.draft(None, self._max_pixel_bounds(w, h, max_pixels))
                downscaled = True
            image = normalize_image(image)
            w, h = image.size
            if max_pixels > 0 and w * h > max_pixels:
                image.thumbnail(
                    self._max_pixel_bounds(w, h, max_pixels),
                    Image.Resampling.BILINEAR,
                )
                downscaled = True
            image.load()
            converted = self._convert_image_mode(image)
        except (OSError, Image.UnidentifiedImageError) as e:
            raise ValueError(f"Failed to load image: {e}") from e

        io_config: dict[str, object] | None = None
        if converted is not image:
            io_config = {
                "image_mode": self.image_mode,
                "rgba_background_color": self.rgba_background_color,
            }
        if downscaled:
            io_config = {**(io_config or {}), "max_pixels": max_pixels}
        return converted, io_config

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]:
        max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
        digest = (
            blake3(data, max_threads=blake3.AUTO).digest()
            if len(data) >= 524288
            else blake3(data).digest()
        )
        cache_key = (
            digest,
            self.image_mode,
            self.rgba_background_color,
            max_pixels,
        )
        with _VALIDATED_IMAGE_META_LOCK:
            cached = _VALIDATED_IMAGE_META_CACHE.get(cache_key)
            if cached is not None:
                _VALIDATED_IMAGE_META_CACHE.move_to_end(cache_key)

        if cached is not None:
            size, mode, cached_io_config = cached
            io_config = dict(cached_io_config) if cached_io_config is not None else None
            lazy_img = _LazyPILImage(
                data,
                lambda raw: self._decode_and_convert_bytes(raw)[0],
                size,
                mode,
            )
            return MediaWithBytes(lazy_img, data, io_config)

        converted, io_config = self._decode_and_convert_bytes(data)
        if Image.ExifTags.Base.ImageID not in converted.getexif():
            with _VALIDATED_IMAGE_META_LOCK:
                _VALIDATED_IMAGE_META_CACHE[cache_key] = (
                    converted.size,
                    converted.mode,
                    dict(io_config) if io_config is not None else None,
                )
                _VALIDATED_IMAGE_META_CACHE.move_to_end(cache_key)
                while len(_VALIDATED_IMAGE_META_CACHE) > _VALIDATED_IMAGE_META_MAX_ENTRIES:
                    _VALIDATED_IMAGE_META_CACHE.popitem(last=False)

        return MediaWithBytes(converted, data, io_config)

    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]:
        return self.load_bytes(filepath.read_bytes())

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def _load_pickled_torch(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(buffer, weights_only=True)
            return safe_to_dense(tensor, parameter="image_embeds")

    def _load_numpy(self, data: bytes) -> torch.Tensor:
        with BytesIO(data) as buffer:
            return torch.from_numpy(np.load(buffer))

    def load_bytes(self, data: bytes) -> torch.Tensor:
        if data[:6] == MAGIC_NUMPY_PREFIX:
            return self._load_numpy(data)

        return self._load_pickled_torch(data)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        if filepath.suffix == ".npy":
            return torch.from_numpy(np.load(filepath))

        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(filepath, weights_only=True)
            return safe_to_dense(tensor, parameter="image_embeds")

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
