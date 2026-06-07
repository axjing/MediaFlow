"""Clipper-specific helpers that build on top of :mod:`mediaflow.common`."""
from __future__ import annotations

import base64
import logging
import os
import re
from io import BytesIO
from typing import TYPE_CHECKING, Optional

from .markdown import MD

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from PIL import Image


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents."""
    with open(file_path, "r", encoding=encoding) as fh:
        return fh.read()


def gen_web(camera):
    """Video streaming generator function for MJPEG endpoints."""
    while True:
        frame = camera.get_frame()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


def base64_to_pil(img_base64: str) -> "Image.Image":
    """Decode a base64 data-URL image into a :class:`PIL.Image.Image`."""
    from PIL import Image

    image_data = re.sub(r"^data:image/.+;base64,", "", img_base64)
    return Image.open(BytesIO(base64.b64decode(image_data)))


def np_to_base64(img_np: "np.ndarray") -> str:
    """Encode an RGB ``numpy`` frame as a PNG data-URL string."""
    from PIL import Image
    import numpy as np

    img = Image.fromarray(img_np.astype("uint8"), "RGB")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def str_to_bool(s: str) -> bool:
    """Parse a strict ``"true"`` / ``"false"`` string into a boolean."""
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"Cannot parse bool from {s!r}")


def is_video(filename: str) -> bool:
    """Return ``True`` if ``filename`` has a known video extension."""
    _, ext = os.path.splitext(filename)
    return ext in [".mp4", ".mov", ".mkv", ".avi", ".flv", ".f4v", ".webm"]


def is_audio(filename: str) -> bool:
    """Return ``True`` if ``filename`` has a known audio extension."""
    _, ext = os.path.splitext(filename)
    return ext in [".ogg", ".wav", ".mp3", ".flac", ".m4a"]


def change_ext(filename: str, new_ext: str) -> str:
    """Replace the extension of ``filename`` with ``new_ext`` (with leading dot)."""
    base, _ = os.path.splitext(filename)
    if not new_ext.startswith("."):
        new_ext = "." + new_ext
    return base + new_ext


def add_cut(filename: str) -> str:
    """Append a ``_cut`` marker to the stem of ``filename``."""
    base, ext = os.path.splitext(filename)
    if base.endswith("_cut"):
        base = base[:-4] + "_" + base[-4:]
    else:
        base += "_cut"
    return base + ext


def check_exists(output: str, force: bool) -> bool:
    """Return ``True`` when ``output`` exists and we should not overwrite it."""
    if os.path.exists(output):
        if force:
            logging.info(f"{output} exists. Will overwrite it")
        else:
            logging.info(
                f"{output} exists, skipping... Use the --force flag to overwrite"
            )
            return True
    return False


def compact_rst(sub_fn: str, encoding: str = "utf-8") -> Optional[str]:
    """Toggle an SRT file between verbose and compact forms.

    Returns the path of the file written, or ``None`` if the format was rejected.
    """
    import opencc
    import srt

    base, ext = os.path.splitext(sub_fn)
    COMPACT = "_compact"
    if ext != ".srt":
        logging.fatal("only .srt file is supported")
        return None

    if base.endswith(COMPACT):
        with open(sub_fn, encoding=encoding) as fh:
            lines = fh.readlines()
        subs = []
        for line in lines:
            items = line.split(" ")
            if len(items) < 4:
                continue
            subs.append(
                srt.Subtitle(
                    index=0,
                    start=srt.srt_timestamp_to_timedelta(items[0]),
                    end=srt.srt_timestamp_to_timedelta(items[2]),
                    content=" ".join(items[3:]).strip(),
                )
            )
        target = base[: -len(COMPACT)] + ext
        with open(target, "wb") as fh:
            fh.write(srt.compose(subs).encode(encoding, "replace"))
        return target

    cc = opencc.OpenCC("t2s")
    with open(sub_fn, encoding=encoding) as fh:
        subs = srt.parse(fh.read())
    target = base + COMPACT + ext
    with open(target, "wb") as fh:
        for s in subs:
            fh.write(
                f"{srt.timedelta_to_srt_timestamp(s.start)} --> "
                f"{srt.timedelta_to_srt_timestamp(s.end)} "
                f"{cc.convert(s.content.strip())}\n".encode(encoding, "replace")
            )
    return target


def trans_srt_to_md(
    encoding: str,
    force: bool,
    srt_fn: str,
    video_fn: Optional[str] = None,
) -> str:
    """Convert an SRT subtitle file into a markdown editing checklist.

    Returns the path of the generated ``.md`` file.
    """
    import srt

    base, ext = os.path.splitext(srt_fn)
    if ext != ".srt":
        logging.fatal("only .srt file is supported")
    md_fn = base + ext.split(".")[0] + ".md"

    check_exists(md_fn, force)

    with open(srt_fn, encoding=encoding) as fh:
        subs = srt.parse(fh.read())

    md = MD(md_fn, encoding)
    md.clear()
    md.add_done_editing(False)
    if video_fn:
        if not is_video(video_fn):
            logging.fatal(f"{video_fn} may not be a video")
        md.add_video(os.path.basename(video_fn))
    md.add(
        f"\nTexts generated from [{os.path.basename(srt_fn)}]({os.path.basename(srt_fn)})."
        "Mark the sentences to keep for autocut.\n"
        "The format is [subtitle_index,duration_in_second] subtitle context.\n\n"
    )

    for s in subs:
        sec = s.start.seconds
        pre = f"[{s.index},{sec // 60:02d}:{sec % 60:02d}]"
        md.add_task(False, f"{pre:11} {s.content.strip()}")
    md.write()
    return md_fn
