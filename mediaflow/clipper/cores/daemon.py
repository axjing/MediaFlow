import copy
import glob
import os
import time

from mediaflow.clipper.cores import cut, transcriber
from mediaflow.common import (
    Logging,
    MD,
    add_cut,
    change_ext,
    is_audio,
    is_video,
)
logger = Logging(__name__).get_logger()


class Daemon:
    def __init__(self, args):
        self.args = args
        self.sleep = 1

    def run(self):
        assert len(self.args.inputs) == 1, "Must provide a single folder"
        while True:
            self._iter()
            time.sleep(self.sleep)
            self.sleep = min(60, self.sleep + 1)

    def _iter(self):
        folder = self.args.inputs[0]
        files = sorted(list(glob.glob(os.path.join(folder, "*"))))
        media_files = [f for f in files if is_video(f) or is_audio(f)]
        args = copy.deepcopy(self.args)
        for f in media_files:
            srt_fn = change_ext(f, "srt")
            md_fn = change_ext(f, "md")
            is_video_file = is_video(f)
            if srt_fn not in files or md_fn not in files:
                args.inputs = [f]
                try:
                    transcriber.Transcribe(args).run()
                    self.sleep = 1
                    break
                except RuntimeError as e:
                    logger.warn(
                        "Failed, may be due to the video is still on recording"
                    )
                    pass
            if md_fn in files:
                if add_cut(md_fn) in files:
                    continue
                md = MD(md_fn, self.args.encoding)
                ext = "mp4" if is_video_file else "mp3"
                if not md.done_editing() or os.path.exists(
                    change_ext(add_cut(f), ext)
                ):
                    continue
                args.inputs = [f, md_fn, srt_fn]
                cut.Cutter(args).run()
                self.sleep = 1

        args.inputs = [os.path.join(folder, "autocut.md")]
        merger = cut.Merger(args)
        merger.write_md(media_files)
        merger.run()
