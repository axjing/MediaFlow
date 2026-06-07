import os
import re

import srt



from moviepy import editor


from mediaflow.common import Logging, MD, add_cut, change_ext, check_exists, is_video
logger = Logging(__name__).get_logger()


# Merge videos
class Merger:
    def __init__(self, args):
        self.args = args

    def write_md(self, videos):
        md = MD(self.args.inputs[0], self.args.encoding)
        num_tasks = len(md.tasks())
        # Not overwrite if already marked as down or no new videos
        if md.done_editing() or num_tasks == len(videos) + 1:
            return

        md.clear()
        md.add_done_editing(False)
        md.add("\nSelect the files that will be used to generate `autocut_final.mp4`\n")
        base = lambda fn: os.path.basename(fn)
        for f in videos:
            md_fn = change_ext(f, "md")
            video_md = MD(md_fn, self.args.encoding)
            # select a few words to scribe the video
            desc = ""
            if len(video_md.tasks()) > 1:
                for _, t in video_md.tasks()[1:]:
                    m = re.findall(r"\] (.*)", t)
                    if m and "no speech" not in m[0].lower():
                        desc += m[0] + " "
                    if len(desc) > 50:
                        break
            md.add_task(
                False,
                f'[{base(f)}]({base(md_fn)}) {"[Edited]" if video_md.done_editing() else ""} {desc}',
            )
        md.write()

    def run(self):
        md_fn = self.args.inputs[0]
        md = MD(md_fn, self.args.encoding)
        if not md.done_editing():
            return

        videos = []
        for m, t in md.tasks():
            if not m:
                continue
            m = re.findall(r"\[(.*)\]", t)
            if not m:
                continue
            fn = os.path.join(os.path.dirname(md_fn), m[0])
            logger.info(f"Loading {fn}")
            videos.append(editor.VideoFileClip(fn))

        dur = sum([v.duration for v in videos])
        logger.info(f"Merging into a video with {dur / 60:.1f} min length")

        merged = editor.concatenate_videoclips(videos)
        fn = os.path.splitext(md_fn)[0] + "_merged.mp4"
        merged.write_videofile(
            fn, audio_codec="aac", bitrate=self.args.bitrate
        )  # logger=None,
        logger.info(f"Saved merged video to {fn}")



class Cutter:
    """_summary_
        Cut media
    """
    def __init__(self, args):
        self.args = args

    def run(self):
        fns = {"srt": None, "media": None, "md": None}
        for fn in self.args.inputs:
            ext = os.path.splitext(fn)[1][1:]
            fns[ext if ext in fns else "media"] = fn

        assert fns["media"], "must provide a media filename"
        assert fns["srt"], "must provide a srt filename"

        is_video_file = is_video(fns["media"])
        outext = "mp4" if is_video_file else "mp3"
        output_fn = change_ext(add_cut(fns["media"]), outext)
        if check_exists(output_fn, self.args.force_write):
            return

        with open(fns["srt"], encoding=self.args.encoding) as f:
            subs = list(srt.parse(f.read()))

        if fns["md"]:
            md = MD(fns["md"], self.args.encoding)
            if not md.done_editing():
                return
            index = []
            for mark, sent in md.tasks():
                if not mark:
                    continue
                m = re.match(r"\[(\d+)", sent.strip())
                if m:
                    index.append(int(m.groups()[0]))
            subs = [s for s in subs if s.index in index]
            logger.info(f'Cut {fns["media"]} based on {fns["srt"]} and {fns["md"]}')
        else:
            logger.info(f'Cut {fns["media"]} based on {fns["srt"]}')

        segments = []
        # Avoid disordered subtitles
        subs.sort(key=lambda x: x.start)
        for x in subs:
            if len(segments) == 0:
                segments.append(
                    {"start": x.start.total_seconds(), "end": x.end.total_seconds()}
                )
            else:
                if x.start.total_seconds() - segments[-1]["end"] < 0.5:
                    segments[-1]["end"] = x.end.total_seconds()
                else:
                    segments.append(
                        {"start": x.start.total_seconds(), "end": x.end.total_seconds()}
                    )

        if is_video_file:
            media = editor.VideoFileClip(fns["media"])
        else:
            media = editor.AudioFileClip(fns["media"])

        # Add a fade between two clips. Not quite necessary. keep code here for reference
        # fade = 0
        # segments = _expand_segments(segments, fade, 0, video.duration)
        # clips = [video.subclip(
        #         s['start'], s['end']).crossfadein(fade) for s in segments]
        # final_clip = editor.concatenate_videoclips(clips, padding = -fade)

        clips = [media.subclip(s["start"], s["end"]) for s in segments]
        if is_video_file:
            final_clip: editor.VideoClip = editor.concatenate_videoclips(clips)
            logger.info(
                f"Reduced duration from {media.duration:.1f} to {final_clip.duration:.1f}"
            )

            aud = final_clip.audio.set_fps(44100)
            final_clip = final_clip.without_audio().set_audio(aud)
            final_clip = final_clip.fx(editor.afx.audio_normalize)


            # 修复：明确指定 FPS 参数
            # 获取原始视频的 FPS，如果无法获取则使用默认值 24
            fps = media.fps if media.fps else 24
             # 确保 FPS 是浮点数
            fps = float(fps)
            logger.info(f"Using FPS: {fps}, bitrate: {self.args.bitrate}")
            
            # 修复：显式设置 final_clip 的 fps 属性，并确保它被正确设置
            try:
                final_clip.set_fps(fps)
                # 验证 FPS 是否被正确设置
                if hasattr(final_clip, 'fps') and final_clip.fps is not None:
                    logger.info(f"Final clip FPS successfully set to: {final_clip.fps}")
                else:
                    logger.warning("Failed to set FPS on final clip, using parameter only")
            except Exception as e:
                logger.warning(f"Error setting FPS on final clip: {e}")
            
            print(f"final_clip.fps:{final_clip.fps},fps:{fps}")
            # an alternative to birate is use crf, e.g. ffmpeg_params=['-crf', '18']
            final_clip.write_videofile(
                output_fn,fps=fps, audio_codec="aac", bitrate=self.args.bitrate
            )
        else:
            final_clip: editor.AudioClip = editor.concatenate_audioclips(clips)
            logger.info(
                f"Reduced duration from {media.duration:.1f} to {final_clip.duration:.1f}"
            )

            final_clip = final_clip.fx(editor.afx.audio_normalize)
            final_clip.write_audiofile(
                output_fn, codec="libmp3lame", fps=44100, bitrate=self.args.bitrate
            )

        media.close()
        logger.info(f"Saved media to {output_fn}")
        return 0
