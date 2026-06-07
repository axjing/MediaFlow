# -*- coding: utf-8 -*-
"""Gradio web UI for the MediaFlow clipper subcommand."""
import os

import gradio as gr

from mediaflow.common import YamlParser, read_file

CURRENT_DIR = os.path.dirname(__file__)

transcribe_config_path = os.path.join(CURRENT_DIR, "caches/transcribe_config")


def transcriber_web_api(video_path, audio_path, yaml_file=transcribe_config_path):
    print(transcribe_config_path)
    args = YamlParser(cfg_name="", path=yaml_file)
    if video_path:
        file_path = video_path
    elif audio_path:
        file_path = audio_path
    else:
        print("ERROR: no input provided")

    args.inputs = [file_path]

    from mediaflow.clipper.cores.transcriber import Transcribe

    Transcribe(args).run()

    srt_file = os.path.splitext(file_path)[0] + ".srt"

    txt = read_file(srt_file)

    return txt


demo = gr.Interface(
    fn=transcriber_web_api,
    inputs=[
        gr.Video(source="upload", label="In", interactive=True),
        gr.Audio(source="upload", label="In", type="filepath", interactive=True),
    ],
    outputs="text",
    css="footer {visibility: hidden}",
)

if __name__ == "__main__":
    demo.launch(share=False)
