import datetime
import os
import time

import opencc
import srt
import torch
import whisper
from tqdm import tqdm

from mediaflow.clipper.cores.translator import SubtitleTranslator
from mediaflow.common import (
    Logging,
    MD,
    check_exists,
    expand_segments,
    merge_adjacent_segments,
    remove_short_segments,
)
logger = Logging(__name__).get_logger()


def process(whisper_model, audio, seg, lang, prompt):
    r = whisper_model.transcribe(
        audio[int(seg["start"]) : int(seg["end"])],
        task="transcribe",
        language=lang,
        initial_prompt=prompt,
    )
    r["origin_timestamp"] = seg
    return r


class Transcribe:
    def __init__(self, args):
        self.args = args
        self.sampling_rate = 16000
        self.whisper_model = None
        self.translator = None
        self.vad_model = None
        self.detect_speech = None
        self.is_translator = False
        if hasattr(self.args, 'translate') and self.args.translate:
            self.is_translator = True
            # 创建翻译器配置
            self.translator_config = {
                'translator_type': getattr(self.args, 'translator_type', 'hunyuanmt'),
                'bilingual_subtitles': getattr(self.args, 'bilingual', True),
                'bilingual_format': getattr(self.args, 'bilingual_format', 'dual_line'),
                'translate_batch_size': 2
            }
            
        self.torch_home = os.environ.get("TORCH_HOME")
        self.transformer_home=os.environ.get("TRANSFORMERS_HOME")
            
        
        
    def run(self):
        try:
            for input in self.args.inputs:
                logger.info(f"Transcribing {input}")
                
                name, _ = os.path.splitext(input)
                if check_exists(name + ".md", self.args.force_write):
                    continue
                
                audio = whisper.load_audio(input, sr=self.sampling_rate)
                if (
                    self.args.use_VAD == "1"
                    or self.args.use_VAD == "auto"
                    and not name.endswith("_cut")
                ):
                    speech_timestamps = self._detect_voice_activity(audio)
                else:
                    speech_timestamps = [{"start": 0, "end": len(audio)}]
                    
                logger.info(
                    f"Detected {len(speech_timestamps)} segments, sampling rate {self.sampling_rate}"
                )
                transcribe_results = self._transcribe(audio, speech_timestamps)
                self._cleanup()
                print(f"self.args.translate: {self.args.translate},\nself.args:{self.args}")
                
                
                # 检查是否启用双语字幕
                if self.is_translator:
                    output = f"{name}_bilingual.srt"
                    logger.info(f"Generating bilingual subtitles for {input}")
                else:
                    output = f"{name}.srt"
                
                
                srt_result = self._save_srt(output, transcribe_results)
                logger.info(f"Transcribed {input} to {output}")
                
                self._save_md(f"{name}.md", output, input)
                logger.info(f'Saved texts to {name + ".md"} to mark sentences')
                self._cleanup()
                return srt_result
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            self._cleanup()

    def _detect_voice_activity(self, audio):
        """Detect segments that have voice activities"""
        tic = time.time()
        if self.vad_model is None or self.detect_speech is None:
            # torch load limit https://github.compytorch/vision/issues/4156
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            # repo_or_dir = "snakers4/silero-vad"
            # repo_or_dir_path = os.path.join(self.torch_home, "hub",f"{repo_or_dir.replace('/', '_')}_master")
            # if os.path.exists(repo_or_dir_path):
            #     self.vad_model, funcs = torch.hub.load(
            #         repo_or_dir=repo_or_dir_path, model="silero_vad", trust_repo=False
            #     )
            # else:
            self.vad_model, funcs = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
            )

            self.detect_speech = funcs[0]

        speeches = self.detect_speech(
            audio, self.vad_model, sampling_rate=self.sampling_rate
        )

        # Remove too short segments
        speeches = remove_short_segments(speeches, 1.0 * self.sampling_rate)

        # Expand to avoid to tight cut. You can tune the pad length
        speeches = expand_segments(
            speeches, 0.2 * self.sampling_rate, 0.0 * self.sampling_rate, audio.shape[0]
        )

        # Merge very closed segments
        speeches = merge_adjacent_segments(speeches, 0.5 * self.sampling_rate)

        logger.info(f"Done voice activity detection in {time.time() - tic:.1f} sec")
        return speeches if len(speeches) > 1 else [{"start": 0, "end": len(audio)}]

    def _transcribe(self, audio, speech_timestamps):
        tic = time.time()
        logger.info(
            f"Transcribing..."
        )
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(
                self.args.whisper_model, self.args.device
            )

        res = []
        if self.args.device == "cpu" and len(speech_timestamps) > 1:
            from multiprocessing import Pool

            pbar = tqdm(total=len(speech_timestamps))

            pool = Pool(processes=4)
            # TODO, a better way is merging these segments into a single one, so whisper can get more context
            for seg in speech_timestamps:
                r = self.whisper_model.transcribe(
                    audio[int(seg["start"]) : int(seg["end"])],
                    task="transcribe",
                    language=self.args.language,
                    initial_prompt=self.args.prompt,
                    verbose=False if len(speech_timestamps) == 1 else None,
                )
                r["origin_timestamp"] = seg
                res.append(r)
            #     res.append(
            #         pool.apply_async(
            #             process,
            #             (
            #                 self.whisper_model,
            #                 audio,
            #                 seg,
            #                 self.args.language,
            #                 self.args.prompt,
            #             ),
            #             callback=lambda x: pbar.update(),
            #         )
            #     )
            # pool.close()
            # pool.join()
            # pbar.close()
            # res_l=[]
            # for i in res:
            #     res_l.append(i.get())
            #     del i
            # return res_l
            logger.info(f"Done transcription in {time.time() - tic:.1f} sec")
            return res
        else:
            for seg in (
                speech_timestamps
                if len(speech_timestamps) == 1
                else tqdm(speech_timestamps)
            ):
                logger.info(
                    f"Transcribing {seg['start'] / self.sampling_rate:.1f} - {seg['end'] / self.sampling_rate:.1f} sec"
                )
                r = self.whisper_model.transcribe(
                    audio[int(seg["start"]) : int(seg["end"])],
                    task="transcribe",
                    language=self.args.language,
                    initial_prompt=self.args.prompt,
                    verbose=False if len(speech_timestamps) == 1 else None,
                )
                r['text']=r['text'].lstrip()
                r["origin_timestamp"] = seg
                print(f"r: {r}")
                res.append(r)
            logger.info(f"Done transcription in {time.time() - tic:.1f} sec")
            return res

    def _save_srt(self, output, transcribe_results):
        subs = []
        # whisper sometimes generate traditional chinese, explicitly convert
        cc = opencc.OpenCC("t2s")

        def _add_sub(start, end, text, translation=None):
            # 如果有翻译，创建双语字幕
            if translation and translation.strip() and translation != text:
                content = f"{text}\n{translation}"
            else:
                content = text
            # print(f"[_add_sub] content: {content}")
            subs.append(
                srt.Subtitle(
                    index=0,
                    start=datetime.timedelta(seconds=start),
                    end=datetime.timedelta(seconds=end),
                    content=cc.convert(content.strip()),
                )
            )

        prev_end = 0
        # texts_to_translate = []
        subtitle_data = []
        
        # 收集需要翻译的文本
        for r in transcribe_results:
            origin = r["origin_timestamp"]
            for s in r["segments"]:
                start = s["start"] + origin["start"] / self.sampling_rate
                end = min(
                    s["end"] + origin["start"] / self.sampling_rate,
                    origin["end"] / self.sampling_rate,
                )
                if start > end:
                    continue
                    
                if start > prev_end + 1.0:
                    _add_sub(prev_end, start, "< No Speech >")
                
                # texts_to_translate.append(s["text"])
                subtitle_data.append({
                    'text': s["text"],
                    'start': start,
                    'end': end,
                    'prev_end': prev_end
                })
                prev_end = end

        # 批量翻译
        if self.is_translator:
            # 准备字幕格式
            subtitles_for_translation = [
                {'text': data['text'], 'start': data['start'], 'end': data['end']}
                for data in subtitle_data
            ]
            
            if self.translator is None:
                self.translator = SubtitleTranslator(self.translator_config)
            
            # 批量翻译
            translated_subtitles = self.translator.translate_subtitles(
                subtitles_for_translation, 
                source_lang='auto', 
                target_lang='zh'
            )
            
            # 生成双语字幕
            for i, (data, translated) in enumerate(zip(subtitle_data, translated_subtitles)):
                # print(f"{data} ----> {translated}")
                _add_sub(data['start'], data['end'], data['text'], translated['translated_text']) # translated['text']
        else:
            # 不使用翻译
            for data in subtitle_data:
                _add_sub(data['start'], data['end'], data['text'])

        with open(output, "wb") as f:
            f.write(srt.compose(subs).encode(self.args.encoding, "replace"))
            
        return subs


    def _save_md(self, md_fn, srt_fn, video_fn):
        with open(srt_fn, encoding=self.args.encoding) as f:
            subs = srt.parse(f.read())

        md = MD(md_fn, self.args.encoding)
        md.clear()
        md.add_done_editing(False)
        md.add_video(os.path.basename(video_fn))
        
        # 检查是否是双语字幕
        if "_bilingual" in srt_fn:
            md.add(
                f"\n**Bilingual Subtitles**\n"
                f"Texts generated from [{os.path.basename(srt_fn)}]({os.path.basename(srt_fn)}).\n"
                "Each line contains original text followed by translation.\n"
                "Mark the sentences to keep for autocut.\n"
                "The format is [subtitle_index,duration_in_second] subtitle context.\n\n"
            )
        else:
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
        
    def _cleanup(self):
        """清理所有模型资源"""
        import gc
        import torch
        
        # 清理Whisper模型
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            
        # 清理VAD模型
        if self.vad_model is not None:
            del self.vad_model
            self.vad_model = None
            self.detect_speech = None
            
        # 清理翻译器
        if self.translator is not None:
            if hasattr(self.translator, 'cleanup'):
                self.translator.cleanup()
            del self.translator
            self.translator = None
            
        # 强制垃圾回收
        gc.collect()
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("模型资源已清理")
