<div align="center">
# 🎬 MediaFlow
**一站式自媒体内容生产工具 · 集成音色克隆 / 文字转语音 / 自动字幕 / 智能剪辑**
</div>

---

## 核心功能

| 模块 | 能力 | 一句话介绍 |
|------|------|------------|
| 🎙️ **自动字幕** | Whisper 多语种转写 + VAD 静音检测 | 一行命令把视频变成 `.srt` + `.md` |
| ✂️ **口气词剪除** | 嗯/啊/呃/uh/um 等口水话自动剔除 | 自动清理"那个、然后、就是说" |
| 🌍 **字幕翻译** | OpenCC / OpenAI / DeepL / Google / 混元 | 一次生成中英双语字幕 |
| 🎬 **视频剪辑** | SRT/MD 勾选驱动 | 标记保留片段 → 一次性输出 |
| 🗣️ **文字转语音** | IndexTTS2 + IndexTTS1 双版本 | 中英日韩多语种、FP16/DeepSpeed 加速 |
| 🧬 **音色克隆** | 5 秒参考音频零样本克隆 | "用你自己的声音说任何话" |
| 🎞️ **配音视频** | 字幕 → TTS → 烧字幕 → 合成 | 完整一条龙，自动拼装 + 可选背景音 |
| 🪟 **Web 界面** | Gradio 统一面板 | 无需写命令，点点鼠标即可 |

---

## 目录结构

```
MediaFlow/
├── mediaflow/                 # 统一 Python 包
│   ├── common/                # 跨子包共享工具（日志 / YAML / MD / 段操作）
│   ├── tts/                   # IndexTTS / IndexTTS2 推理代码
│   │   ├── BigVGAN/           # 声码器
│   │   ├── gpt/               # GPT2 backbone（含 transformers 5.x 兼容垫片）
│   │   ├── s2mel/             # 梅尔频谱解码器（CFM / DiT / WavNet）
│   │   └── utils/             # 文本归一化 / 特征提取 / 采样器
│   ├── clipper/               # autocut 流水线
│   │   ├── cores/             # 翻译 / 转写 / 切割 / 守护
│   │   ├── caches/            # 默认 YAML 配置
│   │   ├── main.py            # CLI 入口
│   │   └── webui.py           # Gradio Web UI
│   ├── cli.py                 # 统一 CLI 调度器
│   └── __main__.py            # python -m mediaflow
├── tests/                     # pytest 用例
├── examples/                  # 用例与样例
├── docs/                      # 文档与图片
├── checkpoints/               # 模型权重（用户自行下载）
├── outputs/                   # 生成结果默认输出
├── prompts/                   # 提示词样例
├── pyproject.toml             # 统一构建/依赖清单
├── README.md
└── LICENSE
```

---

## 安装

```bash
# 1. 创建并激活 Python 3.10+ 虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. 安装（CPU/GPU 通用）
pip install -e .

# 可选：DeepSpeed 加速
pip install -e ".[deepspeed]"
```

> 模型权重不在仓库中，请按 `examples/` 或各子包文档的指引下载到
> `checkpoints/` 目录。

---

## 统一命令行

```bash
# 显式子命令
mediaflow tts "你好，欢迎使用 MediaFlow" -v prompts/voice.wav -o outputs/hello.wav
mediaflow clipper input.mp4 --cfg=config

# 快捷方式：第一个非标志参数默认进入 TTS
mediaflow "你好，欢迎使用 MediaFlow" -v prompts/voice.wav -o outputs/hello.wav

# 也可以作为模块运行
python -m mediaflow tts --help
python -m mediaflow clipper --help
```

---

## Python 程序化 API

可以直接在 Python 脚本里调用，**不需要走命令行**：

```python
from mediaflow.tts import SynthesisConfig, synthesize

config = SynthesisConfig(
    text="你好，欢迎使用 MediaFlow",
    voice_prompt="prompts/voice.wav",
    output_path="outputs/hello.wav",
    fp16=True,
)
synthesize(config)
```

剪辑器同样提供程序化入口：

```python
from mediaflow.clipper import ClipperConfig, transcribe, cut_media, to_markdown

cfg = ClipperConfig(inputs=["input.mp4"], cfg="config", cut_mode="transcribe")
transcribe(cfg)                # 生成 .srt / .md
# cut_media(cfg)               # 根据标记好的 .md 切割
# to_markdown(cfg)             # 把 .srt 转为 .md 任务列表
```

`mediaflow.tts.synthesize` 和 `mediaflow.clipper.{transcribe, cut_media,
run_daemon, to_markdown, compact_subtitles}` 是所有 CLI 背后的真正实现；
CLI 只是它们的一个薄包装。

---

## 运行测试

```bash
pip install pytest
pytest tests/
```

测试不要求 GPU / 网络 / 重量级模型。涉及 torch + opencc 的子集
会在依赖缺失时自动跳过。

---

## 旧版目录

`mediaflow/index-tts/` 与 `mediaflow/clipperX/` 已在重构中删除；
所有功能均已迁入 `mediaflow/tts/` 与 `mediaflow/clipper/`。

## 许可

见 [LICENSE](LICENSE)。
