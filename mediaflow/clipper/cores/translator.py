"""
字幕翻译模块
支持多种翻译API，支持英文字幕翻译成中文，保持中英文字幕同时存在
"""
import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from mediaflow.common import Logging
logger = Logging(__name__).get_logger()


class BaseTranslator(ABC):
    """翻译器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # self.supported_languages = {'en', 'zh'}
        self.supported_languages_map = {'zh': 'Chinese', 'en': 'English'}

    @abstractmethod
    def translate(self, text: str, target_lang: str = 'zh') -> str:
        """翻译单条文本"""
        pass

    @abstractmethod
    def translate_batch(self, texts: List[str], target_lang: str = 'zh') -> List[str]:
        """批量翻译文本"""
        pass

    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测逻辑
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'[^\w\s]', '', text))

        if total_chars == 0:
            return 'unknown'

        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.3 else 'en'

    def cleanup(self) -> None:
        """Release any model resources held by the translator."""
        for attr in ("model", "tokenizer"):
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    del value
                setattr(self, attr, None)
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass




class GoogleTranslator(BaseTranslator):
    """Google翻译器（使用免费API）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('google_api_key')
        self.base_url = "https://translation.googleapis.com/language/translate/v2"

    def translate(self, text: str, target_lang: str = 'zh') -> str:
        if not text.strip():
            return text

        try:
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests module not available")

            params = {
                'q': text,
                'target': 'zh-cn' if target_lang == 'zh' else target_lang,
                'key': self.api_key,
                'format': 'text'
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if 'data' in data and 'translations' in data['data']:
                return data['data']['translations'][0]['translatedText']
            else:
                logger.warning(f"Google translate API response error: {data}")
                return text

        except Exception as e:
            logger.error(f"Google translate error: {e}")
            return text

    def translate_batch(self, texts: List[str], target_lang: str = 'zh') -> List[str]:
        return [self.translate(text, target_lang) for text in texts]


class DeepLTranslator(BaseTranslator):
    """DeepL翻译器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('deepl_api_key')
        self.base_url = "https://api-free.deepl.com/v2/translate"

    def translate(self, text: str, target_lang: str = 'zh') -> str:
        if not text.strip():
            return text

        try:
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests module not available")

            target = 'ZH' if target_lang == 'zh' else target_lang.upper()
            data = {
                'text': [text],
                'target_lang': target,
                'auth_key': self.api_key
            }

            response = requests.post(self.base_url, json=data, timeout=15)
            response.raise_for_status()

            result = response.json()
            if 'translations' in result:
                return result['translations'][0]['text']
            else:
                logger.warning(f"DeepL translate API response error: {result}")
                return text

        except Exception as e:
            logger.error(f"DeepL translate error: {e}")
            return text

    def translate_batch(self, texts: List[str], target_lang: str = 'zh') -> List[str]:
        target = 'ZH' if target_lang == 'zh' else target_lang.upper()

        try:
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests module not available")

            data = {
                'text': texts,
                'target_lang': target,
                'auth_key': self.api_key
            }

            response = requests.post(self.base_url, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            if 'translations' in result:
                return [t['text'] for t in result['translations']]
            else:
                logger.warning(f"DeepL batch translate error: {result}")
                return texts

        except Exception as e:
            logger.error(f"DeepL batch translate error: {e}")
            return texts
class HunyuanMT(BaseTranslator):
    """HunyuanMT翻译器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name_or_path = config.get('model_name_or_path', "tencent/Hunyuan-MT-7B")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto",torch_dtype=torch.bfloat16)
    def translate(self, text: str, target_lang: str = 'zh') -> str:
        if not text.strip():
            return text
        
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers module not available")
            target_lang_map = {'zh': 'Chinese', 'en': 'English'}
            target_name = target_lang_map.get(target_lang, target_lang)
            user_prompt=f"Translate the following segment into {target_name}, without additional explanation.\n\n{text}"
            messages = [
                {"role": "user", "content": user_prompt},
            ]
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            # 计算输入token长度
            input_length = tokenized_chat.shape[1]
            print(f"===> 输入token长度: {input_length}")
            
            # 检查是否超出模型最大长度
            max_model_length = self.model.config.max_position_embeddings
            if input_length >= max_model_length:
                # 截断过长的输入
                tokenized_chat = tokenized_chat[:, :max_model_length - 100]
                input_length = tokenized_chat.shape[1]
                print(f"===> 输入过长，已截断至: {input_length} tokens")
            
            # 动态计算最大生成token数
            # 中文翻译通常比英文短，但为安全起见，设置合理的上限
            estimated_output_tokens = min(len(text) * 3, 4096)  # 更合理的估计
            max_new_tokens = min(estimated_output_tokens, max_model_length - input_length - 10)
            
            print(f"===> 设置max_new_tokens: {max_new_tokens}")
            print(f"===> 原始输入长度: {len(text)}")
            outputs = self.model.generate(
                tokenized_chat.to(self.model.device),
                max_new_tokens=max_new_tokens,  # 更严格的长度限制
                do_sample=True,
                temperature=0.3,  # 降低温度以获得更稳定的输出
                top_p=0.9,
                repetition_penalty=1.1,
            )
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # print(f"原始输出: {output_text}")
            translation_text = self._split_mt_text_by_separator(output_text)
            # print(f"提取的翻译: {translation_text}")
            # 如果提取失败，尝试直接清理输出
            if not translation_text:
                translation_text = self._clean_translation_output(output_text)
                # print(f"清理后的输出: {translation_text}")
            # 进一步清理翻译结果
            translation_text = self._post_process_translation(translation_text, text)
            print(f"===> 最终翻译: {translation_text}")
            
            return translation_text
        except Exception as e:
            logger.error(f"HunyuanMT translate error: {e}")
            return text
    def _split_mt_text_by_separator(self, text):
        """提取字符串中 <|extra_0|> 和 <|eos|> 之间的文本"""
        pattern = r'<\|extra_0\|>(.*?)<\|eos\|>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            middle_text = match.group(1).strip()
            return middle_text
        else:
            return ""

    def translate_batch(self, texts: List[str], target_lang: str = 'zh') -> List[str]:
        """批量翻译使用组合文本"""
        if not texts:
            return []
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers module not available")


            # 组合文本以提高效率
            combined_text = "\n\n---\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])
            print(f"Combined text for translation: {combined_text}")
            result=self.translate(combined_text, target_lang)
            
            logger.info(f"HunyuanMT batch translate result: {result}")
            # 解析结果
            translations = []
            for line in result.split('\n'):
                # print(f"Processing line: {line}")
                match = re.match(r'\[\d+\]\s*(.+)', line.strip())
                if match:
                    translations.append(match.group(1))
                    # print(f"Extracted translation: {match.group(1)}")
            # print(f"Texts:{texts[len(translations)]}")
            # 确保结果数量匹配
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])
            print(f"Final translations: {translations},translation length:{len(translations)},texts length:{len(texts)}")
            return translations[:len(texts)]
            
        except Exception as e:
            logger.error(f"HunyuanMT batch translate error: {e}")
            return [self.translate(text, target_lang) for text in texts]
            

            
    def _clean_translation_output(self, text):
        """清理翻译输出，移除特殊标记"""
        # 移除模型特定的特殊标记
        clean_text = re.sub(r'<\|.*?\|>', '', text)
        # 移除多余的空格和换行
        clean_text = ' '.join(clean_text.split()).strip()
        return clean_text
    def _post_process_translation(self, translation, original):
        """后处理翻译结果，确保质量"""
        # 移除可能的多余内容
        translation = re.sub(r'^(翻译|译文|中文):?\s*', '', translation, flags=re.IGNORECASE)
        translation = translation.strip()
        
        # 如果翻译结果与原文相同或为空，返回原文
        if not translation or translation == original:
            return original
            
        # 限制翻译长度（不超过原文长度的3倍）
        max_length = len(original) * 3
        if len(translation) > max_length:
            translation = translation[:int(max_length)] + "..."
            
        return translation
            
class OpenAITranslator(BaseTranslator):
    """OpenAI翻译器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('openai_api_key')
        self.model = config.get('openai_model', 'gpt-3.5-turbo')
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def translate(self, text: str, target_lang: str = 'zh') -> str:
        if not text.strip():
            return text

        try:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai module not available")

            target_lang_map = {'zh': 'Chinese', 'en': 'English'}
            target_name = target_lang_map.get(target_lang, target_lang)
            
            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the given text to {target_name}. Only return the translation, no explanations."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=len(text) * 2
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI translate error: {e}")
            return text

    def translate_batch(self, texts: List[str], target_lang: str = 'zh') -> List[str]:
        """批量翻译使用组合文本"""
        if not texts:
            return []

        try:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai module not available")

            target_lang_map = {'zh': 'Chinese', 'en': 'English'}
            target_name = target_lang_map.get(target_lang, target_lang)

            # 组合文本以提高效率
            combined_text = "\n\n---\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])

            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the numbered texts to {target_name}. Return only the translations in the same numbered format, preserving the line breaks."},
                    {"role": "user", "content": combined_text}
                ],
                temperature=0.3,
                max_tokens=len(combined_text) * 2
            )

            result = response.choices[0].message.content.strip()

            # 解析结果
            translations = []
            for line in result.split('\n'):
                match = re.match(r'\[\d+\]\s*(.+)', line.strip())
                if match:
                    translations.append(match.group(1))

            # 确保结果数量匹配
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])

            return translations[:len(texts)]

        except Exception as e:
            logger.error(f"OpenAI batch translate error: {e}")
            return texts


class LocalTranslator(BaseTranslator):
    """本地翻译器（使用OpenCC）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import opencc
            self.converter = opencc.OpenCC('t2s')
            self.s2t_converter = opencc.OpenCC('s2t')
        except ImportError:
            logger.error("OpenCC not available for local translation")
            self.converter = None
            self.s2t_converter = None

    def translate(self, text: str, target_lang: str = 'zh') -> str:
        if not self.converter:
            return text

        try:
            # 简繁转换（如果已经是中文则不需要翻译）
            if self.detect_language(text) == 'en':
                # 英文到中文需要外部翻译，这里只做占位
                return text
            else:
                # 中文转换
                if target_lang == 'zh':
                    return self.converter.convert(text)
                else:
                    return self.s2t_converter.convert(text)
        except Exception as e:
            logger.error(f"Local translate error: {e}")
            return text

    def translate_batch(self, texts: List[str], target_lang: str = 'zh') -> List[str]:
        return [self.translate(text, target_lang) for text in texts]


class SubtitleTranslator:
    """字幕翻译管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.translator = self._create_translator()
        self.translation_cache = {}

    def _create_translator(self) -> BaseTranslator:
        """根据配置创建翻译器"""
        translator_type = self.config.get('translator_type', 'local')

        if translator_type == 'openai' and OPENAI_AVAILABLE:
            return OpenAITranslator(self.config)
        elif translator_type == 'deepl' and REQUESTS_AVAILABLE:
            return DeepLTranslator(self.config)
        elif translator_type == 'google' and REQUESTS_AVAILABLE:
            return GoogleTranslator(self.config)
        
        elif translator_type == 'hunyuanmt' and TRANSFORMERS_AVAILABLE:
            return HunyuanMT(self.config)
        else:
            return LocalTranslator(self.config)

    def translate_subtitles(self, subtitles: List[Dict[str, Any]],
                          source_lang: str = 'auto',
                          target_lang: str = 'zh') -> List[Dict[str, Any]]:
        """翻译字幕列表"""
        if not self.translator or not subtitles:
            return subtitles

        logger.info(f"开始翻译 {len(subtitles)} 条字幕")

        translated_subtitles = []
        texts_to_translate = []
        subtitle_indices = []

        # 第一遍：检测需要翻译的字幕
        for i, subtitle in enumerate(subtitles):
            original_text = subtitle.get('text', '').strip()
            logger.info(f"检测字幕 {i + 1}: {original_text}")

            if not original_text:
                translated_subtitles.append(subtitle.copy())
                continue

            detected_lang = self.translator.detect_language(original_text) if source_lang == 'auto' else source_lang

            if detected_lang == target_lang:
                # 目标语言，不需要翻译
                translated_subtitles.append(subtitle.copy())
            elif detected_lang != 'en' and target_lang == 'zh':
                # 已经是中文，不需要翻译
                translated_subtitles.append(subtitle.copy())
            else:
                # 需要翻译
                texts_to_translate.append(original_text)
                subtitle_indices.append(i)
                translated_subtitles.append(None)  # 占位
        
        # 批量翻译
        if texts_to_translate:
            logger.info(f"翻译 {len(texts_to_translate)} 条文本")

            # 分批处理避免API限制
            batch_size = self.config.get('translate_batch_size', 10)
            translated_texts = []

            for i in range(0, len(texts_to_translate), batch_size):
                batch = texts_to_translate[i:i+batch_size]
                try:
                    logger.info(f"开始翻译批次 {i//batch_size + 1},{batch},batch_size:{batch_size}")
                    batch_translated = self.translator.translate_batch(batch, target_lang)
                    logger.info(f"翻译结果：\n{batch_translated}")
                    logger.info(f"{'🎉'*3} [Successfully translated batch.] {'🎉'*3}")
                    translated_texts.extend(batch_translated)
                    time.sleep(1)  # 避免API限制
                except Exception as e:
                    logger.error(f"翻译批次 {i//batch_size + 1} 失败: {e}")
                    translated_texts.extend(batch)  # 使用原文

            # 替换需要翻译的字幕
            for idx, translated_text in zip(subtitle_indices, translated_texts):
                if idx < len(translated_subtitles) and translated_subtitles[idx] is None:
                    original_subtitle = subtitles[idx].copy()

                    # 创建双语字幕
                    original_text = original_subtitle.get('text', '')
                    if self.config.get('bilingual_subtitles', True):
                        # 双语字幕格式
                        bilingual_text = self._format_bilingual_subtitle(original_text, translated_text, target_lang)
                        original_subtitle['text'] = bilingual_text
                        original_subtitle['translated_text'] = translated_text
                        original_subtitle['original_text'] = original_text
                    else:
                        # 单语字幕（仅翻译）
                        original_subtitle['text'] = translated_text
                        original_subtitle['translated_text'] = translated_text

                    original_subtitle['translated'] = True
                    translated_subtitles[idx] = original_subtitle

        # 清理None值
        result = [sub for sub in translated_subtitles if sub is not None]
        logger.info(f"翻译完成，共处理 {len(result)} 条字幕")
        logger.info(f"{result}\n'-'*100")

        return result

    def _format_bilingual_subtitle(self, original: str, translated: str, target_lang: str) -> str:
        """格式化双语字幕"""
        if not translated or translated == original:
            return original

        bilingual_format = self.config.get('bilingual_format', 'dual_line')

        if bilingual_format == 'dual_line':
            if target_lang == 'zh':
                return f"{translated}\n{original}"
            else:
                return f"{original}\n{translated}"
        elif bilingual_format == 'brackets':
            if target_lang == 'zh':
                return f"{translated} ({original})"
            else:
                return f"{original} ({translated})"
        elif bilingual_format == 'slash':
            return f"{translated} / {original}"
        else:
            return translated  # 默认返回翻译

    def get_translation_stats(self, subtitles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取翻译统计信息"""
        total = len(subtitles)
        translated = sum(1 for sub in subtitles if sub.get('translated', False))
        bilingual = sum(1 for sub in subtitles if 'original_text' in sub)

        return {
            'total_subtitles': total,
            'translated_subtitles': translated,
            'bilingual_subtitles': bilingual,
            'translation_rate': translated / total if total > 0 else 0,
            'bilingual_rate': bilingual / total if total > 0 else 0
        }