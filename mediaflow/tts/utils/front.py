# -*- coding: utf-8 -*-
from functools import lru_cache
import os
import traceback
import re
from typing import List, Union, overload
import warnings
from mediaflow.tts.utils.common import tokenize_by_CJK_char, de_tokenized_by_CJK_char
from sentencepiece import SentencePieceProcessor


class TextNormalizer:
    def __init__(self, enable_glossary=False):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "锛?: ",",
            "锛?: ",",
            ";": ",",
            "锛?: ",",
            "銆?: ".",
            "锛?: "!",
            "锛?: "?",
            "\n": " ",
            "路": "-",
            "銆?: ",",
            "...": "鈥?,
            ",,,": "鈥?,
            "锛岋紝锛?: "鈥?,
            "鈥︹€?: "鈥?,
            "鈥?: "'",
            "鈥?: "'",
            '"': "'",
            "鈥?: "'",
            "鈥?: "'",
            "锛?: "'",
            "锛?: "'",
            "(": "'",
            ")": "'",
            "銆?: "'",
            "銆?: "'",
            "銆?: "'",
            "銆?: "'",
            "[": "'",
            "]": "'",
            "鈥?: "-",
            "锝?: "-",
            "~": "-",
            "銆?: "'",
            "銆?: "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }
        self.enable_glossary = enable_glossary
        # 鏈璇嶆眹琛細鐢ㄦ埛鍙嚜瀹氫箟涓撲笟鏈鐨勮娉?
        # 鏍煎紡: {"鍘熷鏈": {"en": "鑻辨枃璇绘硶", "zh": "涓枃璇绘硶"}}
        # "M.2": {"en": "M dot two", "zh": "M 浜?},
        # "PCIe 5.0": {"en": "PCIE five", "zh": "PCIE 浜旂偣闆?},
        # "PCIe 4.0": {"en": "PCIE four", "zh": "PCIE 鍥涚偣闆?},
        # "AHCI": "A H C I",
        # "TTS": "T T S",
        # "Inc.": {"en": "Ink"},
        # ".json": {"en": " dot Jay-Son", "zh": "鐐?Jay-Son"},
        # "C++": {"en": "C plus plus", "zh": "C 鍔犲姞"},
        # "C#": "C sharp"
        # self.term_glossary = {
        #     "C++": {"en": "C plus plus", "zh": "C 鍔犲姞"},
        #     "C#": "C sharp",
        #     "CMake": "C Make",
        # }
        self.term_glossary = dict()

    def match_email(self, email):
        # 姝ｅ垯琛ㄨ揪寮忓尮閰嶉偖绠辨牸寮忥細鏁板瓧鑻辨枃@鏁板瓧鑻辨枃.鑻辨枃
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiou眉v]|[ae]i|u[aio]|ao|ou|i[aue]|[u眉v]e|[uv眉]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    鍖归厤鎷奸煶澹拌皟鏍煎紡锛歱inyin+鏁板瓧锛屽０璋?-5锛?琛ㄧず杞诲０
    渚嬪锛歺uan4, jve2, ying1, zhong4, shang5
    涓嶅尮閰嶏細beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-路鈥擼[\u4e00-\u9fff]+){1,2}"
    """
    鍖归厤浜哄悕锛屾牸寮忥細涓枃路涓枃锛屼腑鏂嚶蜂腑鏂?涓枃
    渚嬪锛氬厠閲屾柉鎵樺紬路璇哄叞锛岀害鐟熷か路楂樼櫥-鑾辩淮鐗?
    """

    TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
    """
    鍖归厤鎶€鏈湳璇紝鏍煎紡锛氬瓧姣嶅紑澶?(瀛楁瘝鎴栨暟瀛?*+(-瀛楁瘝鎴栨暟瀛?+
    渚嬪锛欸PT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2
    蹇呴』浠ュ瓧姣嶅紑澶达紝閬垮厤鍖归厤绾暟瀛楋紙濡傜數璇濆彿鐮?135-4567-8900锛?
    鐢ㄤ簬淇濇姢杩炲瓧绗︾粨鏋勶紝闃叉涓枃normalizer灏嗚繛瀛楃瑙ｆ瀽涓哄噺鍙凤紙濡?璐熶簲鍑?锛?
    """

    # 鍖归厤甯歌鑻辫缂╁啓 's锛屼粎鐢ㄤ簬鏇挎崲涓?is锛屼笉鍖归厤鎵€鏈?'s
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"


    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() != "Linux":  # Mac and Windows
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, ".gitignore"), "w") as f:
                    f.write("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            # 搴旂敤鏈璇嶆眹琛紙浼樺厛绾ф渶楂橈紝鍦ㄦ墍鏈変繚鎶や箣鍓嶏級
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang="zh")
            # 淇濇姢鎶€鏈湳璇紙濡?GPT-5-nano锛夐伩鍏嶈涓枃normalizer閿欒澶勭悊
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)

            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # 鎭㈠浜哄悕
            result = self.restore_names(result, original_name_list)
            # 鎭㈠鎷奸煶澹拌皟
            result = self.restore_pinyin_tones(result, pinyin_list)
            # 鎭㈠鎶€鏈湳璇?
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                # 搴旂敤鏈璇嶆眹琛紙浼樺厛绾ф渶楂橈紝鍦ㄦ墍鏈変繚鎶や箣鍓嶏級
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                # 淇濇姢鎶€鏈湳璇紙濡?GPT-5-Nano锛夐伩鍏嶈鑻辨枃normalizer閿欒澶勭悊
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                # 鎭㈠鎶€鏈湳璇?
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin: str):
        """
        灏?jqx 鐨勯煹姣嶄负 u/眉 鐨勬嫾闊宠浆鎹负 v
        濡傦細ju -> jv , que -> qve, x眉n -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # 鍖归厤 jqx 鐨勯煹姣嶄负 u/眉 鐨勬嫾闊?
        pattern = r"([jqx])[u眉](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        鏇挎崲浜哄悕涓哄崰浣嶇 <n_a>銆?<n_b>, ...
        渚嬪锛氬厠閲屾柉鎵樺紬路璇哄叞 -> <n_a>
        """
        # 浜哄悕
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 鏇挎崲鍗犱綅绗?<n_a>銆?<n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        """
        鎭㈠浜哄悕涓哄師鏉ョ殑鏂囧瓧
        渚嬪锛?n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 鏇挎崲涓哄崰浣嶇 <n_a>銆?<n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_tech_terms(self, original_text):
        """
        淇濇姢鎶€鏈湳璇腑鐨勮繛瀛楃锛岄槻姝㈣涓枃normalizer瑙ｆ瀽涓哄噺鍙?
        绛栫暐锛氬皢鏈涓殑杩炲瓧绗︽浛鎹负鐗规畩鍗犱綅绗?H>锛屾暟瀛椾粛鍙姝ｅ父澶勭悊
        渚嬪锛欸PT-5-nano -> GPT<H>5<H>nano锛岀劧鍚?5 琚浆鎹负 浜?
        鏈€缁堟仮澶嶄负锛欸PT-浜?nano
        """
        tech_pattern = re.compile(TextNormalizer.TECH_TERM_PATTERN)
        original_tech_list = tech_pattern.findall(original_text)
        if len(original_tech_list) == 0:
            return (original_text, None)

        # 鍘婚噸骞舵寜闀垮害闄嶅簭鎺掑垪锛堥伩鍏嶇煭鍖归厤鍏堟浛鎹㈠鑷撮棶棰橈級
        original_tech_list = sorted(set(original_tech_list), key=len, reverse=True)
        transformed_text = original_text

        # 灏嗘湳璇腑鐨勮繛瀛楃鏇挎崲涓哄崰浣嶇 <H>
        for term in original_tech_list:
            # 灏?GPT-5-nano 鏇挎崲涓?GPT<H>5<H>nano
            protected_term = term.replace("-", "<H>")
            transformed_text = transformed_text.replace(term, protected_term)

        return transformed_text, original_tech_list

    def restore_tech_terms(self, normalized_text, original_tech_list):
        """
        鎭㈠鎶€鏈湳璇腑鐨勮繛瀛楃
        灏嗗崰浣嶇 <H> 鎭㈠涓鸿繛瀛楃 -
        鍚屾椂娓呯悊 normalizer 鍙兘鍦ㄥ崰浣嶇鍛ㄥ洿娣诲姞鐨勫浣欑┖鏍?
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text

        # 娓呯悊 <H> 鍛ㄥ洿鍙兘鐨勭┖鏍硷紝鐒跺悗鎭㈠涓鸿繛瀛楃
        # 澶勭悊妯″紡: " <H> " -> "-", " <H>" -> "-", "<H> " -> "-", "<H>" -> "-"
        transformed_text = re.sub(r'\s*<H>\s*', '-', normalized_text)
        return transformed_text

    def apply_glossary_terms(self, text, lang="zh"):
        """
        搴旂敤鏈璇嶆眹琛紝灏嗕笓涓氭湳璇浛鎹负瀵瑰簲璇█鐨勮娉?

        Args:
            text: 寰呭鐞嗘枃鏈?
            lang: 璇█绫诲瀷 "zh" 鎴?"en"

        Returns:
            澶勭悊鍚庣殑鏂囨湰

        Example:
            "M.2 NVMe SSD" -> (zh) "M 浜?NVMe SSD"
            "M.2 NVMe SSD" -> (en) "M dot two NVMe SSD"
        """
        if not self.term_glossary:
            return text

        # 鎸夋湳璇暱搴﹂檷搴忔帓鍒楋紝閬垮厤鐭湳璇厛鍖归厤瀵艰嚧闀挎湳璇棤娉曞尮閰?
        # 渚嬪锛?PCIe 5.0" 搴旇鍦?"PCIe" 涔嬪墠鍖归厤
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)
        @lru_cache(maxsize=42)
        def get_term_pattern(term: str):
            return re.compile(re.escape(term), re.IGNORECASE)
        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            # 浣跨敤姝ｅ垯杩涜澶у皬鍐欎笉鏁忔劅鐨勬浛鎹?
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)

        return transformed_text

    def load_glossary(self, glossary_dict):
        """
        鍔犺浇澶栭儴鏈璇嶆眹琛?

        Args:
            glossary_dict: 鏈璇嶅吀锛屾牸寮忎负 {"鏈": {"en": "鑻辨枃璇绘硶", "zh": "涓枃璇绘硶"}}

        Example:
            normalizer.load_glossary({
                "M.2": {"en": "M dot two", "zh": "M 浜?},
                "PCIe": {"en": "PCIE", "zh": "PCIE"}
            })
        """
        if glossary_dict and isinstance(glossary_dict, dict):
            self.term_glossary.update(glossary_dict)

    def load_glossary_from_yaml(self, glossary_path):
        """
        浠?YAML 鏂囦欢鍔犺浇鏈璇嶆眹琛?

        Args:
            glossary_path: YAML 鏂囦欢璺緞

        Example:
            normalizer.load_glossary_from_yaml("checkpoints/glossary.yaml")

        YAML 鏂囦欢鏍煎紡:
            M.2:
              en: M dot two
              zh: M 浜?
            NVMe: N-V-M-E  # 涓嫳鏂囩浉鍚岃娉?
        """
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, 'r', encoding='utf-8') as f:
                external_glossary = yaml.safe_load(f)
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_glossary_to_yaml(self, glossary_path):
        """
        淇濆瓨鏈璇嶆眹琛ㄥ埌 YAML 鏂囦欢

        Args:
            glossary_path: YAML 鏂囦欢璺緞
        """
        import yaml
        with open(glossary_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.term_glossary, f, allow_unicode=True, default_flow_style=False)

    def save_pinyin_tones(self, original_text):
        """
        鏇挎崲鎷奸煶澹拌皟涓哄崰浣嶇 <pinyin_a>, <pinyin_b>, ...
        渚嬪锛歺uan4 -> <pinyin_a>
        """
        # 澹版瘝闊垫瘝+澹拌皟鏁板瓧
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 鏇挎崲涓哄崰浣嶇 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        鎭㈠鎷奸煶涓殑闊宠皟鏁板瓧锛?-5锛変负鍘熸潵鐨勬嫾闊?
        渚嬪锛?pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 鏇挎崲鍗犱綅绗?<pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # 鍔犺浇璇嶈〃
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [
            # 棰勫鐞嗗櫒
            tokenize_by_CJK_char,
        ]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        # 棰勫鐞?
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        # 棰勫鐞?
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_segments_by_token(
        tokenized_str: List[str],
        split_tokens: List[str],
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int = 0
    ) -> List[List[str]]:
        """
        灏唗okenize鍚庣殑缁撴灉鎸夌壒瀹歵oken杩涗竴姝ュ垎鍓?
        """
        # 澶勭悊鐗规畩鎯呭喌
        if len(tokenized_str) == 0:
            return []
        segments: List[List[str]] = []
        current_segment = []
        current_segment_tokens_len = 0
        for i in range(len(tokenized_str)):
            token = tokenized_str[i]
            current_segment.append(token)
            current_segment_tokens_len += 1
            if not  ("," in split_tokens or "鈻?" in split_tokens ) and ("," in current_segment or "鈻?" in current_segment): 
                # 濡傛灉褰撳墠tokens涓湁,锛屽垯鎸?鍒嗗壊
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment, [",", "鈻?"], max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
                )
            elif "-" not in split_tokens and "-" in current_segment:
                # 娌℃湁,锛屽垯鎸?鍒嗗壊
                sub_segments = TextTokenizer.split_segments_by_token(
                    current_segment, ["-"], max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
                )
            elif current_segment_tokens_len <= max_text_tokens_per_segment:
                if token in split_tokens and current_segment_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "鈻?"]:
                            # 鍚庣画token鏄?锛屽垯涓嶅垏鍒?
                            current_segment.append(tokenized_str[i + 1])
                            i += 1
                    segments.append(current_segment)
                    current_segment = []
                    current_segment_tokens_len = 0
                continue
            # 濡傛灉褰撳墠tokens鐨勯暱搴﹁秴杩囨渶澶ч檺鍒?
            else:
                # 鎸夌収闀垮害鍒嗗壊
                sub_segments = []
                for j in range(0, len(current_segment), max_text_tokens_per_segment):
                    if j + max_text_tokens_per_segment < len(current_segment):
                        sub_segments.append(current_segment[j : j + max_text_tokens_per_segment])
                    else:
                        sub_segments.append(current_segment[j:])
                warnings.warn(
                    f"The tokens length of segment exceeds limit: {max_text_tokens_per_segment}, "
                    f"Tokens in segment: {current_segment}."
                    "Maybe unexpected behavior",
                    RuntimeWarning,
                )
            segments.extend(sub_segments)
            current_segment = []
            current_segment_tokens_len = 0
        if current_segment_tokens_len > 0:
            assert current_segment_tokens_len <= max_text_tokens_per_segment
            segments.append(current_segment)
        # 濡傛灉鐩搁偦鐨勫彞瀛愬姞璧锋潵闀垮害灏忎簬鏈€澶ч檺鍒讹紝涓旀鍓峵oken鎬绘暟瓒呰繃quick_streaming_tokens锛屽垯鍚堝苟
        merged_segments = []
        total_token = 0
        for segment in segments:
            total_token += len(segment)
            if len(segment) == 0:
                continue
            if len(merged_segments) == 0:
                merged_segments.append(segment)
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment and total_token > quick_streaming_tokens:
                merged_segments[-1] = merged_segments[-1] + segment
            # 鎴栧皬浜庢渶澶ч暱搴﹂檺鍒剁殑涓€鍗婏紝鍒欏悎骞?
            elif len(merged_segments[-1]) + len(segment) <= max_text_tokens_per_segment / 2:
                merged_segments[-1] = merged_segments[-1] + segment
            else:
                merged_segments.append(segment)
        return merged_segments

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        "鈻?",
        # "鈻?", # unk
        "鈻?",
        "鈻?..", # ellipsis
    ]
    def split_segments(self, tokenized: List[str], max_text_tokens_per_segment=120, quick_streaming_tokens = 0) -> List[List[str]]:
        return TextTokenizer.split_segments_by_token(
            tokenized, self.punctuation_marks_tokens, max_text_tokens_per_segment=max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens
        )


if __name__ == "__main__":
    # 娴嬭瘯绋嬪簭

    text_normalizer = TextNormalizer(enable_glossary=True)

    cases = [
        "IndexTTS 姝ｅ紡鍙戝竷1.0鐗堟湰浜嗭紝鏁堟灉666",
        "鏅昘UAN4鏄竴绉岹AN3瑙?,
        "鎴戠埍浣狅紒",
        "I love you!",
        "鈥滄垜鐖变綘鈥濈殑鑻辫鏄€淚 love you鈥?,
        "2.5骞虫柟鐢电嚎",
        "鍏?65绡囷紝绾?15涓囧瓧",
        "2002骞寸殑绗竴鍦洪洩锛屼笅鍦ㄤ簡2003骞?,
        "閫熷害鏄?0km/h",
        "鐜板湪鏄寳浜椂闂?025骞?1鏈?1鏃?20:00",
        "浠栬繖鏉¤￥瀛愭槸2012骞翠拱鐨勶紝鑺变簡200鍧楅挶",
        "鐢佃瘽锛?35-4567-8900",
        "1閿?杩?,
        "浠栬繖鏉¤棰戠偣璧?000+锛岃瘎璁?000+锛屾敹钘?00+",
        "杩欐槸1024鍏冪殑鎵嬫満锛屼綘瑕佸悧锛?,
        "鍙椾笉liao3浣犱簡",
        "鈥滆。瑁斥€濅笉璇昏。chang2锛岃€屾槸璇昏。shang5",
        "鏈€zhong4瑕佺殑鏄細涓嶈chong2韫堣杈?,
        "涓峼uo1姝诲氨涓嶄細姝?,
        "See you at 8:00 AM",
        "8:00 AM 寮€浼?,
        "Couting down 3, 2, 1, go!",
        "鏁板埌3灏卞紑濮嬶細1銆?銆?",
        "This sales for 2.5% off, only $12.5.",
        "5G缃戠粶鏄?G缃戠粶鐨勫崌绾х増锛?G缃戠粶鏄?G缃戠粶鐨勫墠韬?,
        "鑻规灉浜?030/1/2鍙戝竷鏂?iPhone 2X 绯诲垪鎵嬫満锛屾渶浣庡敭浠蜂粎 楼12999",
        "杩欓厭...閲?..鏈夋瘨...",
        # 寮傚父case
        "鍙湁,,,鎵嶆槸鏈€濂界殑",
        "babala2鏄粈涔堬紵",  # babala浜屾槸浠€涔?
        "鐢╞eta1娴嬭瘯",  # 鐢╞eta涓€娴嬭瘯
        "have you ever been to beta2?",  # have you ever been to beta two?
        "where's the money?",  # where is the money?
        "who's there?",  # who is there?
        "which's the best?",  # which is the best?
        "how's it going?",  # how is it going?
        "浠婂ぉ鏄釜濂芥棩瀛?it's a good day",  # 浠婂ぉ鏄釜濂芥棩瀛?it is a good day
        # 鏈
        "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",  # such as xtts,cosyvoice two,fish-speech,and f five-tts
        "GPT-5-Nano is the smallest and fastest variant in the GPT-5 model family.",  # GPT-five-Nano is the smallest and fastest variant in the GPT-five model family
        "GPT-5-Nano 鏄?GPT-5 妯″瀷瀹舵棌涓渶灏忎笖閫熷害鏈€蹇殑鍙樹綋",  # GPT-浜?Nano 鏄?GPT-浜?绯荤粺涓渶灏忎笖閫熷害鏈€蹇殑鍙樹綋
        "2025/09/08 IndexTTS-2 鍏ㄧ悆鍙戝竷",  # 浜岄浂浜屼簲骞翠節鏈堝叓鏃?IndexTTS-浜屽叏鐞冨彂甯?
        "Here are some highly-rated M.2 NVMe SSDs: Samsung 9100 PRO PCIe 5.0 SSD M.2, $139.99",  # Here are some highly-rated M dot two NVMe SSD's, Samsung nine thousand one hundred PRO PCIE five SSD M dot two . one hundred and thirty nine dollars and ninety nine cents
        "we dive deep into the showdown between DisplayPort 1.4 and HDMI 2.1 to determine which is the best choice for gaming enthusiasts",
        # 浜哄悕
        "绾︾憻澶烽珮鐧?鑾辩淮鐗癸紙Joseph Gordon-Levitt is an American actor锛?,
        "钂傝帿瑗柯峰攼绾冲痉路搴撳厠锛堣嫳鏂囧悕锛歍imothy Donald Cook锛夛紝閫氱О钂傚路搴撳厠锛圱im Cook锛夛紝缇庡浗鍟嗕笟缁忕悊銆佸伐涓氬伐绋嬪笀鍜屽伐涓氬紑鍙戝晢锛岀幇浠昏嫻鏋滃叕鍙搁甯墽琛屽畼銆?,
        # 闀垮彞瀛?
        "銆婄洍姊︾┖闂淬€嬫槸鐢辩編鍥藉崕绾冲厔寮熷奖鐗囧叕鍙稿嚭鍝佺殑鐢靛奖锛岀敱鍏嬮噷鏂墭寮椔疯鍏版墽瀵煎苟缂栧墽锛岃幈鏄傜撼澶毬疯开鍗℃櫘閲屽ゥ銆佺帥涓芥槀路姝岃开浜氥€佺害鐟熷か路楂樼櫥-鑾辩淮鐗广€佽壘鍒╁ゥ鐗孤蜂僵鍚夈€佹堡濮喡峰搱杩瓑鑱旇涓绘紨锛?010骞?鏈?6鏃ュ湪缇庡浗涓婃槧锛?010骞?鏈?鏃ュ湪涓浗鍐呭湴涓婃槧锛?020骞?鏈?8鏃ュ湪涓浗鍐呭湴閲嶆槧銆傚奖鐗囧墽鎯呮父璧颁簬姊﹀涓庣幇瀹炰箣闂达紝琚畾涔変负鈥滃彂鐢熷湪鎰忚瘑缁撴瀯鍐呯殑褰撲唬鍔ㄤ綔绉戝够鐗団€濓紝璁茶堪浜嗙敱鑾辨槀绾冲路杩崱鏅噷濂ユ壆婕旂殑閫犳ⅵ甯堬紝甯﹂鐗瑰伐鍥㈤槦杩涘叆浠栦汉姊﹀锛屼粠浠栦汉鐨勬綔鎰忚瘑涓洍鍙栨満瀵嗭紝骞堕噸濉戜粬浜烘ⅵ澧冪殑鏁呬簨銆?,
        "娓呮櫒鎷夊紑绐楀笜锛岄槼鍏夋磼鍦ㄧ獥鍙扮殑Bloomixy鑺辫壓绀肩洅涓娾€斺€旇柊琛ｈ崏棣欒柊铚＄儧鍞ら啋鍡呰锛屾案鐢熻姳鏉熸姌灏勫嚭鏅ㄩ湶鑸厜娉姐€傝璁″笀灏嗏€滆嚜鐒剁唤鏀剧編瀛︹€濊瀺鍏ユ瘡涓粏鑺傦細鎵嬪伐闄剁摲鑺辩摱鍙綔棣栭グ鏀剁撼锛岄钖扮簿娌瑰惈渚濆叞渚濆叞鑸掔紦閰嶆柟銆傞檺閲忔闄勮禒銆?65澶╂彃鑺辩伒鎰熸墜鍐屻€嬶紝璁╂瘡涓钩鍑℃棩瀛愰兘鏈夎姳寮€浠紡鎰熴€俓n瀹翠細鍘呯伅鍏夋殫涓嬬殑鍒归偅锛孏limmeria鏄熸湀绯诲垪鑰冲潬寮€濮嬪彂鍏夆€斺€旂憺澹喎鐝愮悈宸ヨ壓璁╄摑瀹濈煶濡傞摱娌虫祦鍔紝閽涘悎閲戦鏋朵粎3.2g鏃犺礋閲嶆劅銆傝璁″笀绉樺瘑锛氬唴缃井鍨嬮噸鍔涙劅搴斿櫒锛岄殢姝ヤ紣浜х敓0.01mm鎸箙锛屾墦閫犫€滆璧扮殑鏄熷厜鈥濄€備竷澶曢檺瀹氱ぜ鐩掑惈鏄熷骇瀹氬埗閾墝锛岃鐖辨剰濡傛槦杈版案鎭掗棯鑰€銆?,
        "鐢靛奖1锛氣€滈粦鏆楅獞澹€濓紙婕斿憳锛氬厠閲屾柉钂傚畨路璐濆皵銆佸笇鏂疯幈鏉帮紱瀵兼紨锛氬厠閲屾柉鎵樺紬路璇哄叞锛夛紱鐢靛奖2锛氣€滅洍姊︾┖闂粹€濓紙婕斿憳锛氳幈鏄傜撼澶毬疯开鍗℃櫘閲屽ゥ锛涘婕旓細鍏嬮噷鏂墭寮椔疯鍏帮級锛涚數褰?锛氣€滈挗鐞村鈥濓紙婕斿憳锛氳壘寰烽噷瀹壜峰竷娲涜开锛涘婕旓細缃楁浖路娉㈠叞鏂熀锛夛紱鐢靛奖4锛氣€滄嘲鍧﹀凹鍏嬪彿鈥濓紙婕斿憳锛氳幈鏄傜撼澶毬疯开鍗℃櫘閲屽ゥ锛涘婕旓細瑭瑰鏂峰崱姊呴殕锛夛紱鐢靛奖5锛氣€滈樋鍑¤揪鈥濓紙婕斿憳锛氳惃濮喡锋矁杈涢】锛涘婕旓細瑭瑰鏂峰崱姊呴殕锛夛紱鐢靛奖6锛氣€滃崡鏂瑰叕鍥細澶х數褰扁€濓紙婕斿憳锛氶┈鐗孤锋柉閫氥€佹墭椹柉路鑹炬仼鏍肩憺锛涘婕旓細鐗归浄路甯曞厠锛?,
    ]
    # 娴嬭瘯鍒嗚瘝鍣?
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/bpe.model",
        normalizer=text_normalizer,
    )

    codes = tokenizer.batch_encode(
        cases,
        out_type=int,
    )

    print(f"vocab_size: {tokenizer.vocab_size}")
    # print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"unk_token: {tokenizer.unk_token}, unk_token_id: {tokenizer.unk_token_id}")
    # 娴嬭瘯鎷奸煶 (8474-10201)
    for id in range(8474, 10201):
        pinyin = tokenizer.convert_ids_to_tokens(id)
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, pinyin, re.IGNORECASE) is None:
            print(f"{pinyin} should be matched")
    for badcase in [
        "beta1", "better1", "voice2", "bala2", "babala2", "hunger2"
    ]:
        if re.match(TextNormalizer.PINYIN_TONE_PATTERN, badcase, re.IGNORECASE) is not None:
            print(f"{badcase} should not be matched!")
    # 涓嶅簲璇ユ湁 unk_token_id
    for t in set([*TextTokenizer.punctuation_marks_tokens, ",", "鈻?", "-", "鈻?.."]):
        tokens = tokenizer.convert_tokens_to_ids(t)
        if tokenizer.unk_token_id in tokens:
            print(f"Warning: {t} is unknown token")
        print(f"`{t}`", "->", tokens, "->", tokenizer.convert_ids_to_tokens(tokens))
    for ch in set(tokenizer.normalizer.zh_char_rep_map.values()):
        # 娴嬭瘯 normalize鍚庣殑瀛楃鑳借鍒嗚瘝鍣ㄨ瘑鍒?
        print(f"`{ch}`", "->", tokenizer.sp_model.Encode(ch, out_type=str))
        print(f"` {ch}`", "->", tokenizer.sp_model.Encode(f" {ch}", out_type=str))
    max_text_tokens_per_segment=120
    for i in range(len(cases)):
        print(f"鍘熷鏂囨湰: {cases[i]}")
        print(f"Normalized: {text_normalizer.normalize(cases[i])}")
        tokens = tokenizer.tokenize(cases[i])
        print("Tokenzied: ", ", ".join([f"`{t}`" for t in tokens]))
        segments = tokenizer.split_segments(tokens, max_text_tokens_per_segment=max_text_tokens_per_segment)
        print("Segments count:", len(segments))
        if len(segments) > 1:
            for j in range(len(segments)):
                print(f"  {j}, count:", len(segments[j]), ", tokens:", "".join(segments[j]))
                if len(segments[j]) > max_text_tokens_per_segment:
                    print(f"Warning: segment {j} is too long, length: {len(segments[j])}")
        #print(f"Token IDs (first 10): {codes[i][:10]}")
        if tokenizer.unk_token in codes[i]:
            print(f"Warning: `{cases[i]}` contains UNKNOWN token")
        print(f"Decoded: {tokenizer.decode(codes[i], do_lower_case=True)}")
        print("-" * 50)
