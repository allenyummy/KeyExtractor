# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Tokenizaton

import logging
from abc import ABC, abstractmethod
from typing import List, Union
from ckip_transformers.nlp import CkipWordSegmenter
from transformers import AutoTokenizer

# from ckiptagger import data_utils, construct_dictionary, WS
# import spacy
# import jieba
# import monpa

logger = logging.getLogger(__name__)


TOKENIZER_MODEL_MAP = {
    "ckiptagger": "model/ckiptagger/",
    "ckip-transformers-albert-tiny": "ckiplab/albert-tiny-chinese-ws",
    "ckip-transformers-albert-base": "ckiplab/albert-base-chinese-ws",
    "ckip-transformers-bert-base": "ckiplab/bert-base-chinese-ws",
    "bert-tokenizer": "hfl/chinese-bert-wwm",
    "spacy-zh_core_web_sm_3.0.0": "model/spacy/zh_core_web_sm-3.0.0/zh_core_web_sm/zh_core_web_sm-3.0.0/",
    "spacy-zh_core_web_md_3.0.0": "model/spacy/zh_core_web_md-3.0.0/zh_core_web_md/zh_core_web_md-3.0.0",
    "spacy-zh_core_web_lg_3.0.0": "model/spacy/zh_core_web_lg-3.0.0/zh_core_web_lg/zh_core_web_lg-3.0.0",
    "spacy-zh_core_web_trf_3.0.0": "model/spacy/zh_core_web_trf-3.0.0/zh_core_web_trf/zh_core_web_trf-3.0.0",
    "jieba": None,
    "monpa": None,
    "nltk": None,
}


def TokenizerFactory(name: str):

    model_path = TOKENIZER_MODEL_MAP[name]

    LOCALIZERS = {
        # "ckiptagger": Ckip_Tagger_Tokenizer,
        "ckip-transformers-albert-tiny": Ckip_Transformers_Tokenizer,
        "ckip-transformers-albert-base": Ckip_Transformers_Tokenizer,
        "ckip-transformers-bert-base": Ckip_Transformers_Tokenizer,
        "bert-tokenizer": Bert_Tokenizer,
        # "spacy-zh_core_web_sm_3.0.0": Spacy_Chinese_Tokenizer,
        # "spacy-zh_core_web_md_3.0.0": Spacy_Chinese_Tokenizer,
        # "spacy-zh_core_web_lg_3.0.0": Spacy_Chinese_Tokenizer,
        # "spacy-zh_core_web_trf_3.0.0": Spacy_Chinese_Tokenizer,
        # "jieba": Jieba_Tokenizer,
        # "monpa": Monpa_Tokenizer,
        # "nltk": NLTK_Tokenizer,
    }
    return LOCALIZERS[name](model_path) if model_path else LOCALIZERS[name]()


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:
        raise NotImplementedError


# class Ckip_Tagger_Tokenizer(Tokenizer):
#     def __init__(self, model_path: str):
#         self.ws_model = WS(model_path)

#     def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:

#         if isinstance(text, str):
#             text = [text]
#         elif isinstance(text, list):
#             pass
#         else:
#             raise ValueError(
#                 f"Expect text type (str or List[str]) but got {type(text)}"
#             )
#         tokenized_text = self.ws_model(text)
#         return tokenized_text


class Ckip_Transformers_Tokenizer(Tokenizer):

    MODEL_LEVEL_MAP = {
        "ckiplab/albert-tiny-chinese-ws": 1,
        "ckiplab/albert-base-chinese-ws": 2,
        "ckiplab/bert-base-chinese-ws": 3,
    }

    def __init__(self, model_path: str):
        if model_path not in self.MODEL_LEVEL_MAP.keys():
            raise ValueError(
                f"Expect {self.MODEL_LEVEL_MAP.keys()} but got {model_path}."
            )
        self.ws_model = CkipWordSegmenter(
            level=self.MODEL_LEVEL_MAP[model_path], device=-1
        )

    def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:

        if isinstance(text, str):
            text = [text]
        elif isinstance(text, list):
            pass
        else:
            raise ValueError(
                f"Expect text type (str or List[str]) but got {type(text)}"
            )
        tokenized_text = self.ws_model(text)
        return tokenized_text


class Bert_Tokenizer(Tokenizer):
    def __init__(self, model_path: str):
        self.ws_model = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:
        if isinstance(text, str):
            tokenized_text = self.ws_model(text)

        elif isinstance(text, list):
            tokenized_text = list()
            for doc_text in text:
                doc = self.ws_model.tokenize(doc_text)
                tokenized_text.append(doc)
        else:
            raise ValueError(
                f"Expect text type (str or List[str]) but got {type(text)}"
            )
        return tokenized_text


# class Spacy_Chinese_Tokenizer(Tokenizer):
#     def __init__(self, model_path: str):
#         self.ws_model = spacy.load(model_path)

#     def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:

#         if isinstance(text, str):
#             doc = self.ws_model(text)
#             tokenized_text = [[token.text for token in doc]]

#         elif isinstance(text, list):
#             tokenized_text = list()
#             for doc_text in text:
#                 doc = self.ws_model(doc_text)
#                 res = [token.text for token in doc]
#                 tokenized_text.append(res)
#         else:
#             raise ValueError(
#                 f"Expect text type (str or List[str]) but got {type(text)}"
#             )
#         return tokenized_text


# class Jieba_Tokenizer(Tokenizer):
#     def __init__(self):
#         pass

#     def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:

#         if isinstance(text, str):
#             doc = jieba.cut(text)
#             tokenized_text = [[token for token in doc]]

#         elif isinstance(text, list):
#             tokenized_text = list()
#             for doc_text in text:
#                 doc = jieba.cut(doc_text)
#                 res = [token for token in doc]
#                 tokenized_text.append(res)
#         else:
#             raise ValueError(
#                 f"Expect text type (str or List[str]) but got {type(text)}"
#             )
#         return tokenized_text


# class Monpa_Tokenizer(Tokenizer):
#     def __init__(self):
#         pass

#     def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:

#         if isinstance(text, str):
#             doc = monpa.cut(text)
#             tokenized_text = [[token for token in doc]]

#         elif isinstance(text, list):
#             tokenized_text = list()
#             for doc_text in text:
#                 doc = monpa.cut(doc_text)
#                 res = [token for token in doc]
#                 tokenized_text.append(res)
#         else:
#             raise ValueError(
#                 f"Expect text type (str or List[str]) but got {type(text)}"
#             )
#         return tokenized_text


# class NLTK_Tokenizer(Tokenizer):
#     def __init__(self):
#         raise NotImplementedError

#     def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:
#         raise NotImplementedError
