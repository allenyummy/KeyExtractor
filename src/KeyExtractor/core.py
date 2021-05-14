# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: KeyExtractor

import logging
import os
from typing import List, Tuple, Union, Optional
import torch
from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    TransformerWordEmbeddings,
    # TransformerDocumentEmbeddings,
)
from .utils import (
    utility as ut,
    evaluation as ev,
    struct as st,
    # tokenization as tk,
)


logger = logging.getLogger(__name__)


class KeyExtractor:
    def __init__(self, embedding_method_or_model: str):

        # self.tokenizer = tk.TokenizerFactory(tokenization_method_or_model)
        self.word_embed_model = TransformerWordEmbeddings(embedding_method_or_model)
        self.doc_embed_model = DocumentPoolEmbeddings([self.word_embed_model])
        # self.doc_embed_model = TransformerDocumentEmbeddings(embedding_method_or_model)

    def extract_keywords(
        self,
        text: List[str],
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
        n_gram: Optional[int] = 1,
        top_n: Optional[int] = 5,
    ):

        """ Preprocess """
        logger.debug("======== [[ PREPROCESS ]] ========")
        _, n_gram_text = self._preprocess(text, stopwords, load_default, n_gram)

        """ Evaluate """
        logger.debug("======== [[ EVALUATE ]] ========")
        results = self._evaluate(text, n_gram_text)

        """ PostProcess """
        logger.debug("======== [[ POSTPROCESS ]] ========")
        ret = self._postprocess(results, top_n)

        return ret

    def _preprocess(
        self,
        text: List[str],
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
        n_gram: Optional[int] = 1,
    ) -> Union[List[Tuple[int, str]], List[List[Tuple[int, str]]]]:

        ## Check if tokenized_text is tokenized
        if not (isinstance(text, list) and all(isinstance(i, str) for i in text)):
            raise ValueError(
                "Text must be tokenized ! Expected text to be List[str], but got type(text). "
            )

        ## Get/Load stopwords
        logger.debug("[Step 1] Loading stopwords ...")
        stopwords_list = self._load_stopwords(stopwords, load_default)
        logger.debug("[Step 1] Finish.")

        ## Get content words by removing stopwords
        logger.debug("[Step 2] Getting content words by removing stopwords ...")
        content_text = list()
        for i, token in enumerate(text):
            if token not in stopwords_list:
                content_text.append((i, token))
        logger.debug(content_text)
        logger.debug(f"Finally we got {len(content_text)} content words.")
        logger.debug("[Step 2] Finish.")

        ## Get N-gram
        logger.debug(f"[Step 3] Getting {n_gram}-gram from content words.")
        n_gram_text = list()
        for i in range(len(content_text) - n_gram + 1):
            n_gram_text.append(content_text[i : i + n_gram])
        logger.debug(n_gram_text)
        logger.debug(f"Finally we got {len(n_gram_text)} {n_gram}-gram combinations.")
        logger.debug("[Step 3] Finish.")

        return content_text, n_gram_text

    def _load_stopwords(
        self,
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
    ) -> List[str]:

        if not stopwords and not load_default:
            return []

        given = list()
        if stopwords:
            logger.debug("Loading stopwords given ...")
            given.extend(ut.load(stopwords))
            given = list(set(given))
            logger.debug(f"We got {len(given)} stopwords given.")
        else:
            logger.debug("No stopwords given.")

        default = list()
        if load_default:
            from .utils.stopwords.zh import baidu, hit, scu, zhcn, zhtw

            logger.debug("Loading default stopwords (baidu, hit, scu, zhcn, zhtw) ...")
            for stopwords_set in [
                baidu.stopwords,
                hit.stopwords,
                scu.stopwords,
                zhcn.stopwords,
                zhtw.stopwords,
            ]:
                default.extend(list(stopwords_set))
            default = list(set(default))
            logger.debug(f"We got {len(default)} stopwords.")
        else:
            logger.debug("No default keywords.")

        ret = list()
        ret.extend(given)
        ret.extend(default)
        ret = list(set(ret))
        logger.debug(f"Finally we got {len(ret)} stopwords.")

        return ret

    def _evaluate(self, text: List[str], n_gram_text: List[List[Tuple[int, str]]]):

        ## Get doc embeddings
        logger.debug("[Step 1] Loading document embeddings ...")
        doc_embeddings = self._get_doc_embeddings(text)
        logger.debug(f"We got {doc_embeddings.size()} document embeddings.")
        logger.debug("[Step 1] Finish.")

        ## Get word embeddings of each n gram
        logger.debug("[Step 2] Loading word embeddings of each n_gram combination ...")
        word_embeddings_list = self._get_word_embeddings(text, n_gram_text)
        logger.debug(
            f"We got {len(word_embeddings_list)} word embeddings list. (same size as n_gram)"
        )
        logger.debug(
            f"We got {word_embeddings_list[0].size()} word embeddings of each n gram combination."
        )
        logger.debug("[Step 2] Finish.")

        ## Calculate score and format results
        logger.debug("[Step 3] Calculating score ...")
        results = list()
        for i, (each_n_gram, word_embeddings) in enumerate(
            zip(n_gram_text, word_embeddings_list)
        ):
            score = ev.cosineSimilarity(doc_embeddings, word_embeddings)
            results.append(
                st.KeyStruct(
                    id=i,
                    keyword=[token for (_, token) in each_n_gram],
                    score=round(float(score), 4),
                    embeddings=word_embeddings,
                )
            )
        logger.debug("[Step 3] Finish.")

        return results

    def _get_doc_embeddings(self, text: List[str]) -> torch.tensor:

        doc = Sentence(text)
        self.doc_embed_model.embed(doc)
        return doc.embedding  ## torch.size([768])

    def _get_word_embeddings(
        self, text: List[str], n_gram_text: List[List[Tuple[int, str]]]
    ) -> List[torch.tensor]:

        doc = Sentence(text)
        self.word_embed_model.embed(doc)

        word_embedding_list = list()
        for each_n_gram in n_gram_text:
            embedding_list = [
                doc[token_idx].embedding for token_idx, token in each_n_gram
            ]
            stack_embedding = torch.stack(embedding_list, dim=0)  ## [N, 768]
            mean_embedding = torch.mean(stack_embedding, dim=0)  ## [768]
            word_embedding_list.append(mean_embedding)
        return word_embedding_list

    def _postprocess(self, input: List[st.KeyStruct], top_n: Optional[int] = 5):

        ## Sort
        logger.debug("[Step 1] Sorting ...")
        input = sorted(input, key=lambda k: k.score, reverse=True)
        logger.debug("[Step 1] Finish.")

        ## Remove duplicates
        logger.debug("[Step 2] Removing duplicates ...")
        input_wo_dup = list()
        input_w_dup = list()
        for ipt in input:
            if ipt not in input_wo_dup:  ## st.__eq__()
                input_wo_dup.append(ipt)
            else:
                input_w_dup.append(ipt)
        logger.debug(f"Duplicaes: {input_w_dup}")
        logger.debug(f"Finally we removed {len(input_w_dup)} duplicates.")
        logger.debug("[Step 2] Finish.")

        ## Top N
        top_n = min(len(input_wo_dup), top_n)
        logger.debug(f"[Step 3] Getting Top {top_n} Keywords ...")
        ret = input_wo_dup[:top_n]
        for k in ret:
            logger.debug(k)
        logger.debug("[Step 3] Finish.")

        return ret