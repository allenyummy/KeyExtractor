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
        """
        Init KeyExtractor.

        KeyExtractor can extract keywords from chinese documents.

        Args:
            `embedding_method_or_model`: a model name from huggingface model hub or a model local path.
        Type:
            `embedding_method_or_model`: string.
        Return:
            None
        """

        ## Tokenization could be a feature, but it could make operation slower.
        ## So I think it would be better to do tokenization outside KeyExtractor.
        # self.tokenizer = tk.TokenizerFactory(tokenization_method_or_model)

        """ Word Embedding Model """
        ## Use Flair framework to load transformers-based word embeddings model
        self.word_embed_model = TransformerWordEmbeddings(embedding_method_or_model)

        """ Document Embedding Model """
        ## Use Flair framework to load document pooling embeddings model
        self.doc_embed_model = DocumentPoolEmbeddings([self.word_embed_model])
        ## Use Flair framework to load transformers-based document embeddings model.
        ## TBMS, embeddings of [CLS] token is used as document embeddings.
        ## I test this before but got a lower cosine similarity score. This may be because I do not fine tune it.
        ## Another version of Flair, which has not yet released, support transformers-based pooling embeddings.
        # self.doc_embed_model = TransformerDocumentEmbeddings(embedding_method_or_model)

    def extract_keywords(
        self,
        text: List[str],
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
        n_gram: Optional[int] = 1,
        top_n: Optional[int] = 5,
    ) -> List[st.KeyStruct]:
        """
        Extract Keywords.

        Args:
            `text`        : An input text that is tokenized already.
            `stopwords`   : A custom stopwords that you think they must not be keywords.
                            It can take "STOPWORDS", ["STOPWORDS1", "STOPWORDS2", ..], "DIR/STOPWORDS.txt" or ["DIR/STOPWORDS.txt", ...] as input.
            `load_default`: Whether to load default stopwords. It can be seen from utils/stopwords/zh/*.
            `n_gram`      : N gram for content words as a keyword candidate.
            `top_n`       : Top N keywords extracted.
        Type:
            `text`        : list of string
            `stopwords`   : string or list of string (Default: None)
            `load_default`: bool (Default: True)
            `n_gram`      : integer (Default: 1)
            `top_n`       : integer (Default: 5)
        Return:
            List of Keywords.
            rtype: list of st.KeyStruct
        """

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
        """
        Preprocess tokenized text into content text and n_gram text.
        This function is for internal use.

        Args:
            `text`        : An input text that is tokenized already.
            `stopwords`   : A custom stopwords that you think they must not be keywords.
                            It can take "STOPWORDS", ["STOPWORDS1", "STOPWORDS2", ..], "DIR/STOPWORDS.txt" or ["DIR/STOPWORDS.txt", ...] as input.
            `load_default`: Whether to load default stopwords. It can be seen from utils/stopwords/zh/*.
            `n_gram`      : N gram for content words as a keyword candidate.
        Type:
            `text`        : list of string
            `stopwords`   : string or list of string (Default: None)
            `load_default`: bool (Default: True)
            `n_gram`      : integer (Default: 1)
        Return:
            Content text and N-gram text.
            rtype1: list of tuple of [integer, string]
            rtype2: lisr of list of tuple of [integer, string]
        """

        """ Check args """
        ## Input text must be tokenized by users.
        if not (isinstance(text, list) and all(isinstance(i, str) for i in text)):
            raise ValueError(
                "Text must be tokenized ! Expected text to be List[str], but got type(text). "
            )

        """ Get/Load Stopwords """
        ## Load custom stopwords if necessary.
        ## Load default stopwords if load_default is True.
        logger.debug("[Step 1] Loading stopwords ...")
        stopwords_list = self._load_stopwords(stopwords, load_default)
        logger.debug("[Step 1] Finish.")

        """ Get Content Words """
        ## Remove stopwords loaded above from the input tokenized text.
        ## Obtain content words.
        ## Also, add original position of each token in content words.
        logger.debug("[Step 2] Getting content words by removing stopwords ...")
        content_text = list()
        for i, token in enumerate(text):
            if token not in stopwords_list:
                content_text.append((i, token))
        logger.debug(content_text)
        logger.debug(f"Finally we got {len(content_text)} content words.")
        logger.debug("[Step 2] Finish.")

        """ Get N-Gram Combinations """
        ## Get N-gram combinaitons based on content words.
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
        """
        Load Custom Stopwords as well as Default Stopwords.
        This function is for internal use.

        Args:
            `stopwords`   : A custom stopwords that you think they must not be keywords.
                            It can take "STOPWORDS", ["STOPWORDS1", "STOPWORDS2", ..], "DIR/STOPWORDS.txt" or ["DIR/STOPWORDS.txt", ...] as input.
            `load_default`: Whether to load default stopwords. It can be seen from utils/stopwords/zh/*.
        Type:
            `stopwords`   : string or list of string (Default: None)
            `load_default`: bool (Default: True)
        Return:
            List of stopwords.
            rtype: list of string
        """

        """ Check args """
        if not stopwords and not load_default:
            return []

        """ Load Custom Stopwords """
        given = list()
        if stopwords:
            logger.debug("Loading stopwords given ...")
            given.extend(ut.load(stopwords))
            given = list(set(given))
            logger.debug(f"We got {len(given)} stopwords given.")
        else:
            logger.debug("No stopwords given.")

        """ Load Default Stopwords """
        ## Default stopwords are stored in utils/stopwords/zh/*.
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

        """ Integration """
        ## Use set operation to ensure there are no duplicates stopwords in the return list.
        ret = list()
        ret.extend(given)
        ret.extend(default)
        ret = list(set(ret))
        logger.debug(f"Finally we got {len(ret)} stopwords.")

        return ret

    def _evaluate(
        self, text: List[str], n_gram_text: List[List[Tuple[int, str]]]
    ) -> List[st.KeyStruct]:
        """
        Evaluate cosine similarity score between document and n-gram text based on their embeddings.
        This function is for internal use.

        Args:
            `text`        : An input text that is tokenized already.
            `n_gram_text` : N gram combinations.
        Type:
            `text`        : list of string
            `n_gram_text` : list of list of tuple of (integer, string)
        Return:
            List of Keywords.
            rtype: list of st.KeyStruct
        """

        """ Get Document Embeddings """
        logger.debug("[Step 1] Loading document embeddings ...")
        doc_embeddings = self._get_doc_embeddings(text)
        logger.debug(f"We got {doc_embeddings.size()} document embeddings.")
        logger.debug("[Step 1] Finish.")

        """ Get Word Embeddings"""
        ## Get word embeddings of each n gram combination
        logger.debug("[Step 2] Loading word embeddings of each n_gram combination ...")
        word_embeddings_list = self._get_word_embeddings(text, n_gram_text)
        logger.debug(
            f"We got {len(word_embeddings_list)} word embeddings list. (same size as n_gram)"
        )
        logger.debug(
            f"We got {word_embeddings_list[0].size()} word embeddings of each n gram combination."
        )
        logger.debug("[Step 2] Finish.")

        """ Calculate Score """
        ## Calculate cosine similarity score between document embeddings and word embeddings of each n-gram combination
        ## Also, format the results as st.KeyStruct.
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
        """
        Get Document Embeddings.
        This function is for internal use.

        Args:
            `text`: An input text that is tokenized already.
        Type:
            `text`: list of string
        Return:
            Document Embeddings.
            rtype: torch.tensor
            rsize: torcn.size([768]]
        """

        doc = Sentence(text)
        self.doc_embed_model.embed(doc)
        return doc.embedding  ## torch.size([768])

    def _get_word_embeddings(
        self, text: List[str], n_gram_text: List[List[Tuple[int, str]]]
    ) -> List[torch.tensor]:
        """
        Get Word Embeddings of each n-gram combination.
        Note that Transformers-based Word Embeddings is dynamic base on nearby words.
        This function is for internal use.

        Args:
            `text`        : An input text that is tokenized already.
            `n_gram_text` : N gram combinations.
        Type:
            `text`        : list of string
            `n_gram_text` : list of list of tuple of (integer, string)
        Return:
            List of Embeddings.
            rtype: list of torch.tensor
            rsize: list of torch.size([768])
        """

        """ Get EACH Token Embeddings """
        doc = Sentence(text)
        self.word_embed_model.embed(doc)

        """ Get EACH N-Gram Embeddings """
        ## stack token embeddings and mean them
        word_embedding_list = list()
        for each_n_gram in n_gram_text:
            embedding_list = [
                doc[token_idx].embedding for token_idx, token in each_n_gram
            ]
            stack_embedding = torch.stack(embedding_list, dim=0)  ## [N, 768]
            mean_embedding = torch.mean(stack_embedding, dim=0)  ## [768]
            word_embedding_list.append(mean_embedding)
        return word_embedding_list

    def _postprocess(
        self, input: List[st.KeyStruct], top_n: Optional[int] = 5
    ) -> List[st.KeyStruct]:
        """
        Postprocess results.
        This function is for internal use.

        Args:
            `input`: A result list after _preprocess and _evaluate function.
            `top_n`: Top N keywords extracted.
        Type:
            `input`: list of st.KeyStruct
            `top_n`: integer (Default: 5)
        Return:
            List of Keywords.
            rtype: list of st.KeyStruct
        """

        """ Sort """
        ## Sort first and then we could remove duplicates which has lower score.
        logger.debug("[Step 1] Sorting ...")
        input = sorted(input, key=lambda k: k.score, reverse=True)
        logger.debug("[Step 1] Finish.")

        """ Remove Duplicates """
        ## Use st.__eq__() to determine duplicates.
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

        """ Choose TOP N """
        top_n = min(len(input_wo_dup), top_n)
        logger.debug(f"[Step 3] Getting Top {top_n} Keywords ...")
        ret = input_wo_dup[:top_n]
        for k in ret:
            logger.debug(k)
        logger.debug("[Step 3] Finish.")

        return ret

    def doc_embeddings(self, text: List[str]) -> torch.tensor:
        return self._get_doc_embeddings(text)

    def word_embeddings_from_text(
        self, text: List[str], token_idx: int
    ) -> torch.tensor:
        doc = Sentence(text)
        self.word_embed_model.embed(doc)
        return doc[token_idx].embedding

    def word_embeddings(self, text: str) -> torch.tensor:
        word = Sentence(text)
        self.word_embed_model.embed(word)
        return word[0].embedding