# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test Embeddings

import logging
import torch
from KeyExtractor.utils import evaluation as ev

logger = logging.getLogger(__name__)


def test_document_embeddings_size(testcase_all, extractor):
    for _, testcase in testcase_all.items():
        tokenized_text = testcase["tokenized_text"]
        doc_embeddings = extractor.doc_embeddings(tokenized_text)
        assert doc_embeddings.size() == torch.Size([768])


def test_word_embeddings_size(testcase_all, extractor):
    for _, testcase in testcase_all.items():
        token = testcase["tokenized_text"][0]
        word_embeddings = extractor.word_embeddings(token)
        assert word_embeddings.size() == torch.Size([768])


def test_word_embeddings_quality(testcase2, extractor):
    tokenized_text = testcase2["tokenized_text"]
    logger.warning(f"token 19: {tokenized_text[19]}")  ##漫畫
    logger.warning(f"token 22: {tokenized_text[22]}")  ##漫畫
    word_embeddings1 = extractor.word_embeddings_from_text(tokenized_text, 19)
    word_embeddings2 = extractor.word_embeddings_from_text(tokenized_text, 22)
    word_embeddings3 = extractor.word_embeddings("漫畫")
    cs_score12 = ev.cosineSimilarity(word_embeddings1, word_embeddings2)
    cs_score13 = ev.cosineSimilarity(word_embeddings1, word_embeddings3)
    cs_score23 = ev.cosineSimilarity(word_embeddings2, word_embeddings3)
    logger.warning(f"cs_score12: {cs_score12}")
    logger.warning(f"cs_score13: {cs_score13}")
    logger.warning(f"cs_score23: {cs_score23}")
    assert cs_score12 >= 0.80
    assert cs_score13 >= 0.80
    assert cs_score23 >= 0.80
