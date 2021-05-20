# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test Embeddings

import logging
import torch
from KeyExtractor.utils import evaluation as ev

logger = logging.getLogger(__name__)


def test_word_embeddings_quality(testcase2, extractor):
    tokenized_text = testcase2["tokenized_text"]
    logger.debug(f"token 19: {tokenized_text[19]}")  ##漫畫
    logger.debug(f"token 22: {tokenized_text[22]}")  ##漫畫
    word_embeddings1 = extractor.word_embeddings_from_text(tokenized_text, 19)
    word_embeddings2 = extractor.word_embeddings_from_text(tokenized_text, 22)
    word_embeddings3 = extractor.word_embeddings("漫畫")
    cs_score12 = ev.cosineSimilarity(word_embeddings1, word_embeddings2)
    cs_score13 = ev.cosineSimilarity(word_embeddings1, word_embeddings3)
    cs_score23 = ev.cosineSimilarity(word_embeddings2, word_embeddings3)
    logger.debug(f"cs_score12: {cs_score12}")
    logger.debug(f"cs_score13: {cs_score13}")
    logger.debug(f"cs_score23: {cs_score23}")
    assert cs_score12 >= 0.80
    assert cs_score13 >= 0.80
    assert cs_score23 >= 0.80
