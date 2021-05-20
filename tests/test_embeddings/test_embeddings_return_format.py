# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test Embeddings

import logging
import torch
from KeyExtractor.utils import evaluation as ev

logger = logging.getLogger(__name__)


def test_document_embeddings_format(testcase_all, extractor):
    for _, testcase in testcase_all.items():
        tokenized_text = testcase["tokenized_text"]
        doc_embeddings = extractor.doc_embeddings(tokenized_text)
        assert torch.is_tensor(doc_embeddings)
        assert doc_embeddings.size() == torch.Size([768])


def test_word_embeddings_format(testcase_all, extractor):
    for _, testcase in testcase_all.items():
        token = testcase["tokenized_text"][0]
        word_embeddings = extractor.word_embeddings(token)
        assert torch.is_tensor(word_embeddings)
        assert word_embeddings.size() == torch.Size([768])
