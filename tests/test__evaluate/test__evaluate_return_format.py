# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test evaluate function

import logging
import torch
from KeyExtractor.utils import struct as st

logger = logging.getLogger(__name__)


def test__evaluate_return_format(testcase3, extractor):
    tokenized_text = testcase3["tokenized_text"]
    _, n_gram_text = extractor._preprocess(tokenized_text)
    results = extractor._evaluate(tokenized_text, n_gram_text)
    assert all(isinstance(res, st.KeyStruct) for res in results)
    for res in results:
        assert isinstance(res.id, int)
        assert isinstance(res.keyword, list)
        assert all(isinstance(i, str) for i in res.keyword)
        assert isinstance(res.score, float)
        assert torch.is_tensor(res.embeddings)
