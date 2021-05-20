# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test extract_keywords function

import logging
import torch
from KeyExtractor.utils import struct as st

logger = logging.getLogger(__name__)


def test_extract_keywords_return_format(testcase3, extractor):
    tokenized_text = testcase3["tokenized_text"]
    rets = extractor.extract_keywords(tokenized_text)
    assert all(isinstance(ret, st.KeyStruct) for ret in rets)
    for ret in rets:
        assert isinstance(ret.id, int)
        assert isinstance(ret.keyword, list)
        assert all(isinstance(i, str) for i in ret.keyword)
        assert isinstance(ret.score, float)
        assert torch.is_tensor(ret.embeddings)
