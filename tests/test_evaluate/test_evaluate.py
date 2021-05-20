# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test evaluate function

import logging
from KeyExtractor.utils import struct as st

logger = logging.getLogger(__name__)


def test_return(testcase3, extractor):
    tokenized_text = testcase3["tokenized_text"]
    _, n_gram_text = extractor._preprocess(tokenized_text)
    results = extractor._evaluate(tokenized_text, n_gram_text)
    assert all(isinstance(res, st.KeyStruct) for res in results)


def test_calculate_score(testcase3, extractor):
    tokenized_text = testcase3["tokenized_text"]
    _, n_gram_text = extractor._preprocess(tokenized_text)
    results = extractor._evaluate(tokenized_text, n_gram_text)
    score_list = [res.score for res in results]
    expected_score_list = [
        0.7383,
        0.6668,
        0.7027,
        0.7007,
        0.7288,
        0.7039,
        0.6966,
        0.706,
    ]
    assert score_list == expected_score_list