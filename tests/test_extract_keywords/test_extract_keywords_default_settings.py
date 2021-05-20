# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test extract_keywords function

import logging
from KeyExtractor.utils import struct as st

logger = logging.getLogger(__name__)


def test_extract_keywords_default_settings(testcase3, extractor):
    tokenized_text = testcase3["tokenized_text"]
    rets = extractor.extract_keywords(tokenized_text)
    keyword_list = [ret.keyword for ret in rets]
    score_list = [ret.score for ret in rets]

    expected_keyword_list = [["中國"], ["中華民國"], ["銀行"], ["大型"], ["商業"]]
    expected_score_list = [0.7383, 0.7288, 0.706, 0.7039, 0.7027]

    assert len(keyword_list) == 5
    assert keyword_list == expected_keyword_list
    assert score_list == expected_score_list