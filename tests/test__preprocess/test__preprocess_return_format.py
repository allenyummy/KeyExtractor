# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test preprocess function

import logging

logger = logging.getLogger(__name__)


def test__preprocess_return_format(testcase3, extractor):
    tokenized_text = testcase3["tokenized_text"]
    content_text, n_gram_text = extractor._preprocess(tokenized_text)

    assert isinstance(content_text, list)
    for element in content_text:
        assert isinstance(element, tuple)
        assert len(element) == 2
        token_idx = element[0]
        token = element[1]
        assert isinstance(token_idx, int)
        assert isinstance(token, str)

    assert isinstance(n_gram_text, list)
    for each_n_gram in n_gram_text:
        assert isinstance(each_n_gram, list)

        for element in each_n_gram:
            assert isinstance(element, tuple)
            assert len(element) == 2
            token_idx = element[0]
            token = element[1]
            assert isinstance(token_idx, int)
            assert isinstance(token, str)
