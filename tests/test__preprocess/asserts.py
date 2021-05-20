# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: assert functions for testing preprocess function

import logging

logger = logging.getLogger(__name__)


def assertEquals(
    content_text, expected_content_text, n_gram_text, expected_n_gram_text
):
    assert len(content_text) == len(expected_content_text)
    assert content_text == expected_content_text

    assert len(n_gram_text) == len(expected_n_gram_text)
    assert len(n_gram_text[0]) == len(expected_n_gram_text[0])
    assert n_gram_text == expected_n_gram_text
