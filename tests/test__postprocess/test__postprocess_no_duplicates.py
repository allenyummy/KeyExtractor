# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test postprocess function

import logging
import torch
from KeyExtractor.utils import struct as st

logger = logging.getLogger(__name__)


def test__postprocess_no_duplicates(testcase2, extractor):
    tokenized_text = testcase2["tokenized_text"]
    _, n_gram_text = extractor._preprocess(tokenized_text)
    results = extractor._evaluate(tokenized_text, n_gram_text)
    post_results = extractor._postprocess(results)
    assert len(post_results) == 5

    another_post_results = list()
    for res in post_results:
        if res not in another_post_results:  ## st.__eq__()
            another_post_results.append(res)
    assert len(post_results) == len(another_post_results)
    assert post_results == another_post_results
