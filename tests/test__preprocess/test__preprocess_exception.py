# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test preprocess function

import logging
import pytest

logger = logging.getLogger(__name__)


def test__preprocess_exception1(extractor):
    with pytest.raises(ValueError) as excinfo:
        extractor._preprocess(123)
    assert (
        str(excinfo.value)
        == "Text must be tokenized ! Expected text to be List[str], but got <class 'int'>."
    )


def test__preprocess_exception2(extractor):
    with pytest.raises(ValueError) as excinfo:
        extractor._preprocess(["輸入", 123])
    assert (
        str(excinfo.value)
        == "Text must be tokenized ! Expected text to be List[str], but got <class 'list'>."
    )