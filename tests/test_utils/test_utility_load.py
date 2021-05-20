# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test utility function

import logging
from KeyExtractor.utils import utility as ut

logger = logging.getLogger(__name__)


def test_load_string():
    ans = ut.load("輸入")
    expected_ans = ["輸入"]
    assert set(ans) == set(expected_ans)


def test_load_string_file():
    ans = ut.load("tests/test_utils/test_file.txt")
    expected_ans = ["輸入1", "輸入2", "輸入3"]
    assert set(ans) == set(expected_ans)


def test_load_list_of_string():
    ans = ut.load(["輸入1", "輸入2"])
    expected_ans = ["輸入1", "輸入2"]
    assert set(ans) == set(expected_ans)


def test_load_list_of_string_that_contains_duplicates():
    ans = ut.load(["輸入1", "輸入2", "輸入1"])
    expected_ans = ["輸入1", "輸入2"]
    assert set(ans) == set(expected_ans)


def test_load_list_of_string_that_mixs_words_and_filenames():
    ans = ut.load(["輸入0", "tests/test_utils/test_file.txt"])
    expected_ans = ["輸入0", "輸入1", "輸入2", "輸入3"]
    assert set(ans) == set(expected_ans)