# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test utility function

import logging
import pytest
from KeyExtractor.utils import utility as ut

logger = logging.getLogger(__name__)

test_data = [
    (
        "TEST-0",
        "我是字串",
        {
            "is_string": True,
            "is_list_of_string": False,
        },
    ),
    (
        "TEST-1",
        123,
        {
            "is_string": False,
            "is_list_of_string": False,
        },
    ),
    (
        "TEST-2",
        ["我是字串"],
        {
            "is_string": False,
            "is_list_of_string": True,
        },
    ),
    (
        "TEST-3",
        [123, "我是字串"],
        {
            "is_string": False,
            "is_list_of_string": False,
        },
    ),
    (
        "TEST-4",
        [["123"], ["我是字串"]],
        {
            "is_string": False,
            "is_list_of_string": False,
        },
    ),
    (
        "TEST-5",
        ["123", ["我是字串"]],
        {
            "is_string": False,
            "is_list_of_string": False,
        },
    ),
    (
        "TEST-6",
        [[123], ["我是字串"]],
        {
            "is_string": False,
            "is_list_of_string": False,
        },
    ),
]


@pytest.mark.parametrize(
    argnames=("name, text, expected_ans"),
    argvalues=test_data,
    ids=[f"{i[0]}" for i in test_data],
)
def test_is_string(name, text, expected_ans):
    assert ut.is_string(text) == expected_ans["is_string"]


@pytest.mark.parametrize(
    argnames=("name, text, expected_ans"),
    argvalues=test_data,
    ids=[f"{i[0]}" for i in test_data],
)
def test_is_list_of_string(name, text, expected_ans):
    assert ut.is_list_of_string(text) == expected_ans["is_list_of_string"]
