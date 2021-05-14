# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Utility Function

import logging
from typing import List, Union

logger = logging.getLogger(__name__)


def is_string(input):
    if isinstance(input, str):
        return True
    return False


def is_list_of_string(input):
    if isinstance(input, list) and all(isinstance(i, str) for i in input):
        return True
    return False


def load(input: Union[str, List[str]]) -> List[str]:

    if not input:
        return []

    if not is_string(input) and not is_list_of_string(input):
        raise ValueError(f"Expected string or list of string, but got {type(input)}")

    def load_file(file_path: str):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            ret = [w.rstrip() for w in f.readlines()]
            f.close()
        return ret

    ret = list()
    if is_string(input):
        if input.endswith(".txt"):
            ret.extend(load_file(input))
        else:
            ret.append(input)

    elif is_list_of_string(input):
        for ipt in input:
            if ipt.endswith(".txt"):
                ret.extend(load_file(ipt))
            else:
                ret.append(ipt)

    return list(set(ret))
