# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test struct

import logging
import torch
from KeyExtractor.utils import struct as st

logger = logging.getLogger(__name__)


def test___eq__1():
    ks1 = st.KeyStruct(0, ["輸入1"], 0.81, torch.rand(1))
    ks2 = st.KeyStruct(1, ["輸入2"], 0.83, torch.rand(1))
    assert ks1 != ks2


def test___eq__2():
    ks1 = st.KeyStruct(0, ["輸入1"], 0.81, torch.rand(1))
    ks2 = st.KeyStruct(0, ["輸入1"], 0.81, torch.rand(1))
    assert ks1 == ks2


def test___eq__3():
    ks1 = st.KeyStruct(0, ["輸入1"], 0.81, torch.rand(1))
    ks2 = st.KeyStruct(1, ["輸入1"], 0.83, torch.rand(1))
    assert ks1 == ks2


def test___eq__4():
    ks1 = st.KeyStruct(0, ["輸入1"], 0.81, torch.rand(1))
    ks2 = st.KeyStruct(1, ["輸入1", "輸入2"], 0.83, torch.rand(1))
    assert ks1 != ks2


def test___repr__():
    ks = st.KeyStruct(0, ["輸入1"], 0.81, torch.rand(1))
    assert str(ks) == (
        f"\n"
        f"[ID]: 0\n"
        f"[KEYWORD]: ['輸入1']\n"
        f"[SCORE]: 0.81\n"
        f"[EMBEDDINGS]: torch.Size([1])"
    )
