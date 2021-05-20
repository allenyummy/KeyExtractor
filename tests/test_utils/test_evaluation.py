# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test evaluation function

import logging
import torch
from KeyExtractor.utils import evaluation as ev

logger = logging.getLogger(__name__)


def test_cosineSimilarity1():
    e1 = torch.tensor([1], dtype=torch.float32)
    e2 = torch.tensor([1], dtype=torch.float32)

    score = ev.cosineSimilarity(e1, e2)
    expected_score = 1.0
    assert score == expected_score


def test_cosineSimilarity2():
    e1 = torch.tensor([0.1, 0.5], dtype=torch.float32)
    e2 = torch.tensor([-0.1, -0.5], dtype=torch.float32)

    score = ev.cosineSimilarity(e1, e2)
    expected_score = -1.0
    assert score == expected_score


def test_cosineSimilarity3():
    e1 = torch.tensor([0.3, 0.4], dtype=torch.float32)
    e2 = torch.tensor([0, 0.5], dtype=torch.float32)

    score = ev.cosineSimilarity(e1, e2)
    expected_score = 0.80
    assert round(float(score), 2) == expected_score