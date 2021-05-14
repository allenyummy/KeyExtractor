# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Data Structure

import logging
from typing import List, NamedTuple
import torch
from torch.autograd.grad_mode import F

logger = logging.getLogger(__name__)


class KeyStruct(NamedTuple):

    id: int
    keyword: List[str]
    score: float
    embeddings: torch.tensor

    def __eq__(self, other):
        return self.keyword == other.keyword

    def __repr__(self):
        return (
            f"\n"
            f"[ID]: {self.id}\n"
            f"[KEYWORD]: {self.keyword}\n"
            f"[SCORE]: {self.score}\n"
            f"[EMBEDDINGS]: {self.embeddings.size()}"
        )