# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test preprocess functions with settings of n-gram

import logging
from asserts import assertEquals

logger = logging.getLogger(__name__)


def test_n_gram(testcase2, extractor):
    tokenized_text = testcase2["tokenized_text"]
    content_text, n_gram_text = extractor._preprocess(
        tokenized_text,
        n_gram=2,
    )
    expected_content_text = [
        (1, "進擊"),
        (3, "巨人"),
        (6, "日語"),
        (8, "進撃"),
        (9, "の"),
        (10, "巨人"),
        (13, "日本"),
        (14, "漫畫家"),
        (15, "諫山"),
        (16, "創"),
        (17, "創作"),
        (19, "漫畫"),
        (20, "作品"),
        (22, "漫畫"),
        (24, "2009年"),
        (25, "9月"),
        (27, "2021年"),
        (28, "4月間"),
        (30, "講談社"),
        (33, "冊"),
        (34, "少年"),
        (35, "Magazine"),
        (38, "連載"),
        (40, "故事"),
        (41, "建立"),
        (43, "人類"),
        (45, "巨人"),
        (47, "衝突"),
        (50, "人類"),
        (51, "居住"),
        (54, "高"),
        (55, "牆"),
        (56, "包圍"),
        (58, "城市"),
        (60, "對抗"),
        (62, "食"),
        (65, "巨人"),
    ]
    expected_n_gram_text = [
        [(1, "進擊"), (3, "巨人")],
        [(3, "巨人"), (6, "日語")],
        [(6, "日語"), (8, "進撃")],
        [(8, "進撃"), (9, "の")],
        [(9, "の"), (10, "巨人")],
        [(10, "巨人"), (13, "日本")],
        [(13, "日本"), (14, "漫畫家")],
        [(14, "漫畫家"), (15, "諫山")],
        [(15, "諫山"), (16, "創")],
        [(16, "創"), (17, "創作")],
        [(17, "創作"), (19, "漫畫")],
        [(19, "漫畫"), (20, "作品")],
        [(20, "作品"), (22, "漫畫")],
        [(22, "漫畫"), (24, "2009年")],
        [(24, "2009年"), (25, "9月")],
        [(25, "9月"), (27, "2021年")],
        [(27, "2021年"), (28, "4月間")],
        [(28, "4月間"), (30, "講談社")],
        [(30, "講談社"), (33, "冊")],
        [(33, "冊"), (34, "少年")],
        [(34, "少年"), (35, "Magazine")],
        [(35, "Magazine"), (38, "連載")],
        [(38, "連載"), (40, "故事")],
        [(40, "故事"), (41, "建立")],
        [(41, "建立"), (43, "人類")],
        [(43, "人類"), (45, "巨人")],
        [(45, "巨人"), (47, "衝突")],
        [(47, "衝突"), (50, "人類")],
        [(50, "人類"), (51, "居住")],
        [(51, "居住"), (54, "高")],
        [(54, "高"), (55, "牆")],
        [(55, "牆"), (56, "包圍")],
        [(56, "包圍"), (58, "城市")],
        [(58, "城市"), (60, "對抗")],
        [(60, "對抗"), (62, "食")],
        [(62, "食"), (65, "巨人")],
    ]

    assertEquals(content_text, expected_content_text, n_gram_text, expected_n_gram_text)
