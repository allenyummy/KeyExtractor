# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test preprocess functions with setting of adding custom stopwords

import logging
from asserts import assertEquals

logger = logging.getLogger(__name__)


def test__preprocess_add_custom_stopwords(testcase2, extractor):
    tokenized_text = testcase2["tokenized_text"]
    content_text, n_gram_text = extractor._preprocess(
        tokenized_text,
        stopwords=["進擊", "巨人", "人類"],
    )
    expected_content_text = [
        (6, "日語"),
        (8, "進撃"),
        (9, "の"),
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
        (47, "衝突"),
        (51, "居住"),
        (54, "高"),
        (55, "牆"),
        (56, "包圍"),
        (58, "城市"),
        (60, "對抗"),
        (62, "食"),
    ]
    expected_n_gram_text = [
        [(6, "日語")],
        [(8, "進撃")],
        [(9, "の")],
        [(13, "日本")],
        [(14, "漫畫家")],
        [(15, "諫山")],
        [(16, "創")],
        [(17, "創作")],
        [(19, "漫畫")],
        [(20, "作品")],
        [(22, "漫畫")],
        [(24, "2009年")],
        [(25, "9月")],
        [(27, "2021年")],
        [(28, "4月間")],
        [(30, "講談社")],
        [(33, "冊")],
        [(34, "少年")],
        [(35, "Magazine")],
        [(38, "連載")],
        [(40, "故事")],
        [(41, "建立")],
        [(47, "衝突")],
        [(51, "居住")],
        [(54, "高")],
        [(55, "牆")],
        [(56, "包圍")],
        [(58, "城市")],
        [(60, "對抗")],
        [(62, "食")],
    ]

    assertEquals(content_text, expected_content_text, n_gram_text, expected_n_gram_text)
