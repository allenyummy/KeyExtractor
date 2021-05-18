# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Load global variable

import logging
import pytest
from KeyExtractor.utils import tokenization as tk
from KeyExtractor.core import KeyExtractor

logger = logging.getLogger(__name__)


####################################################
##############    ADD TESTCASE HERE   ##############
####################################################

text0 = "詐欺犯吳朱傳甫獲釋又和同夥林志成假冒檢警人員，向新營市黃姓婦人詐財一百八十萬元，事後黃婦驚覺上當報警處理，匯寄的帳戶被列警示帳戶，凍結資金往返；四日兩嫌再冒名要黃婦領五十萬現金交付，被埋伏的警員當場揪住。"
text1 = """監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。"""
text2 = "《進擊的巨人》（日語：進撃の巨人）是日本漫畫家諫山創創作的漫畫作品。漫畫於2009年9月至2021年4月間在講談社《別冊少年Magazine》上連載。故事建立在人類與巨人的衝突上，人類居住在由高牆包圍的城市，對抗會食人的巨人。"

text_list = [text0, text1, text2]


####################################################
##################    TOKENIZER    #################
####################################################


@pytest.fixture(scope="session")
def tokenizer():
    logger.warning("Loading Tokenizer ...")
    return tk.TokenizerFactory(name="ckip-transformers-albert-tiny")


####################################################
##################    EXTRACTOR    #################
####################################################


@pytest.fixture(scope="session")
def extractor():
    logger.warning("Loading Extractor ...")
    return KeyExtractor(embedding_method_or_model="ckiplab/bert-base-chinese")


####################################################
######    TOKENIZED ALL TESTCASEs AT ONCE     ######
####################################################


@pytest.fixture(scope="session")
def testcase_all(tokenizer):
    tokenized_text_list = tokenizer.tokenize(text_list)
    return {
        i: {"text": text, "tokenized_text": tokenized_text}
        for i, (text, tokenized_text) in enumerate(zip(text_list, tokenized_text_list))
    }


@pytest.fixture(scope="session")
def testcase0(testcase_all):
    return testcase_all[0]


@pytest.fixture(scope="session")
def testcase1(testcase_all):
    return testcase_all[1]


@pytest.fixture(scope="session")
def testcase2(testcase_all):
    return testcase_all[2]
