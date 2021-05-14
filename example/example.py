# encoding=utf-8
import logging
import logging.config
from src.KeyExtractor.utils import tokenization as tk
from src.KeyExtractor.core import KeyExtractor

logging.config.fileConfig("logging.conf")


tokenizer = tk.TokenizerFactory(name="ckip-transformers-albert-tiny")
ke = KeyExtractor(embedding_method_or_model="ckiplab/bert-base-chinese")

text1 = "詐欺犯吳朱傳甫獲釋又和同夥林志成假冒檢警人員，向新營市黃姓婦人詐財一百八十萬元，事後黃婦驚覺上當報警處理，匯寄的帳戶被列警示帳戶，凍結資金往返；四日兩嫌再冒名要黃婦領五十萬現金交付，被埋伏的警員當場揪住。"
text2 = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
"""

text_list = [text1, text2]
# text_list = text2

tokenized_text_list = tokenizer.tokenize(text_list)
for tokenized_text in tokenized_text_list:
    print(tokenized_text)
    keywords = ke.extract_keywords(
        tokenized_text, stopwords=["中", "對象"], n_gram=2, top_n=5
    )
    for key in keywords:
        print(key)
    print("###########################")
