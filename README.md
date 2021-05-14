[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/KeyExtractor/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/allenyummy/KeyExtractor/blob/main/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/KeyExtractor)](https://pypi.org/project/KeyExtractor/)
<!-- [![Build](https://img.shields.io/github/workflow/status/MaartenGr/keyBERT/Code%20Checks/master)](https://pypi.org/project/keybert/) -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OxpgwKqSzODtO3vS7Xe1nEmZMCAIMckX?usp=sharing) -->

# KeyExtractor
KeyExtractor 是一個十分簡單且好用的關鍵字詞抽取器，該模組透過 Transformer-based 模型，以零訓練的方式，抽取中文文件之關鍵字詞，無需標記資料與GPU資源即可操作。

KeyExtractor performs keyword extraction for chinese documents with state-of-the-art transformer models without training and labeled data.

<a name="toc"/></a>
## Table of Contents
* [About the Project](#about)
* [Getting Started](#gettingstarted)
    * [Installation](#installation)
    * [Example](#example)
    * [Basic Usage](#usage)
    * [Tokenizer](#tokenizer)
    * [Embeddings](#embeddings)
    * [Logger](#logger)

<a name="about"/></a>
## About
[Go Back](#toc)

在企業即將滿第一個年頭，感受到的文化與學術界差異甚大，比如說在中研院時，大家追求的是更好更完善的模型或演算法，為了能夠與經典論文上的模型或演算法較勁，大家無不使用相同、公開且乾淨的標記資料集，並實作自己的模型，跑出各種實驗數據，證明模型之間的優劣勝敗。有趣的是，彼此差距往往在不到 1% 之內，不難想像為何那麼難投稿上頂尖研討會了吧！

而在企業裡，每個案子有各自獨特的梳理邏輯，與其相應的資料集，而且多半是不完整且骯髒的資料，甚至，這些資料連標記都沒有，導致在導入 AI/NLP 技術時，路途困難重重，大概在清理資料階段、或是人工標記資料階段時就陣亡了，遑論使用最新穎的技術。

因為上述現象，企業在敏捷開發的狀態下，時常使用 Rule-Based 的方式解決，久而久之，幾乎很少導入新穎的技術，接著一個案子一個案子就這麼過去了。身為初入社會的我來說，不太習慣這樣的做法，但是若要有所突破，我仍然得面臨資料集一樣骯髒、一樣沒有標記、一樣不完整，那我該怎麼辦呢？

這是我接手的第二個案子：新聞閱讀，無標記資料，新聞資料還要自己爬蟲下來。我將此案子其中一部份拉出來作成公開套件：新聞關鍵字詞抽取器，給定一篇中文新聞，經由模組，生出若干關鍵字。

2018年末，預訓練與微調機制擄獲人心，便一路盛行至今。為了撇開對標記資料的依賴，我撇開微調機制，僅採用預訓練機制。預訓練模型使用中研院ckip實驗室的模型，並將其當作詞向量抽取器，接著使用 Cosine Similarity 去一一比較文本與各個候選字詞的向量夾角，作為判斷是否為關鍵字詞的依據。

上述的想法源自於 [KeyBERT](https://github.com/MaartenGr/KeyBERT)，但是因為它沒有支援中文和中文斷詞，才讓我想自己動手做一個套件出來。

<a name="gettingstarted"></a>
## Getting Started
[Go Back](#toc)

<a name="installation"></a>
### Installation

Installation can be done using [pypi KeyExtractor](https://pypi.org/project/KeyExtractor/).
```
$ pip install KeyExtractor
```

<a name="example"/></a>
### Example
```
$ PYTHONPATH=./::./src python example/example.py
```

<a name="usage"/></a>
### Usage
[Go Back](#toc)

* Single Document

Input text should be tokenized as properly as possible before extracting keywords from it.
```
from KeyExtractor.utils import tokenization as tk

tokenizer = tk.TokenizerFactory(name="ckip-transformers-albert-tiny")
text = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
"""
tokenized_text_list = tokenizer.tokenize(text_list)[0]  ## Return nested list of tokenized results
```

Extract Keywords from document tokenized before.

```
from KeyExtractor.core import KeyExtractor
ke = KeyExtractor(embedding_method_or_model="ckiplab/bert-base-chinese")
keywords = ke.extract_keywords(tokenized_text, n_gram=1, top_n=5)
```

Return keywords as a list of struct.KeyStruct.

```
>>> [print(key) for key in keywords]
[ID]: 29
[KEYWORD]: ['學習']
[SCORE]: 0.7103
[EMBEDDINGS]: torch.Size([768])

[ID]: 33
[KEYWORD]: ['對象']
[SCORE]: 0.6965
[EMBEDDINGS]: torch.Size([768])

[ID]: 31
[KEYWORD]: ['範例']
[SCORE]: 0.6923
[EMBEDDINGS]: torch.Size([768])

[ID]: 28
[KEYWORD]: ['監督']
[SCORE]: 0.6849
[EMBEDDINGS]: torch.Size([768])

[ID]: 46
[KEYWORD]: ['分析']
[SCORE]: 0.6834
[EMBEDDINGS]: torch.Size([768])
```

N-gram could be 2 or 3 or more.
```
keywords = ke.extract_keywords(tokenized_text, n_gram=2, top_n=5)

>>> [print(key) for key in keywords]
[ID]: 30
[KEYWORD]: ['中', '範例']
[SCORE]: 0.8059
[EMBEDDINGS]: torch.Size([768])

[ID]: 31
[KEYWORD]: ['範例', '輸入']
[SCORE]: 0.8006
[EMBEDDINGS]: torch.Size([768])

[ID]: 28
[KEYWORD]: ['監督', '學習']
[SCORE]: 0.7888
[EMBEDDINGS]: torch.Size([768])

[ID]: 32
[KEYWORD]: ['輸入', '對象']
[SCORE]: 0.7825
[EMBEDDINGS]: torch.Size([768])

[ID]: 29
[KEYWORD]: ['學習', '中']
[SCORE]: 0.7816
[EMBEDDINGS]: torch.Size([768])
```

It could add custom stopwords that you think they must not be keyword candidates. They would be removed in the preprocessing stage.

```
keywords = ke.extract_keywords(tokenized_text, stopwords=["中", "對象"], n_gram=2, top_n=5)

>>> [print(key) for key in keywords]
[ID]: 28
[KEYWORD]: ['學習', '範例']
[SCORE]: 0.8039
[EMBEDDINGS]: torch.Size([768])

[ID]: 29
[KEYWORD]: ['範例', '輸入']
[SCORE]: 0.8006
[EMBEDDINGS]: torch.Size([768])

[ID]: 27
[KEYWORD]: ['監督', '學習']
[SCORE]: 0.7888
[EMBEDDINGS]: torch.Size([768])

[ID]: 24
[KEYWORD]: ['推斷出', '函數']
[SCORE]: 0.7738
[EMBEDDINGS]: torch.Size([768])

[ID]: 18
[KEYWORD]: ['訓練', '數據']
[SCORE]: 0.7677
[EMBEDDINGS]: torch.Size([768])
```

Also, we have default zh-cn/zh-tw stopwords (`load_default` is set to `True`). You can check them in the `utils/stopwords/zh/`. If you don't want them as stopwords, just simply set `load_default` to `False`.

* Multiple Documents

You can feel safe to send multiple documents into tokenizer. They can process multiple documents more efficiently than processing single document at one time.

```
from KeyExtractor.utils import tokenization as tk

tokenizer = tk.TokenizerFactory(name="ckip-transformers-albert-tiny")
text = """
    監督學習是機器學習任務，它學習基於範例輸入-範例輸出組合，將輸入映射到輸出的函數。
    [1]它從標記的訓練數據（由一組訓練範例組成）中推斷出函數。
    [2]在監督學習中，每個範例都是一對，由輸入對象（通常是矢量）和期望的輸出值（也稱為監督信號）組成。
    監督學習演算法分析訓練數據並產生一個推斷函數，該函數可用於映射新範例。
    最佳方案將使演算法能夠正確確定未見實例的類標籤。
    這就要求學習算法以“合理”的方式將訓練數據推廣到看不見的情況（見歸納偏差）。
"""
text2 = "詐欺犯吳朱傳甫獲釋又和同夥林志成假冒檢警人員，向新營市黃姓婦人詐財一百八十萬元，事後黃婦驚覺上當報警處理，匯寄的帳戶被列警示帳戶，凍結資金往返；四日兩嫌再冒名要黃婦領五十萬現金交付，被埋伏的警員當場揪住。"

text_list = [text1, text2]
tokenized_text_list = tokenizer.tokenize(text_list)  ## Return nested list

for tokenized_text in tokenized_text_list:
    keywords = ke.extract_keywords(tokenized_text, n_gram=2, top_n=5)
    for key in keywords:
        print(key)
```

[Go Back](#toc)

<a name="tokenizer"/></a>
### Tokenizer

I use as ckip-transformers as my backbone tokenizer. Please check details from this [repo](https://github.com/ckiplab/ckip-transformers).

<a name="embeddings"/></a>
### Embeddings

I use flair framework to get word and document embeddings from `ckiplab/bert-base-chinese`. Their model could be seen in huggingface model hub [here](https://huggingface.co/ckiplab/bert-base-chinese). Feel free to get your own pretrained model or another one.

<a name="logger"/></a>
### Logger

If you want to check details of operation, you can set up logger.level as logging.DEBUG. 

**Github Repos**:  
* https://github.com/MaartenGr/KeyBERT
* https://github.com/flairNLP/flair
* https://github.com/ckiplab/ckip-transformers

[Go Back](#toc)