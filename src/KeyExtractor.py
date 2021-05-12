# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: KeyExtractor

import logging
import os
from typing import List, Tuple, Union, Optional
import torch
from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    TransformerWordEmbeddings,
    TransformerDocumentEmbeddings,
)
from src.utils import (
    tokenization as tk,
    utility as ut,
    evaluation as ev,
    datastruct as ds,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KeyExtractor:
    def __init__(
        self, tokenization_method_or_model: str, embedding_method_or_model: str
    ):

        self.tokenizer = tk.TokenizerFactory(tokenization_method_or_model)
        self.word_embed_model = TransformerWordEmbeddings(embedding_method_or_model)
        self.doc_embed_model = DocumentPoolEmbeddings([self.word_embed_model])
        # self.doc_embed_model = TransformerDocumentEmbeddings(embedding_method_or_model)

    def extract_keywords(
        self,
        text: Union[str, List[str]],
        is_split: Optional[bool] = False,
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
        n_gram: Optional[int] = 1,
        top_n: Optional[int] = 5,
    ):

        """ Preprocess """
        tokenized_text, content_text, n_gram_text = self._preprocess(
            text, is_split, stopwords, load_default, n_gram
        )

        """ Evaluate """
        results = self._evaluate(tokenized_text, n_gram_text)

        """ PostProcess """
        ret = self._postprocess(results, top_n)

        return ret

    def _preprocess(
        self,
        text: Union[str, List[str]],
        is_split: Optional[bool] = False,
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
        n_gram: Optional[int] = 1,
    ) -> Union[List[str], List[Tuple[int, str]], List[List[Tuple[int, str]]]]:

        """ Tokenization """
        tokenized_text = list()
        if isinstance(text, str):
            if is_split:
                raise ValueError("Wrong Input ! text -> str, is_split -> False")
            tokenized_text = self.tokenizer.tokenize(text)[0]
        if isinstance(text, list) and all(isinstance(i, str) for i in text):
            if not is_split:
                raise ValueError("Wrong Input ! text -> List[str], is_split -> True")
            tokenized_text = text
        logger.debug("==== TOKENIZATION ====")
        logger.debug(tokenized_text)

        """ Get/Load stopwords """
        stopwords_list = self._load_stopwords(stopwords, load_default)
        logger.debug("==== STOPWORDS ====")
        logger.debug(f"{len(stopwords_list)}")

        """ Get content words by removing stopwords """
        content_text = list()
        for i, token in enumerate(tokenized_text):
            if token not in stopwords_list:
                content_text.append((i, token))
        logger.debug("==== CONTENT TEXT ====")
        logger.debug(content_text)

        """ Get N-gram """
        n_gram_text = list()
        for i in range(len(content_text) - n_gram + 1):
            n_gram_text.append(content_text[i : i + n_gram])
        logger.debug("==== N GRAM ====")
        logger.debug(n_gram_text)

        return tokenized_text, content_text, n_gram_text

    def _load_stopwords(
        self,
        stopwords: Optional[Union[str, List[str]]] = None,
        load_default: Optional[bool] = True,
    ) -> List[str]:

        if not stopwords and not load_default:
            return []

        ret = list()
        if stopwords:
            ret.extend(ut.load(stopwords))

        if load_default:
            file_abs_dirname = os.path.join("src", "utils", "stopwords")
            file_path_list = [
                os.path.join(file_abs_dirname, file)
                for file in os.listdir(file_abs_dirname)
                if file.endswith(".txt")
            ]
            ret.extend(ut.load(file_path_list))

        return list(set(ret))

    def _evaluate(
        self, tokenized_text: List[str], n_gram_text: List[List[Tuple[int, str]]]
    ):

        """ Get doc embeddings """
        doc_embeddings = self._get_doc_embeddings(tokenized_text)
        logger.debug(f"DOC EMBEDDINGS: {doc_embeddings.size()}")

        """ Get word embeddings of each n gram """
        word_embeddings_list = self._get_word_embeddings(tokenized_text, n_gram_text)
        logger.debug(f"WORD EMBEDDINGS LIST: {len(word_embeddings_list)}")
        logger.debug(f"WORD EMBEDDINGS[0]: {word_embeddings_list[0].size()}")

        """ Calculate score and format results """
        results = list()
        for i, (each_n_gram, word_embeddings) in enumerate(
            zip(n_gram_text, word_embeddings_list)
        ):
            score = ev.cosineSimilarity(doc_embeddings, word_embeddings)
            results.append(
                ds.KeyStruct(
                    id=i,
                    keyword=[token for token in each_n_gram],
                    score=round(score, 4),
                    embeddings=word_embeddings,
                )
            )

        return results

    def _get_doc_embeddings(self, text: List[str]) -> torch.tensor:

        doc = Sentence(text)
        self.doc_embed_model.embed(doc)
        return doc.embedding  ## torch.size([768])

    def _get_word_embeddings(
        self, text: List[str], n_gram_text: List[List[Tuple[int, str]]]
    ) -> List[torch.tensor]:

        doc = Sentence(text)
        self.word_embed_model.embed(doc)

        word_embedding_list = list()
        for each_n_gram in n_gram_text:
            embedding_list = [
                doc[token_idx].embedding for token_idx, token in each_n_gram
            ]
            stack_embedding = torch.stack(embedding_list, dim=0)  ## [N, 768]
            mean_embedding = torch.mean(stack_embedding, dim=0)  ## [768]
            word_embedding_list.append(mean_embedding)
        return word_embedding_list

    def _postprocess(self, input: List[ds.KeyStruct], top_n: Optional[int] = 5):

        """ Sort """
        input = sorted(input, key=lambda k: k.score, reverse=True)

        """ Remove duplicates """
        input_wo_dup = list()
        for ipt in input:
            if ipt not in input_wo_dup:  ## st.__eq__()
                input_wo_dup.append(ipt)

        """ Top N """
        ret = input_wo_dup[:top_n]

        for k in ret:
            logger.debug(k)

        return ret


if __name__ == "__main__":

    text = "詐欺犯吳朱傳甫獲釋又和同夥林志成假冒檢警人員，向新營市黃姓婦人詐財一百八十萬元，事後黃婦驚覺上當報警處理，匯寄的帳戶被列警示帳戶，凍結資金往返；四日兩嫌再冒名要黃婦領五十萬現金交付，被埋伏的警員當場揪住。"
    text = "報導指出，這起詐騙事件行騙對象主要是馬來西亞華人，嫌犯於2011年7月至9月間撥打6857通馬來西亞電話，詐騙金額人民幣3615.7萬元。"
    text = "晶圓代工龍頭台積電日前核准資本預算28.87億美元，將在中國南京廠擴建28奈米成熟製程產能，卻引起中國通信業分析師項立剛強烈反對，\
            指此舉將打擊中國晶片產業。中國國台辦發言人馬曉光今日表示，相信中國是法治國家，會依法辦事，依法依規來處理。項立剛日前發表「強烈呼籲制止台積電南京廠擴產」一文，\
            稱台積電正在推行高端晶片控制、低端晶片傾銷壓制的策略，呼籲中國官方進行研究、審查，保護中國晶片製造企業，防止台積電的市場壟斷行為。\
            今日在國台辦例行記者會上，有媒體提問，台積電日前宣佈要在南京擴產28奈米全球製程產能帶，中芯國際日前也曾宣佈要在深圳合作擴產28奈米晶片製程，\
            中國有專家表示，台積電擴產或對中芯國際產生排擠競爭效應，還有學者呼籲中國應禁止台積電這個方案，請問對此有何評論？馬曉光則回應，「相信我們是法治國家，會依法辦事，依法依規來處理」。\
            他強調，關於台資台企來中國投資，有完備的法律、法規和管理機制，對於具體個案，有關部門會科學評估，依法依規處理。"
    text = "墾丁悠活麗緻渡假村的三至六區以集合住宅名義興建，目前仍屬停業狀態，僅一、二區共161間房正常營業。屏東縣政府觀光傳播處指出，業者正在辦理使用執照變更成為旅館用途，日前已經通過環評，但仍待墾丁國家公園管理處核准開發許可，最終還需回到縣府申請旅館登記證，屆時才能恢復營業。"
    text = "悠活渡假公司前董事長曾忠信涉及掏空公司資產、詐領資策會補助款及故買盜伐林木等3案件，不法獲益上億元，台南地檢署歷經1年多的偵查，29日將曾忠信等6人依違反證交法、詐欺、背信罪及故買贓物等罪起訴，並請法官從重量刑。"
    text = "法尼新創科技股份有限公司負責人田書旗、黃淑霞夫妻倆，涉嫌以投資「乙太幣拓礦礦機」、「認購飲料店、咖啡店、火鍋店及餐廳之股份」等名義，吸引民眾投資，估計違法吸金17億餘元，受害投資民眾恐多達2000人，桃園地檢署今天偵結，已依違反銀行法、非法經營期貨經理事業等罪嫌，將田書旗夫妻與員工共13人提起公訴。田書旗（40歲）、黃淑霞（42歲）從106年7月起，設計「乙太幣拓礦礦機」、「認購公司股份或特別股股份」、「認購飲料店、咖啡店、火鍋店及餐廳之股份」、「購買奇歐外匯期貨」等方案，由陳育勝等人擔任講師，四處召開說明會，以每年有18%到86%不等的高獲利，吸引民眾投資。透過說明會、網路與通訊軟體發布的訊息，田書旗夫妻倆迅速吸收許多民眾加入投資行列，檢方說，夫妻倆以「吸後金補前金」的方式，按約定將紅利撥付給投資民眾，其餘款項全挪為己用。桃園地檢署接獲檢舉後，指揮法務部調查局台北市調查處追查，偵結前已查證262位投資民眾，而他們的總投資金額已高達3億3500餘萬元，檢方說，比對扣案檔案、外部金流等資料，估計田書旗夫妻倆吸金總金額高達17億元，投資民眾被牽連受害的恐怕超過2000人，已因田書旗等人違反非銀行法「不得經營準收受存款業務達1億元以上」、「非法經營期貨經理事業」等罪嫌，將他們提起公訴。"

    ke = KeyExtractor(
        tokenization_method_or_model="ckip-transformers-albert-tiny",
        embedding_method_or_model="ckiplab/bert-base-chinese",
    )
    ke.extract_keywords(text, stopwords=["「", "」"], n_gram=2, top_n=5)
