from posseg import dl_posseg
from keywords import dl_keywords


def execute(text, topK=10):
    # 分词
    posseg_res, word_all, word_meaning = dl_posseg(text=text)
    # 关键词提取
    keyword_res = dl_keywords(word_meaning)
    # 信息抽取
    # 情感倾向分析