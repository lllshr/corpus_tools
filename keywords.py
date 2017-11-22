import os
import re
import hashlib
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from assist import textprocessing
import newwords


def get_corpus(path, seg=True, stop=True):
    """
    获取语料库
    path: 语料文件夹路径
    """
    tp = textprocessing.TextProcessing()
    corpus = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            corpus.append(' '.join(tp.textprocess(''.join(f.readlines()), seg=seg, stop=stop)))
    # print('共有%d篇语料' % len(corpus))
    return corpus


def train_tfidf(corpus, min_df=0.05, max_df=0.95):
    """
    训练tf-idf模型
    :param corpus: 语料库
    :param min_df: 最低频率
    :param max_df: 最高频率
    """
    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    print('保存词袋模型...')
    # 保存
    joblib.dump(vectorizer, 'model_save/vec.m')
    joblib.dump(transformer, 'model_save/tfidf.m') 
    

def dl_keywords(word_meaning, topK=10, vec_model='model_save/vec.m',
                tfidf_model='model_save/tfidf.m', out_weight=True):
    if len(word_meaning) == 0:
        return ''
    # 加载模型
    vectorizer = joblib.load(vec_model)
    transformer = joblib.load(tfidf_model)
    # 计算tfidf值
    tfidf = transformer.transform(vectorizer.transform([' '.join(word_meaning)]))
    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    # 获取本文本在词典里存在的词
    indices = tfidf.indices
    dic = {}
    for index in indices:
        dic[words[index]] = weight[0][index]
    # 按权重由高到低排序
    keywords = sorted(dic.items(), key=lambda dic:dic[1], reverse=True)
    # 关键词数目
    num = min(len(keywords), topK)
    if num == 0:
        return ''
    head = math.ceil(num/2)
    tail = math.floor(num/2)
    if out_weight:
        if num == 1:
            return '\n'.join([keyword+str(weight) for keyword, weight in keywords[:head]])
        else:
            return '\n'.join([keyword+str(weight) for keyword, weight in keywords[:head]]) + '\n' + \
                   '\n'.join([keyword+str(weight) for keyword, weight in keywords[-tail:]])
    else:
        if num == 1:
            return ' '.join([keyword for keyword, weight in keywords[:head]])
        else:
            return ' '.join([keyword for keyword, weight in keywords[:head]]) + ' ' + \
                   ' '.join([keyword for keyword, weight in keywords[-tail:]])


if __name__ == "__main__":
    #src = r'D:\Work\003_语义语料库\Data\original'
    src=r'D:\Work\009_信通产业集团\Data\textdata\zongbuyuyin_text\2016-01-01'
    """
    n = 7
    min_count = 128
    min_proba = {2:5, 3:25, 4:125, 5:125, 6:125, 7:125}
    new_words = newwords.get_newwords(src, n, min_count, min_proba)
    """
    corpus = get_corpus(src)
    train_tfidf(corpus)
    text = ''
    with open(r'D:\Work\\003_语义语料库\Data\original\\NB,0087028f829a00ae,540573,2016-11-23.txt', 'r', encoding='utf-8-sig') as fin:
        text = fin.read()
    key_words = dl_keywords(text)
    print(key_words)
