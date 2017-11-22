import os
import re
from tqdm import tqdm
import hashlib
import numpy as np
from collections import defaultdict
from assist import textprocessing

md5 = lambda s: hashlib.md5(s).hexdigest()


def texts(src=r'D:\Work\009_信通产业集团\Data\textdata\zongbuyuyin_text\2016-01-01'):
    texts_set = set()
    # for a in tqdm(db.find(no_cursor_timeout=True).limit(3000000)):
    #src = r'D:\开源代码&工具\源码\python\201706_newwords\test'
    file_list = os.listdir(src)
    for index in tqdm(range(len(file_list))):
        file_path = os.path.join(src, file_list[index])
        with open(file_path, 'r', encoding='utf-8-sig') as fin:
            text = ''.join([line.strip() for line in fin.readlines()])
        if md5(text.encode('utf-8')) in texts_set:
            continue
        else:
            texts_set.add(md5(text.encode('utf-8')))
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', text):
                if t:
                    yield t
    print ('最终计算了%s篇文章'%len(texts_set))


def select_by_minCount(src, n=7, min_count=128):
    """
    筛选出现次数大于min_count的词语
    """
    ngrams = defaultdict(int)

    for t in texts(src):
        for i in range(len(t)):
            for j in range(1, n+1):
                if i+j <= len(t):
                    ngrams[t[i:i+j]] += 1

    ngrams = {i:j for i,j in ngrams.items() if j >= min_count}
    return ngrams



# 内部凝固度计算

def is_keep(s, ngrams, total, min_proba):
    if len(s) >= 2:
        score = min([total*ngrams[s]/(ngrams[s[:i+1]]*ngrams[s[i+1:]]) for i in range(len(s)-1)])
        if score > min_proba[len(s)]:
            return True
    else:
        return False

"""
定义切分函数，并进行切分统计
"""
def cut(s, ngrams_, n):
    """
    :param s: 文本
    :param n: 最长词语长度
    """
    r = np.array([0]*(len(s)-1))
    for i in range(len(s)-1):
        for j in range(2, n+1):
            if s[i:i+j] in ngrams_:
                r[i:i+j-1] += 1
    w = [s[0]]
    for i in range(1, len(s)):
        if r[i-1] > 0:
            w[-1] += s[i]
        else:
            w.append(s[i])
    return w


def is_real(s, n, ngrams_):
    if len(s) >= 3:
        for i in range(3, n+1):
            for j in range(len(s)-i+1):
                if s[j:j+i] not in ngrams_:
                    return False
        return True
    else:
        return True


def get_words(src, n=7, min_count=128, min_proba={2:5, 3:25, 4:125, 5:125, 6:125, 7:125}):
    ngrams = select_by_minCount(src=src, n=n, min_count=min_count)

    total = 1.*sum([j for i,j in ngrams.items() if len(i) == 1])
    # 不同长度，分别考虑阈值
    ngrams_ = set(i for i,j in ngrams.items() if is_keep(i, ngrams, total, min_proba))
    
    words = defaultdict(int)
    for t in texts(src=src):
        for i in cut(t, ngrams_, n):
           words[i] += 1

    words = {i:j for i,j in words.items() if j >= min_count}
    w = {i:j for i,j in words.items() if is_real(i, n, ngrams_) and len(i) > 1}
    return w


def get_corpus(path, seg=True, stop=True):
    """
    获取语料库
    path: 语料文件夹路径
    """
    tp = textprocessing.TextProcessing()
    corpus = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8-sig') as f:
            corpus.append(' '.join(tp.textprocess(''.join(f.readlines()), seg=seg, stop=stop)))
    # print('共有%d篇语料' % len(corpus))
    return corpus


def get_newwords(src, n=7, min_count=128, min_proba={2:5, 3:25, 4:125, 5:125, 6:125, 7:125}):
    """
    发现新词
    """
    w = get_words(src=src, n=n, min_count=128, min_proba=min_proba)
    # compare,利用现有词典进行分词，比较上面得到的词语与已分得的词
    new_words = set()
    corpus = get_corpus(src, seg=True, stop=False)
    old_words = set()
    old_list = []
    for wordl in corpus:
        old_list.extend(wordl.split(' '))
    old_words = set(old_list)

    print(old_words)
    with open('new.txt', 'w', encoding='utf-8-sig') as fout:
        for word in w.keys():
            if word not in old_words:
                fout.write(word)
                fout.write('\n')
                new_words.add(word)
    return new_words

if __name__ == "__main__":
    src = r'D:\Work\003_语义语料库\Data\original'
    min_count = 128
    min_proba={2:5, 3:25, 4:125, 5:125, 6:125, 7:125}
    n = 7
    new_words = get_newwords(src, n, min_count, min_proba)
    print(new_words)


