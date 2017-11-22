import jieba
import jieba.posseg
import jieba.analyse


stop_pos = ('w', 'm', 'q', 'y', 'x', 'u', 'd', 't', 'c', 'mq', 'f', 'z', 'o', 'e', 'p', 'uj', 'df', 'r',
                         'ad', 'ud', 'ns', 's', 'l', 'a', 'nr', 'nz')


def load_stopwords():
    with open('swords.txt', 'r', encoding='utf-8-sig') as f:
        stopwords = set(word.strip() for word in f.readlines())
    return stopwords


def useless(self, word, posseg):
    if word in self.stopwords or posseg in stop_pos or len(word) < 2 or word.endswith(('到', '了', '呢', '吗', '嘛')):
        return True
    return False


def dl_posseg(text):
    jieba.load_userdict('newwords.txt')
    word_all = []
    res = ''
    word_meaning = []

    if len(text) > 0:
        # print(datetime.datetime.now())
        ob_list = jieba.posseg.lcut(text)
        for ob in ob_list:
            word_all.append(ob.word)
            res = res + ' ' + ob.word + '/' + ob.flag
            if useless(ob.word, ob.flag) is False:
                word_meaning.append(ob.word)
    return res, word_all, word_meaning


