import os
import re
import jieba
import jieba.posseg


stop_pos = ('w', 'm', 'q', 'y', 'x', 'u', 'd', 't', 'c', 'mq', 'f', 'z', 'o', 'e', 'p', 'uj', 'df', 'r',
                         'ad', 'ud', 'ns', 's', 'l', 'a', 'nr', 'nz')


class TextProcessing(object):
    def __init__(self, new_words=[], new_path='newwords.txt', stop_path='swords.txt'):
        self.stopwords = set()
        #jieba.enable_parallel()
        if len(new_words):
            [jieba.add_word(word) for word in new_words]
        if len(new_path):
            jieba.load_userdict(new_path)
        if len(stop_path):
            with open(stop_path, 'r', encoding='utf-8-sig') as f:
                self.stopwords = set(word.strip() for word in f.readlines())

    def loadDir(self, dir_name, seg=False, stop=False, type='all'):
        data_dic = {}
        n_processed = 0
        file_list = os.listdir(dir_name)
        pat = '(\d{5,})_(\d{4}-\d{2}-\d{2})-(\d{2})-\d{2}-\d{2}_(.*).txt'
        for file_name in file_list:
            n_processed += 1
            desc = re.findall(pat, file_name)
            if desc and len(desc[0]) == 4:
                area_no = desc[0][0]
                sta_date = desc[0][1]
                sta_clock = desc[0][2]
                record_id = desc[0][3]
            data_dic.setdefault(area_no, {})
            data_dic[area_no].setdefault(sta_date, {})
            data_dic[area_no][sta_date].setdefault(sta_clock, {})
            data_dic[area_no][sta_date][sta_clock] = (record_id, self.loadFile('\\'.join([dir_name, file_name]), seg, stop, type))
            if n_processed % 100 == 0:
                print('已处理%d篇文本' % n_processed)
        return data_dic

    def loadFile(self, file_name, seg=False, stop=False, type='all'):
        text = ''
        with open(file_name, 'r', encoding='utf-8-sig') as file:
            if type == 'all':
                text = ''.join([line.strip() for line in file.readlines()])
            elif type == 'custom':
                text = ''.join([line.strip() for line in file.readlines() if line.startswith('客户：')])
            elif type == 'server':
                text = ''.join([line.strip() for line in file.readlines() if line.startswith('坐席：')])
            return self.textprocess(text, seg, stop)

    def textprocess(self, text, seg=False, stop=False):
        # 如果不以二进制打开，则类型为str，需要编码，如以二进制打开，类型为bytes
        if seg:
            if type(text) == str:
                text = text.encode()
            if stop:
                word_list = [ob.word for ob in jieba.posseg.cut(text) if self.useless(ob.word, ob.flag) is False]
            else:
                word_list = [ob.word for ob in jieba.posseg.cut(text)]
            return word_list
        else:
            return text

    def useless(self, word, posseg):
        if word in self.stopwords \
                or posseg in stop_pos or len(word) < 2 \
                or word.endswith(('到', '了', '呢', '吗', '嘛')):
            return True
        return False

if __name__ == '__main__':
    tp = TextProcessing()
    file_list = os.listdir(r'D:\Work\003_语义语料库\20170717_语料标注\文本语料')
    word_num = 0
    for file in file_list:
        file_path = os.path.join(r'D:\Work\003_语义语料库\20170717_语料标注\文本语料', file)
        with open(file_path, 'r', encoding='utf-8-sig') as fin:
            text = ''.join(fin.readlines())
            word_list = tp.textprocess(text, seg=True, stop=True)
            word_num += len(word_list)
    print(word_num)




