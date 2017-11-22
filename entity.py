import sys
import os
import jieba
import jieba.posseg
import jieba.analyse
import re
from pyltp import SentenceSplitter, Postagger, NamedEntityRecognizer
import logging
from sklearn.externals import joblib

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




TIME_ENTITY = 0
LOCATION_ENTITY = 'Ns'
PERSON_ENTITY = 'Nh'
ORGANIZATION_ENTITY = 'Ni'
EVENT_ENTITY = 3


def entity(text, time_flag=True, location_flag=True, person_flag=True, event_flag=False):
    """
    获取文本中的时间、地点、人物及事件实体
    :param in_text: 输入文本
    :param time_flag: 是否返回时间实体
    :param location_flag: 是否返回地点实体
    :param person_flag: 是否返回人物实体
    :param event_flag: 是否返回事件实体
    :return:
    """
    try:
        if len(text) == 0:
            return ''
        res = ''
        time_list = []
        location_list = []
        person_list = []
        event_list = []
        jieba.load_userdict('newwords.txt')
        # 分句分析
        for sentence in SentenceSplitter.split(text):
            if len(sentence.strip()) == 0:
                continue
            # 分词
            ob_list = jieba.posseg.cut(sentence.strip())
            posseg_list = [ob.word + '/' + ob.flag for ob in ob_list]
            posseg_text = ' '.join(posseg_list)
            word_list = [ob[:ob.find('/')] for ob in posseg_list]
            ner_dic = {}
            # 获取时间实体
            if time_flag:
                time_list.extend(time_entity(sentence, posseg_list))
            # 地点或人物实体
            if location_flag or person_flag:
                ner_dic = person_location_entity(word_list)
            # 获取地点实体
            if location_flag:
                location_list.extend([' '.join(val) for key, val in ner_dic.items() if key == 'Ns'])
            # 获取人物实体
            if person_flag:
                person_list.extend([' '.join(val) for key, val in ner_dic.items() if key == 'Nh' or key == 'Ni'])
            # 获取事件实体
            if event_flag:
                event_list.append(event_entity(word_list))
        # 返回结果
        if time_flag:
            res += '时间实体：' + ' '.join(time_list) + '\n'
        if location_flag:
            res += '地点实体：' + ' '.join(location_list) + '\n'
        if person_flag:
            res += '人物实体：' + ' '.join(person_list) + '\n'
        if event_flag:
            res += '事件实体：' + ' '.join(event_list) + '\n'
        return res
    except Exception as e:
        print(str(e))
        return str(e)


def time_entity_d(in_text):
    """
    返回时间实体，内容包括：以及x年x月x日
    :param in_text: 输入文本
    :return:
    """
    # 输出含年、月、日的句子，看语料情况
    time_l = []
    res = set()
    # pat = '(.*[年、月、日].*)'
    pat = '.*?([一二三四五六七八九十]{1,}[点]{,1}[年,月,日,号,时,点,分,半,点]).*?'
    pat_m = '(^[一二三四五六七八九十]{1,4}).*?'
    # pat_n = '.*?([一二三四五六七八九十]{1,}[点,时]{1,})([一二三四五六七八九十]{0,4}).*?'
    # print(os.path.abspath(r'../../../Data/trans1'))

    # 去掉混淆项
    in_text = in_text.strip('一号键')
    # 匹配到点，但类似“七点三十”，则只能匹配到七点，三十会丢掉
    matcher = re.findall(pat, in_text)
    if len(matcher) > 0:
        # 连接列表，如果在原文中可以找到，则证明是一个完整的时间，如['十一月', '二十号', '九点']
        if in_text.find(''.join(matcher)) != -1:
            time_p = ''.join(matcher)
            # 判断整点后是否带分钟，如“九点二十”，正则只能匹配到九点
            matcher_part = re.findall(pat_m, in_text[in_text.find(time_p) + len(time_p):])
            if len(matcher_part) > 0:
                time_l.append(time_p + matcher_part[0])
            # 判断后面是否为特定文字，以排除如“三号楼”
            elif in_text[in_text.find(time_p) + len(time_p):].startswith('楼') is False:
                time_l.append(time_p)
        # 当原文中为月份，匹配会变成月
        elif in_text.find(''.join(matcher).replace('月', '月份')) != -1:
            time_p = ''.join(matcher).replace('月', '月份')
            # 判断整点后是否带分钟，如“九点二十”，正则只能匹配到九点
            matcher_part = re.findall(pat_m, in_text[in_text.find(time_p) + len(time_p):])
            if len(matcher_part) > 0:
                time_l.append(time_p + matcher_part[0])
            else:
                time_l.append(time_p)
        else:
            time_l = matcher
        for time in time_l:
            # 如果只有“时”，则去掉
            pat_c = '(.*)([年、时、点、分]).*'
            # 以大于四开头的日或号删掉
            pat_d = '.*?[四五六七八九十]{1}[一二三四五六七八九十][号,日]'
            # 大于三且多位开头的点去掉
            pat_t = '.*?[三四五六七八九十]{2,}点'
            end = re.findall(pat_c, time)
            # 去年只有时，只有分，以及点前太长（有可能是小数点）
            if len(end) == 1 and (end[0][1] == '时' or end[0][1] == '分' or
                                      (end[0][1] == '点' and len(end[0][0]) > 3)):
                continue
            # 年前面数字必须两个以上
            if len(end) == 1 and len(end[0][0]) < 2:
                continue
            day = re.findall(pat_d, time)
            if len(day) > 0:
                continue
            # 不能为三十几点。。。。
            if len(re.findall(pat_t, time)) > 0:
                continue
            res.add(time)
    return res


def time_entity_t(posseg_list):
    """
    返回时间实体，内容为词性为“t”的词语
    :param posseg_list:分词后词语/词性列表
    :return:
    """
    time_l = []
    res = set()
    non = {'下来', '过去', '过后', '目前', '正在', '平时', '白天', '九五', '双前', '七天', '将近', '北段',
           '下去', '东四', '过来', '北往', '过年', '中叶', '秋', '往前',
           '本来', '上去', '未来', '东汉', '青年', '上来', '上去', '最晚', '下次', '上次', '事后', '后来', '日期', '事后'}

    word_list = posseg_list
    time = ''
    for word in word_list:
        # 合并两个连续的时间
        if word.split('/')[1] == 't':
            time += word.split('/')[0]
        elif len(time) > 1 and time.startswith(('当', '刚', '目', '现')) is not True \
                and time.endswith(('年', '月', '日', '号')) is not True:
            time_l.append(time)
            time = ''
    res = set(time_l)
    # res = set([word.split('/')[0] for word in word_list if word.split('/')[1] == 't'])
    res = res.difference(non)
    return res


def time_entity(text, posseg_list):
    """
    提取时间实体
    :param text: 原始文本
    :param posseg_list: 分词后词语/词性列表
    :return:
    """
    entity_d = time_entity_d(text)
    entity_t = time_entity_t(posseg_list)
    return entity_d.union(entity_t)


def person_location_entity(word_list):
    """
    利用ltp获取人物及地点实体
    :param word_list: 分词后的词语列表
    :return:返回实体字典，key为：Nh,Ns,Ni，value为列表
    """
    logging.info('enter person_location_entity...')
    ner_dic = {}
    ner = ''
    if len(word_list) == 0:
        return ner_dic
    MODEL_PATH = r'/home/yanlei/IdeaProjects/hotpot/ltp_model'
    pos_model_path = os.path.join(MODEL_PATH, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    ner_model_path = os.path.join(MODEL_PATH, 'ner.model')  # 实体识别模型路径，模型名称为`ner.model`
    # 1.初始化实例
    logging.info('initilizing...')
    postagger = Postagger()
    recognizer = NamedEntityRecognizer()
    # 2.加载模型及字典
    logging.info('loading...')
    postagger.load(pos_model_path)
    recognizer.load(ner_model_path)
    # 3.词性标注,remove函数无返回值
    logging.info('postaging...')
    if word_list.count('\n') > 0:
        word_list.remove('\n')
    postags = postagger.postag(word_list)
    # 4.实体识别
    logging.info('recognizering...')
    netags = recognizer.recognize(word_list, postags)  # 命名实体识别
    # print ('\t'.join(netags))
    # 5.结果处理
    logging.info('result operating...')
    index = 0  # 词语索引
    for tag in netags:
        # 如果找不到-，直接越过
        if tag.find('-') == -1:
            continue
        # 以’-’分隔，前面为词语在实体中的位置（B 表示实体开始词，I表示实体中间词，E表示实体结束词，S表示单独成实体），
        # 后面为实体类型（人名（Nh）、地名（Ns）、机构名（Ni））
        position, type = tag.split('-')
        if position == 'S':
            ner_dic.setdefault(type, [])
            ner_dic[type].append(word_list[index])
        elif position == 'B':
            ner = word_list[index]
        elif position == 'I':
            ner += word_list[index]
        elif position == 'E':
            ner += word_list[index]
            ner_dic.setdefault(type, [])
            ner_dic[type].append(ner)
            ner = ''
        index += 1
    # 按规则过滤
    for type, value in ner_dic.items():
        ner_dic[type] = filter_entity(ner_dic.get(type), type)
    # print(ner_dic)
    logging.info('releasing...')
    postagger.release()  # 释放模型
    recognizer.release()  # 释放模型
    return ner_dic


def event_entity(word_list):
    vectorizer = joblib.load('model_save/vec.m')
    transformer = joblib.load('model_save/tfidf.m')
    clf = joblib.load('model_save/GaussianNB.m')
    tfidf = transformer.transform(vectorizer.transform(' '.join(word_list)))
    weight = tfidf.toarray()
    result = clf.predict(weight[0])
    # 将判定结果（数字）转为事件说明文字
    return result


def filter_entity(ner_list, type):
    """
    实体过滤，定制一些规则过滤实体识别结果
    :param ner: 实体列表
    :param type: 实体类型
    :return: 过滤后的实体列表
    """
    if type == TIME_ENTITY:
        return ner_list
    if type == LOCATION_ENTITY:
        fil = {'客户'}
        return list(set(ner_list).difference(fil))
    if type == PERSON_ENTITY or type == ORGANIZATION_ENTITY:
        fil = {'客户'}
        return list(set(ner_list).difference(fil))
    if type == EVENT_ENTITY:
        return ner_list
