# 特征值13-16

from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
from mediawiki import MediaWiki
wikipedia = MediaWiki()

def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def read_corpus():
    url = 'E:/实验相关代码/m3代码/m3_code/data/concept_pairs.xlsx'
    df = pd.read_excel(url)
    g = df['A'].tolist()
    h = df['B'].tolist()
    con_list = list(dict.fromkeys(g+h))
    corpus =[]
    for i in con_list:
        try:
            s = wikipedia.page(i).content
        except:
            print('概念' + i + '没找到对应的维基百科页面')
        sentence = re.findall(r'(?:[^\W\d_]+\d|\d+[^\W\d_])[^\W_]*|[ ^\W\d_]+', s)
        istToStr = ' '.join([str(elem) for elem in sentence])
        corpus.append(istToStr)
    return corpus, g, h, con_list

if __name__ == "__main__":
    # 存储读取语料 一行语料为一个文档 文档当中词语词之间用空格分隔
    # corpus = []
    # for line in open('D1_wiki.txt', 'r', encoding = "utf8").readlines():
    # # print line
    # corpus.append(line.strip())
    # # print(corpus)

    corpus, g, h, con_list = read_corpus()
    topic_num = 100

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # print(vectorizer)
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()
    # print(len(weight))
    # print(weight[:35, :35])
    # LDA算法
    # print('LDA:')
    import numpy as np
    import lda
    # import lda.datasets

    np.set_printoptions(suppress=True)

    model = lda.LDA(n_topics=topic_num, n_iter=500, random_state=1)
    model.fit(np.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    # print("------")
    # print(model)
    # print("------")

    # print(topic_word.shape)

    word = vectorizer.get_feature_names_out()

    label = []
    # 主题-词语（Topic-Word）分布
    for n in range(topic_num):

        # topic_most_pr = topic_word[n]
        topic_most_pr = topic_word[n].argmax()
        label.append(topic_most_pr)
        print("topic: {} word: {}".format(n, word[topic_most_pr]))

    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    # print(doc_topic)

    # 选择对应词的大小值
    # for i in doc_topic:
    # print(list(i), end=",")
    # x 选择主题最大概率的值
    import math
    LDAVector_dict = {}
    ha_dict = {}
    count = -1
    for i in doc_topic:
        count += 1
        ha_sum = 0
        LDAVector_dict[con_list[count]] = i.tolist()
        # print(i.tolist())
        for j in i.tolist():
            ha_sum += j * math.log(j)
        ha_dict[con_list[count]] = 0 - ha_sum
    value_13 = []
    value_14 = []
    HAB_list = []
    HBA_list = []
    for i in g:
        value_13.append(ha_dict[i])
    for i in h:
        value_14.append(ha_dict[i])
    for i, j in enumerate(h):
        A = g[i]  # A概念
        B = j  # B概念
        HAB_list.append(ha_dict[A] + KL(LDAVector_dict[A], LDAVector_dict[B]))
        HBA_list.append(ha_dict[B] + KL(LDAVector_dict[B], LDAVector_dict[A]))
    df = pd.DataFrame({'13': value_13, '14': value_14, '15': HAB_list, '16': HBA_list})
    df.to_excel('E:/实验相关代码/m3代码/m3_code/data/M2/13-16Features.xlsx')


