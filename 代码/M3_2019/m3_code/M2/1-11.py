# 特征值1-11
# !pip install pymediawiki
import json

import pandas as pd
import re
import numpy as np
# import nltk

# nltk.download('brown')
# nltk.download('punkt')
from mediawiki import MediaWiki

wikipedia = MediaWiki()
from textblob import TextBlob

list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
list9 = []
list_10 = []
list_11 = []
list_12 = []
conceptLinksBackLinksLength_dict = {}

def compute_jaccard_similarity_score(x, y):
    """
     Jaccard Similarity J (A,B) = | Intersection (A,B) | /
     | Union (A,B) |
     """
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)

def read_data_dict():
    # url = r"C:\LK\test.xlsx"
    url = 'E:/实验相关代码/m3代码/m3_code/data/concept_pairs.xlsx'
    df = pd.read_excel(url)
    A_list = df['A'].tolist()
    B_list = df['B'].tolist()

    con_list = list(dict.fromkeys(A_list + B_list))
    con_dict = {}
    for i in con_list:
        try:
            p_wiki = wikipedia.page(i)  # 概念对象
            content_str = p_wiki.content  # 概念正文
        except:
            pass

        con_dict[i.replace('_', ' ')] = content_str
    A_list = [w.replace('_', ' ') for w in A_list]
    B_list = [w.replace('_', ' ') for w in B_list]

    # 文件名暂时不动
    with open('E:/实验相关代码/m3代码/m3_code/data/M2/M2概念对-test.json', 'w') as outfile:
        json.dump(con_dict, outfile)

    print('概念content抓取完成')
    return A_list, B_list, con_dict

# 1.概念B是否出现在概念a的词条中
# 2.概念A是否出现在概念B的词条中
def cal1_2(concept, content):
    if " " + concept.lower() in content or concept + " " in content:
        return 1
    return 0


# 3.概念B是否出现在概念a的词条中的次数(大小写都要)
# 4.概念A是否出现在概念B的词条中的次数
def cal3_4(concept, content):
    return content.count(concept) + content.count(concept.lower())
# 5.概念B是否出现在概念a的词条中的第一行(第1行限制前400个字符)
# 6.概念A是否出现在概念B的词条中的次数
def cal5_6(concept, content):
    first_line = content[:500]

    if concept in first_line or concept.lower() in first_line:
        return 1
    return 0


# 7.概念B是否出现在概念A中
def cal7(conceptA, conceptB):
    if conceptB in conceptA or conceptB.lower() in conceptA:
        return 1
    return 0


# 8.统计概念a的单词数
# 9.统计概念b的单词数
def cal8_9(conceptContent):
    # using regex (findall()) to count words in string
    res = len(re.findall(r'\w+', conceptContent))
    return res


# 10. 概念a和概念b词条页面的Jaccard similarity
def cal_10(pageA, pageB):
    segWordA_list = re.findall(r'\w+', pageA) # 概念正文分词

    segWordB_list = re.findall(r'\w+', pageB) # 概念正文分词

    return compute_jaccard_similarity_score(np.array(segWordA_list), np.array(segWordB_list))


# 11. 概念a和概念b词条页面(名词)的Jaccard similarity
def cal_11(pageA, pageB):
    txt_list = re.findall(r'\w+', pageA)  # 概念正文分词
    txt = ' '.join([str(elem) for elem in txt_list])
    blob = TextBlob(txt)
    segWordA_list = blob.noun_phrases
    txt_list = re.findall(r'\w+', pageB)  # 概念正文分词
    txt = ' '.join([str(elem) for elem in txt_list])
    blob = TextBlob(txt)
    segWordB_list = blob.noun_phrases
    return compute_jaccard_similarity_score(np.array(segWordA_list), np.array(segWordB_list))

    pass

# 返回概念A集合、概念B集合、概念-词条的字典
if __name__ == "__main__":
    g, h, con_dict = read_data_dict()
    count = 0
    for i, j in enumerate(h):
        A = g[i]
        B = j
        A_page = con_dict[A]
        B_page = con_dict[B]
        # print(cal1_2(B, A_page))
        list1.append(cal1_2(B, A_page))
        # print('特征值1计算完成')
        list2.append(cal1_2(A, B_page))
        # print('特征值2计算完成')
        list3.append(cal3_4(B, A_page))
        # print('特征值3计算完成')
        list4.append(cal3_4(A, B_page))
        # print('特征值4计算完成')
        list5.append(cal5_6(B, A_page))
        # print('特征值5计算完成')
        list6.append(cal5_6(A, B_page))
        # print('特征6计算完成')
        list7.append(cal7(A, B))
        # print('特征值7计算完成')
        list8.append(cal8_9(A_page))
        # print('特征值8计算完成')
        list9.append(cal8_9(B_page))
        # print('特征值9计算完成')
        list_10.append(cal_10(A_page, B_page))
        # print('特征值10计算完成')
        list_11.append(cal_11(A_page, B_page))
        # print('特征值11计算完成')
        # 12.RefD代码在另一个文件
        count += 1
        print(count)
    # 错误效应
    for i in range(len(list1)):
        print(list1[i], list2[i], list3[i], list4[i], list5[i], list6[i], list7[i],
              list8[i], list9[i], list_10[i], list_11[i])
    # 存储到excel
    df = pd.DataFrame(
        {'1': list1, '2': list2, '3': list3, '4': list4, '5': list5,
         '6': list6, '7': list7, '8': list8, '9': list9,
         '10': list_10, '11': list_11})
    df.to_excel('E:/实验相关代码/m3代码/m3_code/data/M2/M2的11个特征值.xlsx')