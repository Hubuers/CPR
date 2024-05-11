import re
import requests
import pandas as pd
import csv
from mediawiki import MediaWiki


def get_wiki_page_first_400_words(title):
    wikipedia = MediaWiki()

    p_wiki = wikipedia.page(title)

    # 获取页面内容
    content_str = p_wiki.content

    # 使用正则表达式将文本拆分为单词
    words = content_str.split()

    # 截取前 400 个单词
    first_400_words = ' '.join(words[:400])

    return first_400_words


# 创建概念对应的维基百科描述文件
def generate_new_dataset(input_file, output_file):
    # 读取输入CSV文件，假设CSV文件中有一列名为 '概念'
    df = pd.read_csv(input_file, header=None, names=['概念'], encoding='gbk')

    # 打开输出CSV文件，准备写入概念名称和描述
    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        # 写入CSV文件的表头
        csv_writer = csv.writer(f_out)
        csv_writer.writerow(['概念', '概念的维基百科前400个单词'])

        # 遍历每一行，获取概念的前400个单词，并将结果写入新文件中
        for index, row in df.iterrows():
            concept = row['概念']
            try:
                concept_wiki = get_wiki_page_first_400_words(concept)
                print(concept_wiki)
                # 写入CSV文件
                csv_writer.writerow([concept, concept_wiki])
            except:
                print(f"Error: cannot retrieve information for {concept}")


import pandas as pd

def process_dataset(concept_pairs_file, concept_wiki_file, output_file):
    # 读取概念对文件
    concept_pairs_df = pd.read_excel(concept_pairs_file)

    # 读取概念描述文件，指定列名
    concept_wiki_df = pd.read_csv(concept_wiki_file, encoding='utf-8')

    # 将概念描述映射到概念对
    concept_pairs_df = concept_pairs_df.merge(concept_wiki_df, left_on='A', right_on='概念', how='left')
    concept_pairs_df = concept_pairs_df.rename(columns={'概念的维基百科前400个单词': 'A_indexWords'}).drop(['概念'], axis=1)
    concept_pairs_df = concept_pairs_df.merge(concept_wiki_df, left_on='B', right_on='概念', how='left')
    concept_pairs_df = concept_pairs_df.rename(columns={'概念的维基百科前400个单词': 'B_indexWords'}).drop(['概念'], axis=1)

    # 存储结果到新文件，只包含概念A的描述和概念B的描述
    concept_pairs_df[['A_indexWords', 'B_indexWords']].to_excel(output_file, index=False)



if __name__ == '__main__':
    print(get_wiki_page_first_400_words('Assembly language'))

    # 创建概念对应的维基百科描述文件
    # generate_new_dataset('E:/实验相关代码/m3代码/m3_code/data/University_Course_Concept.csv', 'E:/实验相关代码/m3代码/m3_code/data/concept_wiki.csv')

    # 调用函数并传入文件路径
    # process_dataset('E:/实验相关代码/m3代码/m3_code/data/concept_pairs.xlsx', 'E:/实验相关代码/m3代码/m3_code/data/concept_wiki.csv', 'E:/实验相关代码/m3代码/m3_code/data/概念前400单词.xlsx')
