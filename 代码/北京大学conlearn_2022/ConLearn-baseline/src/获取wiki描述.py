import pandas as pd
from mediawiki import MediaWiki
import certifi
import requests
import ssl
import os

print(ssl.OPENSSL_VERSION)
# 确保requests使用最新的CA证书
requests.get('https://en.wikipedia.org', verify=certifi.where())

def get_wiki_page_first_200_words(title):
    wikipedia = MediaWiki()
    try:
        p_wiki = wikipedia.page(title)
        print(f"成功获取'{title}'的描述。")
        content_str = p_wiki.content
        words = content_str.split()
        first_200_words = ' '.join(words[:200])
        return first_200_words
    except:
        print(f"未找到'{title}'的描述。")
        return "No description available"

def update_progress(concept, description, filename='进度保存.csv'):
    # 转义描述中的双引号
    description_escaped = description.replace('"', '""')
    # 将转义后的描述包裹在双引号中，并使用逗号作为分隔符写入文件
    with open(filename, 'a', encoding='utf-8', newline='') as file:
        file.write(f'"{concept}","{description_escaped}"\n')


# 读取CSV文件
df = pd.read_csv('大学课程(概念名).csv', header=None)

# 检查进度文件是否存在，如果存在，读取已处理的概念
progress_file = '进度保存.csv'
if os.path.exists(progress_file):
    processed_df = pd.read_csv(progress_file, header=None, encoding='utf-8', quotechar='"')
    processed_concepts = set(processed_df[0])
else:
    processed_concepts = set()

descriptions = []
# 对于每个概念获取其维基百科描述，并更新进度
for concept in df[0]:
    if concept not in processed_concepts:
        description = get_wiki_page_first_200_words(concept)
        descriptions.append(description)
        update_progress(concept, description)
    else:
        # 如果已经处理过，直接从进度文件中取得描述
        description = processed_df[processed_df[0] == concept].iloc[0, 1]
        descriptions.append(description)

# 将描述添加为新列
df[1] = descriptions

# 写入新的CSV文件
df.to_csv('大学课程(概念+描述).csv', index=False, header=False, encoding='utf-8')

print("描述已写入!")
