import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

# 设定模型和设备
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

def get_embedding(text):
    # 分词并将输入数据移到适当的设备
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成嵌入
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float().to(device)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    # 转换为numpy数组并返回
    return mean_embeddings.cpu().numpy()

# 接下来，读取CSV，计算嵌入，保存数据
def process_and_save_embeddings(csv_path, output_csv_path):
    df = pd.read_csv(csv_path)

    # 在DataFrame的最左边添加一个ID列，其值从0开始递增
    df.insert(0, 'node_id', range(0, len(df)))

    # 为每个概念描述计算嵌入
    df['node_emb'] = df['description'].apply(lambda x: get_embedding(x)[0])

    # 将嵌入向量转换为字符串格式以便保存
    df['node_emb'] = df['node_emb'].apply(lambda x: ','.join(map(str, x)))

    df['lm_emb'] = df['node_emb']

    # 保存修改后的DataFrame
    df.to_csv(output_csv_path, index=False)

# 调用函数
process_and_save_embeddings('大学课程(概念+描述).csv', '大学课程(id+name+lm+node).csv')
