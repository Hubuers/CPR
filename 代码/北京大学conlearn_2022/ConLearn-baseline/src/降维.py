# 768维bert嵌入降维至300维  lm_emb

import pandas as pd
import torch
import torch.nn as nn


class LinearDimReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDimReducer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def load_embeddings(csv_file):
    # 正确加载lm_emb列并将其转换为张量
    df = pd.read_csv(csv_file)
    embeddings = [torch.tensor([float(i) for i in emb.split(',')], dtype=torch.float) for emb in df['lm_emb']]
    embeddings = torch.stack(embeddings)
    return df['node_id'], df['node_name'], embeddings, df['node_emb']


def save_embeddings(node_ids, node_names, reduced_embeddings, node_embs, output_file):
    embeddings_list = reduced_embeddings.tolist()
    embeddings_str = [','.join(map(str, emb)) for emb in embeddings_list]
    df = pd.DataFrame({
        'node_id': node_ids,
        'node_name': node_names,
        'lm_emb': embeddings_str,
        'node_emb': node_embs,
    })
    df.to_csv(output_file, index=False)


def reduce_embeddings(input_file_path, output_file_path, input_dim=768, output_dim=300):
    # 加载数据
    node_ids, node_names,  embeddings, node_embs = load_embeddings(input_file_path)

    # 初始化模型并设置为评估模式
    model = LinearDimReducer(input_dim, output_dim)
    model.eval()  # 设置为评估模式，关闭dropout等

    # 应用模型降维
    with torch.no_grad():  # 不需要计算梯度
        reduced_embeddings = model(embeddings)

    # 保存降维后的嵌入向量及其他信息
    save_embeddings(node_ids, node_names, reduced_embeddings, node_embs, output_file_path)
    return output_file_path

# 示例使用
if __name__ == '__main__':
    input_file = r'C:\Users\Windows 10\Desktop\my_model\test_model\gated GNN\data\lectureBankData\lectureBank(id+name+lm+node).csv'  # 输入文件的路径
    output_file = '../Dataset/lectureBank/lectureBank(id+name+lm+node).csv'  # 输出文件的路径
    reduced_file = reduce_embeddings(input_file, output_file)
    print(f"Reduced embeddings are saved to {reduced_file}")
