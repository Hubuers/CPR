import torch
import torch.nn as nn
import pandas as pd
import math
import torch.nn.functional as F
from math import sqrt

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self.fc_out = nn.Linear(dim_v, dim_in)  # Transform back to input dimension
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

        # Additional components for non-linearity and residual connection
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_in),
            nn.ReLU(),
            nn.Linear(2 * dim_in, dim_in),
        )
        self.layer_norm1 = nn.LayerNorm(dim_in)
        self.layer_norm2 = nn.LayerNorm(dim_in)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)

        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)

        # Apply the output linear transformation
        att = self.fc_out(att)

        # Incorporate the residual connection and apply the first layer normalization
        x = self.layer_norm1(x + att)

        # Apply the feed-forward network
        x = self.feed_forward(x)

        # Incorporate the residual connection again and apply the second layer normalization
        x = self.layer_norm2(x + att)

        return x


# class TransformerBlock(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(TransformerBlock, self).__init__()
#         self.attention = MultiHeadSelfAttention(embed_size, heads)
#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, 2 * embed_size),
#             nn.ReLU(),
#             nn.Linear(2 * embed_size, embed_size),
#         )
#
#     def forward(self, value, key, query, mask):
#         attention = self.attention(value, key, query, mask)
#         x = self.norm1(attention + query)
#         forward = self.feed_forward(x)
#         out = self.norm2(forward + x)
#         return out

# 读取CSV文件并转换嵌入向量
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    embeddings = []
    names = df['node_name'].tolist()
    for emb_str in df['embeddings']:
        emb = torch.tensor([float(e) for e in emb_str.split(',')], dtype=torch.float32)
        embeddings.append(emb)
    embeddings = torch.stack(embeddings)
    return names, embeddings

def update_embeddings(embeddings, model):
    # 假设 embeddings 的形状为 (num_nodes, embed_size)
    # 添加一个维度来表示序列长度，这里设置为 1
    embeddings = embeddings.unsqueeze(1)  # 新形状为 (num_nodes, 1, embed_size)
    updated_embeddings = model(embeddings)
    # 删除添加的序列长度维度，以便恢复到原始形状 (num_nodes, embed_size)
    updated_embeddings = updated_embeddings.squeeze(1)
    return updated_embeddings


def save_updated_embeddings(names, updated_embeddings, output_csv_path):
    updated_embeddings_list = updated_embeddings.tolist()
    updated_embeddings_str = [','.join(map(str, emb)) for emb in updated_embeddings_list]
    df_updated = pd.DataFrame({'node_name': names, 'embeddings': updated_embeddings_str})
    df_updated.to_csv(output_csv_path, index=False)

# 路径设置
input_csv_path = 'final_node_embeddings.csv'
output_csv_path = 'path_to_your_output.csv'

# 步骤1: 加载数据
names, embeddings = load_embeddings(input_csv_path)

# 步骤2: 定义模型
dim_in = embeddings.size(1)  # 假设所有嵌入的大小相同
model = MultiHeadSelfAttention(dim_in, dim_k=dim_in, dim_v=dim_in, num_heads=8)

# 步骤3: 更新嵌入
updated_embeddings = update_embeddings(embeddings, model)

# 步骤4: 保存更新后的嵌入
save_updated_embeddings(names, updated_embeddings, output_csv_path)
