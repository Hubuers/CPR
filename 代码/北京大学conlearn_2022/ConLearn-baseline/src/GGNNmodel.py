import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd
import numpy as np

# 将字符串形式的向量转换为浮点数列表
def preprocess_embeddings(embeddings_str):
    embeddings = [float(x) for x in embeddings_str.strip('[]').split(',')]
    return torch.tensor(embeddings, dtype=torch.float)

# 获取GGNN邻接矩阵A，A_in，A_out
def create_adjacency_matrix(edges_path, num_nodes, device='cuda'):
    edges_df = pd.read_csv(edges_path)

    A_in = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_out = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 记录每个节点出度的数组
    out_degree = np.zeros(num_nodes, dtype=np.float32)

    # 遍历每条边，计算出度
    for _, row in edges_df.iterrows():
        src = int(row['source_node_id'])
        out_degree[src] += 1

    # 遍历每条边，填充邻接矩阵
    for _, row in edges_df.iterrows():
        src = int(row['source_node_id'])
        dst = int(row['target_node_id'])

        # 归一化出边权重
        A_out[src, dst] = 1.0 / out_degree[src] if out_degree[src] > 0 else 0

        # 进边不需要归一化
        A_in[dst, src] = 1

    # 将numpy数组转换为torch张量，并放到指定的设备上
    A_in_tensor = torch.tensor(A_in, device=device)
    A_out_tensor = torch.tensor(A_out, device=device)

    A = torch.cat([A_in_tensor.unsqueeze(-1), A_out_tensor.unsqueeze(-1)], dim=-1)

    print(A_in.shape)
    print(A_out.shape)
    print(A.shape)

    return A_in_tensor, A_out_tensor, A

class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Propogator(nn.Module):

    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()
        self.state_dim = state_dim

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        # 从三维邻接矩阵中提取A_in和A_out，假设A是[num_nodes, num_nodes, 2]
        A_in = A[:, :, 0]  # 获取所有入边，形状应为[89, 89]
        A_out = A[:, :, 1]  # 获取所有出边，形状应为[89, 89]


        a_in = torch.matmul(A_in, state_in)
        a_out = torch.matmul(A_out, state_in)
        a = torch.cat((a_in, a_out, state_in), 1)  # 沿特征维度拼接

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in.squeeze(0), a_out.squeeze(0), r * state_cur), 1)
        h_hat = self.transform(joined_input)

        output = (1 - z) * state_cur + z * h_hat
        return output


class GGNN(nn.Module):
    def __init__(self, opt):
        super(GGNN, self).__init__()
        self.state_dim = opt['state_dim']
        self.annotation_dim = opt['annotation_dim']
        self.n_edge_types = opt['n_edge_types']
        self.n_node = opt['n_node']
        self.n_steps = opt['n_steps']
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Initialize incoming and outgoing edge embeddings
        for i in range(self.n_edge_types):
            setattr(self, "in_{}".format(i), nn.Linear(self.state_dim, self.state_dim))
            setattr(self, "out_{}".format(i), nn.Linear(self.state_dim, self.state_dim))

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, state_cur, A):
        for i_step in range(self.n_steps):
            state_in = state_out = state_cur
            for i in range(self.n_edge_types):
                in_transform = getattr(self, "in_{}".format(i))
                out_transform = getattr(self, "out_{}".format(i))
                state_in = in_transform(state_in)
                state_out = out_transform(state_out)

            state_cur = self.propogator(state_in, state_out, state_cur, A)

        return state_cur   #直接用prop_state（更新后的节点状态）作为输出，因为它现在包含了每个节点的特征向量。

# 加载数据集
data_path = '../data/MOOC/concept_pair.csv'     #原版完整数据集，概念对
node_path = '../data/MOOC/MOOC(id+name+lm+node).csv'   #节点数据集
edges_path= '../data/MOOC/edges_dataset.csv'    #通过训练集得到的边数据集
# 读取节点数据
nodes_df = pd.read_csv(node_path)

# 获取节点数量
num_nodes = nodes_df['node_id'].nunique()

print(nodes_df['lm_emb'].dtype)
# 应用预处理
nodes_df['lm_emb'] = nodes_df['lm_emb'].apply(preprocess_embeddings)        #文本嵌入
nodes_df['node_emb'] = nodes_df['node_emb'].apply(preprocess_embeddings)

# 读取边数据
edges_df = pd.read_csv(edges_path)
# 假设node_id是连续的并且从0开始
num_nodes = nodes_df['node_id'].nunique()
# 构建边的COO表示（这需要边的索引是从0开始的整数）
edge_index = torch.tensor([edges_df['source_node_id'].values, edges_df['target_node_id'].values], dtype=torch.long)
# 节点特征是文本嵌入
node_features = torch.stack(nodes_df['lm_emb'].tolist())
# 创建PyG的Data对象
data = Data(x=node_features, edge_index=edge_index)

def save_node_embeddings(node_ids, node_names, embeddings, file_path):
    # 将Tensor转换为列表形式的字符串
    embeddings_list = embeddings.detach().cpu().numpy().tolist()
    embeddings_str = [",".join(map(str, emb)) for emb in embeddings_list]

    # 创建DataFrame并保存到CSV
    df = pd.DataFrame({
        'node_id': node_ids,
        'node_name': node_names,
        'embeddings': embeddings_str
    })
    df.to_csv(file_path, index=False)
    print(f"Embeddings saved to {file_path}")

opt = {
        'state_dim': 1024,
        'annotation_dim': 0,
        'n_edge_types': 1,
        'n_node': num_nodes,
        'n_steps': 2  # Number of propagation steps
    }
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GGNN(opt).to(device)
data = data.to(device)  # data是图数据
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
A_in_tensor, A_out_tensor, A = create_adjacency_matrix(edges_path, num_nodes, 'cuda')

criterion = nn.BCEWithLogitsLoss()  # 二分类任务的损失函数
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    state_cur = model(data.x, A)  # 更新节点向量

    # 反向传播和优化
    # 假设我们已经有了一个目标值和损失计算
    # 例如: loss = criterion(state_cur, targets)
    # loss.backward()
    optimizer.step()

    if epoch == num_epochs - 1:
        # 只在最后一个epoch保存节点向量
        save_node_embeddings(nodes_df['node_id'].tolist(), nodes_df['node_name'].tolist(), state_cur, "final_node_embeddings.csv")



